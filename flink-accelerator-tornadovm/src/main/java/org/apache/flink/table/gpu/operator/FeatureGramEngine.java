/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.table.gpu.operator;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;

import java.util.Arrays;

/**
 * A generated kernel and a cuBLAS call in one task graph, over one set of device buffers.
 *
 * <p>This is the shape the offload operator cannot reach. A {@code Calc} is elementwise, so the
 * best a library can offer it is BLAS-1, which loses to a kernel that fuses the whole expression
 * into a single pass. Put an aggregate behind the Calc and that changes: the aggregate contracts
 * over rows, and a contraction is what GEMM exists for.
 *
 * <pre>{@code
 * SELECT SUM(f0*f0), SUM(f0*f1), ...            <- aggregate: contracts over rows, GEMM
 * FROM (SELECT SIN(c0) AS f0, SIN(c1) AS f1, ...  <- Calc: elementwise, generated kernel
 *       FROM t)
 * }</pre>
 *
 * <p>Both halves are real work and neither tool does the other's job well. Computing the feature
 * map is transcendental arithmetic per element, which is where a generated kernel wins and where a
 * BLAS has no entry point at all. Contracting the result is a rank-{@code k} update, where cuBLAS
 * tiles and reuses through shared memory and a naive generated kernel reloads a column per output.
 *
 * <p>What makes this worth doing on a device at all is that the intermediate never lands: the
 * features are written to a device buffer that cuBLAS reads directly. On the host the same pipeline
 * would materialise a {@code rows x columns} intermediate and read it back a second time.
 *
 * <h2>Batching, and why the tail is padded</h2>
 *
 * <p>The graph is built once, for a fixed number of rows, because a {@link TornadoExecutionPlan}
 * fixes buffer sizes. A short final batch is therefore zero-filled rather than resized. That is
 * exact, not an approximation: a zero row contributes zero to every entry of {@code A'A}, and
 * {@code sin(0)} is zero, so the padding survives the feature map as padding.
 *
 * <h2>Float, not double</h2>
 *
 * <p>The binding exposes no FP64 GEMM, so this path is FP32 throughout while the {@code Calc}
 * offload stages doubles. That is a real difference in kind, not a detail: a Gram matrix
 * accumulates {@code rows} products per entry, and FP32 loses digits there in a way the Calc path
 * was specifically designed to avoid. Callers get partial sums as doubles and should compare them
 * against a double-precision reference before believing them.
 */
public final class FeatureGramEngine implements AutoCloseable {

    private final int columns;
    private final int batchSize;

    /** Column-major staging, {@code batchSize} rows by {@code columns}, reused every batch. */
    private final FloatArray data;

    /** The per-batch result, {@code columns x columns}, overwritten every execution. */
    private final FloatArray gram;

    /** Running total across batches, in double, because the per-batch results are FP32. */
    private final double[] total;

    private TornadoExecutionPlan plan;
    private int staged;
    private long batches;
    private long rows;

    public FeatureGramEngine(int columns, int batchSize) {
        this.columns = columns;
        this.batchSize = batchSize;
        this.data = new FloatArray(batchSize * columns);
        this.gram = new FloatArray(columns * columns);
        this.total = new double[columns * columns];
    }

    public void open() {
        // C = A' * A. A is batchSize x columns held column-major, so its leading dimension is the
        // row count and the transpose costs nothing -- cuBLAS reads the buffer the feature kernel
        // has just written, in place, without a copy.
        TaskGraph graph =
                new TaskGraph("feature-gram")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, data)
                        .task("features", FeatureGramEngine::features, data, batchSize, columns)
                        .libraryTask(
                                "gemm",
                                CuBlas::cublasSgemm,
                                CuBlasOperation.CUBLAS_OP_T.operation(),
                                CuBlasOperation.CUBLAS_OP_N.operation(),
                                columns,
                                columns,
                                batchSize,
                                1.0f,
                                data,
                                batchSize,
                                data,
                                batchSize,
                                0.0f,
                                gram,
                                columns)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, gram);
        plan = new TornadoExecutionPlan(graph.snapshot());
    }

    /** Stages one value. Column-major, so a column's rows are contiguous for cuBLAS. */
    public void set(int column, double value) {
        data.set(column * batchSize + staged, (float) value);
    }

    /** Ends a row. Returns true if the batch filled and was executed. */
    public boolean rowComplete() {
        staged++;
        rows++;
        if (staged == batchSize) {
            execute();
            return true;
        }
        return false;
    }

    /** Runs whatever is staged, padding the tail with zeros. Safe to call on an empty batch. */
    public void flush() {
        if (staged > 0) {
            execute();
        }
    }

    private void execute() {
        for (int c = 0; c < columns; c++) {
            for (int r = staged; r < batchSize; r++) {
                data.set(c * batchSize + r, 0.0f);
            }
        }
        try {
            plan.execute();
        } catch (Exception e) {
            throw new IllegalStateException("feature-gram execution failed", e);
        }
        for (int i = 0; i < total.length; i++) {
            total[i] += gram.get(i);
        }
        batches++;
        staged = 0;
    }

    /**
     * The feature map: elementwise, one thread per row.
     *
     * <p>Parallel over rows rather than over the {@code rows x columns} elements because the inner
     * loop is a handful of iterations and the row count is the dimension with enough work to fill a
     * device.
     */
    public static void features(FloatArray data, int rows, int columns) {
        for (@Parallel int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                int i = c * rows + r;
                data.set(i, TornadoMath.sin(data.get(i)));
            }
        }
    }

    /** The accumulated {@code columns x columns} matrix, column-major. */
    public double[] total() {
        return Arrays.copyOf(total, total.length);
    }

    public long batchCount() {
        return batches;
    }

    public long rowCount() {
        return rows;
    }

    @Override
    public void close() throws Exception {
        if (plan != null) {
            plan.close();
            plan = null;
        }
    }
}
