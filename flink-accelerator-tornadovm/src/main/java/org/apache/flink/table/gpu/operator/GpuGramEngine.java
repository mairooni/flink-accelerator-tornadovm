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

import org.apache.flink.table.gpu.codegen.GpuKernelSource;
import org.apache.flink.table.gpu.codegen.GpuValueType;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;

import java.lang.reflect.Method;

/**
 * A Gram matrix over a generated feature map: {@code A = f(X)} on the device, then {@code A'A} by
 * cuBLAS, in one task graph.
 *
 * <h2>How this differs from {@link FeatureGramEngine}, which it generalises</h2>
 *
 * <p>That engine hardcodes {@code sin}. It was written to measure whether the composition is worth
 * anything — §T15 says it is, by 12.09x at sixty-four features — and it answers no query, because
 * the feature map a query writes is whatever the query wrote. Here the feature kernel is
 * <em>generated</em> from the IR, by the same {@link
 * org.apache.flink.table.gpu.codegen.AccelKernelGenerator} that serves every Calc.
 *
 * <p>Which turned out to need nothing new. The generator already has a packed output mode — every
 * computed column into one buffer, {@code packedStride} rows apart, column-major — added so that a
 * contraction over <em>k</em> columns is one task rather than <em>k</em>. Column-major with a
 * stride of the row count is exactly the matrix cuBLAS reads, so the generated kernel writes {@code
 * A} in place and no packing pass exists.
 *
 * <h2>Why {@code cublasDgemm} and not {@code cublasSgemm}</h2>
 *
 * <p>{@link FeatureGramEngine} is FP32 throughout and accumulates its per-batch results in double
 * to compensate. That was never written down as a contract, and a Gram matrix is the left-hand side
 * of a normal equation — the one place a narrowed contraction actually propagates. The generator
 * emits {@code double} for a {@code DOUBLE} expression and the binding ships {@code cublasDgemm},
 * so the whole path is FP64 and the question does not arise. It costs throughput a consumer card
 * would rather keep, and against §T15's margin that is affordable.
 *
 * <h2>The tail batch</h2>
 *
 * <p>{@code k} — the contraction length — is captured when the graph is built, like every other
 * library task here. A partition's last batch is short, so it gets a graph of its own with the
 * smaller {@code k}; the leading dimension stays the full batch size, because that is the stride
 * the kernel wrote at. Nothing has to be zeroed: rows past {@code k} are simply not read, which is
 * the difference between this and {@link FeatureGramEngine}'s zero-padding — that engine pads
 * because {@code sin(0)} is zero, and a generated feature map has no such guarantee.
 */
public final class GpuGramEngine implements AutoCloseable {

    private final GpuKernelSource kernel;
    private final int features;
    private final int batchSize;

    /** One staging buffer per input column the feature expressions read. */
    private Object[] inputs;

    /** {@code A}: the features, {@code batchSize x features}, column-major. */
    private DoubleArray packed;

    /** {@code A'A} for one batch, {@code features x features}. */
    private DoubleArray gram;

    /** The live row count, which the kernel reads so a short batch launches short. */
    private IntArray rows;

    /** Running total across batches. */
    private final double[] total;

    private GeneratedKernel generated;
    private Method entry;
    private Object[] kernelArgs;

    private TornadoExecutionPlan plan;
    private WorkerGrid1D grid;

    private TornadoExecutionPlan tailPlan;
    private WorkerGrid1D tailGrid;
    private int tailLength = -1;

    private long batches;
    private long rowCount;

    public GpuGramEngine(GpuKernelSource kernel, int features, int batchSize) {
        this.kernel = kernel;
        this.features = features;
        this.batchSize = batchSize;
        this.total = new double[features * features];
    }

    public void open() throws Exception {
        inputs = new Object[kernel.inputFieldIndexes().length];
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = GeneratedKernel.allocate(kernel.inputTypes()[i], batchSize);
        }
        packed = new DoubleArray(batchSize * features);
        packed.init(0.0);
        gram = new DoubleArray(features * features);
        gram.init(0.0);
        rows = new IntArray(1);
        rows.set(0, batchSize);

        generated = new GeneratedKernel(kernel);
        entry = generated.compile();

        // Inputs, the packed output, the live row count. No mask and no validity words: a Gram
        // matrix is refused over a filter or a nullable column, so neither can be present.
        kernelArgs = new Object[inputs.length + 2];
        System.arraycopy(inputs, 0, kernelArgs, 0, inputs.length);
        kernelArgs[inputs.length] = packed;
        kernelArgs[inputs.length + 1] = rows;

        plan = buildPlan(batchSize, "gram");
        grid = gridOf(plan, "gram");
    }

    private TornadoExecutionPlan buildPlan(int contraction, String name) {
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, inputs)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, rows)
                        .task("features", entry, kernelArgs)
                        // C = A' * A. A is batchSize x features held column-major, so its leading
                        // dimension is the batch size and the transpose costs nothing: cuBLAS
                        // reads the buffer the feature kernel has just written, in place.
                        .libraryTask(
                                "gemm",
                                CuBlas::cublasDgemm,
                                CuBlasOperation.CUBLAS_OP_T.operation(),
                                CuBlasOperation.CUBLAS_OP_N.operation(),
                                features,
                                features,
                                contraction,
                                1.0,
                                packed,
                                batchSize,
                                packed,
                                batchSize,
                                0.0,
                                gram,
                                features)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, gram);
        return new TornadoExecutionPlan(graph.snapshot());
    }

    private WorkerGrid1D gridOf(TornadoExecutionPlan target, String name) {
        // Stated rather than inferred, for the reason recorded in GeneratedKernelEngine: since
        // M2.5 the loop bound is read from a buffer, so inference gives up and emits a sequential
        // loop -- silently, with correct results about a thousand times slower.
        WorkerGrid1D worker = new WorkerGrid1D(batchSize);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(name + ".features", worker);
        target.withGridScheduler(scheduler);
        return worker;
    }

    /** One staged input column, for the gather to write into. */
    public Object inputColumn(int column) {
        return inputs[column];
    }

    /** Runs one batch of {@code count} rows and folds its Gram matrix into the total. */
    public void execute(int count) throws Exception {
        rows.set(0, count);
        if (count == batchSize) {
            grid.setGlobalWork(count, 1, 1);
            plan.execute();
        } else {
            if (tailPlan == null || tailLength != count) {
                closeTail();
                tailPlan = buildPlan(count, "gram-tail");
                tailGrid = gridOf(tailPlan, "gram-tail");
                tailLength = count;
            }
            tailGrid.setGlobalWork(count, 1, 1);
            tailPlan.execute();
        }
        for (int i = 0; i < total.length; i++) {
            total[i] += gram.get(i);
        }
        batches++;
        rowCount += count;
    }

    /** Entry {@code (i, j)} of the accumulated matrix. Symmetric, so the order does not matter. */
    public double entry(int i, int j) {
        return total[j * features + i];
    }

    public long batchCount() {
        return batches;
    }

    public long rowsSeen() {
        return rowCount;
    }

    private void closeTail() throws Exception {
        if (tailPlan != null) {
            tailPlan.close();
            tailPlan = null;
        }
    }

    @Override
    public void close() throws Exception {
        if (plan != null) {
            plan.close();
            plan = null;
        }
        closeTail();
        if (generated != null) {
            generated.close();
            generated = null;
        }
    }

    /** Which staging width each input column is held at, for the gather to match. */
    public GpuValueType[] inputTypes() {
        return kernel.inputTypes();
    }
}
