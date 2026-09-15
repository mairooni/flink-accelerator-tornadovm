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

package org.apache.flink.table.gpu;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;

import java.util.Random;

/**
 * Where a library call beats a generated kernel, and where it does not.
 *
 * <p>Every kernel this project generates so far is elementwise: one output per row, no index shared
 * between rows. That shape can only ever reach BLAS-1, where a library has nothing to offer over a
 * kernel that fuses the whole expression into one pass. The covariance matrix is the smallest
 * realistic query that escapes it.
 *
 * <pre>{@code
 * SELECT SUM(c1*c1), SUM(c1*c2), ..., SUM(cd*cd) FROM t   -- centred: the Gram matrix A'A
 * }</pre>
 *
 * <p>The contraction is over <em>rows</em>, which is what an aggregate does and a Calc cannot, and
 * the arithmetic intensity is {@code d/2} flops per byte in FP32: linear in the table's width. At
 * {@code d = 4} it is memory-bound and a library is pointless; the interesting question is where
 * along {@code d} that stops being true, which is what this sweeps.
 *
 * <h2>What makes it hybrid</h2>
 *
 * <p>Centring the columns is elementwise and belongs in a generated kernel. The contraction is a
 * GEMM and belongs in cuBLAS. The point of the {@link TaskGraph} here is that they are the same
 * graph over the same device buffers: the centred data is never copied back to the host between
 * them.
 *
 * <pre>{@code
 * .task("centre", ...)                       generated, @Parallel over N
 * .libraryTask("gemm", CuBlas::cublasSgemm)  cuBLAS, contracts over N
 * }</pre>
 *
 * <h2>Why float</h2>
 *
 * <p>The offload operator stages doubles, because {@code GpuCalcSpec} chose exactness. The cuBLAS
 * binding exposes no FP64 GEMM -- Sgemm, SgemmTF32, batched Sgemm and the FP16/BF16 GemmEx family,
 * nothing wider -- so this harness is FP32 throughout and reports the error against a double
 * reference rather than hiding it. Accumulating hundreds of thousands of products in FP32 is
 * exactly where a covariance computation loses digits, so that number is part of the result, not a
 * footnote to it.
 *
 * <p>Run with TornadoVM's argfile, as a plain Java main:
 *
 * <pre>{@code
 * java @tornado.args -cp target/classes:<deps> \
 *     org.apache.flink.table.gpu.GramMatrixSweep [rows]
 * }</pre>
 */
public final class GramMatrixSweep {

    /** Table widths to sweep. Intensity is d/2 flops per byte, so this spans both sides of it. */
    private static final int[] WIDTHS = {4, 8, 16, 32, 64, 128};

    private static final int WARMUP = 3;
    private static final int RUNS = 10;

    private GramMatrixSweep() {}

    public static void main(String[] args) {
        int rows = args.length > 0 ? Integer.parseInt(args[0]) : 262_144;

        System.out.printf("rows=%,d  (batch size of the offload operator)%n%n", rows);
        System.out.printf(
                "%5s %12s %12s %12s %10s %12s %12s%n",
                "d", "intensity", "generated", "cublas", "speedup", "gen err", "blas err");

        for (int d : WIDTHS) {
            float[] host = randomColumnMajor(rows, d);
            double[] reference = referenceGram(host, rows, d);

            Result generated = time(rows, d, host, false);
            Result library = time(rows, d, host, true);

            System.out.printf(
                    "%5d %10.1f/B %10.2f ms %10.2f ms %9.2fx %12.2e %12.2e%n",
                    d,
                    d / 2.0,
                    generated.millis,
                    library.millis,
                    generated.millis / library.millis,
                    relativeError(generated.gram, reference),
                    relativeError(library.gram, reference));
        }
    }

    /** One timed configuration: centre on the device, contract on the device, read back d*d. */
    private static Result time(int rows, int d, float[] host, boolean useLibrary) {
        FloatArray data = new FloatArray(rows * d);
        for (int i = 0; i < rows * d; i++) {
            data.set(i, host[i]);
        }
        FloatArray means = new FloatArray(d);
        FloatArray gram = new FloatArray(d * d);

        // Column means on the host: d reductions over N, once, outside the timed region. On the
        // device this would be its own kernel; it is not what the sweep is measuring.
        for (int c = 0; c < d; c++) {
            double sum = 0.0;
            for (int r = 0; r < rows; r++) {
                sum += host[c * rows + r];
            }
            means.set(c, (float) (sum / rows));
        }

        TaskGraph graph =
                new TaskGraph("gram")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, data, means)
                        .task("centre", GramMatrixSweep::centre, data, means, rows, d);

        if (useLibrary) {
            // C = A' * A, with A held column-major as rows x d, so lda is the row count and the
            // transpose is free -- cuBLAS reads the same buffer the kernel just wrote.
            graph =
                    graph.libraryTask(
                            "gemm",
                            CuBlas::cublasSgemm,
                            CuBlasOperation.CUBLAS_OP_T.operation(),
                            CuBlasOperation.CUBLAS_OP_N.operation(),
                            d,
                            d,
                            rows,
                            1.0f,
                            data,
                            rows,
                            data,
                            rows,
                            0.0f,
                            gram,
                            d);
        } else {
            graph = graph.task("gram", GramMatrixSweep::gram, data, gram, rows, d);
        }

        graph = graph.transferToHost(DataTransferMode.EVERY_EXECUTION, gram);

        double best = Double.MAX_VALUE;
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            for (int i = 0; i < WARMUP + RUNS; i++) {
                // Re-centring a centred buffer would drift, so the input is restored each run.
                for (int j = 0; j < rows * d; j++) {
                    data.set(j, host[j]);
                }
                long start = System.nanoTime();
                plan.execute();
                double millis = (System.nanoTime() - start) / 1e6;
                if (i >= WARMUP) {
                    best = Math.min(best, millis);
                }
            }
        } catch (Exception e) {
            throw new IllegalStateException(
                    "device execution failed for d=" + d + (useLibrary ? " (cublas)" : ""), e);
        }

        double[] out = new double[d * d];
        for (int i = 0; i < d * d; i++) {
            out[i] = gram.get(i);
        }
        return new Result(best, out);
    }

    /**
     * Subtracts each column's mean, in place.
     *
     * <p>Elementwise, so this is the half of the computation a generated kernel is right for. It is
     * also why the graph is hybrid rather than a single library call: cuBLAS has no primitive for
     * it that would not cost another pass over global memory.
     */
    public static void centre(FloatArray data, FloatArray means, int rows, int d) {
        for (@Parallel int r = 0; r < rows; r++) {
            for (int c = 0; c < d; c++) {
                data.set(c * rows + r, data.get(c * rows + r) - means.get(c));
            }
        }
    }

    /**
     * The same contraction as the GEMM, generated.
     *
     * <p>One thread per output element, each summing over all rows. Correct, and the honest
     * comparison for the library call: no tiling, no shared-memory reuse, so it reloads a column
     * per output element where cuBLAS loads a tile once and reuses it. That difference is the whole
     * hypothesis being tested.
     */
    public static void gram(FloatArray data, FloatArray out, int rows, int d) {
        for (@Parallel int i = 0; i < d; i++) {
            for (@Parallel int j = 0; j < d; j++) {
                float sum = 0.0f;
                for (int r = 0; r < rows; r++) {
                    sum += data.get(i * rows + r) * data.get(j * rows + r);
                }
                out.set(j * d + i, sum);
            }
        }
    }

    /** Column-major rows x d, values around 1.0 so centring leaves something to cancel. */
    private static float[] randomColumnMajor(int rows, int d) {
        Random random = new Random(42);
        float[] values = new float[rows * d];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) (1.0 + random.nextGaussian());
        }
        return values;
    }

    /** Double-precision Gram matrix of the centred data, to measure what FP32 costs. */
    private static double[] referenceGram(float[] host, int rows, int d) {
        double[] means = new double[d];
        for (int c = 0; c < d; c++) {
            double sum = 0.0;
            for (int r = 0; r < rows; r++) {
                sum += host[c * rows + r];
            }
            means[c] = sum / rows;
        }
        // The kernel centres in float, so the reference does too: this isolates the error of the
        // contraction, which is what differs between the two paths, from the error of centring,
        // which they share.
        double[] out = new double[d * d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                double sum = 0.0;
                for (int r = 0; r < rows; r++) {
                    sum +=
                            (double) ((float) (host[i * rows + r] - (float) means[i]))
                                    * (float) (host[j * rows + r] - (float) means[j]);
                }
                out[j * d + i] = sum;
            }
        }
        return out;
    }

    /** Largest elementwise error relative to the reference's own magnitude. */
    private static double relativeError(double[] actual, double[] expected) {
        double worst = 0.0;
        for (int i = 0; i < expected.length; i++) {
            double scale = Math.max(1.0, Math.abs(expected[i]));
            worst = Math.max(worst, Math.abs(actual[i] - expected[i]) / scale);
        }
        return worst;
    }

    private static final class Result {
        private final double millis;
        private final double[] gram;

        private Result(double millis, double[] gram) {
            this.millis = millis;
            this.gram = gram;
        }
    }
}
