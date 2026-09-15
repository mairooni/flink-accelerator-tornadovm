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

import org.apache.flink.table.gpu.codegen.GpuCalcSpec;
import org.apache.flink.table.gpu.codegen.GpuKernelSource;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.metrics.OffloadMetrics;
import org.apache.flink.table.gpu.operator.GeneratedKernelEngine;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.RowType;

/**
 * Arithmetic-intensity sweep through the generated-kernel path, used to check the cost floor.
 *
 * <h2>Why this does not run SQL</h2>
 *
 * <p>The obvious version of this — submit the same query with offload on and off and compare wall
 * times — does not work. Measured with the {@code datagen} sequence source, a CPU-only job takes
 * 4.9 s at 20k rows, 20.6 s at 200k and 100.6 s at 1M: about 100 µs per row for {@code val*2+1}.
 * The source dominates by orders of magnitude, so a whole-job comparison measures Flink's record
 * plumbing and says nothing about the operator.
 *
 * <p>This drives {@link GeneratedKernelEngine} directly, as the original calibration drove the
 * hand-written kernel, so the two are comparable and neither includes job overhead.
 *
 * <p>The kernel sources here are written in the shape {@code GpuKernelGenerator} emits rather than
 * produced by it: the generator lives in the planner, which has Calcite but not TornadoVM, and this
 * module has TornadoVM but not Calcite. The weights come from the real estimator, read out of
 * EXPLAIN for the equivalent SQL.
 */
public final class GeneratedKernelSweep {

    /** Weighted cost the planner's estimator assigns to each level, read from EXPLAIN. */
    private static final int[] LEVELS = {0, 1, 2, 4, 8, 16};

    private static final int[] WEIGHTS = {3, 47, 91, 179, 355, 707};

    public static void main(String[] args) throws Exception {
        final int rows = args.length > 0 ? Integer.parseInt(args[0]) : 4_000_000;
        final int batchSize = args.length > 1 ? Integer.parseInt(args[1]) : 262_144;

        double[] input = new double[rows];
        for (int i = 0; i < rows; i++) {
            input[i] = 1.0 + i * (10.0 / rows);
        }

        System.out.printf("rows=%,d  batch=%,d%n%n", rows, batchSize);
        System.out.printf(
                "%-7s %-9s %10s %10s %9s %10s %10s%n",
                "terms", "weighted", "cpu_ms", "gpu_ms", "speedup", "kernel_ms", "compile_ms");

        for (int level = 0; level < LEVELS.length; level++) {
            int terms = LEVELS[level];
            double cpuMs = cpuBaseline(input, terms);

            long compileStart = System.nanoTime();
            GpuCalcSpec spec = spec(terms, batchSize);
            try (GeneratedKernelEngine engine = new GeneratedKernelEngine(spec, true)) {
                engine.open();
                double compileMs = (System.nanoTime() - compileStart) / 1e6;

                // One untimed batch so kernel compilation on the device is not in the numbers.
                stage(engine, input, 0, Math.min(batchSize, rows));
                engine.execute(batchSize);

                double gpuMs = run(engine, input, rows, batchSize);
                OffloadMetrics m = engine.metrics();
                System.out.printf(
                        "%-7d %-9d %10.0f %10.0f %8.2fx %10.1f %10.0f%n",
                        terms,
                        WEIGHTS[level],
                        cpuMs,
                        gpuMs,
                        cpuMs / gpuMs,
                        m.getKernelNanos() / 1e6,
                        compileMs);
            }
        }
    }

    /** Total attributed time: staging, transfer, kernel, drain — everything the offload costs. */
    private static double run(
            GeneratedKernelEngine engine, double[] input, int rows, int batchSize) {
        long total = 0;
        for (int start = 0; start < rows; start += batchSize) {
            int count = Math.min(batchSize, rows - start);

            long t0 = System.nanoTime();
            stage(engine, input, start, count);
            long gather = System.nanoTime() - t0;

            GeneratedKernelEngine.Execution execution = engine.execute(batchSize);

            long d0 = System.nanoTime();
            int emitted = 0;
            double sink = 0;
            for (int i = 0; i < count; i++) {
                if (engine.selected(i)) {
                    sink += ((Number) engine.output(0, i)).doubleValue();
                    emitted++;
                }
            }
            long drain = System.nanoTime() - d0;
            if (Double.isNaN(sink)) {
                throw new AssertionError("keeps the drain from being optimised away");
            }
            engine.recordBatch(count, gather, execution, emitted, drain);
            total += gather + drain;
        }
        OffloadMetrics m = engine.metrics();
        return (total + m.getCopyInNanos() + m.getKernelNanos() + m.getCopyOutNanos()) / 1e6;
    }

    private static void stage(GeneratedKernelEngine engine, double[] input, int start, int count) {
        for (int i = 0; i < count; i++) {
            engine.inputColumn(0).set(i, input[start + i]);
        }
    }

    /** The scalar loop the offload has to beat. */
    private static double cpuBaseline(double[] input, int terms) {
        long t0 = System.nanoTime();
        double sink = 0;
        for (double v : input) {
            if (v > 1.0) {
                double h = v * 2.0 + 1.0;
                for (int k = 1; k <= terms; k++) {
                    h += Math.sin(v + k) * Math.cos(v + k);
                }
                sink += h;
            }
        }
        long elapsed = System.nanoTime() - t0;
        if (Double.isNaN(sink)) {
            throw new AssertionError("keeps the loop from being optimised away");
        }
        return elapsed / 1e6;
    }

    /** Kernel source in the shape {@code GpuKernelGenerator} emits. */
    private static GpuCalcSpec spec(int terms, int batchSize) {
        StringBuilder expr = new StringBuilder("((c0 * 2.0) + 1.0)");
        for (int k = 1; k <= terms; k++) {
            expr.append(" + (TornadoMath.sin(c0 + ")
                    .append(k)
                    .append(".0) * TornadoMath.cos(c0 + ")
                    .append(k)
                    .append(".0))");
        }
        String className = "GpuCalcKernel$Sweep" + terms;
        String source =
                "import uk.ac.manchester.tornado.api.annotations.Parallel;\n"
                        + "import uk.ac.manchester.tornado.api.math.TornadoMath;\n"
                        + "import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;\n"
                        + "import uk.ac.manchester.tornado.api.types.arrays.IntArray;\n\n"
                        + "public final class "
                        + className
                        + " {\n\n"
                        + "    public static void evaluate(DoubleArray c0_in, DoubleArray out0,"
                        + " IntArray mask, IntArray rows) {\n"
                        + "        final int n = rows.get(0);\n"
                        + "        for (@Parallel int i = 0; i < n; i++) {\n"
                        + "            double c0 = c0_in.get(i);\n"
                        + "            out0.set(i, "
                        + expr
                        + ");\n"
                        + "            if ((c0 > 1.0)) {\n"
                        + "                mask.set(i, 1);\n"
                        + "            } else {\n"
                        + "                mask.set(i, 0);\n"
                        + "            }\n"
                        + "        }\n"
                        + "    }\n"
                        + "}\n";

        GpuKernelSource kernel =
                new GpuKernelSource(
                        className,
                        "evaluate",
                        source,
                        new int[] {0},
                        new GpuValueType[] {GpuValueType.DOUBLE},
                        new GpuValueType[] {GpuValueType.DOUBLE},
                        true,
                        new int[] {GpuCalcSpec.COMPUTED});
        return new GpuCalcSpec(
                kernel, new int[] {GpuCalcSpec.COMPUTED}, RowType.of(new DoubleType()), batchSize);
    }

    private GeneratedKernelSweep() {}
}
