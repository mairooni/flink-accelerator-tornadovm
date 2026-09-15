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

import org.apache.flink.table.gpu.codegen.GpuAggregateSpec;
import org.apache.flink.table.gpu.codegen.GpuKernelSource;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.operator.FusedAggregateEngine;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

import java.util.Arrays;

/**
 * Drives {@link FusedAggregateEngine} directly, with no Flink around it.
 *
 * <p>Written after four hypotheses were tested against a cluster at a quarter of an hour each and
 * none was right. A hand-built approximation of the same graph ran correctly, so approximating the
 * shape was not enough; this runs the actual engine, which is what found the fault -- the operator
 * had been staging every row at position 0.
 *
 * <p>It now sweeps the number of contracted columns, because that is where the second failure was.
 * A buffer per column meant a reduction task per column, and at 136 sums -- the gram query at
 * sixteen feature columns -- TornadoVM answers {@code Tornado Graph resize not implemented yet}.
 * Packed into one matrix the contraction is a single task at any width, and these widths say so.
 *
 * <pre>{@code
 * java @$TORNADO_SDK/tornado-argfile -cp <classes> org.apache.flink.table.gpu.FusedAggregateSweep
 * }</pre>
 *
 * <p>Every staged row is 1.0 and column <em>j</em> is {@code c0 * (j + 1)}, so column <em>j</em>
 * must sum to {@code (j + 1) * rows} and any shortfall says how much of the buffer was reduced.
 */
public final class FusedAggregateSweep {

    private static final int ROWS = 1_000_000;
    private static final int BATCH = 262_144;

    private FusedAggregateSweep() {}

    private static GpuAggregateSpec spec(int columns) {
        StringBuilder writes = new StringBuilder();
        for (int j = 0; j < columns; j++) {
            writes.append("            out.set(")
                    .append(j * BATCH)
                    .append(" + i, (c0 * ")
                    .append(j + 1)
                    .append(".0));\n");
        }
        String source =
                "import uk.ac.manchester.tornado.api.annotations.Parallel;\n"
                        + "import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;\n\n"
                        + "public final class GpuCalcKernel$Sweep {\n\n"
                        + "    public static void evaluate(DoubleArray c0_in, DoubleArray out) {\n"
                        + "        for (@Parallel int i = 0; i < c0_in.getSize(); i++) {\n"
                        + "            double c0 = c0_in.get(i);\n"
                        + writes
                        + "        }\n"
                        + "    }\n"
                        + "}\n";

        GpuValueType[] outputs = new GpuValueType[columns];
        Arrays.fill(outputs, GpuValueType.DOUBLE);
        int[] layout = new int[columns];
        Arrays.fill(layout, -1);
        int[] sums = new int[columns];
        LogicalType[] fields = new LogicalType[columns];
        String[] names = new String[columns];
        for (int j = 0; j < columns; j++) {
            sums[j] = j;
            fields[j] = new DoubleType();
            names[j] = "f" + j;
        }

        GpuKernelSource kernel =
                new GpuKernelSource(
                        "GpuCalcKernel$Sweep",
                        "evaluate",
                        source,
                        new int[] {0},
                        new GpuValueType[] {GpuValueType.DOUBLE},
                        outputs,
                        false,
                        layout,
                        BATCH);
        RowType row = RowType.of(fields, names);
        return new GpuAggregateSpec(kernel, sums, row, row, BATCH);
    }

    public static void main(String[] args) throws Exception {
        for (int columns : new int[] {1, 8, 36, 136, 256}) {
            try (FusedAggregateEngine engine = new FusedAggregateEngine(spec(columns))) {
                engine.open();
                for (int i = 0; i < ROWS; i++) {
                    engine.inputColumn(0).set(engine.position(), 1.0);
                    engine.rowComplete();
                }
                engine.flush();
                int wrong = 0;
                for (int j = 0; j < columns; j++) {
                    if (engine.total(j) != (j + 1.0) * ROWS) {
                        wrong++;
                    }
                }
                System.out.printf(
                        "columns=%3d  first=%.1f last=%.1f  %s%n",
                        columns,
                        engine.total(0),
                        engine.total(columns - 1),
                        wrong == 0 ? "OK" : wrong + " of " + columns + " WRONG");
            } catch (Exception e) {
                System.out.printf("columns=%3d  FAILED: %s%n", columns, e);
            }
        }
    }
}
