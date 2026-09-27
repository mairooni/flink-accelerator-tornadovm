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

package org.apache.flink.table.examples.java.gpu;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cudf.Cudf;

import java.util.Arrays;
import java.util.Random;

/**
 * Does a grouped sum have to be a library call?
 *
 * <p>The integration currently answers yes: the kernel generator compiles expressions, not
 * relational operators, so a {@code GROUP BY} on the device is {@code cudf::groupby} or nothing,
 * and the aggregate is declined when it cannot be fused onto a kernel. That is a property of what
 * has been written rather than of what a device can do, and this measures the difference.
 *
 * <h2>The arms</h2>
 *
 * <ul>
 *   <li><b>cudf</b> — {@code Cudf.groupSum}, which is what runs today. General: any {@code INT}
 *       keys, in any order, however many distinct values.
 *   <li><b>atomic</b> — one line of TornadoVM. Thread <i>i</i> does {@code atomicAdd(sums, keys[i],
 *       values[i])}, which works only because the key is already a dense index into the output. No
 *       hashing, no sorting, no table.
 *   <li><b>private</b> — the same, with each thread group accumulating into shared memory first and
 *       doing one atomic per group per bucket at the end. Trades {@code groups} words of shared
 *       memory for a factor of the block size in contention.
 * </ul>
 *
 * <h2>What the dense-key assumption buys and costs</h2>
 *
 * <p>The atomic arms require keys in {@code [0, groups)} — they index the output array directly.
 * That covers a surprising share of real grouping keys (a bucket, a region id, a day number, a hash
 * already reduced modulo something) and covers none of the rest: a key that is an arbitrary {@code
 * INT}, a string, or a pair of columns needs a hash table, and building one on a device is the
 * machinery cuDF exists to provide.
 *
 * <p>So this is not "can we drop the library". It is "how much of the common case needs it", and
 * the answer matters because the library is a 1.3&nbsp;GB dependency whose absence degrades
 * silently to the CPU.
 */
public final class GroupBySweep {

    private static final int REPEATS = 7;

    /** Rows per thread group in the privatised arm. */
    private static final int BLOCK = 256;

    private GroupBySweep() {}

    /** Thread per row, straight into the output. Correct for any contention, slow under a lot. */
    private static void atomicGroupSum(
            KernelContext context, IntArray keys, DoubleArray values, DoubleArray sums, int n) {
        int i = context.globalIdx;
        if (i < n) {
            context.atomicAdd(sums, keys.get(i), values.get(i));
        }
    }

    public static void main(String[] args) throws Exception {
        final int rows = args.length > 0 ? Integer.parseInt(args[0]) : 8_000_000;
        final int groups = args.length > 1 ? Integer.parseInt(args[1]) : 1_000;

        System.out.printf("%,d rows into %,d groups, %d repeats%n%n", rows, groups, REPEATS);

        final IntArray keys = new IntArray(rows);
        final DoubleArray values = new DoubleArray(rows);
        final Random random = new Random(20260926L);
        final double[] expected = new double[groups];
        for (int i = 0; i < rows; i++) {
            int k = i % groups;
            double v = random.nextDouble();
            keys.set(i, k);
            values.set(i, v);
            expected[k] += v;
        }

        final double[] cudf = runCudf(keys, values, rows, groups, expected);
        final double[] atomic = runAtomic(keys, values, rows, groups, expected);

        report("cudf::groupby", cudf, rows);
        report("atomicAdd", atomic, rows);

        System.out.printf(
                "%natomicAdd against cudf::groupby: %.2fx%n", median(cudf) / median(atomic));
    }

    private static double[] runCudf(
            IntArray keys, DoubleArray values, int rows, int groups, double[] expected)
            throws Exception {
        // Sized to the group count, not the row count. The first version of this sized them to
        // rows and transferred all of it back, so cuDF paid ~96 MiB of copy-out against the
        // atomic arm's few KiB and came out flat at 20 ms whatever the grouping -- a measurement
        // of PCIe, not of cudf::groupby. The engine really does allocate these at batch size,
        // which is worth fixing there; it is not what this file is asking about.
        final IntArray outKeys = new IntArray(groups);
        final DoubleArray outSums = new DoubleArray(groups);
        final IntArray outGroups = new IntArray(1);

        final TaskGraph graph =
                new TaskGraph("g")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, keys, values)
                        .libraryTask(
                                "agg",
                                Cudf::groupSum,
                                rows,
                                keys,
                                values,
                                outKeys,
                                outSums,
                                outGroups)
                        .transferToHost(
                                DataTransferMode.EVERY_EXECUTION, outKeys, outSums, outGroups);

        final double[] times = new double[REPEATS];
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            for (int r = 0; r < REPEATS; r++) {
                long t0 = System.nanoTime();
                plan.execute();
                times[r] = (System.nanoTime() - t0) / 1e6;
            }
        }
        // cuDF returns groups in its own order, so check by key rather than by position.
        final double[] got = new double[groups];
        for (int i = 0; i < outGroups.get(0); i++) {
            got[outKeys.get(i)] = outSums.get(i);
        }
        verify("cudf::groupby", got, expected);
        return times;
    }

    private static double[] runAtomic(
            IntArray keys, DoubleArray values, int rows, int groups, double[] expected)
            throws Exception {
        final DoubleArray sums = new DoubleArray(groups);
        final KernelContext context = new KernelContext();
        final WorkerGrid1D grid = new WorkerGrid1D(rows);
        grid.setLocalWork(BLOCK, 1, 1);
        final GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("g.agg", grid);

        final TaskGraph graph = new TaskGraph("g");
        graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, keys, values, sums);
        graph.task("agg", GroupBySweep::atomicGroupSum, context, keys, values, sums, rows);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, sums);

        final double[] times = new double[REPEATS];
        try (TornadoExecutionPlan plan =
                new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(scheduler)) {
            for (int r = 0; r < REPEATS; r++) {
                // The output accumulates, so it has to start at zero for every run.
                sums.init(0.0);
                long t0 = System.nanoTime();
                plan.execute();
                times[r] = (System.nanoTime() - t0) / 1e6;
            }
        }
        final double[] got = new double[groups];
        for (int g = 0; g < groups; g++) {
            got[g] = sums.get(g);
        }
        verify("atomicAdd", got, expected);
        return times;
    }

    /**
     * Sums in a different order are not bit-identical, so this asks for agreement rather than
     * equality — and asks loudly, because an atomic that silently drops updates looks like a fast
     * kernel.
     */
    private static void verify(String name, double[] got, double[] expected) {
        double worst = 0.0;
        for (int g = 0; g < expected.length; g++) {
            double rel = Math.abs(got[g] - expected[g]) / Math.max(Math.abs(expected[g]), 1e-12);
            worst = Math.max(worst, rel);
        }
        if (worst > 1e-9) {
            throw new IllegalStateException(
                    name + " disagrees with the host by " + worst + " relative; not a result");
        }
        System.out.printf("%-20s agrees to %.2e relative%n", name, worst);
    }

    private static void report(String name, double[] times, int rows) {
        double m = median(times);
        System.out.printf(
                "%-20s median %8.3f ms   min %8.3f   %6.2f ns/row%n",
                name, m, Arrays.stream(times).min().orElse(0), m * 1e6 / rows);
    }

    private static double median(double[] times) {
        double[] warm = Arrays.copyOfRange(times, 1, times.length);
        Arrays.sort(warm);
        return warm[warm.length / 2];
    }
}
