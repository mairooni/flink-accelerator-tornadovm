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

import uk.ac.manchester.tornado.api.DataRange;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cudf.Cudf;

import java.util.Arrays;
import java.util.Random;

/**
 * Where a tuned library beats anything you would reasonably write.
 *
 * <p>{@link GroupBySweep} found the opposite result and it is worth stating why. A grouped sum over
 * dense keys is reduction-shaped: every row's destination is known from its key, so one {@code
 * atomicAdd} a row does it and the specialised kernel beats {@code cudf::groupby} by up to 1.7x.
 * Nothing about that generalises.
 *
 * <p>A sort is the other kind of problem. Where each element ends up is not a function of the
 * element — it depends on every other element — so the parallelism is irregular and the good
 * algorithms are involved. cuDF sorts with a radix sort that has had years of tuning behind it.
 * What a competent person writes instead is a bitonic sort: correct, genuinely parallel, <em>O(n
 * log²n)</em> comparisons against radix's <em>O(n)</em> passes, and it is the honest comparison
 * because it is what this project could generate today.
 *
 * <h2>The arms</h2>
 *
 * <ul>
 *   <li><b>bitonic</b> — one {@code @Parallel} kernel, executed once per merge stage. For 2²²
 *       elements that is 253 launches over the same resident buffer. The stage parameters live in a
 *       device buffer the host updates between launches, because a scalar argument is captured when
 *       the graph is built.
 *   <li><b>cudf</b> — {@code cudf::stable_sorted_order} followed by a one-pass gather to
 *       materialise the sorted values, so both arms end with sorted ints on the device.
 * </ul>
 *
 * <p>Neither arm's timing includes the copy back: the question is which algorithm is faster on the
 * device, not how fast PCIe is.
 */
public final class SortSweep {

    private static final int REPEATS = 5;

    private SortSweep() {}

    /**
     * One bitonic merge step. {@code params} holds {@code k} and {@code j} for this launch.
     *
     * <p>Each thread owns index {@code i} and compares with {@code i ^ j}; only the lower of the
     * pair acts, so no element is written twice within a launch and no barrier is needed. The
     * direction comes from bit {@code k} of the index, which is what makes the sequence bitonic.
     */
    private static void bitonicStep(IntArray params, IntArray a, int n) {
        int k = params.get(0);
        int j = params.get(1);
        for (@Parallel int i = 0; i < n; i++) {
            int partner = i ^ j;
            if (partner > i) {
                int x = a.get(i);
                int y = a.get(partner);
                boolean ascending = (i & k) == 0;
                if ((ascending && x > y) || (!ascending && x < y)) {
                    a.set(i, y);
                    a.set(partner, x);
                }
            }
        }
    }

    /** Materialises the permutation cuDF produced, so both arms end in the same place. */
    private static void gather(IntArray source, IntArray order, IntArray out, int n) {
        for (@Parallel int i = 0; i < n; i++) {
            out.set(i, source.get(order.get(i)));
        }
    }

    public static void main(String[] args) throws Exception {
        // Bitonic sort needs a power of two, so both arms use one.
        final int n = args.length > 0 ? Integer.parseInt(args[0]) : 1 << 22;
        if (Integer.bitCount(n) != 1) {
            throw new IllegalArgumentException("n must be a power of two, got " + n);
        }
        System.out.printf("%,d ints, %d repeats%n%n", n, REPEATS);

        final int[] source = new int[n];
        final Random random = new Random(20260926L);
        for (int i = 0; i < n; i++) {
            source[i] = random.nextInt();
        }
        final int[] expected = source.clone();
        Arrays.sort(expected);

        final double[] bitonic = runBitonic(source, n, expected);
        final double[] cudf = runCudf(source, n, expected);

        report("bitonic kernel", bitonic, n);
        report("cudf::sorted_order", cudf, n);
        System.out.printf("%ncudf against the kernel: %.2fx%n", median(bitonic) / median(cudf));
    }

    private static double[] runBitonic(int[] source, int n, int[] expected) throws Exception {
        final IntArray a = new IntArray(n);
        final IntArray params = new IntArray(2);

        final WorkerGrid1D grid = new WorkerGrid1D(n);
        final GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("s.step", grid);

        // The buffer stays on the device across all 253 launches; only the two stage parameters
        // are re-sent. Copying 32 MiB back and forth per stage would measure PCIe 253 times.
        final TaskGraph graph =
                new TaskGraph("s")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, a)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, params)
                        .task("step", SortSweep::bitonicStep, params, a, n)
                        .persistOnDevice(a);

        final double[] times = new double[REPEATS];
        TornadoExecutionResult last = null;
        try (TornadoExecutionPlan plan =
                new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(scheduler)) {
            for (int r = 0; r < REPEATS; r++) {
                for (int i = 0; i < n; i++) {
                    a.set(i, source[i]);
                }
                long t0 = System.nanoTime();
                int launches = 0;
                for (int k = 2; k <= n; k <<= 1) {
                    for (int j = k >> 1; j > 0; j >>= 1) {
                        params.set(0, k);
                        params.set(1, j);
                        last = plan.execute();
                        launches++;
                    }
                }
                times[r] = (System.nanoTime() - t0) / 1e6;
                if (r == 0) {
                    System.out.printf("bitonic: %d kernel launches a sort%n", launches);
                }
            }
            last.transferToHost(new DataRange(a));
        }
        verify("bitonic kernel", a, expected, n);
        return times;
    }

    private static double[] runCudf(int[] source, int n, int[] expected) throws Exception {
        final IntArray a = new IntArray(n);
        for (int i = 0; i < n; i++) {
            a.set(i, source[i]);
        }
        final IntArray order = new IntArray(n);
        final IntArray out = new IntArray(n);

        final WorkerGrid1D grid = new WorkerGrid1D(n);
        final GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("c.gather", grid);

        final TaskGraph graph =
                new TaskGraph("c")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, a)
                        .libraryTask("order", Cudf::sortedOrder, n, a, order)
                        .task("gather", SortSweep::gather, a, order, out, n)
                        .persistOnDevice(out);

        final double[] times = new double[REPEATS];
        try (TornadoExecutionPlan plan =
                new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(scheduler)) {
            TornadoExecutionResult last = null;
            for (int r = 0; r < REPEATS; r++) {
                long t0 = System.nanoTime();
                last = plan.execute();
                times[r] = (System.nanoTime() - t0) / 1e6;
            }
            last.transferToHost(new DataRange(out));
        }
        verify("cudf::sorted_order", out, expected, n);
        return times;
    }

    /** A sort that drops or duplicates an element looks like a very fast sort. */
    private static void verify(String name, IntArray got, int[] expected, int n) {
        for (int i = 0; i < n; i++) {
            if (got.get(i) != expected[i]) {
                throw new IllegalStateException(
                        name
                                + " is not sorted: element "
                                + i
                                + " is "
                                + got.get(i)
                                + ", expected "
                                + expected[i]);
            }
        }
        System.out.printf("%-20s sorted correctly%n", name);
    }

    private static void report(String name, double[] times, int n) {
        double m = median(times);
        System.out.printf(
                "%-20s median %9.2f ms   min %9.2f   %6.1f ns/element%n",
                name, m, Arrays.stream(times).min().orElse(0), m * 1e6 / n);
    }

    private static double median(double[] times) {
        double[] warm = Arrays.copyOfRange(times, 1, times.length);
        Arrays.sort(warm);
        return warm[warm.length / 2];
    }
}
