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

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Ordering a partition with a generated kernel instead of a library, so the two can be compared.
 *
 * <p>The device sort is normally {@code cudf::stable_sorted_order}. This is the same job written as
 * a kernel: a bitonic sort, which is what a compiler for this project could reasonably emit, since
 * every step is one data-parallel pass with no cooperation between threads.
 *
 * <p>It is <b>slower, by design of the algorithm rather than of this code</b>. Measured against
 * cuDF over the same integers, 4.5x slower at 8M elements and 9.6x at 1M. Bitonic needs <i>O(n
 * log&sup2;n)</i> comparisons spread over {@code log n * (log n + 1) / 2} kernel launches — 253 of
 * them for four million rows — while a radix sort is a handful of linear passes. That gap is the
 * argument for binding a library at all, and it is worth being able to run rather than cite.
 *
 * <p>Selected with {@code -Dflink.accelerator.tornadovm.sortKernel=true}. Off by default: it is
 * here to be measured against, not to be used.
 *
 * <h2>Why it sorts pairs</h2>
 *
 * <p>The operator drains its staged columns in permutation order, so what a sort has to produce is
 * the permutation, not sorted keys — and the key column has to survive, because it is one of the
 * columns being drained. So this sorts a <em>copy</em> of the keys with the identity alongside it,
 * swapping both whenever it swaps either. The copy ends up sorted and is thrown away; the identity
 * ends up as the permutation.
 *
 * <h2>Why it pads</h2>
 *
 * <p>Bitonic sorting is defined on a power of two. A partition is not, so the arrays are rounded up
 * and the tail filled with {@link Integer#MAX_VALUE}, which sorts to the end and is never read: the
 * caller takes the first {@code n} entries of the permutation. Padding with the maximum is what
 * makes that safe — any smaller filler would interleave with real keys.
 */
public final class BitonicSort {

    /** Off by default: this exists to be measured against {@code cudf::sorted_order}. */
    public static final boolean ENABLED =
            Boolean.getBoolean("flink.accelerator.tornadovm.sortKernel");

    private BitonicSort() {}

    /**
     * Seeds the permutation with the identity and pads the key copy.
     *
     * <p>Runs over the padded length: the tail has to be written, not left at whatever the buffer
     * held from a previous partition.
     */
    public static void prepare(IntArray keys, IntArray scratch, IntArray order, int n, int padded) {
        for (@Parallel int i = 0; i < padded; i++) {
            order.set(i, i);
            scratch.set(i, i < n ? keys.get(i) : Integer.MAX_VALUE);
        }
    }

    /**
     * One bitonic merge step, over the whole padded array.
     *
     * <p>Thread {@code i} pairs with {@code i ^ j} and only the lower index of the pair writes, so
     * no element is touched twice in a launch and no barrier is needed. Bit {@code k} of the index
     * gives the direction, which is what makes each subsequence bitonic.
     *
     * <p>{@code k} and {@code j} come from a buffer rather than as arguments because a scalar is
     * captured when the task graph is built, and these change on every one of the launches.
     */
    public static void step(IntArray params, IntArray keys, IntArray order, int padded) {
        final int k = params.get(0);
        final int j = params.get(1);
        for (@Parallel int i = 0; i < padded; i++) {
            final int partner = i ^ j;
            if (partner > i) {
                final int x = keys.get(i);
                final int y = keys.get(partner);
                final boolean ascending = (i & k) == 0;
                if ((ascending && x > y) || (!ascending && x < y)) {
                    keys.set(i, y);
                    keys.set(partner, x);
                    final int xi = order.get(i);
                    final int yi = order.get(partner);
                    order.set(i, yi);
                    order.set(partner, xi);
                }
            }
        }
    }

    /** The smallest power of two that holds {@code n}, which is the length bitonic needs. */
    public static int paddedLength(int n) {
        return n <= 1 ? 1 : Integer.highestOneBit(n - 1) << 1;
    }

    /** Launches a bitonic sort needs: one per merge step, over all stages. */
    public static int launches(int padded) {
        int count = 0;
        for (int k = 2; k <= padded; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                count++;
            }
        }
        return count;
    }

    /**
     * Orders {@code n} keys and returns the permutation.
     *
     * <p>One graph, executed once per merge step with the step's parameters re-sent. The arrays
     * stay resident across the launches — copying a partition back and forth 253 times would
     * measure the bus rather than the sort.
     *
     * @param keys the staged key column, which is read and not modified
     * @param scratch a padded scratch buffer for the key copy this destroys
     * @param order a padded buffer that ends holding the permutation
     */
    public static int[] order(IntArray keys, IntArray scratch, IntArray order, int n)
            throws Exception {
        final int padded = paddedLength(n);
        final IntArray params = new IntArray(2);
        params.set(0, 2);
        params.set(1, 1);

        final WorkerGrid1D grid = new WorkerGrid1D(padded);
        final GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("bitonic.prepare", grid);
        scheduler.addWorkerGrid("bitonic.step", grid);

        final TaskGraph prep =
                new TaskGraph("bitonic")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, keys, params)
                        .task("prepare", BitonicSort::prepare, keys, scratch, order, n, padded)
                        .task("step", BitonicSort::step, params, scratch, order, padded)
                        .persistOnDevice(scratch, order);

        try (TornadoExecutionPlan plan =
                new TornadoExecutionPlan(prep.snapshot()).withGridScheduler(scheduler)) {
            // The first execution does prepare and the first step together; every later one
            // repeats the step with new parameters. prepare is idempotent only before any step
            // has run, so the two are separated after the first launch.
            plan.execute();
            final TaskGraph merge =
                    new TaskGraph("bitonic")
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, params)
                            .task("step", BitonicSort::step, params, scratch, order, padded)
                            .persistOnDevice(scratch, order);
            try (TornadoExecutionPlan steps =
                    new TornadoExecutionPlan(merge.snapshot()).withGridScheduler(scheduler)) {
                boolean first = true;
                for (int k = 2; k <= padded; k <<= 1) {
                    for (int j = k >> 1; j > 0; j >>= 1) {
                        if (first) {
                            // Already done by the graph above, which had to run prepare first.
                            first = false;
                            continue;
                        }
                        params.set(0, k);
                        params.set(1, j);
                        steps.execute();
                    }
                }
                steps.execute().transferToHost(order);
            }
        }

        final int[] permutation = new int[n];
        for (int i = 0; i < n; i++) {
            permutation[i] = order.get(i);
        }
        return permutation;
    }
}
