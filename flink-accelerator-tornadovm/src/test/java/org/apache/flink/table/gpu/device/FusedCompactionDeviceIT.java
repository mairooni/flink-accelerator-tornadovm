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

package org.apache.flink.table.gpu.device;

import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.util.OneInputStreamOperatorTestHarness;
import org.apache.flink.table.accelerator.AccelCall;
import org.apache.flink.table.accelerator.AccelFilter;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelLiteral;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.operator.GpuCalcOperator;
import org.apache.flink.table.runtime.accelerator.AcceleratorContext;
import org.apache.flink.table.runtime.accelerator.AcceleratorPlan;
import org.apache.flink.table.types.logical.BooleanType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The fused path: a filter that compacts on the device, in one execution plan with the projection.
 *
 * <h2>What is actually being tested</h2>
 *
 * <p>The rows are the same either way — that is the problem. A projection with a {@code WHERE}
 * produces one answer, and whether the survivors were selected by cuDF on the device or skipped by
 * a loop on the host is invisible in the output. So every case here asserts both: that the fused
 * path was the one that ran, and that what it emitted is what the same predicate emits in Java.
 *
 * <p>The mechanism underneath — two task graphs in one {@code TornadoExecutionPlan}, a projection
 * and a mask persisted by the first and consumed by the second, {@code Cudf.selectedIndices} over
 * the persisted mask, and a gather — is pinned separately in {@code
 * FusedCompactionMechanismDeviceIT}. This is about the operator that uses it.
 *
 * <h2>The boundaries chosen, and why each is a boundary</h2>
 *
 * <ul>
 *   <li><b>Nothing survives.</b> cuDF writes a count of zero and the gather must emit nothing; the
 *       obvious failure is a gather launched over the batch that writes garbage at position 0.
 *   <li><b>Everything survives.</b> The compaction is an identity permutation, which is the case a
 *       gather that ignores its index array still passes.
 *   <li><b>Empty input.</b> No batch is flushed at all, so the second graph never runs.
 *   <li><b>A short last batch.</b> The compaction's row count is captured when the graph is built
 *       and is always the full batch, so the mask's tail has to be cleared or the partition emits
 *       rows from the previous batch. This is the case the design is most likely to get wrong.
 *   <li><b>NaN and infinities.</b> A comparison against NaN is false in SQL and in IEEE 754, so
 *       those rows must be dropped rather than kept; infinities compare normally and must survive.
 *   <li><b>Duplicate values.</b> Compaction is order-preserving, and equal values must not be
 *       reordered or coalesced.
 * </ul>
 */
class FusedCompactionDeviceIT {

    private static final RowType ROW = RowType.of(new IntType(false), new DoubleType(false));

    private static final DoubleType DOUBLE = new DoubleType(false);

    /** Small enough that a partition can straddle it in one test and fit inside it in another. */
    private static final int BATCH = 1024;

    /**
     * The threshold, in one place, because the IR and the host reference must not drift apart.
     *
     * <p>They did while this was being written: a test that fed rows the IR's predicate would drop
     * and a Java predicate that kept them reported a wrong row count and looked like a compaction
     * bug. Two statements of one predicate is one too many.
     */
    private static final double THRESHOLD = 2.0;

    private static final AcceleratorContext CONTEXT =
            new AcceleratorContext() {
                @Override
                public RowType outputType() {
                    return ROW;
                }

                @Override
                public int maxBatchSize() {
                    return BATCH;
                }

                @Override
                public ClassLoader userCodeClassLoader() {
                    return FusedCompactionDeviceIT.class.getClassLoader();
                }

                @Override
                public boolean providesOffHeap() {
                    return false;
                }

                @Override
                public ByteBuffer allocateOffHeap(int bytes) {
                    throw new IllegalStateException("this context provides none");
                }
            };

    @Test
    @DisplayName("a selective filter compacts on the device and agrees with the host exactly")
    void selectiveFilter() throws Exception {
        // Two thirds dropped, so the interconnect carries a third of what it otherwise would.
        run("a selective filter", 4096, i -> 1.0 + (i % 3));
    }

    @Test
    @DisplayName("nothing survives: a count of zero, and no row emitted")
    void nothingSurvives() throws Exception {
        run("nothing survives", 2048, i -> 1.0);
    }

    @Test
    @DisplayName("everything survives: the permutation is the identity and must still be applied")
    void everythingSurvives() throws Exception {
        run("everything survives", 2048, i -> 3.0 + i);
    }

    @Test
    @DisplayName("empty input produces no batch, no graph execution and no row")
    void emptyInput() throws Exception {
        Outcome outcome = run("empty input", 0, i -> 1.0);
        assertThat(outcome.emitted).isEmpty();
    }

    @Test
    @DisplayName("a partition that straddles a batch boundary, so the last batch is short")
    void shortLastBatch() throws Exception {
        // 1024 + 37: the tail of the second batch's mask still holds the first batch's verdicts,
        // and cuDF looks at the whole buffer whatever the live row count is.
        run("a short last batch", BATCH + 37, i -> 1.0 + (i % 5));
    }

    @Test
    @DisplayName("a partition exactly one batch long, where no tail exists to clear")
    void exactlyOneBatch() throws Exception {
        run("exactly one batch", BATCH, i -> 1.0 + (i % 5));
    }

    @Test
    @DisplayName("NaN fails every comparison, so those rows are dropped and not kept")
    void nanIsDropped() throws Exception {
        run("NaN in the predicate", 1024, i -> i % 7 == 0 ? Double.NaN : 1.0 + (i % 4));
    }

    @Test
    @DisplayName("infinities compare normally and survive or not on their own merits")
    void infinitiesCompareNormally() throws Exception {
        run(
                "infinities",
                1024,
                i -> {
                    if (i % 11 == 0) {
                        return Double.POSITIVE_INFINITY;
                    }
                    if (i % 11 == 1) {
                        return Double.NEGATIVE_INFINITY;
                    }
                    return 1.0 + (i % 4);
                });
    }

    @Test
    @DisplayName("duplicates keep their order and are not coalesced")
    void duplicatesKeepTheirOrder() throws Exception {
        run("duplicates", 2048, i -> 3.0);
    }

    // -------------------------------------------------------------------------------------------

    /** What one run produced, alongside what the same predicate produces in Java. */
    private static final class Outcome {
        private final List<RowData> emitted;

        Outcome(List<RowData> emitted) {
            this.emitted = emitted;
        }
    }

    @SuppressWarnings("unchecked")
    private Outcome run(String label, int rows, java.util.function.IntToDoubleFunction value)
            throws Exception {
        DeviceAssumptions.requireCudf();

        AccelNode subtree = plan(rows);
        Optional<AcceleratorPlan> offered =
                DeviceAssumptions.provider().accept(subtree, work(subtree, rows));
        assertThat(offered)
                .as("the provider must express a projection over a filter; if not, it regressed")
                .isPresent();

        StreamOperatorFactory<RowData> factory =
                DeviceAssumptions.provider().createOperator(offered.get(), CONTEXT);

        List<RowData> emitted = new ArrayList<>();
        String timings;
        try (OneInputStreamOperatorTestHarness<RowData, RowData> harness =
                new OneInputStreamOperatorTestHarness<>(factory, 1, 1, 0)) {
            harness.setup();
            harness.open();
            GpuCalcOperator operator = (GpuCalcOperator) harness.getOneInputOperator();

            assertThat(operator.compactedOnDevice())
                    .as(
                            "the fused path must be the one under test; a host drain would pass "
                                    + "every assertion below and prove nothing")
                    .isTrue();

            for (int i = 0; i < rows; i++) {
                GenericRowData row = new GenericRowData(2);
                row.setField(0, i);
                row.setField(1, value.applyAsDouble(i));
                harness.processElement(new StreamRecord<>(row));
            }
            operator.endInput();

            timings = operator.metrics().report("fused compaction -- " + label);

            harness.getOutput().stream()
                    .map(o -> ((StreamRecord<RowData>) o).getValue())
                    .forEach(emitted::add);
        }

        List<RowData> expected = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            double v = value.applyAsDouble(i);
            if (v > THRESHOLD) {
                GenericRowData out = new GenericRowData(2);
                out.setField(0, i);
                out.setField(1, v * 2.0);
                expected.add(out);
            }
        }

        assertThat(emitted)
                .as("%s: %d rows in, %d expected out", label, rows, expected.size())
                .hasSize(expected.size());
        for (int j = 0; j < expected.size(); j++) {
            assertThat(emitted.get(j).getInt(0))
                    .as("%s: survivor %d kept its identity and its position", label, j)
                    .isEqualTo(expected.get(j).getInt(0));
            assertThat(emitted.get(j).getDouble(1))
                    .as("%s: survivor %d computed the same value the host did", label, j)
                    .isEqualTo(expected.get(j).getDouble(1));
        }

        // Printed rather than asserted: staging, transfers, device work and drain are what the
        // fused path exists to move, and a number nobody can see is a claim nobody can check.
        System.out.println(timings);
        return new Outcome(emitted);
    }

    /** {@code SELECT id, v * 2.0 FROM t WHERE v > 2.0} in the accelerator IR, near enough. */
    private static AccelNode plan(int rows) {
        AccelNode input =
                new AccelFilter(
                        new AccelCall(
                                AccelFunction.GREATER_THAN,
                                Arrays.asList(
                                        new AccelInputRef(1, DOUBLE),
                                        new AccelLiteral(THRESHOLD, DOUBLE)),
                                new BooleanType(false)),
                        new AccelInput(ROW),
                        ROW);
        return new AccelProject(
                Arrays.asList(
                        new AccelInputRef(0, new IntType(false)),
                        new AccelCall(
                                AccelFunction.TIMES,
                                Arrays.asList(
                                        new AccelInputRef(1, DOUBLE),
                                        new AccelLiteral(2.0, DOUBLE)),
                                DOUBLE)),
                input,
                ROW);
    }

    private static AccelWorkProfile work(AccelNode subtree, int rows) {
        int[] ops = new int[AccelFunction.values().length];
        ops[AccelFunction.TIMES.ordinal()] = 1;
        ops[AccelFunction.GREATER_THAN.ordinal()] = 1;
        return new AccelWorkProfile(ops, 12, 12, Math.max(1, rows));
    }
}
