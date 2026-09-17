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
import org.apache.flink.table.accelerator.AccelAggCall;
import org.apache.flink.table.accelerator.AccelAggFunction;
import org.apache.flink.table.accelerator.AccelAggregate;
import org.apache.flink.table.accelerator.AccelCall;
import org.apache.flink.table.accelerator.AccelExpression;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelLiteral;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.operator.GpuGroupedAggregateOperator;
import org.apache.flink.table.runtime.accelerator.AcceleratorContext;
import org.apache.flink.table.runtime.accelerator.AcceleratorPlan;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;

/**
 * A local {@code GROUP BY} running as a generated kernel and a cuDF group-by in one task graph.
 *
 * <p>What this is really testing is the composition. Both halves are already covered — the kernel
 * by {@link AcceleratedCalcDeviceIT}, {@code Cudf.groupSum} by TornadoVM's own suite — and neither
 * catches the thing that can go wrong between them: the group-by reads the columns the kernel wrote
 * where they lie on the device, so a mistake in which buffer holds which field is silent, produces
 * plausible numbers, and shows up only as a wrong sum.
 *
 * <p>Skips without a device, and separately without the cuDF shim, which is the usual case even on
 * a machine with a card. See {@link DeviceAssumptions#requireCudf()}.
 */
class AcceleratedGroupDeviceIT {

    /** Deliberately not a multiple of the batch size, so the short tail batch runs too. */
    private static final int ROWS = 50_000;

    private static final int BATCH = 16_384;

    /** How many distinct keys the rows fall into. */
    private static final int GROUPS = 64;

    private static final LogicalType DOUBLE = new DoubleType(false);
    private static final LogicalType INT = new IntType(false);

    /** The projection's row, which is the aggregate's input: (key, weighted value). */
    private static final RowType PROJECTED = RowType.of(INT, DOUBLE);

    /** What the aggregate emits: the key, then the partial sum. */
    private static final RowType GROUPED = RowType.of(INT, DOUBLE);

    private static final AcceleratorContext CONTEXT =
            new AcceleratorContext() {
                @Override
                public RowType outputType() {
                    return GROUPED;
                }

                @Override
                public int maxBatchSize() {
                    return BATCH;
                }

                @Override
                public ClassLoader userCodeClassLoader() {
                    return AcceleratedGroupDeviceIT.class.getClassLoader();
                }

                @Override
                public boolean providesOffHeap() {
                    return false;
                }

                @Override
                public ByteBuffer allocateOffHeap(int bytes) {
                    throw new IllegalStateException("no managed memory in this harness");
                }
            };

    @BeforeEach
    void requireCudf() {
        DeviceAssumptions.requireCudf();
    }

    @Test
    @DisplayName("the device's group sums agree with the host's")
    void groupedSumsAgreeWithTheHost() throws Exception {
        AccelNode subtree = groupedPlan();

        Optional<AcceleratorPlan> offered =
                DeviceAssumptions.provider().accept(subtree, work(subtree));
        assertThat(offered).as("the provider declined a shape it is meant to serve").isPresent();

        StreamOperatorFactory<RowData> factory =
                DeviceAssumptions.provider().createOperator(offered.get(), CONTEXT);

        Map<Integer, Double> expected = new HashMap<>();
        Map<Integer, Double> actual = new HashMap<>();
        int emitted = 0;
        long batches;
        long kernelNanos;
        try (OneInputStreamOperatorTestHarness<RowData, RowData> harness =
                new OneInputStreamOperatorTestHarness<>(factory, 1, 1, 0)) {
            harness.setup();
            harness.open();

            for (int i = 0; i < ROWS; i++) {
                GenericRowData row = new GenericRowData(2);
                row.setField(0, i);
                row.setField(1, value(i));
                harness.processElement(new StreamRecord<>(row));
                expected.merge(i % GROUPS, weigh(value(i)), Double::sum);
            }
            GpuGroupedAggregateOperator operator =
                    (GpuGroupedAggregateOperator) harness.getOneInputOperator();
            operator.endInput();
            // Not that the operator was built, which proves only that the provider did not throw,
            // but that batches ran and a kernel spent time on them.
            batches = operator.metrics().getBatches();
            kernelNanos = operator.metrics().getKernelNanos();

            for (Object o : harness.getOutput()) {
                @SuppressWarnings("unchecked")
                RowData out = ((StreamRecord<RowData>) o).getValue();
                actual.merge(out.getInt(0), out.getDouble(1), Double::sum);
                emitted++;
            }
        }

        // More rows than groups is not a defect: this is the local half of the aggregate and it
        // emits each batch's groups as it produces them. The merge stage downstream is what makes
        // one row per key, and summing by key here is exactly what it does.
        assertThat(batches)
                .as("no batch ran, so nothing reached the device")
                .isEqualTo(ROWS / BATCH + 1);
        assertThat(kernelNanos).as("batches ran but no kernel time was spent").isPositive();
        assertThat(emitted)
                .as("a partial aggregate emits per batch, so four batches of 64 keys are expected")
                .isGreaterThan(GROUPS);
        assertThat(actual.keySet()).isEqualTo(expected.keySet());
        for (Map.Entry<Integer, Double> group : expected.entrySet()) {
            // A tolerance, not an exact match, and only because the addition order differs: the
            // device sums a group in whatever order cuDF reduces it and the host sums it in row
            // order. Every individual term is bit-exact, which AcceleratedCalcDeviceIT asserts.
            assertThat(actual.get(group.getKey()))
                    .as("group " + group.getKey())
                    .isCloseTo(group.getValue(), offset(1e-6));
        }
    }

    @Test
    @DisplayName("two aggregate calls are declined rather than half-served")
    void twoAggregatesAreDeclined() {
        AccelAggCall sum = new AccelAggCall(AccelAggFunction.SUM, 1, DOUBLE);
        AccelNode subtree =
                new AccelAggregate(
                        new int[] {0},
                        Arrays.asList(sum, sum),
                        projection(),
                        RowType.of(INT, DOUBLE, DOUBLE));

        assertThat(DeviceAssumptions.provider().accept(subtree, work(subtree)))
                .as("Cudf.groupSum computes one sum; claiming two would emit a wrong column")
                .isEmpty();
    }

    @Test
    @DisplayName("a COUNT(*) is declined, because the binding sums and does not count")
    void countStarIsDeclined() {
        AccelNode subtree =
                new AccelAggregate(
                        new int[] {0},
                        Collections.singletonList(
                                new AccelAggCall(
                                        AccelAggFunction.COUNT_STAR,
                                        AccelAggCall.NO_INPUT_FIELD,
                                        new BigIntType(false))),
                        projection(),
                        RowType.of(INT, new BigIntType(false)));

        assertThat(DeviceAssumptions.provider().accept(subtree, work(subtree))).isEmpty();
    }

    // --------------------------------------------------------------------------------------
    // The plan under test, and the host's answer to it
    // --------------------------------------------------------------------------------------

    /** {@code SELECT id % 64, SUM(val * 2.0 + 1.0) ... GROUP BY id % 64}, as IR. */
    private static AccelNode groupedPlan() {
        return new AccelAggregate(
                new int[] {0},
                Collections.singletonList(new AccelAggCall(AccelAggFunction.SUM, 1, DOUBLE)),
                projection(),
                GROUPED);
    }

    private static AccelProject projection() {
        AccelExpression id = new AccelInputRef(0, INT);
        // id - (id / 64) * 64 -- the integer remainder, written the way the planner lowers MOD.
        AccelExpression key =
                new AccelCall(
                        AccelFunction.MINUS,
                        Arrays.asList(
                                id,
                                new AccelCall(
                                        AccelFunction.TIMES,
                                        Arrays.asList(
                                                new AccelCall(
                                                        AccelFunction.INT_DIVIDE,
                                                        Arrays.asList(
                                                                id, new AccelLiteral(GROUPS, INT)),
                                                        INT),
                                                new AccelLiteral(GROUPS, INT)),
                                        INT)),
                        INT);
        AccelExpression weighted =
                new AccelCall(
                        AccelFunction.PLUS,
                        Arrays.asList(
                                new AccelCall(
                                        AccelFunction.TIMES,
                                        Arrays.asList(
                                                new AccelInputRef(1, DOUBLE),
                                                new AccelLiteral(2.0, DOUBLE)),
                                        DOUBLE),
                                new AccelLiteral(1.0, DOUBLE)),
                        DOUBLE);
        return new AccelProject(
                Arrays.asList(key, weighted), new AccelInput(RowType.of(INT, DOUBLE)), PROJECTED);
    }

    private static double value(int i) {
        return 1.0 + (i % 997) * 0.125;
    }

    private static double weigh(double value) {
        return value * 2.0 + 1.0;
    }

    private static AccelWorkProfile work(AccelNode subtree) {
        int[] ops = new int[AccelFunction.values().length];
        countInto(subtree, ops);
        return new AccelWorkProfile(ops, 12, 12, ROWS);
    }

    private static void countInto(AccelNode node, int[] ops) {
        if (node instanceof AccelProject) {
            ((AccelProject) node).projections().forEach(e -> countInto(e, ops));
        }
        node.inputs().forEach(child -> countInto(child, ops));
    }

    private static void countInto(AccelExpression expression, int[] ops) {
        if (expression instanceof AccelCall) {
            AccelCall call = (AccelCall) expression;
            ops[call.function().ordinal()]++;
            call.operands().forEach(operand -> countInto(operand, ops));
        }
    }
}
