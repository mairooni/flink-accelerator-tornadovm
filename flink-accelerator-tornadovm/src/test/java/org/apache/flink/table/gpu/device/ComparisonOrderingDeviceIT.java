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
import org.apache.flink.table.accelerator.AccelExpression;
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
import org.apache.flink.table.types.logical.DecimalType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
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
 * SQL has two orderings for a comparison, and a kernel has to generate the one Flink would.
 *
 * <p>{@code WHERE val > 0.5} and {@code WHERE val > CAST(0.5 AS DOUBLE)} look like the same
 * predicate and are not. A bare literal is {@code DECIMAL}, and Flink compares an approximate
 * numeric against a decimal with {@code Double.compare} — a total order in which NaN is greater
 * than everything and equal to itself, and {@code -0.0} is strictly below {@code 0.0}. Written as a
 * {@code DOUBLE}, the same predicate is IEEE: every comparison involving NaN is false and the two
 * zeros are equal.
 *
 * <p>Generating IEEE for both changes query results rather than rounding them — a NaN row Flink
 * keeps is one the device drops. This runs both orderings, over both hazards, through all six
 * comparison operators, on a real device, and checks each against what {@code Double.compare} and
 * what IEEE say in Java.
 *
 * <p>The host-side half of the same guarantee is {@code AcceleratorConformanceIT} in the Flink
 * repository, which drives whole SQL queries through the planner. This one goes through the kernel
 * generator, which is the part that has to emit the extra terms.
 */
class ComparisonOrderingDeviceIT {

    private static final DoubleType DOUBLE = new DoubleType(false);

    /** Whatever a bare {@code 0.0} or {@code 1.0} arrives as: exact, small, and never NaN. */
    private static final DecimalType DECIMAL = new DecimalType(false, 2, 1);

    private static final RowType ROW = RowType.of(new IntType(false), new DoubleType(false));

    private static final AcceleratorContext CONTEXT =
            new AcceleratorContext() {
                @Override
                public RowType outputType() {
                    return ROW;
                }

                @Override
                public int maxBatchSize() {
                    return 1024;
                }

                @Override
                public ClassLoader userCodeClassLoader() {
                    return ComparisonOrderingDeviceIT.class.getClassLoader();
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

    /** Every value that makes the two orderings disagree, plus ordinary ones that must not. */
    private static final double[] VALUES = {
        Double.NaN,
        -0.0,
        0.0,
        1.0,
        -1.0,
        0.5,
        Double.POSITIVE_INFINITY,
        Double.NEGATIVE_INFINITY,
        2.0,
        -2.0
    };

    @Test
    @DisplayName("a decimal literal compares as a total order, NaN greatest, on the device")
    void decimalLiteralUsesTheTotalOrder() throws Exception {
        for (AccelFunction op : comparisons()) {
            for (double literal : new double[] {0.0, 1.0}) {
                List<Double> kept = runFilter(op, literal, DECIMAL);
                List<Double> expected = new ArrayList<>();
                for (double v : VALUES) {
                    if (totalOrderHolds(op, Double.compare(v, literal))) {
                        expected.add(v);
                    }
                }
                assertSameValues(op + " vs decimal " + literal, expected, kept);
            }
        }
    }

    @Test
    @DisplayName("a DOUBLE literal compares with IEEE rules, NaN failing everything")
    void doubleLiteralUsesIeee() throws Exception {
        for (AccelFunction op : comparisons()) {
            for (double literal : new double[] {0.0, 1.0}) {
                List<Double> kept = runFilter(op, literal, DOUBLE);
                List<Double> expected = new ArrayList<>();
                for (double v : VALUES) {
                    if (ieeeHolds(op, v, literal)) {
                        expected.add(v);
                    }
                }
                assertSameValues(op + " vs double " + literal, expected, kept);
            }
        }
    }

    @Test
    @DisplayName("the literal on the left is the same comparison read backwards")
    void theLiteralMayBeOnTheLeft() throws Exception {
        for (AccelFunction op : comparisons()) {
            List<Double> kept = runFilterLiteralFirst(op, 0.0);
            List<Double> expected = new ArrayList<>();
            for (double v : VALUES) {
                // literal OP value, so the comparison result is compare(literal, value).
                if (totalOrderHolds(op, Double.compare(0.0, v))) {
                    expected.add(v);
                }
            }
            assertSameValues(op + " with the literal first", expected, kept);
        }
    }

    // -------------------------------------------------------------------------------------------

    private static AccelFunction[] comparisons() {
        return new AccelFunction[] {
            AccelFunction.GREATER_THAN,
            AccelFunction.GREATER_OR_EQUAL,
            AccelFunction.LESS_THAN,
            AccelFunction.LESS_OR_EQUAL,
            AccelFunction.EQUALS,
            AccelFunction.NOT_EQUALS
        };
    }

    /** What {@code DecimalDataUtils.compare(l, r) op 0} answers, given the comparison's sign. */
    private static boolean totalOrderHolds(AccelFunction op, int compare) {
        switch (op) {
            case GREATER_THAN:
                return compare > 0;
            case GREATER_OR_EQUAL:
                return compare >= 0;
            case LESS_THAN:
                return compare < 0;
            case LESS_OR_EQUAL:
                return compare <= 0;
            case EQUALS:
                return compare == 0;
            case NOT_EQUALS:
                return compare != 0;
            default:
                throw new IllegalArgumentException(op.name());
        }
    }

    /** What Java's own operators answer, which is what Flink emits for two approximate numerics. */
    private static boolean ieeeHolds(AccelFunction op, double a, double b) {
        switch (op) {
            case GREATER_THAN:
                return a > b;
            case GREATER_OR_EQUAL:
                return a >= b;
            case LESS_THAN:
                return a < b;
            case LESS_OR_EQUAL:
                return a <= b;
            case EQUALS:
                return a == b;
            case NOT_EQUALS:
                return a != b;
            default:
                throw new IllegalArgumentException(op.name());
        }
    }

    private static void assertSameValues(String what, List<Double> expected, List<Double> actual) {
        assertThat(actual)
                .as("%s: kept the wrong rows", what)
                .usingElementComparator(Double::compare)
                .containsExactlyInAnyOrderElementsOf(expected);
    }

    private List<Double> runFilter(AccelFunction op, double literal, LogicalType literalType)
            throws Exception {
        return run(
                plan(
                        new AccelCall(
                                op,
                                Arrays.asList(
                                        new AccelInputRef(1, DOUBLE),
                                        new AccelLiteral(literal, literalType)),
                                new BooleanType(false))));
    }

    private List<Double> runFilterLiteralFirst(AccelFunction op, double literal) throws Exception {
        return run(
                plan(
                        new AccelCall(
                                op,
                                Arrays.asList(
                                        new AccelLiteral(literal, DECIMAL),
                                        new AccelInputRef(1, DOUBLE)),
                                new BooleanType(false))));
    }

    /**
     * The projection carries the value through a multiply by one, which is not decoration.
     *
     * <p>A projection of nothing but input references has no computed column, and the provider
     * declines it — there would be no kernel to generate. Multiplying by {@code 1.0} gives it one
     * while preserving every value under test exactly: NaN stays NaN, an infinity stays itself, and
     * {@code -0.0 * 1.0} is still {@code -0.0}, which is the whole point of half these rows.
     */
    private static AccelNode plan(AccelExpression condition) {
        return new AccelProject(
                Arrays.asList(
                        new AccelInputRef(0, new IntType(false)),
                        new AccelCall(
                                AccelFunction.TIMES,
                                Arrays.asList(
                                        new AccelInputRef(1, DOUBLE),
                                        new AccelLiteral(1.0, DOUBLE)),
                                DOUBLE)),
                new AccelFilter(condition, new AccelInput(ROW), ROW),
                ROW);
    }

    private static void countInto(Object node, int[] ops) {
        if (node instanceof AccelCall) {
            AccelCall call = (AccelCall) node;
            ops[call.function().ordinal()]++;
            call.operands().forEach(operand -> countInto(operand, ops));
            return;
        }
        if (node instanceof AccelProject) {
            ((AccelProject) node).projections().forEach(e -> countInto(e, ops));
        } else if (node instanceof AccelFilter) {
            countInto(((AccelFilter) node).condition(), ops);
        }
        if (node instanceof AccelNode) {
            ((AccelNode) node).inputs().forEach(child -> countInto(child, ops));
        }
    }

    @SuppressWarnings("unchecked")
    private List<Double> run(AccelNode subtree) throws Exception {
        DeviceAssumptions.requireDevice();

        int[] ops = new int[AccelFunction.values().length];
        countInto(subtree, ops);
        // A row count the provider will take seriously. Ten rows is what this test actually feeds
        // -- the values are chosen, not generated -- and a provider is entitled to decline a
        // partition that cannot repay setting a device up. The semantics under test do not depend
        // on how many rows carry them.
        Optional<AcceleratorPlan> offered =
                DeviceAssumptions.provider()
                        .accept(subtree, new AccelWorkProfile(ops, 12, 12, 4_000_000L));
        assertThat(offered)
                .as("the provider must express a comparison against a literal")
                .isPresent();

        StreamOperatorFactory<RowData> factory =
                DeviceAssumptions.provider().createOperator(offered.get(), CONTEXT);

        List<Double> kept = new ArrayList<>();
        try (OneInputStreamOperatorTestHarness<RowData, RowData> harness =
                new OneInputStreamOperatorTestHarness<>(factory, 1, 1, 0)) {
            harness.setup();
            harness.open();
            GpuCalcOperator operator = (GpuCalcOperator) harness.getOneInputOperator();
            for (int i = 0; i < VALUES.length; i++) {
                GenericRowData row = new GenericRowData(2);
                row.setField(0, i);
                row.setField(1, VALUES[i]);
                harness.processElement(new StreamRecord<>(row));
            }
            operator.endInput();
            harness.getOutput().stream()
                    .map(o -> ((StreamRecord<RowData>) o).getValue())
                    .forEach(r -> kept.add(r.getDouble(1)));
        }
        return kept;
    }
}
