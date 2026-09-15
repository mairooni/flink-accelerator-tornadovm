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
import org.apache.flink.table.runtime.accelerator.AcceleratorProvider;
import org.apache.flink.table.runtime.accelerator.AcceleratorProviders;
import org.apache.flink.table.types.logical.BooleanType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.DoubleUnaryOperator;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The offload path, on a real device.
 *
 * <p>Everything else in this project runs on a CPU. Flink's own suites run against a host-side test
 * provider, which is the right way to test the <em>plumbing</em> — it makes the accelerator path
 * exercisable on any machine — but it means a green build says nothing about whether a kernel
 * compiles, runs, or returns the right numbers. This is the test that does, and it is the reason
 * the {@code device} profile exists.
 *
 * <p>Skips rather than fails without a card. See {@link DeviceAssumptions}.
 *
 * <p>Note what is asserted at the end: not that the operator was <em>built</em>, which proves only
 * that the provider did not throw, but that batches ran and a kernel spent time. Those come from
 * the engine's own metrics, and they are the difference between "a device path exists" and "a
 * device ran this".
 */
class AcceleratedCalcDeviceIT {

    private static final int ROWS = 50_000;
    private static final LogicalType DOUBLE = new DoubleType();

    /** What a TaskManager tells a provider about the task it is being built into. */
    private static final AcceleratorContext CONTEXT =
            new AcceleratorContext() {
                @Override
                public RowType outputType() {
                    return RowType.of(new IntType(), new DoubleType());
                }

                @Override
                public int maxBatchSize() {
                    return 16_384;
                }

                @Override
                public ClassLoader userCodeClassLoader() {
                    return AcceleratedCalcDeviceIT.class.getClassLoader();
                }
            };

    @BeforeEach
    void requireDevice() {
        DeviceAssumptions.requireDevice();
    }

    @Test
    @DisplayName("ServiceLoader finds this provider, which is how a TaskManager will")
    void discoverableThroughTheSpi() {
        Optional<AcceleratorProvider> found =
                AcceleratorProviders.find(getClass().getClassLoader());

        assertThat(found).isPresent();
        assertThat(found.get().name()).isEqualTo("tornadovm");
    }

    @Test
    @DisplayName("a transfer-bound expression is declined, and being declined is an answer")
    void declinesWhatIsNotWorthMoving() {
        // val * 2.0 + 1.0: three operations over 24 bytes a row. The bus costs more than the
        // arithmetic saves, and the provider says so rather than accepting and losing.
        AccelNode subtree =
                plan(
                        call(
                                AccelFunction.PLUS,
                                call(AccelFunction.TIMES, col(1), lit(2.0)),
                                lit(1.0)),
                        null);

        Optional<AcceleratorPlan> offered =
                DeviceAssumptions.provider().accept(subtree, work(subtree));

        assertThat(offered).isPresent();
        assertThat(offered.get().cost().speedupOverCpu())
                .as("a claim below 1.0 is the provider declining to win, not an error")
                .isLessThan(1.0);
    }

    @Test
    @DisplayName("the headline expression agrees with the host bit for bit")
    void headlineAgreesExactly() throws Exception {
        Result result = runOnDevice(headline(), AcceleratedCalcDeviceIT::headlineOnHost);

        assertThat(result.batches).as("no batch ran, so nothing reached the device").isPositive();
        assertThat(result.kernelNanos).as("batches ran but no kernel time was spent").isPositive();
        assertThat(result.emitted).hasSameSizeAs(result.expected);

        for (int i = 0; i < result.expected.size(); i++) {
            assertThat(result.emitted.get(i).getInt(0))
                    .as("pass-through column at row " + i)
                    .isEqualTo(result.expected.get(i).getInt(0));
            // Exact, with no tolerance anywhere near it. This is the expression every document in
            // the project uses as the example, and if a user could tell which processor ran it the
            // premise of the whole path would be gone.
            assertThat(result.emitted.get(i).getDouble(1))
                    .as("computed column at row " + i)
                    .isEqualTo(result.expected.get(i).getDouble(1));
        }
    }

    /**
     * How far a long arithmetic chain drifts from the host, measured rather than assumed.
     *
     * <p><b>This does not say the drift is acceptable.</b> It bounds it, because an unbounded
     * regression would be far worse and nothing else in the project can see this at all: Flink's
     * differential harness runs against a host-side provider that evaluates with {@code Math} and
     * therefore agrees with the CPU by construction.
     *
     * <p>What was measured, and the shape is odd enough to be worth writing down:
     *
     * <ul>
     *   <li>{@code val * 2.0 + 1.0} — exact.
     *   <li>{@code val * 1.0009765625 + 0.5} — exact, product inexact and it still agrees.
     *   <li>twenty multiplies, no adds — exact.
     *   <li>twenty multiplies followed by twenty adds — differs, about 1 ulp.
     *   <li>twenty interleaved multiply-adds — differs, about 1 ulp.
     * </ul>
     *
     * <p>FMA contraction is the obvious suspect and the third and fourth lines do not fit it
     * cleanly, so <b>the cause is not established</b> and this comment does not claim one. Chasing
     * it further belongs in the blocker recorded in the project's TASKS.md, not here.
     */
    @Test
    @DisplayName("a long arithmetic chain drifts from the host, and by how much is measured")
    void longChainDriftIsBounded() throws Exception {
        assertDriftWithin(runOnDevice(longChain(), AcceleratedCalcDeviceIT::longChainOnHost), 4);
    }

    /**
     * The same for transcendentals, where a divergence is expected rather than surprising.
     *
     * <p>Java specifies {@code exp}, {@code log}, {@code sin}, {@code cos} and {@code pow} to
     * within 1 ulp and semi-monotonic; CUDA specifies its own to a few ulp; neither is required to
     * match the other, and they do not. First run: 2.1756620143894665 on the host against
     * 2.175662014389466 on the device.
     *
     * <p>Unresolved, and it matters more here than it would in a library. Refusing transcendentals,
     * shipping a correctly-rounded device implementation, or documenting the divergence the way
     * signed zero already is — that is a design decision, recorded as a blocker, not something this
     * test settles by widening until it passes.
     */
    @Test
    @DisplayName("transcendentals diverge from the host, and by how much is measured")
    void transcendentalDriftIsBounded() throws Exception {
        assertDriftWithin(runOnDevice(heavy(), AcceleratedCalcDeviceIT::hostReference), 4);
    }

    private static void assertDriftWithin(Result result, long maxUlps) {
        assertThat(result.batches).as("no batch ran, so nothing reached the device").isPositive();
        assertThat(result.emitted).hasSameSizeAs(result.expected);

        long exact = 0;
        long worstUlps = 0;
        for (int i = 0; i < result.expected.size(); i++) {
            double got = result.emitted.get(i).getDouble(1);
            double want = result.expected.get(i).getDouble(1);
            if (Double.compare(got, want) == 0) {
                exact++;
                continue;
            }
            worstUlps = Math.max(worstUlps, ulpsBetween(got, want));
        }

        assertThat(worstUlps)
                .as(
                        "drifted %d ulp from the host (%d of %d rows exact); a few ulp is the"
                                + " measured state, a large number is a regression",
                        worstUlps, exact, result.expected.size())
                .isLessThanOrEqualTo(maxUlps);
    }

    /** Distance between two doubles in representable steps. */
    private static long ulpsBetween(double a, double b) {
        long x = Double.doubleToLongBits(a);
        long y = Double.doubleToLongBits(b);
        if ((x < 0) != (y < 0)) {
            // Straddles zero; the bit patterns are not comparable as a signed distance.
            return Math.abs(x) + Math.abs(y);
        }
        return Math.abs(x - y);
    }

    /** What came back from the device, and what the host says it should have been. */
    private static final class Result {
        final List<RowData> emitted;
        final List<RowData> expected;
        final long batches;
        final long kernelNanos;

        Result(List<RowData> emitted, List<RowData> expected, long batches, long kernelNanos) {
            this.emitted = emitted;
            this.expected = expected;
            this.batches = batches;
            this.kernelNanos = kernelNanos;
        }
    }

    @SuppressWarnings("unchecked")
    private Result runOnDevice(AccelExpression projection, DoubleUnaryOperator onHost)
            throws Exception {
        AccelNode subtree =
                plan(projection, predicate(AccelFunction.GREATER_THAN, col(1), lit(1.0)));

        Optional<AcceleratorPlan> offered =
                DeviceAssumptions.provider().accept(subtree, work(subtree));
        assertThat(offered)
                .as("the provider must be able to express this; if not, the generator regressed")
                .isPresent();

        StreamOperatorFactory<RowData> factory =
                DeviceAssumptions.provider().createOperator(offered.get(), CONTEXT);

        List<RowData> emitted = new ArrayList<>();
        long batches;
        long kernelNanos;

        try (OneInputStreamOperatorTestHarness<RowData, RowData> harness =
                new OneInputStreamOperatorTestHarness<>(factory, 1, 1, 0)) {
            harness.setup();
            harness.open();
            GpuCalcOperator operator = (GpuCalcOperator) harness.getOneInputOperator();

            for (int i = 0; i < ROWS; i++) {
                harness.processElement(new StreamRecord<>(row(i, value(i))));
            }
            operator.endInput();

            // Read the metrics before close(), which releases the engine.
            batches = operator.metrics().getBatches();
            kernelNanos = operator.metrics().getKernelNanos();

            harness.getOutput().stream()
                    .map(o -> ((StreamRecord<RowData>) o).getValue())
                    .forEach(emitted::add);
        }

        List<RowData> expected = new ArrayList<>();
        for (int i = 0; i < ROWS; i++) {
            double v = value(i);
            if (v > 1.0) {
                expected.add(row(i, onHost.applyAsDouble(v)));
            }
        }
        return new Result(emitted, expected, batches, kernelNanos);
    }

    // ---------------------------------------------------------------------------------------
    // The expression under test, in three forms that must agree: IR, host reference, and the
    // kernel the provider generates from the IR.
    // ---------------------------------------------------------------------------------------

    /** The project's headline expression: {@code val * 2.0 + 1.0}. */
    private static AccelExpression headline() {
        return call(AccelFunction.PLUS, call(AccelFunction.TIMES, col(1), lit(2.0)), lit(1.0));
    }

    private static double headlineOnHost(double v) {
        return v * 2.0 + 1.0;
    }

    /** Twenty multiply-adds: {@code ((((v*a + b) * a + b) ...))}, 40 operations a row. */
    private static AccelExpression longChain() {
        AccelExpression e = col(1);
        for (int i = 0; i < 20; i++) {
            e = call(AccelFunction.PLUS, call(AccelFunction.TIMES, e, lit(1.0009765625)), lit(0.5));
        }
        return e;
    }

    private static double longChainOnHost(double v) {
        double e = v;
        for (int i = 0; i < 20; i++) {
            e = e * 1.0009765625 + 0.5;
        }
        return e;
    }

    /** EXP(v) * LN(v) + SIN(v) * COS(v) + POWER(v, 3.0) */
    private static AccelExpression heavy() {
        return call(
                AccelFunction.PLUS,
                call(
                        AccelFunction.PLUS,
                        call(
                                AccelFunction.TIMES,
                                call(AccelFunction.EXP, col(1)),
                                call(AccelFunction.LN, col(1))),
                        call(
                                AccelFunction.TIMES,
                                call(AccelFunction.SIN, col(1)),
                                call(AccelFunction.COS, col(1)))),
                call(AccelFunction.POWER, col(1), lit(3.0)));
    }

    private static double hostReference(double v) {
        return Math.exp(v) * Math.log(v) + Math.sin(v) * Math.cos(v) + Math.pow(v, 3.0);
    }

    /** Small and positive: LN needs it positive, EXP needs it small enough not to overflow. */
    private static double value(int i) {
        return 1.0 + (i % 64) * 0.125;
    }

    // ---------------------------------------------------------------------------------------

    private static RowData row(int id, double computed) {
        GenericRowData row = new GenericRowData(2);
        row.setField(0, id);
        row.setField(1, computed);
        return row;
    }

    private static AccelNode plan(AccelExpression projection, AccelExpression condition) {
        RowType inputType = RowType.of(new IntType(), new DoubleType());
        AccelNode input = new AccelInput(inputType);
        if (condition != null) {
            input = new AccelFilter(condition, input, inputType);
        }
        return new AccelProject(
                Arrays.asList(col(0, new IntType()), projection),
                input,
                RowType.of(new IntType(), new DoubleType()));
    }

    /**
     * The histogram Flink would ship, counted from the IR exactly as the planner counts it.
     *
     * <p>Attributing a total to some plausible mix instead does not work, and the first version of
     * this test proved it: the provider prices a transcendental at twenty-four times a multiply, so
     * three operations "mostly sin" claims a speedup and three operations of real arithmetic does
     * not. A cost model that distinguishes functions has to be fed the functions.
     */
    private static AccelWorkProfile work(AccelNode subtree) {
        int[] ops = new int[AccelFunction.values().length];
        countInto(subtree, ops);
        return new AccelWorkProfile(ops, 12, 12, ROWS);
    }

    private static void countInto(AccelNode node, int[] ops) {
        if (node instanceof AccelProject) {
            ((AccelProject) node).projections().forEach(e -> countInto(e, ops));
        } else if (node instanceof AccelFilter) {
            countInto(((AccelFilter) node).condition(), ops);
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

    private static AccelExpression col(int index) {
        return new AccelInputRef(index, DOUBLE);
    }

    private static AccelExpression col(int index, LogicalType type) {
        return new AccelInputRef(index, type);
    }

    private static AccelExpression lit(double value) {
        return new AccelLiteral(value, DOUBLE);
    }

    private static AccelExpression call(AccelFunction function, AccelExpression... operands) {
        return new AccelCall(function, Arrays.asList(operands), DOUBLE);
    }

    private static AccelExpression predicate(AccelFunction function, AccelExpression... operands) {
        return new AccelCall(function, Arrays.asList(operands), new BooleanType());
    }
}
