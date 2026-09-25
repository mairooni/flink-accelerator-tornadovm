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

package org.apache.flink.table.gpu.codegen;

import org.apache.flink.table.accelerator.AccelCall;
import org.apache.flink.table.accelerator.AccelExpression;
import org.apache.flink.table.accelerator.AccelFilter;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelLiteral;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.types.logical.BooleanType;
import org.apache.flink.table.types.logical.DecimalType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * What the default refuses, so that an offloaded predicate cannot change which rows a query
 * returns.
 *
 * <p>The case this guards is measured rather than imagined: on the screening workload a row scored
 * {@code 8.86821620754203} on the host and {@code 8.868216207542035} on the device, and a query
 * with that threshold returned 289,114 rows one way and 289,115 the other. The rule is therefore
 * about the <em>condition</em>, not about arithmetic in general.
 */
class DriftSensitivePredicateTest {

    private static final DoubleType DOUBLE = new DoubleType(false);
    private static final RowType ROW = RowType.of(new IntType(false), new DoubleType(false));

    @AfterEach
    void clearResearchMode() {
        System.clearProperty("tornadovm.accelerator.approximate-predicates");
    }

    @Test
    @DisplayName("a filter over a transcendental is refused by default")
    void transcendentalPredicateIsRefused() {
        for (AccelFunction fn : DriftSensitivePredicate.DRIFT_SENSITIVE) {
            AccelNode plan = filtered(call(fn, col(1)));
            assertThat(DriftSensitivePredicate.refuse(plan))
                    .as("a predicate using %s must be refused", fn)
                    .isNotNull()
                    .contains(fn.name());
        }
    }

    @Test
    @DisplayName("a filter over exact arithmetic is allowed")
    void exactPredicateIsAllowed() {
        // +, -, *, / and the comparisons are exact under IEEE 754 and agree bit for bit.
        assertThat(DriftSensitivePredicate.refuse(filtered(col(1)))).isNull();
        assertThat(
                        DriftSensitivePredicate.refuse(
                                filtered(call(AccelFunction.TIMES, col(1), lit(2.0)))))
                .isNull();
        // ABS, FLOOR, CEIL and SIGN inspect or clear bits; they do not approximate.
        for (AccelFunction fn :
                new AccelFunction[] {
                    AccelFunction.ABS, AccelFunction.FLOOR, AccelFunction.CEIL, AccelFunction.SIGN
                }) {
            assertThat(DriftSensitivePredicate.refuse(filtered(call(fn, col(1)))))
                    .as("%s is exact and must not be refused", fn)
                    .isNull();
        }
    }

    @Test
    @DisplayName("a transcendental nested deep inside the condition is still found")
    void nestedTranscendentalIsFound() {
        AccelExpression deep =
                call(
                        AccelFunction.PLUS,
                        call(AccelFunction.TIMES, col(1), lit(2.0)),
                        call(AccelFunction.EXP, col(1)));
        assertThat(DriftSensitivePredicate.refuse(filtered(deep))).isNotNull().contains("EXP");
    }

    @Test
    @DisplayName("a transcendental in the projection is allowed; only the condition is refused")
    void projectionMayUseTranscendentals() {
        // The projection case is a numerical limitation on an output column, not a membership
        // change, and is documented separately rather than refused.
        AccelNode plan =
                new AccelProject(
                        Arrays.asList(col(0, new IntType(false)), call(AccelFunction.SIN, col(1))),
                        new AccelFilter(
                                predicate(AccelFunction.GREATER_THAN, col(1), lit(2.0)),
                                new AccelInput(ROW),
                                ROW),
                        ROW);
        assertThat(DriftSensitivePredicate.refuse(plan)).isNull();
        assertThat(DriftSensitivePredicate.projectionMayDrift(plan)).isTrue();
    }

    @Test
    @DisplayName("a plan with no filter at all is never refused on this ground")
    void noFilterIsNeverRefused() {
        AccelNode plan =
                new AccelProject(
                        Arrays.asList(col(0, new IntType(false)), call(AccelFunction.EXP, col(1))),
                        new AccelInput(ROW),
                        ROW);
        assertThat(DriftSensitivePredicate.refuse(plan)).isNull();
    }

    @Test
    @DisplayName("the research mode allows it, and is off unless explicitly set")
    void researchModeIsOptIn() {
        AccelNode plan = filtered(call(AccelFunction.EXP, col(1)));
        assertThat(DriftSensitivePredicate.approximateAllowed()).isFalse();
        assertThat(DriftSensitivePredicate.refuse(plan)).isNotNull();

        System.setProperty("tornadovm.accelerator.approximate-predicates", "true");
        assertThat(DriftSensitivePredicate.approximateAllowed()).isTrue();
        assertThat(DriftSensitivePredicate.refuse(plan))
                .as("the research mode is the only way to get the approximate predicate")
                .isNull();
    }

    @Test
    @DisplayName("a decimal-literal operand does not smuggle a transcendental past the rule")
    void operandCoercionDoesNotHideIt() {
        // A bare SQL literal is DECIMAL, and the comparison it takes part in is rendered
        // differently from a DOUBLE one. The refusal is about the function, not the rendering, so
        // both spellings must be refused alike.
        AccelExpression decimalLit = new AccelLiteral(12.0, new DecimalType(false, 4, 1));
        AccelNode plan =
                new AccelProject(
                        Arrays.asList(col(0, new IntType(false)), col(1)),
                        new AccelFilter(
                                predicate(
                                        AccelFunction.GREATER_THAN,
                                        call(AccelFunction.EXP, col(1)),
                                        decimalLit),
                                new AccelInput(ROW),
                                ROW),
                        ROW);
        assertThat(DriftSensitivePredicate.refuse(plan)).isNotNull().contains("EXP");
    }

    @Test
    @DisplayName("comparisons against NaN, infinities and signed zero are still allowed when exact")
    void adversarialLiteralsAreNotRefusedWithoutATranscendental() {
        // These values are exactly where a predicate is most fragile, and exactly why the rule is
        // about the function rather than the operand: comparing against them is bit-for-bit
        // reproducible on both sides, so refusing here would cost coverage and buy nothing.
        for (double v :
                new double[] {
                    Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, -0.0, 0.0
                }) {
            AccelNode plan =
                    new AccelProject(
                            Arrays.asList(col(0, new IntType(false)), col(1)),
                            new AccelFilter(
                                    predicate(AccelFunction.GREATER_THAN, col(1), lit(v)),
                                    new AccelInput(ROW),
                                    ROW),
                            ROW);
            assertThat(DriftSensitivePredicate.refuse(plan))
                    .as("comparing against %s is exact and must not be refused", v)
                    .isNull();
        }
    }

    @Test
    @DisplayName("a null check beside a transcendental is still refused")
    void nullChecksDoNotExemptTheCondition() {
        // IS NOT NULL is exact, but the condition as a whole still decides membership using a
        // drifting value, so the conjunction must be refused rather than the exact half excusing
        // it.
        AccelExpression condition =
                new AccelCall(
                        AccelFunction.AND,
                        Arrays.asList(
                                new AccelCall(
                                        AccelFunction.IS_NOT_NULL,
                                        Arrays.asList(col(1)),
                                        new BooleanType(false)),
                                predicate(
                                        AccelFunction.GREATER_THAN,
                                        call(AccelFunction.EXP, col(1)),
                                        lit(1.0))),
                        new BooleanType(false));
        AccelNode plan =
                new AccelProject(
                        Arrays.asList(col(0, new IntType(false)), col(1)),
                        new AccelFilter(condition, new AccelInput(ROW), ROW),
                        ROW);
        assertThat(DriftSensitivePredicate.refuse(plan)).isNotNull().contains("EXP");
    }

    // -------------------------------------------------------------------------------------------

    private static AccelNode filtered(AccelExpression scored) {
        return new AccelProject(
                Arrays.asList(col(0, new IntType(false)), col(1)),
                new AccelFilter(
                        predicate(AccelFunction.GREATER_THAN, scored, lit(12.0)),
                        new AccelInput(ROW),
                        ROW),
                ROW);
    }

    private static AccelExpression col(int i) {
        return new AccelInputRef(i, DOUBLE);
    }

    private static AccelExpression col(int i, LogicalType t) {
        return new AccelInputRef(i, t);
    }

    private static AccelExpression lit(double v) {
        return new AccelLiteral(v, DOUBLE);
    }

    private static AccelExpression call(AccelFunction fn, AccelExpression... ops) {
        return new AccelCall(fn, Arrays.asList(ops), DOUBLE);
    }

    private static AccelExpression predicate(AccelFunction fn, AccelExpression... ops) {
        return new AccelCall(fn, Arrays.asList(ops), new BooleanType(false));
    }

    @Test
    @DisplayName("SQRT in a filter is allowed: f64 square root is correctly rounded on both sides")
    void squareRootIsNoLongerDriftSensitive() {
        // SQRT was on the drift-sensitive list on the reasoning that whether a device honours
        // IEEE 754's correctly-rounded square root is a property of the toolchain. It is not
        // unknowable: on f64 the CUDA toolchain emits sqrt.rn.f64 whatever it is asked for
        // (--prec-sqrt governs single precision only), and the generator evaluates in double. So
        // the device's result is the correctly rounded one, which is what Math.sqrt promises.
        assertThat(DriftSensitivePredicate.refuse(filtered(call(AccelFunction.SQRT, col(1)))))
                .isNull();
    }
}
