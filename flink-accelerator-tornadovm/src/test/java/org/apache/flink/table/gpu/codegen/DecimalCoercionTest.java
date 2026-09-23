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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The refusal that closes the 24-versus-23 row disagreement.
 *
 * <p>The failing query contained no transcendental. It was {@code -}, {@code *}, {@code +} and
 * {@code LEAST} over a bare decimal literal, which SQL types as {@code DECIMAL} — and Flink does
 * not evaluate arithmetic with a {@code DECIMAL} operand as double arithmetic, while the generated
 * kernel does. Measured on 34 601 rows: the CPU disagrees with <em>itself</em> on 8 680 of them
 * when the same constants are written as {@code DOUBLE} instead, which is the same 8 680 rows the
 * device disagreed on.
 */
class DecimalCoercionTest {

    private static final LogicalType DOUBLE = new DoubleType(false);
    private static final LogicalType DEC = new DecimalType(false, 17, 16);
    private static final RowType ROW = RowType.of(new IntType(false), new DoubleType(false));

    @Test
    @DisplayName("arithmetic on a DECIMAL column is refused: its conversion is not implemented")
    void decimalColumnsAreRefused() {
        // The shape that returned 24 rows on the CPU and 23 on the device, but with the decimals
        // as a COLUMN rather than literals. Literals are now allowed -- their conversion was traced
        // and fixed in AccelIrBuilder -- and a column's is not, so this is where the line sits.
        AccelExpression delta = call(AccelFunction.MINUS, col(1), decCol());
        AccelExpression squared = call(AccelFunction.TIMES, delta, delta);
        AccelNode plan =
                project(
                        new AccelFilter(
                                predicate(AccelFunction.LESS_THAN, squared, decCol()),
                                new AccelInput(ROW),
                                ROW));
        assertThat(DecimalCoercion.refuse(plan)).isNotNull().contains("DECIMAL").contains("CAST");
    }

    @Test
    @DisplayName("the same expression with DOUBLE constants is allowed")
    void doubleConstantsAreAllowed() {
        // Measured at zero difference over 34 601 rows with the device verified to be running, so
        // refusing this too would cost the offload and buy nothing.
        AccelExpression delta = call(AccelFunction.MINUS, col(1), dblLit());
        AccelExpression squared = call(AccelFunction.TIMES, delta, delta);
        AccelNode plan =
                project(
                        new AccelFilter(
                                predicate(AccelFunction.LESS_THAN, squared, dblLit()),
                                new AccelInput(ROW),
                                ROW));
        assertThat(DecimalCoercion.refuse(plan)).isNull();
    }

    @Test
    @DisplayName("a decimal in a projection is refused too, not only in a filter")
    void projectionsAreCoveredAsWellAsPredicates() {
        // A projected column that is wrong is wrong whether or not anything filters on it.
        AccelNode plan =
                new AccelProject(
                        Arrays.asList(
                                col(0, new IntType(false)),
                                call(AccelFunction.PLUS, col(1), decCol())),
                        new AccelInput(ROW),
                        ROW);
        assertThat(DecimalCoercion.refuse(plan)).isNotNull();
    }

    @Test
    @DisplayName("a comparison against a decimal is refused, not only arithmetic")
    void comparisonsCount() {
        AccelNode plan =
                project(
                        new AccelFilter(
                                predicate(AccelFunction.GREATER_THAN, col(1), decCol()),
                                new AccelInput(ROW),
                                ROW));
        assertThat(DecimalCoercion.refuse(plan)).isNotNull();
    }

    @Test
    @DisplayName("a decimal carried through untouched is allowed: nothing computes with it")
    void carryingADecimalIsNotComputingWithOne() {
        AccelNode plan =
                new AccelProject(
                        Arrays.asList(col(0, new IntType(false)), new AccelInputRef(1, DEC)),
                        new AccelInput(ROW),
                        ROW);
        assertThat(DecimalCoercion.refuse(plan)).isNull();
    }

    @Test
    @DisplayName("a decimal nested deep inside a conjunction is still found")
    void nestingDoesNotHideIt() {
        AccelExpression condition =
                new AccelCall(
                        AccelFunction.AND,
                        Arrays.asList(
                                predicate(AccelFunction.GREATER_THAN, col(1), dblLit()),
                                predicate(
                                        AccelFunction.LESS_THAN,
                                        call(AccelFunction.TIMES, col(1), decCol()),
                                        dblLit())),
                        new BooleanType(false));
        AccelNode plan = project(new AccelFilter(condition, new AccelInput(ROW), ROW));
        assertThat(DecimalCoercion.refuse(plan)).isNotNull();
    }

    @Test
    @DisplayName("an expression with no decimal anywhere is allowed")
    void pureDoubleArithmeticIsAllowed() {
        AccelNode plan =
                project(
                        new AccelFilter(
                                predicate(
                                        AccelFunction.LESS_THAN,
                                        call(AccelFunction.TIMES, col(1), col(1)),
                                        dblLit()),
                                new AccelInput(ROW),
                                ROW));
        assertThat(DecimalCoercion.refuse(plan)).isNull();
    }

    @Test
    @DisplayName("SQRT survives: the POWER exponent Calcite rewrites it into is not refused")
    void thePowerExponentIsExempt() {
        // Calcite spells SQRT(x) as POWER(x, 0.5:DECIMAL(2,1)). Refusing that refused every query
        // with a square root in it, for a decimal nobody wrote. Measured identical on 2 000 000
        // rows before the exemption was added.
        AccelExpression sqrt =
                new AccelCall(
                        AccelFunction.POWER,
                        Arrays.asList(
                                col(1),
                                new AccelLiteral(
                                        java.math.BigDecimal.valueOf(0.5),
                                        new DecimalType(false, 2, 1))),
                        DOUBLE);
        AccelNode plan =
                project(
                        new AccelFilter(
                                predicate(AccelFunction.LESS_THAN, sqrt, dblLit()),
                                new AccelInput(ROW),
                                ROW));
        assertThat(DecimalCoercion.refuse(plan)).isNull();
    }

    @Test
    @DisplayName("the exemption is one position only: a decimal BASE of POWER is still refused")
    void onlyTheExponentIsExempt() {
        AccelExpression call =
                new AccelCall(AccelFunction.POWER, Arrays.asList(decCol(), col(1)), DOUBLE);
        AccelNode plan =
                project(
                        new AccelFilter(
                                predicate(AccelFunction.LESS_THAN, call, dblLit()),
                                new AccelInput(ROW),
                                ROW));
        assertThat(DecimalCoercion.refuse(plan)).isNotNull();
    }

    @Test
    @DisplayName("the original 24-versus-23 shape is now allowed: its decimals are literals")
    void theOriginalFailureIsNowSupported() {
        // This is a deliberate change from the first version of this class, which refused it.
        //
        // The refusal existed because the cause was unknown. It is now known: Flink converts a
        // compact DECIMAL to a double as ((double) unscaledLong) / POW10[scale], which rounds a
        // second time, while AccelIrBuilder used the correctly rounded BigDecimal.doubleValue().
        // For 0.9909456437464825 those are one ulp apart, and the kernel then disagreed with the
        // CPU on every row. AccelIrBuilder now calls Flink's own conversion, so a literal carries
        // the double the CPU computes with and refusing it buys nothing.
        //
        // Verified on device over the whole 10M-row population: 0 differing values, where the same
        // query previously differed on 9 966 889 of them.
        AccelExpression delta = call(AccelFunction.MINUS, col(1), decLit());
        AccelExpression squared = call(AccelFunction.TIMES, delta, delta);
        AccelNode plan =
                project(
                        new AccelFilter(
                                predicate(AccelFunction.LESS_THAN, squared, decLit()),
                                new AccelInput(ROW),
                                ROW));
        assertThat(DecimalCoercion.refuse(plan)).isNull();
    }

    private static AccelNode project(AccelNode input) {
        return new AccelProject(Arrays.asList(col(0, new IntType(false)), col(1)), input, ROW);
    }

    private static AccelExpression col(int i) {
        return new AccelInputRef(i, DOUBLE);
    }

    private static AccelExpression col(int i, LogicalType t) {
        return new AccelInputRef(i, t);
    }

    /** A DECIMAL that is not a literal: its conversion is not implemented, so it is refused. */
    private static AccelExpression decCol() {
        return new AccelInputRef(1, DEC);
    }

    private static AccelExpression decLit() {
        return new AccelLiteral(java.math.BigDecimal.valueOf(0.3729116549965876), DEC);
    }

    private static AccelExpression dblLit() {
        return new AccelLiteral(0.3729116549965876, DOUBLE);
    }

    private static AccelExpression call(AccelFunction fn, AccelExpression... ops) {
        return new AccelCall(fn, Arrays.asList(ops), DOUBLE);
    }

    private static AccelExpression predicate(AccelFunction fn, AccelExpression... ops) {
        return new AccelCall(fn, Arrays.asList(ops), new BooleanType(false));
    }
}
