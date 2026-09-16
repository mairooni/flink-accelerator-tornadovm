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
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.BooleanType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.FloatType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Generation of kernel source from Flink's accelerator IR.
 *
 * <p>Moved here from the planner along with the generator itself. It reads better for having moved:
 * the original had to build Calcite {@code RexNode} trees and work around the fact that Flink's
 * {@code LEAST} arrives through a bridging operator rather than Calcite's standard table, and none
 * of that says anything about kernels. The IR is what a provider is actually handed, so the tests
 * now build exactly that.
 */
class AccelKernelGeneratorTest {

    /**
     * Not nullable, and it has to be said out loud.
     *
     * <p>{@code new DoubleType()} is <em>nullable</em> — Flink's logical types default that way.
     * These tests used it throughout and so were quietly building nullable IR, which went unnoticed
     * because the planner refused nullable operands before the generator ever saw one. Since M2.9
     * it does see them, and declines; a test meaning "a NOT NULL column" has to say so.
     */
    private static final LogicalType DOUBLE = new DoubleType(false);

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

    private static AccelExpression call(
            AccelFunction function, LogicalType type, AccelExpression... operands) {
        return new AccelCall(function, Arrays.asList(operands), type);
    }

    private static AccelExpression predicate(AccelFunction function, AccelExpression... operands) {
        return new AccelCall(function, Arrays.asList(operands), new BooleanType(false));
    }

    /**
     * The shape the planner emits: Project over Filter over Input, or Project over Input.
     *
     * <p>The input row type is synthesised wide enough to hold every referenced column, all of them
     * DOUBLE unless a projection says otherwise. The generator only reads the types on the
     * expressions, so this only has to be well-formed, not accurate.
     */
    private static AccelNode plan(List<AccelExpression> projections, AccelExpression condition) {
        List<LogicalType> inputFields = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            inputFields.add(DOUBLE);
        }
        RowType inputType = RowType.of(inputFields.toArray(new LogicalType[0]));
        AccelNode input = new AccelInput(inputType);
        if (condition != null) {
            input = new AccelFilter(condition, input, inputType);
        }
        LogicalType[] outputFields = new LogicalType[projections.size()];
        for (int i = 0; i < projections.size(); i++) {
            outputFields[i] = projections.get(i).outputType();
        }
        return new AccelProject(projections, input, RowType.of(outputFields));
    }

    private static Optional<GpuKernelSource> tryGenerate(
            List<AccelExpression> projections, AccelExpression condition) {
        return AccelKernelGenerator.generate(plan(projections, condition), "T");
    }

    private static GpuKernelSource generate(
            List<AccelExpression> projections, AccelExpression condition) {
        Optional<GpuKernelSource> kernel = tryGenerate(projections, condition);
        assertTrue(kernel.isPresent(), "expected the expressions to be expressible");
        return kernel.get();
    }

    @Test
    @DisplayName("the shape the old catalogue matched still generates")
    void simpleProjectionAndFilter() {
        AccelExpression projection =
                call(AccelFunction.PLUS, call(AccelFunction.TIMES, col(1), lit(2.0)), lit(1.0));
        AccelExpression condition = predicate(AccelFunction.GREATER_THAN, col(1), lit(0.5));

        GpuKernelSource kernel = generate(Arrays.asList(col(0), projection), condition);

        assertArrayEquals(
                new int[] {1},
                kernel.inputFieldIndexes(),
                "only the referenced column is staged; field 0 is projected through");
        assertEquals(1, kernel.outputCount());
        assertTrue(kernel.hasFilter());
        assertTrue(kernel.source().contains("out0.set(i, ((c1 * 2.0) + 1.0));"), kernel.source());
        assertTrue(kernel.source().contains("if ((c1 > 0.5))"), kernel.source());
    }

    @Test
    @DisplayName("LEAST does not use min: min propagates NaN and Flink's LEAST discards it")
    void leastFoldsLeft() {
        // LEAST(c0, c1, 2.0) -- n-ary in SQL, folded left into binary comparisons.
        AccelExpression expr = call(AccelFunction.LEAST, col(0), col(1), lit(2.0));

        GpuKernelSource kernel = generate(Collections.singletonList(expr), null);

        // Not TornadoMath.min: it propagates NaN and Flink's LEAST does not. The device returned
        // different answers from the CPU plan wherever an operand was NaN, which SQRT of a
        // negative produces routinely. See nanOrderingMatchesFlink below.
        assertFalse(kernel.source().contains("TornadoMath.min"), kernel.source());
        assertTrue(kernel.source().contains("((c1 != c1 || c0 < c1) ? c0 : c1)"), kernel.source());
        // Folded left: the inner result is then compared against the third operand.
        assertTrue(kernel.source().contains("2.0) ? "), kernel.source());
    }

    @Test
    @DisplayName("GREATEST folds the same way, onto max")
    void greatestFoldsLeft() {
        AccelExpression expr = call(AccelFunction.GREATEST, col(0), col(1));

        GpuKernelSource kernel = generate(Collections.singletonList(expr), null);

        assertFalse(kernel.source().contains("TornadoMath.max"), kernel.source());
        assertTrue(kernel.source().contains("((c0 != c0 || c0 > c1) ? c0 : c1)"), kernel.source());
    }

    @Test
    @DisplayName("the generated ordering agrees with Flink on NaN and infinities")
    void nanOrderingMatchesFlink() {
        double[] interesting = {
            Double.NaN, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, -1.0, 0.0, 1.0
        };
        for (double a : interesting) {
            for (double b : interesting) {
                // Exactly the expression the generator emits, evaluated on the host.
                double least = (b != b || a < b) ? a : b;
                double greatest = (a != a || a > b) ? a : b;
                assertEquals(flinkLeast(a, b), least, "LEAST(" + a + ", " + b + ")");
                assertEquals(flinkGreatest(a, b), greatest, "GREATEST(" + a + ", " + b + ")");
            }
        }
    }

    /**
     * What {@code ScalarOperatorGens.generateGreatestLeast} produces for two DOUBLEs.
     *
     * <p>{@code Double.compare} rather than {@code <}: it orders NaN above everything and separates
     * the signed zeros, and the generated CPU code uses it.
     */
    private static double flinkLeast(double a, double b) {
        return Double.compare(a, b) <= 0 ? a : b;
    }

    private static double flinkGreatest(double a, double b) {
        return Double.compare(a, b) >= 0 ? a : b;
    }

    @Test
    @DisplayName("an expression the catalogue could never hold")
    void transcendentalExpression() {
        // EXP(c0) * LN(c0) + SIN(c0) * COS(c0)
        AccelExpression expr =
                call(
                        AccelFunction.PLUS,
                        call(
                                AccelFunction.TIMES,
                                call(AccelFunction.EXP, col(0)),
                                call(AccelFunction.LN, col(0))),
                        call(
                                AccelFunction.TIMES,
                                call(AccelFunction.SIN, col(0)),
                                call(AccelFunction.COS, col(0))));

        GpuKernelSource kernel = generate(Collections.singletonList(expr), null);

        assertFalse(kernel.hasFilter());
        assertTrue(kernel.source().contains("TornadoMath.exp(c0)"), kernel.source());
        assertTrue(kernel.source().contains("TornadoMath.log(c0)"), kernel.source());
        assertTrue(kernel.source().contains("TornadoMath.sin(c0)"), kernel.source());
        assertTrue(kernel.source().contains("TornadoMath.cos(c0)"), kernel.source());
    }

    @Test
    @DisplayName("several columns and several outputs")
    void multipleInputsAndOutputs() {
        AccelExpression first = call(AccelFunction.TIMES, col(0), col(2));
        AccelExpression second = call(AccelFunction.DIVIDE, col(2), lit(4.0));

        GpuKernelSource kernel = generate(Arrays.asList(first, second), null);

        assertArrayEquals(new int[] {0, 2}, kernel.inputFieldIndexes());
        assertEquals(2, kernel.outputCount());
        assertTrue(kernel.source().contains("DoubleArray out0, DoubleArray out1"), kernel.source());
        assertTrue(kernel.source().contains("double c2 = c2_in.get(i);"), kernel.source());
    }

    @Test
    @DisplayName("a filter on a different column than the projection is fine")
    void filterOnAnotherColumn() {
        AccelExpression projection = call(AccelFunction.TIMES, col(1), lit(2.0));
        AccelExpression condition = predicate(AccelFunction.LESS_THAN, col(3), lit(0.5));

        GpuKernelSource kernel = generate(Collections.singletonList(projection), condition);

        assertArrayEquals(new int[] {1, 3}, kernel.inputFieldIndexes());
        assertTrue(kernel.source().contains("if ((c3 < 0.5))"), kernel.source());
    }

    @Test
    @DisplayName("BIGINT is refused as an expression input: it does not survive a double")
    void bigintInputRefused() {
        AccelExpression id = col(0, new BigIntType(false));
        AccelExpression expr = call(AccelFunction.TIMES, id, id);

        assertFalse(
                tryGenerate(Collections.singletonList(expr), null).isPresent(),
                "values above 2^53 would differ from the CPU plan");
    }

    /**
     * A nullable value carries a validity bit rather than being refused.
     *
     * <p>Declined at M2.9 and served since M2.11. The bit travels in one packed {@code int} a row —
     * bit <em>k</em> says staged column <em>k</em> is absent — so a kernel over any number of
     * nullable columns moves four bytes a row of validity rather than four per column.
     */
    @Test
    @DisplayName("a nullable operand produces a kernel that carries validity")
    void nullableOperandCarriesValidity() {
        AccelExpression expr =
                new AccelCall(
                        AccelFunction.TIMES,
                        Arrays.asList(new AccelInputRef(1, new DoubleType(true)), lit(2.0)),
                        new DoubleType(true));

        GpuKernelSource kernel = generate(Collections.singletonList(expr), null);

        assertTrue(kernel.carriesValidity());
        assertTrue(kernel.source().contains("IntArray inNulls"), kernel.source());
        assertTrue(kernel.source().contains("IntArray outNulls"), kernel.source());
        // 1 means present, so the stored bit is inverted on the way in.
        assertTrue(
                kernel.source().contains("final int v1 = 1 - ((nulls >>> 0) & 1);"),
                kernel.source());
        // Strict: absent in, absent out. A literal is always present and folds out of the AND.
        assertTrue(
                kernel.source().contains("outNulls.set(i, ((1 - (v1)) << 0));"), kernel.source());
    }

    @Test
    @DisplayName("a NOT NULL expression emits exactly what it did before validity existed")
    void notNullExpressionIsUnchanged() {
        GpuKernelSource kernel =
                generate(
                        Collections.singletonList(call(AccelFunction.TIMES, col(1), lit(2.0))),
                        null);

        // The point of the packed encoding is that declaring NOT NULL stays the cheaper path, not
        // merely the older one: no parameter, no transfer, no instruction.
        assertFalse(kernel.carriesValidity());
        assertFalse(kernel.source().contains("inNulls"), kernel.source());
        assertFalse(kernel.source().contains("outNulls"), kernel.source());
    }

    @Test
    @DisplayName("a null operand makes a condition UNKNOWN, which does not select the row")
    void nullableConditionFoldsValidityIntoTheMask() {
        AccelExpression condition =
                new AccelCall(
                        AccelFunction.GREATER_THAN,
                        Arrays.asList(new AccelInputRef(1, new DoubleType(true)), lit(1.0)),
                        new BooleanType(false));

        GpuKernelSource kernel =
                generate(
                        Collections.singletonList(call(AccelFunction.TIMES, col(0), lit(2.0))),
                        condition);

        // SQL says a comparison with a null is UNKNOWN and UNKNOWN does not select. That is one
        // extra term in the mask rather than a separate mechanism.
        assertTrue(kernel.source().contains("if ((v1 != 0 && (c1 > 1.0)))"), kernel.source());
    }

    @Test
    @DisplayName("IS NULL reads the validity bit, and its own answer is never absent")
    void isNullReadsTheValidityBit() {
        AccelExpression condition =
                new AccelCall(
                        AccelFunction.IS_NOT_NULL,
                        Collections.singletonList(new AccelInputRef(1, new DoubleType(true))),
                        new BooleanType(false));

        GpuKernelSource kernel =
                generate(
                        Collections.singletonList(call(AccelFunction.TIMES, col(0), lit(2.0))),
                        condition);

        // Without this, writing the predicate a user would reach for made their query *less*
        // offloadable than omitting it -- IS NOT NULL had no IR at all. See M2.9.
        assertTrue(kernel.source().contains("if ((v1 != 0))"), kernel.source());
    }

    @Test
    @DisplayName("a projection of only pass-through columns has nothing to offload")
    void passThroughOnly() {
        assertFalse(tryGenerate(Arrays.asList(col(0), col(1)), null).isPresent());
    }

    @Test
    @DisplayName("the emitted unit has the shape the runtime expects")
    void generatedUnitIsWellFormed() {
        AccelExpression expr =
                call(
                        AccelFunction.PLUS,
                        call(AccelFunction.SQRT, col(0)),
                        call(AccelFunction.POWER, col(1), lit(3.0)));
        AccelExpression condition = predicate(AccelFunction.GREATER_THAN, col(0), lit(0.0));

        GpuKernelSource kernel = generate(Collections.singletonList(expr), condition);
        String source = kernel.source();

        assertTrue(source.contains("public final class " + kernel.className()), source);
        assertTrue(source.contains("public static void " + kernel.methodName() + "("), source);
        assertTrue(source.contains("for (@Parallel int i = 0;"), source);
        assertTrue(source.contains("IntArray mask"), source);
        // The loop is bounded by the rows actually staged, never by a buffer's capacity. A
        // capacity bound had the device evaluate the whole expression over the tail of every
        // partial batch -- results nobody reads, arithmetic nobody needs, and one INF away from
        // mattering. M2.5.
        assertTrue(source.contains("IntArray rows"), source);
        assertTrue(source.contains("final int n = rows.get(0);"), source);
        assertTrue(source.contains("i < n; i++"), source);
        assertFalse(source.contains(".getSize()"), source);
        assertEquals(
                countOccurrences(source, "{"), countOccurrences(source, "}"), "unbalanced braces");
        assertEquals(
                countOccurrences(source, "("), countOccurrences(source, ")"), "unbalanced parens");
    }

    private static int countOccurrences(String text, String needle) {
        int count = 0;
        int at = text.indexOf(needle);
        while (at >= 0) {
            count++;
            at = text.indexOf(needle, at + 1);
        }
        return count;
    }

    @Test
    @DisplayName("a column is staged at its declared width, not widened to a double")
    void stagesColumnsAtTheirDeclaredWidth() {
        AccelExpression sum =
                call(
                        AccelFunction.PLUS,
                        call(
                                AccelFunction.PLUS,
                                col(0, new IntType(false)),
                                col(1, new FloatType(false))),
                        col(2, DOUBLE));
        GpuKernelSource kernel = generate(Collections.singletonList(sum), null);

        assertArrayEquals(
                new GpuValueType[] {GpuValueType.INT, GpuValueType.FLOAT, GpuValueType.DOUBLE},
                kernel.inputTypes());
        assertTrue(kernel.source().contains("IntArray c0_in"), kernel.source());
        assertTrue(kernel.source().contains("FloatArray c1_in"), kernel.source());
        assertTrue(kernel.source().contains("DoubleArray c2_in"), kernel.source());
        // Only the buffers are typed: the arithmetic still happens in double, exactly as it did
        // when every column was staged as one.
        assertTrue(kernel.source().contains("double c0 = c0_in.get(i);"), kernel.source());
        assertTrue(kernel.source().contains("double c1 = c1_in.get(i);"), kernel.source());
    }

    @Test
    @DisplayName("a FLOAT result is refused, not narrowed at the end")
    void refusesAFloatResult() {
        AccelExpression sum =
                call(
                        AccelFunction.PLUS,
                        new FloatType(false),
                        col(0, new FloatType(false)),
                        col(1, new FloatType(false)));

        // This used to generate, computing in double and writing `out0.set(i, (float) (...))`.
        // That is not Flink's float arithmetic: Flink rounds at every step, the kernel rounded
        // once. For one operation the two agree, which is why the old test passed and looked
        // right; for a chain they do not. M2.3.
        assertFalse(tryGenerate(Collections.singletonList(sum), null).isPresent());
    }

    @Test
    @DisplayName("a DOUBLE result is written straight through")
    void writesADoubleResultUnnarrowed() {
        AccelExpression sum = call(AccelFunction.PLUS, col(0), col(1));
        GpuKernelSource kernel = generate(Collections.singletonList(sum), null);

        assertArrayEquals(new GpuValueType[] {GpuValueType.DOUBLE}, kernel.outputTypes());
        assertTrue(kernel.source().contains("DoubleArray out0"), kernel.source());
        assertFalse(kernel.source().contains("(float)"), kernel.source());
    }

    @Test
    @DisplayName("the generated source imports only the array classes it uses")
    void importsOnlyTheArrayClassesItUses() {
        AccelExpression sum = call(AccelFunction.PLUS, col(0), col(1));
        String source = generate(Collections.singletonList(sum), null).source();

        assertTrue(
                source.contains("import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;"));
        // IntArray is always used, mask or no mask: since M2.5 the live row count arrives in one.
        assertTrue(source.contains("import uk.ac.manchester.tornado.api.types.arrays.IntArray;"));
        assertFalse(source.contains("FloatArray;"), source);
    }
}
