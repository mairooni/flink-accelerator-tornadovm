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
import org.apache.flink.table.accelerator.AccelLiteral;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.types.logical.LogicalTypeRoot;

import javax.annotation.Nullable;

import java.util.List;

/**
 * Arithmetic with a {@code DECIMAL} operand, which the kernel would compute as if it were a double.
 *
 * <h2>The measurement</h2>
 *
 * <p>A bare decimal in SQL is a {@code DECIMAL} literal, not a {@code DOUBLE} one: the planner
 * types {@code v0 - 0.3729116549965876} as {@code DOUBLE} minus {@code DECIMAL(17,16)}. The kernel
 * generator renders that literal as the nearest double and computes in double, on the assumption
 * that this is what the coercion means.
 *
 * <p>It is not. Measured 2026-09-23 on 34 601 rows, with the device verified to be running and
 * strict compilation in force:
 *
 * <pre>
 *   CPU with DECIMAL literals  vs  CPU with the same values CAST to DOUBLE : 8 680 rows differ
 *   GPU with DECIMAL literals  vs  CPU with DECIMAL literals               : 8 680 rows differ
 *   GPU with DOUBLE  literals  vs  CPU with DOUBLE  literals               :     0 rows differ
 * </pre>
 *
 * <p>The same 25.09%, to the row. So the device computes plain double arithmetic and agrees with
 * Flink exactly when Flink is also computing in double; what it cannot reproduce is what Flink does
 * when a {@code DECIMAL} is in the expression. Whatever that is, Flink's CPU is the reference and
 * this generator does not model it.
 *
 * <p>The difference reached 8 ulp, which is small until a filter sits on it: the same data and the
 * same SQL returned 24 rows on the CPU and 23 on the device.
 *
 * <h2>What this refuses, and what it does not</h2>
 *
 * <p>Refused: any expression in which a {@code DECIMAL} value is an operand of arithmetic or of a
 * comparison. That covers projections as well as predicates, because a projected column that is
 * wrong is wrong whether or not anything filters on it.
 *
 * <p>Not refused: {@code DECIMAL} carried through untouched — selected, or compared to nothing —
 * since nothing is computed and nothing can round differently. Also not refused: the same query
 * written with {@code DOUBLE} operands, which is the case measured at zero difference above.
 *
 * <p>This is a refusal and not a fix. The fix is to model Flink's coercion, and it needs someone to
 * establish what that coercion actually computes; until then the device is used only where it has
 * been shown to agree.
 */
public final class DecimalCoercion {

    private DecimalCoercion() {}

    /** Why this subtree must not be offloaded, or {@code null} if no decimal reaches arithmetic. */
    public static @Nullable String refuse(AccelNode subtree) {
        final String found = walk(subtree);
        if (found == null) {
            return null;
        }
        return "the expression computes with a DECIMAL operand ("
                + found
                + "), and Flink's CPU does not evaluate that as double arithmetic; a kernel that"
                + " does would return different values, and a filter over them different rows."
                + " Writing the value as a DOUBLE (for example CAST(0.5 AS DOUBLE), or a column"
                + " typed DOUBLE) is evaluated identically on both";
    }

    private static @Nullable String walk(AccelNode node) {
        if (node instanceof AccelProject) {
            final String found = inAny(((AccelProject) node).projections());
            if (found != null) {
                return found;
            }
        }
        if (node instanceof AccelFilter) {
            final String found = in(((AccelFilter) node).condition());
            if (found != null) {
                return found;
            }
        }
        for (AccelNode input : node.inputs()) {
            final String found = walk(input);
            if (found != null) {
                return found;
            }
        }
        return null;
    }

    private static @Nullable String inAny(List<AccelExpression> expressions) {
        for (AccelExpression expression : expressions) {
            final String found = in(expression);
            if (found != null) {
                return found;
            }
        }
        return null;
    }

    /**
     * A decimal is only a hazard where something computes with it.
     *
     * <p>So the test is on the <em>operands of a call</em> rather than on every expression: a
     * decimal column that is merely selected is copied, not rounded, and refusing that would cost
     * the offload of a great many queries for no correctness gain.
     */
    private static @Nullable String in(AccelExpression expression) {
        if (!(expression instanceof AccelCall)) {
            return null;
        }
        final AccelCall call = (AccelCall) expression;
        for (int i = 0; i < call.operands().size(); i++) {
            final AccelExpression operand = call.operands().get(i);
            if (isDecimal(operand)
                    && !isMeasuredHarmless(call, i)
                    && !isFaithfullyConvertedLiteral(operand)) {
                return call.function() + " applied to " + operand.outputType();
            }
        }
        for (AccelExpression operand : call.operands()) {
            final String found = in(operand);
            if (found != null) {
                return found;
            }
        }
        return null;
    }

    /**
     * The one DECIMAL position measured to make no difference, so it is not refused.
     *
     * <p>Calcite does not keep {@code SQRT}: it rewrites it to {@code POWER(x, 0.5)}, and that
     * {@code 0.5} is a {@code DECIMAL(2,1)}. Refusing every DECIMAL operand therefore refused every
     * query containing a square root, for a decimal the author never wrote.
     *
     * <p>Measured before allowing it, CPU only, 2 000 000 rows: {@code POWER(ABS(a*b), 0.5)} and
     * {@code POWER(ABS(a*b), 0.5E0)} agree on every row. The exponent of {@code POWER} is converted
     * to a double and the decimal spelling changes nothing, unlike the operands of {@code + - *}
     * where a divergence has been observed. So this exemption is one position of one function,
     * justified by measurement rather than by reasoning about what ought to be equivalent.
     */
    private static boolean isMeasuredHarmless(AccelCall call, int operandIndex) {
        return call.function() == AccelFunction.POWER && operandIndex == 1;
    }

    /**
     * A DECIMAL <em>literal</em>, whose conversion the generator now performs exactly as Flink
     * does.
     *
     * <p>The discrepancy this class was written for has been traced to one cause: Flink converts a
     * compact DECIMAL to a double as {@code ((double) unscaledLong) / POW10[scale]}, a division
     * that rounds, while the generator used {@code BigDecimal.doubleValue()}, which is correctly
     * rounded. For some literals those differ by an ulp — and then they differ on every row. {@code
     * AccelKernelGenerator.asFlinkDouble} now calls Flink's own conversion, so a literal is no
     * longer a hazard and refusing one buys nothing.
     *
     * <p>The refusal stays for every other DECIMAL operand — a decimal column, or a decimal
     * produced by a cast or by arithmetic — because for those the evaluation semantics are
     * <em>not</em> implemented here and have not been traced. The rule is now "what has been
     * implemented is allowed", which is the only version of it that can be defended.
     */
    private static boolean isFaithfullyConvertedLiteral(AccelExpression operand) {
        return operand instanceof AccelLiteral;
    }

    private static boolean isDecimal(AccelExpression expression) {
        return expression.outputType().getTypeRoot() == LogicalTypeRoot.DECIMAL;
    }
}
