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

import org.apache.flink.table.accelerator.AccelAggregate;
import org.apache.flink.table.accelerator.AccelCall;
import org.apache.flink.table.accelerator.AccelExpression;
import org.apache.flink.table.accelerator.AccelFilter;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelLiteral;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;

import java.util.ArrayList;
import java.util.List;

/**
 * Rewrites {@code POWER(x, 2)} into {@code x * x}, which a device computes exactly.
 *
 * <p>Faster, and — which is the reason this exists — <em>more</em> in agreement with what Flink's
 * CPU operator computes, not less.
 *
 * <h2>Why this is a correctness change before it is a speed one</h2>
 *
 * <p>A device's {@code pow} is written to its own accuracy specification and is not correctly
 * rounded, so {@code POWER} is in {@link DriftSensitivePredicate}'s drift-sensitive set and an
 * expression containing one is refused wherever it could decide which rows come back. That is the
 * right call for a general exponent. For these two it is needlessly conservative, because there is
 * an exactly-rounded operation that computes the same thing:
 *
 * <p>A multiply is correctly rounded on every IEEE 754 device, so after this rewrite the device
 * computes bit-for-bit what {@code java.lang.Math} does, where before it computed something within
 * a few ulp of it.
 *
 * <h2>What the CPU side does, measured rather than assumed</h2>
 *
 * <p>The rewrite is applied to the accelerator's IR, not to the planner's {@code RexNode} tree, so
 * Flink's own operator still evaluates {@code Math.pow}. That is only safe if {@code Math.pow}
 * agrees with the operation substituted here, which was measured on JDK 21 over two million random
 * values plus every special case:
 *
 * <p>{@code Math.pow(x, 2.0)} and {@code x * x} are <b>identical for every input tried</b>,
 * including {@code ±0.0}, {@code NaN}, both infinities and negatives.
 *
 * <p>{@code Math.pow} is only specified to within one ulp, so that agreement is a property of an
 * implementation rather than of the language. {@link ExactPowersAgreementTest} asserts it on
 * whatever JVM the build runs on, so a JDK that stops honouring it fails a test instead of silently
 * returning different rows.
 *
 * <h2>Why {@code POWER(x, 0.5)} is <em>not</em> rewritten to {@code SQRT(x)}</h2>
 *
 * <p>It was, and {@link ExactPowersAgreementTest} rejected it. {@code pow} and {@code sqrt} are
 * each correct to their own specification and they disagree at the edges of the domain:
 *
 * <ul>
 *   <li>{@code Math.pow(-0.0, 0.5)} is {@code +0.0}; {@code Math.sqrt(-0.0)} is {@code -0.0}.
 *       Harmless on its own -- the two compare equal, so no row set can depend on it.
 *   <li>{@code Math.pow(-Infinity, 0.5)} is {@code +Infinity}; {@code Math.sqrt(-Infinity)} is
 *       {@code NaN}. That one is not harmless. {@code NaN} fails every comparison, so a filter, a
 *       join key, a grouping key or a sort over such a value selects a different set of rows.
 * </ul>
 *
 * <p>Changing which rows a query returns is the one thing this project will not do, and no cheap
 * guard rules the case out: whether a base can be negative infinity is not decidable from the IR,
 * and the haversine's own argument -- a sum of products of cosines -- is not structurally
 * non-negative even though it is always in range in practice.
 *
 * <p>So the square root stays a {@code POWER} call and stays drift-prone. What is <em>not</em>
 * conditional on this is {@link DriftSensitivePredicate}'s treatment of a directly written {@code
 * SQRT}, which came off the drift-sensitive list on its own evidence.
 */
public final class ExactPowers {

    /** {@code POWER(x, 2)}: a multiply, which is exact. */
    private static final double SQUARE = 2.0;

    private ExactPowers() {}

    /** The subtree with every exactly-computable {@code POWER} replaced. */
    public static AccelNode rewrite(AccelNode node) {
        List<AccelNode> inputs = new ArrayList<>(node.inputs().size());
        boolean inputChanged = false;
        for (AccelNode input : node.inputs()) {
            AccelNode rewritten = rewrite(input);
            inputChanged |= rewritten != input;
            inputs.add(rewritten);
        }

        if (node instanceof AccelProject) {
            AccelProject project = (AccelProject) node;
            List<AccelExpression> projections = rewriteAll(project.projections());
            if (!inputChanged && projections == project.projections()) {
                return node;
            }
            return new AccelProject(projections, inputs.get(0), project.outputType());
        }
        if (node instanceof AccelFilter) {
            AccelFilter filter = (AccelFilter) node;
            AccelExpression condition = rewrite(filter.condition());
            if (!inputChanged && condition == filter.condition()) {
                return node;
            }
            return new AccelFilter(condition, inputs.get(0), filter.outputType());
        }
        // An aggregate carries its grouping and its calls rather than expressions of its own; its
        // projection is the input, which was rewritten above. Anything else is left alone: a node
        // this does not recognise is one whose expressions it cannot safely rebuild, and returning
        // it unchanged costs only the optimisation.
        if (node instanceof AccelAggregate && inputChanged) {
            AccelAggregate aggregate = (AccelAggregate) node;
            return new AccelAggregate(
                    aggregate.grouping(), aggregate.calls(), inputs.get(0), aggregate.outputType());
        }
        return node;
    }

    /** Null when nothing in the list changed, so an untouched subtree is returned as it was. */
    private static List<AccelExpression> rewriteAll(List<AccelExpression> expressions) {
        List<AccelExpression> rewritten = null;
        for (int i = 0; i < expressions.size(); i++) {
            AccelExpression before = expressions.get(i);
            AccelExpression after = rewrite(before);
            if (after != before && rewritten == null) {
                rewritten = new ArrayList<>(expressions);
            }
            if (rewritten != null) {
                rewritten.set(i, after);
            }
        }
        return rewritten == null ? expressions : rewritten;
    }

    private static AccelExpression rewrite(AccelExpression expression) {
        if (!(expression instanceof AccelCall)) {
            return expression;
        }
        AccelCall call = (AccelCall) expression;

        List<AccelExpression> operands = rewriteAll(call.operands());
        AccelCall rebuilt =
                operands == call.operands()
                        ? call
                        : new AccelCall(call.function(), operands, call.outputType());

        if (rebuilt.function() != AccelFunction.POWER || rebuilt.operands().size() != 2) {
            return rebuilt;
        }
        Double exponent = constantExponent(rebuilt.operands().get(1));
        if (exponent == null) {
            return rebuilt;
        }
        AccelExpression base = rebuilt.operands().get(0);
        if (exponent == SQUARE) {
            // The same expression twice. The generator names every operand it emits more than
            // once, so this is one evaluation and one multiply, not two evaluations.
            return new AccelCall(AccelFunction.TIMES, List.of(base, base), rebuilt.outputType());
        }
        return rebuilt;
    }

    /**
     * The exponent as a double, or null when it is not a literal number.
     *
     * <p>Compared by value rather than by the literal's declared type: Flink writes the {@code 2}
     * of {@code POWER(x, 2)} as whatever type the SQL gave it, and an {@code INTEGER} 2, a {@code
     * DECIMAL} 2 and a {@code DOUBLE} 2.0 all describe the same exponent.
     */
    private static Double constantExponent(AccelExpression expression) {
        if (!(expression instanceof AccelLiteral)) {
            return null;
        }
        Object value = ((AccelLiteral) expression).value();
        if (!(value instanceof Number)) {
            return null;
        }
        return ((Number) value).doubleValue();
    }
}
