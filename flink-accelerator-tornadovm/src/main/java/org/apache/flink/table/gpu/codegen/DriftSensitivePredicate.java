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
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelOverAggregate;
import org.apache.flink.table.accelerator.AccelProject;

import javax.annotation.Nullable;

import java.util.EnumSet;
import java.util.Set;

/**
 * Refuses a filter whose outcome could differ from Flink's, rather than quietly changing it.
 *
 * <h2>The defect this exists for</h2>
 *
 * <p>A device transcendental is not required to agree with {@code java.lang.Math} and does not.
 * Measured over two million rows of the screening workload, 36.9% of scores differed from the CPU's
 * and the largest difference was 23 ulp. For a <em>projection</em> that is a numerical difference
 * in an output column. For a <em>predicate</em> it is not: a score on one side of the threshold on
 * the host can land on the other side on the device, and the query then returns a different set of
 * rows.
 *
 * <p>That is not hypothetical. Given a row whose host score is {@code 8.86821620754203} and whose
 * device score is {@code 8.868216207542035}, a query with that threshold returns 289,114 rows on
 * the CPU and 289,115 on the device — differing by exactly the row in question.
 *
 * <p>A changed row set is a wrong answer, not a rounding difference, so the default refuses it. The
 * projection case is left alone and is documented separately: an offloaded {@code SELECT} may
 * differ in the last few ulp of a computed column, which is a stated numerical limitation rather
 * than a membership one.
 *
 * <h2>Why these functions and not others</h2>
 *
 * <p>The set below is not every function the generator can emit. Addition, subtraction,
 * multiplication, division and comparison are exact under IEEE 754 and agree bit for bit on both
 * sides; so do {@code ABS}, {@code FLOOR}, {@code CEIL} and {@code SIGN}, which only inspect or
 * clear bits. What differs is the functions CUDA implements in its own math library to its own
 * accuracy specification — and {@code SQRT} sits with them deliberately even though IEEE 754
 * requires it to be correctly rounded, because whether a given device build honours that is a
 * property of the toolchain rather than of this code, and the conservative reading is the one that
 * cannot silently change a result. {@link #DRIFT_SENSITIVE} is the list, and {@code
 * TranscendentalDriftDeviceIT} measures each of them so the list is evidence-led rather than
 * guessed.
 *
 * <h2>The escape hatch, which is not the default</h2>
 *
 * <p>{@code -Dtornadovm.accelerator.approximate-predicates=true} allows them. It exists so the cost
 * of the refusal can be measured, and it is explicitly a research setting: with it on, an offloaded
 * predicate may select a different set of rows from the same SQL on the CPU. It is off unless
 * someone sets it, it is never set by the planner, and no query can ask for it.
 */
public final class DriftSensitivePredicate {

    /**
     * Functions whose device implementation is not guaranteed to match {@code java.lang.Math}.
     *
     * <p>Every one of these maps onto {@code TornadoMath} and thence to the CUDA math library. CUDA
     * specifies its own accuracy in ulp for each; Java specifies its own; neither is required to
     * match the other, and measurement says they do not.
     */
    public static final Set<AccelFunction> DRIFT_SENSITIVE =
            EnumSet.of(
                    AccelFunction.SQRT,
                    AccelFunction.EXP,
                    AccelFunction.LN,
                    AccelFunction.LOG2,
                    AccelFunction.POWER,
                    AccelFunction.SIN,
                    AccelFunction.COS,
                    AccelFunction.TAN,
                    AccelFunction.ASIN,
                    AccelFunction.ACOS,
                    AccelFunction.ATAN,
                    AccelFunction.ATAN2,
                    AccelFunction.TANH);

    private static final String APPROXIMATE_PROPERTY =
            "tornadovm.accelerator.approximate-predicates";

    private DriftSensitivePredicate() {}

    /**
     * Why this subtree's filter cannot be trusted to select the same rows, or null when it can.
     *
     * <p>Only the <em>condition</em> is examined. A projection alongside it may use anything.
     */
    public static @Nullable String refuse(AccelNode subtree) {
        if (approximateAllowed()) {
            return null;
        }
        AccelExpression condition = conditionOf(subtree);
        if (condition == null) {
            return null;
        }
        AccelFunction offender = firstDriftSensitive(condition);
        if (offender == null) {
            return null;
        }
        return "the filter uses "
                + offender
                + ", whose device implementation is not bit-identical to java.lang.Math; an "
                + "offloaded predicate could select a different set of rows than the same SQL on "
                + "the CPU";
    }

    /** The filter condition anywhere beneath this node, or null when there is no filter. */
    private static @Nullable AccelExpression conditionOf(AccelNode node) {
        if (node instanceof AccelFilter) {
            return ((AccelFilter) node).condition();
        }
        for (AccelNode input : node.inputs()) {
            AccelExpression found = conditionOf(input);
            if (found != null) {
                return found;
            }
        }
        return null;
    }

    private static @Nullable AccelFunction firstDriftSensitive(AccelExpression expression) {
        if (expression instanceof AccelCall) {
            AccelCall call = (AccelCall) expression;
            if (DRIFT_SENSITIVE.contains(call.function())) {
                return call.function();
            }
            for (AccelExpression operand : call.operands()) {
                AccelFunction found = firstDriftSensitive(operand);
                if (found != null) {
                    return found;
                }
            }
        }
        return null;
    }

    /** Whether the research mode is on. Read per call so a test can set and clear it. */
    public static boolean approximateAllowed() {
        String set = System.getProperty(APPROXIMATE_PROPERTY);
        if (set == null) {
            // An environment variable too, so a measurement can enable it for one cluster without
            // editing the distribution's configuration. Same setting, same warning: with it on, an
            // offloaded predicate may select a different set of rows than the CPU.
            set = System.getenv("TORNADOVM_ACCELERATOR_APPROXIMATE_PREDICATES");
        }
        return Boolean.parseBoolean(set);
    }

    /**
     * Whether any drift-sensitive function appears anywhere in the subtree, filter or not.
     *
     * <p>Used to state the <em>projection</em> limitation rather than to refuse on it: a computed
     * column may differ in its last ulp, and callers that care can say so.
     */
    public static boolean projectionMayDrift(AccelNode node) {
        if (node instanceof AccelProject) {
            for (AccelExpression e : ((AccelProject) node).projections()) {
                if (firstDriftSensitive(e) != null) {
                    return true;
                }
            }
        }
        if (node instanceof AccelAggregate || node instanceof AccelOverAggregate) {
            return true;
        }
        for (AccelNode input : node.inputs()) {
            if (projectionMayDrift(input)) {
                return true;
            }
        }
        return false;
    }
}
