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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The assumption {@link ExactPowers} rests on, asserted on whatever JVM the build runs on.
 *
 * <p>{@code ExactPowers} rewrites the accelerator's IR but not the planner's {@code RexNode} tree,
 * so Flink's own operator goes on evaluating {@code Math.pow} while the device evaluates a multiply
 * or a square root. That is only sound while {@code Math.pow} agrees with them — and {@code
 * Math.pow} is specified merely to within one ulp, so the agreement is a property of an
 * implementation rather than of the language.
 *
 * <p>It held over two million random values plus every special case on JDK 21 when the rewrite was
 * written. This is that measurement, kept, so that a JDK on which it stops holding fails a test
 * rather than quietly returning a different set of rows from the same SQL.
 *
 * <p>The second test here records the rewrite that was <em>rejected</em>. {@code POWER(x, 0.5)} to
 * {@code SQRT(x)} looked equally safe and is not: the two disagree at {@code -Infinity}, where
 * {@code pow} gives {@code +Infinity} and {@code sqrt} gives {@code NaN}. This test found that,
 * which is the reason it is worth having rather than a comment.
 *
 * <p>Deliberately not a device test. What is in question here is the <em>host</em> half of the
 * comparison; the device half is IEEE 754 for a multiply and {@code sqrt.rn.f64} for a square root,
 * neither of which is in doubt.
 */
class ExactPowersAgreementTest {

    /** Enough to catch a systematic disagreement; the whole double domain is not reachable. */
    private static final int SAMPLES = 2_000_000;

    @Test
    @DisplayName("Math.pow(x, 2) is exactly x * x, so rewriting POWER(x, 2) changes no CPU result")
    void squareAgreesWithMultiply() {
        for (double x : specialCases()) {
            assertThat(Double.doubleToRawLongBits(Math.pow(x, 2.0)))
                    .withFailMessage(
                            "Math.pow(%s, 2.0) = %s but %s * %s = %s; ExactPowers must stop"
                                    + " rewriting POWER(x, 2) on this JVM",
                            x, Math.pow(x, 2.0), x, x, x * x)
                    .isEqualTo(Double.doubleToRawLongBits(x * x));
        }

        Random random = new Random(11);
        for (int i = 0; i < SAMPLES; i++) {
            double x = (random.nextDouble() - 0.5) * Math.pow(10, random.nextInt(40) - 20);
            assertThat(Double.doubleToRawLongBits(Math.pow(x, 2.0)))
                    .withFailMessage(
                            "Math.pow(%s, 2.0) = %s but %s * %s = %s",
                            x, Math.pow(x, 2.0), x, x, x * x)
                    .isEqualTo(Double.doubleToRawLongBits(x * x));
        }
    }

    @Test
    @DisplayName(
            "pow(x, 0.5) and sqrt(x) disagree at -0.0 and at -Infinity, so that rewrite is refused")
    void rootDoesNotAgreeWithSqrtAtTheEdges() {
        // Kept as a test because it is the reason ExactPowers rewrites POWER(x, 2) and not
        // POWER(x, 0.5). Over the ordinary domain the two are identical -- two million random
        // non-negative values agree below -- so the temptation to rewrite is real, and the case
        // against it lives at exactly two inputs.
        Random random = new Random(7);
        for (int i = 0; i < SAMPLES; i++) {
            double x = Math.abs(random.nextDouble() * Math.pow(10, random.nextInt(40) - 20));
            assertThat(Double.doubleToRawLongBits(Math.pow(x, 0.5)))
                    .withFailMessage("Math.pow(%s, 0.5) and Math.sqrt(%s) differ", x, x)
                    .isEqualTo(Double.doubleToRawLongBits(Math.sqrt(x)));
        }

        // -0.0: a sign difference, and harmless -- they compare equal, so no row set depends on it.
        assertThat(Math.pow(-0.0, 0.5)).isEqualTo(0.0);
        assertThat(Math.sqrt(-0.0)).isEqualTo(-0.0);
        assertThat(Math.pow(-0.0, 0.5) == Math.sqrt(-0.0)).isTrue();

        // -Infinity: not harmless. NaN fails every comparison, so a filter, join key, grouping key
        // or sort over this value selects a different set of rows. This is the disqualifying case.
        assertThat(Math.pow(Double.NEGATIVE_INFINITY, 0.5)).isEqualTo(Double.POSITIVE_INFINITY);
        assertThat(Math.sqrt(Double.NEGATIVE_INFINITY)).isNaN();
        assertThat(Math.pow(Double.NEGATIVE_INFINITY, 0.5) == Math.sqrt(Double.NEGATIVE_INFINITY))
                .isFalse();
    }

    private static double[] specialCases() {
        return new double[] {
            -0.0,
            0.0,
            1.0,
            -1.0,
            2.0,
            -3.0,
            4.0,
            0.1,
            1e-300,
            1e300,
            Double.MIN_VALUE,
            Double.MAX_VALUE,
            Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY,
            Double.NaN
        };
    }
}
