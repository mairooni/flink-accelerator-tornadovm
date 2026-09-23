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

import java.lang.reflect.Field;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Whether the fusion probe can still tell fusion from separate rounding.
 *
 * <p>{@link StrictArithmetic} decides whether to trust a device by running a cancellation on it:
 * {@code c} is exactly minus the rounded product, so two roundings give {@code 0.0} and one gives
 * the rounding error. That only works while the chosen pairs actually lose something in the
 * rounding. A well-meaning edit to the table — rounder numbers, a tidier constant — would leave a
 * probe that passes on every device including the ones that fuse, and nothing would say so.
 *
 * <p>So this checks the data rather than the device: no GPU is needed, and it fails the moment the
 * probe stops being able to discriminate.
 */
class StrictArithmeticProbeTest {

    @Test
    @DisplayName("every probe case distinguishes a fused multiply-add from two rounded ones")
    void everyCaseDiscriminates() throws Exception {
        final double[][] cases = cases();
        assertThat(cases).as("the probe needs cases to be a probe").isNotEmpty();
        for (double[] pair : cases) {
            final double a = pair[0];
            final double b = pair[1];
            final double c = -(a * b);
            final double separate = a * b + c;
            final double fused = Math.fma(a, b, c);
            assertThat(separate)
                    .as("two roundings must cancel exactly for a=%s b=%s", a, b)
                    .isEqualTo(0.0);
            assertThat(fused)
                    .as(
                            "a fused multiply-add must NOT cancel for a=%s b=%s, or this case is blind",
                            a, b)
                    .isNotEqualTo(0.0);
        }
    }

    @Test
    @DisplayName("the cases span magnitudes, so one lucky device cannot pass by accident")
    void theCasesAreNotAllTheSameSize() throws Exception {
        double smallest = Double.MAX_VALUE;
        double largest = 0.0;
        for (double[] pair : cases()) {
            final double magnitude = Math.abs(Math.fma(pair[0], pair[1], -(pair[0] * pair[1])));
            smallest = Math.min(smallest, magnitude);
            largest = Math.max(largest, magnitude);
        }
        assertThat(largest / smallest)
                .as("the residuals should differ by many orders of magnitude")
                .isGreaterThan(1e6);
    }

    private static double[][] cases() throws Exception {
        final Field field = StrictArithmetic.class.getDeclaredField("CASES");
        field.setAccessible(true);
        return (double[][]) field.get(null);
    }
}
