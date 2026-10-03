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

package org.apache.flink.table.gpu.operator;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The kernel between the regexes and the count: the JIT half of a grok region.
 *
 * <p>cuDF writes one byte a row per pattern; this folds them into one survivor flag. It is the half
 * a library has nothing to offer for — a per-row boolean combination is exactly what a generated
 * kernel is good at, and exactly what cuDF would need a separate pass and a separate allocation
 * for.
 *
 * <p>Eight mask parameters rather than an array of them because TornadoVM marshals each task
 * argument separately and has no array-of-arrays; {@code used} says how many carry a pattern, and
 * the rest are one-element buffers that are never read. Eight is the ceiling, not a target.
 */
public final class DeviceGrok {

    /** The most patterns one region serves, set by the kernel's parameter list. */
    public static final int MAX_PATTERNS = 8;

    private DeviceGrok() {}

    public static void combine(
            ByteArray m0,
            ByteArray m1,
            ByteArray m2,
            ByteArray m3,
            ByteArray m4,
            ByteArray m5,
            ByteArray m6,
            ByteArray m7,
            IntArray hits,
            IntArray dims) {
        final int rows = dims.get(0);
        final int used = dims.get(1);
        for (@Parallel int i = 0; i < rows; i++) {
            int all = 1;
            if (used > 0 && m0.get(i) == 0) {
                all = 0;
            }
            if (used > 1 && m1.get(i) == 0) {
                all = 0;
            }
            if (used > 2 && m2.get(i) == 0) {
                all = 0;
            }
            if (used > 3 && m3.get(i) == 0) {
                all = 0;
            }
            if (used > 4 && m4.get(i) == 0) {
                all = 0;
            }
            if (used > 5 && m5.get(i) == 0) {
                all = 0;
            }
            if (used > 6 && m6.get(i) == 0) {
                all = 0;
            }
            if (used > 7 && m7.get(i) == 0) {
                all = 0;
            }
            hits.set(i, all);
        }
    }
}
