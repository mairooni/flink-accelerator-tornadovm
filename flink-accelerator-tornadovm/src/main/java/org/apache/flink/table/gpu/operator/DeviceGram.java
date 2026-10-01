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
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The Gram contraction as a kernel, so it can be measured against {@code cublasGemm}.
 *
 * <p>Exists to answer one question: how much of a device region's advantage is the library and how
 * much is simply that the reduction happens on the device at all. Without a contraction on the
 * device the {@code d(d+1)/2} products cross back per row for the host to sum, which is the cost
 * that sinks the whole shape; with one, only the {@code d x d} matrix crosses, once per file. That
 * is true of any device-side contraction and is not a property of cuBLAS.
 *
 * <p>This is deliberately the naive form -- one thread per output entry, walking the rows -- and is
 * not expected to match a tuned GEMM. A tuned one tiles into shared memory so each loaded value
 * serves many outputs; this reloads both operands from global memory for every product, so it moves
 * {@code 2 n d^2} values where cuBLAS moves closer to {@code n d}. The point is to measure that gap
 * rather than argue about it.
 *
 * <h2>Accuracy</h2>
 *
 * <p>Each entry accumulates sequentially in one thread, which is the same shape as the host's
 * accumulator and therefore carries the same hazard §T46 found: in FP32 a running total that grows
 * far past the size of its increments stops accumulating. cuBLAS blocks its reduction and keeps
 * the partials small. Expect this kernel to be less accurate than the GEMM, and measurably so.
 */
public final class DeviceGram {

    private DeviceGram() {}

    /**
     * {@code C = A' A} for A held column-major, {@code rows} by {@code d}.
     *
     * <p>{@code dims} carries the row count and the width in a buffer rather than as scalars
     * because a scalar is captured when the graph is built and cannot change between executions;
     * the row count changes with the file.
     */
    public static void contractFloat(FloatArray packed, FloatArray gram, IntArray dims) {
        final int rows = dims.get(0);
        final int d = dims.get(1);
        for (@Parallel int i = 0; i < d; i++) {
            for (@Parallel int j = 0; j < d; j++) {
                float acc = 0.0f;
                for (int r = 0; r < rows; r++) {
                    acc += packed.get(i * rows + r) * packed.get(j * rows + r);
                }
                // Column-major, matching what cublasGemm writes, so the drain reads the same way
                // whichever produced it.
                gram.set(j * d + i, acc);
            }
        }
    }

    /** The same contraction in FP64. */
    public static void contractDouble(DoubleArray packed, DoubleArray gram, IntArray dims) {
        final int rows = dims.get(0);
        final int d = dims.get(1);
        for (@Parallel int i = 0; i < d; i++) {
            for (@Parallel int j = 0; j < d; j++) {
                double acc = 0.0;
                for (int r = 0; r < rows; r++) {
                    acc += packed.get(i * rows + r) * packed.get(j * rows + r);
                }
                gram.set(j * d + i, acc);
            }
        }
    }
}
