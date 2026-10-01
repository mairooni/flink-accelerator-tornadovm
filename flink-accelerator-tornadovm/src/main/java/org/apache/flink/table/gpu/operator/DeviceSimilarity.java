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

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The kernels either side of the GEMM in a similarity region, and the two kernels that replace it.
 *
 * <p>Three contractions are here rather than one because the question the region exists to answer
 * is which of them a query should get, and the only honest way to answer it is to run the same
 * query three ways. {@link #scoreTiled} is what a competent kernel generator would emit for an
 * inner product between two matrices; {@link #scoreFused} is what it would emit if it noticed the
 * score matrix is never read as a whole and skipped materialising it. Neither is a straw man, and
 * at narrow widths the fused one wins — see {@code GpuSimilarityJoinSpec.MIN_WIDTH}.
 *
 * <h2>Layout</h2>
 *
 * <p>Everything is column-major, because that is what {@code Cudf::readParquet} writes and what
 * cuBLAS reads, so no pass exists to convert between them. The probe matrix is {@code nQ} rows by
 * {@code d} columns with leading dimension {@code nQ}; the build matrix is {@code nB} by {@code d}
 * with leading dimension {@code nB}; the scores are {@code nQ} by {@code nB} with leading dimension
 * {@code nQ}, so {@code scores[b * nQ + q]} is the score of probe {@code q} against build row
 * {@code b}.
 *
 * <p>That last choice is what makes the epilogue coalesced. One thread per probe row walks the
 * build rows, and at a fixed build row consecutive threads read consecutive addresses. The
 * transpose would have given every thread a contiguous run and the warp a stride of {@code nB},
 * which is the slower way round.
 */
public final class DeviceSimilarity {

    /** A score no inner product of real data reaches, marking a probe row with no partner yet. */
    public static final float NO_MATCH = -Float.MAX_VALUE;

    private DeviceSimilarity() {}

    /**
     * Fold one tile of scores into the running maximum per probe row.
     *
     * <p>{@code dims} carries the counts in a buffer rather than as scalars because a scalar is
     * captured when the graph is built and cannot change between executions, and the row count
     * changes with the file.
     */
    public static void rowMax(FloatArray scores, FloatArray best, IntArray dims) {
        final int nQ = dims.get(0);
        final int nB = dims.get(1);
        for (@Parallel int q = 0; q < nQ; q++) {
            float m = best.get(q);
            for (int b = 0; b < nB; b++) {
                final float v = scores.get(b * nQ + q);
                if (v > m) {
                    m = v;
                }
            }
            best.set(q, m);
        }
    }

    /**
     * The contraction as a tiled kernel: the strongest thing a generator could reasonably emit.
     *
     * <p>Each workgroup computes a {@code TS x TS} block of the score matrix, staging a tile of
     * each operand in local memory so every value loaded serves {@code TS} products instead of one.
     * This is the kernel cuBLAS is measured against, and it is a real one — at {@code d = 32} it is
     * within a factor of two of the library, and the gap only opens as {@code d} grows.
     */
    public static void scoreTiled(
            KernelContext ctx, FloatArray probe, FloatArray build, FloatArray scores,
            int nQ, int nB, int d) {
        final int ts = 16;
        final int localQ = ctx.localIdx;
        final int localB = ctx.localIdy;
        final int q = ts * ctx.groupIdx + localQ;
        final int b = ts * ctx.groupIdy + localB;

        final float[] pSub = ctx.allocateFloatLocalArray(ts * ts);
        final float[] bSub = ctx.allocateFloatLocalArray(ts * ts);

        float acc = 0.0f;
        final int tiles = d / ts;
        for (int t = 0; t < tiles; t++) {
            // Column-major: column (ts*t + localB) of the probe, row q.
            pSub[localB * ts + localQ] = probe.get((ts * t + localB) * nQ + q);
            bSub[localQ * ts + localB] = build.get((ts * t + localQ) * nB + b);
            ctx.localBarrier();
            for (int k = 0; k < ts; k++) {
                acc += pSub[k * ts + localQ] * bSub[k * ts + localB];
            }
            ctx.localBarrier();
        }
        scores.set(b * nQ + q, acc);
    }

    /**
     * The contraction and the maximum in one kernel, with no score matrix at all.
     *
     * <p>The arm that makes the comparison honest. A library has to write its result somewhere, and
     * for this query that result is {@code nQ x nB} values nothing ever reads — so a kernel that
     * fuses the reduction into the product avoids a write the library cannot. At {@code d = 32}
     * that saving is worth more than the library's tuning and this wins. It stops being worth it
     * quickly: the kernel reloads both operands from global memory for every pair, moving
     * {@code O(nQ · nB · d)} where a blocked GEMM moves closer to {@code O(nQ · nB)}, and that
     * gap grows with {@code d} while the saved write does not.
     */
    public static void scoreFused(
            FloatArray probe, FloatArray build, FloatArray best, IntArray dims) {
        final int nQ = dims.get(0);
        final int nB = dims.get(1);
        final int d = dims.get(2);
        for (@Parallel int q = 0; q < nQ; q++) {
            float m = best.get(q);
            for (int b = 0; b < nB; b++) {
                float acc = 0.0f;
                for (int k = 0; k < d; k++) {
                    acc += probe.get(k * nQ + q) * build.get(k * nB + b);
                }
                if (acc > m) {
                    m = acc;
                }
            }
            best.set(q, m);
        }
    }
}
