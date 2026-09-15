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

package org.apache.flink.table.gpu;

import org.apache.flink.table.gpu.operator.ReferenceJoinEngine;

import java.util.Random;

/**
 * {@link ReferenceJoinEngine} on its own, with a CPU reference to check it against.
 *
 * <p>Two jobs. First, correctness: a GEMM has four ways to be wrong that all still produce numbers,
 * and the only way to know which one is running is to compare against a straightforward loop.
 * Second, the transfer-mode A/B without Flink in the way, which is the number the end-to-end
 * benchmark should reproduce and which takes seconds to get here instead of minutes on a cluster.
 *
 * <pre>
 * java @$TORNADO_SDK/tornado-argfile -cp &lt;classes&gt; \
 *     org.apache.flink.table.gpu.ReferenceJoinSweep [refs] [dims] [queries]
 * </pre>
 */
public final class ReferenceJoinSweep {

    private static final int[] BATCHES = {32, 64, 128, 256, 512, 1024, 2048};

    private static final int REPEATS = 5;

    public static void main(String[] args) throws Exception {
        int refCount = args.length > 0 ? Integer.parseInt(args[0]) : 8192;
        int dims = args.length > 1 ? Integer.parseInt(args[1]) : 128;
        int queryRows = args.length > 2 ? Integer.parseInt(args[2]) : 16384;

        double[][] refs = random(refCount, dims, 7);
        double[][] queries = random(queryRows, dims, 13);

        System.out.printf(
                "refs=%,d x %d (%.1f MB)  queries=%,d%n%n",
                refCount, dims, refCount * (double) dims * Float.BYTES / (1 << 20), queryRows);

        correctness(refs, queries, refCount, dims, Math.min(256, queryRows));

        // The first plan built in this JVM pays for compiling both tasks. Whichever arm ran first
        // would otherwise carry that cost and lose to the other for a reason that has nothing to
        // do with transfer modes -- which is exactly what the first version of this sweep reported.
        System.out.print("\nwarming up... ");
        for (int batch : BATCHES) {
            if (batch <= queryRows) {
                run(refs, queries, refCount, dims, batch, true);
                run(refs, queries, refCount, dims, batch, false);
            }
        }
        System.out.println("done");

        System.out.printf(
                "%n%-8s %12s %12s %9s %12s %10s%n",
                "batch", "resident", "reupload", "speedup", "wasted MB", "GFLOP/exec");
        for (int batch : BATCHES) {
            if (batch > queryRows) {
                continue;
            }
            double resident = best(refs, queries, refCount, dims, batch, true);
            double reupload = best(refs, queries, refCount, dims, batch, false);
            long executions = (queryRows + batch - 1) / batch;
            System.out.printf(
                    "%-8d %9.1f ms %9.1f ms %8.2fx %12.1f %10.2f%n",
                    batch,
                    resident,
                    reupload,
                    reupload / resident,
                    (executions - 1) * refCount * (double) dims * Float.BYTES / (1 << 20),
                    (double) batch * refCount * dims * 2 / 1e9);
        }
    }

    /** Compares the device answers against a plain nested loop over the same numbers. */
    private static void correctness(
            double[][] refs, double[][] queries, int refCount, int dims, int rows)
            throws Exception {
        double worstScore = 0.0;
        int disagreements = 0;
        try (ReferenceJoinEngine engine = new ReferenceJoinEngine(refCount, dims, rows, true)) {
            for (int r = 0; r < refCount; r++) {
                for (int d = 0; d < dims; d++) {
                    engine.setReference(r, d, refs[r][d]);
                }
            }
            engine.open();
            for (int q = 0; q < rows; q++) {
                for (int d = 0; d < dims; d++) {
                    engine.set(d, queries[q][d]);
                }
                engine.rowComplete();
            }
            for (int q = 0; q < rows; q++) {
                double top = Double.NEGATIVE_INFINITY;
                int at = -1;
                for (int r = 0; r < refCount; r++) {
                    double dot = 0.0;
                    for (int d = 0; d < dims; d++) {
                        dot += queries[q][d] * refs[r][d];
                    }
                    if (dot > top) {
                        top = dot;
                        at = r;
                    }
                }
                worstScore =
                        Math.max(
                                worstScore,
                                Math.abs(top - engine.bestScore(q)) / Math.max(1.0, Math.abs(top)));
                if (at != engine.bestRef(q)) {
                    disagreements++;
                }
            }
        }
        System.out.printf(
                "correctness over %,d rows: worst score error %.3e, argmax disagreements %d%n",
                rows, worstScore, disagreements);
        if (worstScore > 1e-4) {
            throw new IllegalStateException("device scores are wrong, not merely FP32");
        }
        // A disagreement is legitimate when two references tie to within FP32, so allow a few but
        // not a pattern: a wrong layout disagrees on nearly every row.
        if (disagreements > rows / 100) {
            throw new IllegalStateException(disagreements + " argmax disagreements is a bug");
        }
    }

    /**
     * Fastest of {@link #REPEATS} attempts.
     *
     * <p>Minimum rather than mean: the thing being measured is a fixed cost per execution, and
     * every source of noise on a shared GPU adds to it. The mean would report the noise.
     */
    private static double best(
            double[][] refs,
            double[][] queries,
            int refCount,
            int dims,
            int batch,
            boolean resident)
            throws Exception {
        double best = Double.MAX_VALUE;
        for (int i = 0; i < REPEATS; i++) {
            best = Math.min(best, run(refs, queries, refCount, dims, batch, resident));
        }
        return best;
    }

    /** Runs the whole query at one batch size and returns device execute time. */
    private static double run(
            double[][] refs,
            double[][] queries,
            int refCount,
            int dims,
            int batch,
            boolean resident)
            throws Exception {
        try (ReferenceJoinEngine engine =
                new ReferenceJoinEngine(refCount, dims, batch, resident)) {
            for (int r = 0; r < refCount; r++) {
                for (int d = 0; d < dims; d++) {
                    engine.setReference(r, d, refs[r][d]);
                }
            }
            engine.open();
            for (double[] query : queries) {
                for (int d = 0; d < dims; d++) {
                    engine.set(d, query[d]);
                }
                engine.rowComplete();
            }
            engine.flush();
            return engine.executeMillis();
        }
    }

    private static double[][] random(int rows, int cols, int seed) {
        Random random = new Random(seed);
        double[][] out = new double[rows][cols];
        for (double[] row : out) {
            for (int c = 0; c < cols; c++) {
                row[c] = random.nextDouble() * 2.0 - 1.0;
            }
        }
        return out;
    }

    private ReferenceJoinSweep() {}
}
