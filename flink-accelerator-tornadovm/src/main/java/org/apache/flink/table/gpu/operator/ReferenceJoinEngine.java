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

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;

/**
 * A similarity join against a reference table, where the reference side stays on the device.
 *
 * <p>This engine exists to measure one thing: what {@link DataTransferMode#FIRST_EXECUTION} is
 * worth. Every other benchmark in this project re-uploads on every batch, because every batch is
 * different data. Here one operand is <em>not</em> different -- the query says so.
 *
 * <pre>{@code
 * SELECT q.id, MAX(dot(q, r)) FROM queries q, refs r GROUP BY q.id
 * }</pre>
 *
 * <p>Read relationally, that is a join whose build side is invariant and whose probe side streams.
 * Read numerically it is a GEMM followed by a row-wise maximum. Both readings agree on the thing
 * that matters here: {@code refs} is consumed by every batch and changes for none of them.
 *
 * <h2>Why the reuse count is not an artefact</h2>
 *
 * <p>{@link LogisticRegressionEngine} also keeps data resident, but the number of times it re-reads
 * that data is the iteration count -- a number the caller picks, in a loop the SQL never contains.
 * That makes residency easy to demonstrate and easy to dismiss.
 *
 * <p>Here the reuse count is {@code queryRows / batchSize}. It is set by the cardinality of the
 * probe side and by the batch the operator chose, both properties of the job. Nobody chose to
 * iterate. Streaming a table past a resident reference set is simply what this join <em>is</em>,
 * and a device that had to re-upload {@code refs} per batch would be doing so for no reason
 * expressible in the query.
 *
 * <h2>What the A/B measures</h2>
 *
 * <p>{@code resident} switches {@code refs} between {@link DataTransferMode#FIRST_EXECUTION} and
 * {@link DataTransferMode#EVERY_EXECUTION} and changes nothing else -- same kernels, same device,
 * same arithmetic, same results. The difference between the two arms is {@code refs} crossing PCIe
 * {@code executions} times instead of once.
 *
 * <p>The honest claim that comes out of this is not that residency makes a device faster than a
 * CPU. It is that without residency the transfer of an invariant operand is charged against every
 * batch, and past a certain ratio of operand bytes to per-batch arithmetic it is the whole cost.
 * Residency does not win the comparison; it stops the interconnect from deciding it.
 *
 * <h2>Shapes</h2>
 *
 * <p>All matrices are column-major, which is what cuBLAS reads and what lets the transpose in
 * {@code C = Q R'} cost nothing:
 *
 * <ul>
 *   <li>{@code queries}: {@code batchSize x dims}, leading dimension {@code batchSize}
 *   <li>{@code refs}: {@code refCount x dims}, leading dimension {@code refCount}
 *   <li>{@code scores}: {@code batchSize x refCount}, leading dimension {@code batchSize}
 * </ul>
 *
 * <p>{@code scores} is the largest buffer and never leaves the device -- it is born in the GEMM and
 * dies in the maximum. On the host the same pipeline would materialise it and read it back, which
 * is the other half of why this belongs in one task graph.
 *
 * <h2>Float, not double</h2>
 *
 * <p>FP32, because the binding exposes no FP64 GEMM. A dot product over {@code dims} terms is far
 * more forgiving than the Gram matrix in {@link FeatureGramEngine} -- it accumulates {@code dims}
 * products, not {@code rows} of them -- but a caller comparing scores against a double-precision
 * reference should expect the last few digits to differ, and should expect ties to be broken
 * differently when two references are equidistant.
 */
public final class ReferenceJoinEngine implements AutoCloseable {

    private final int refCount;
    private final int dims;
    private final int batchSize;
    private final boolean resident;

    /**
     * Which contraction runs: {@code "library"} for cuBLAS, {@code "generated"} for the naive
     * kernel below.
     *
     * <p>The point of the second is not to beat cuBLAS. It is that the automatic path cannot call a
     * library at all, so the generated arm is what this query would be worth if the planner could
     * reach the shape but nothing else changed -- which separates "teach the planner this shape"
     * from "teach the planner to call libraries" as two pieces of work with two payoffs.
     */
    private final String contraction;

    /** The invariant operand: {@code refCount x dims}, column-major. Staged once, before open. */
    private final FloatArray refs;

    /** The probe batch: {@code batchSize x dims}, column-major, refilled every batch. */
    private final FloatArray queries;

    /** {@code batchSize x refCount}. Never transferred in either direction. */
    private final FloatArray scores;

    private final FloatArray bestScore;
    private final IntArray bestRef;

    private TornadoExecutionPlan plan;
    private int staged;
    private long executions;
    private long rows;
    private double executeMillis;

    public ReferenceJoinEngine(int refCount, int dims, int batchSize, boolean resident) {
        this(refCount, dims, batchSize, resident, "library");
    }

    public ReferenceJoinEngine(
            int refCount, int dims, int batchSize, boolean resident, String contraction) {
        this.refCount = refCount;
        this.dims = dims;
        this.batchSize = batchSize;
        this.resident = resident;
        this.contraction = contraction;
        this.refs = new FloatArray(refCount * dims);
        this.queries = new FloatArray(batchSize * dims);
        this.scores = new FloatArray(batchSize * refCount);
        this.bestScore = new FloatArray(batchSize);
        this.bestRef = new IntArray(batchSize);
    }

    /** Stages one element of the reference table. Must be complete before {@link #open()}. */
    public void setReference(int ref, int dim, double value) {
        refs.set(dim * refCount + ref, (float) value);
    }

    /**
     * Builds the plan.
     *
     * <p>The only difference between the two arms is the transfer mode of {@code refs}. Under
     * {@link DataTransferMode#FIRST_EXECUTION} the buffer is uploaded once and every later
     * execution finds it already there; under {@link DataTransferMode#EVERY_EXECUTION} the same
     * bytes cross the bus again for each batch.
     */
    public void open() {
        TaskGraph graph = new TaskGraph("reference-join");
        graph =
                resident
                        ? graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, refs)
                        : graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, refs);

        // S = Q R'. Q is batchSize x dims (no transpose), R is refCount x dims so R' is dims x
        // refCount (transpose). Column-major throughout, so neither operand is copied or permuted:
        // the transpose is an argument to the call, not work.
        graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, queries);
        graph =
                "generated".equals(contraction)
                        ? graph.task(
                                "scores",
                                ReferenceJoinEngine::scores,
                                queries,
                                refs,
                                scores,
                                batchSize,
                                refCount,
                                dims)
                        : graph.libraryTask(
                                "scores",
                                CuBlas::cublasSgemm,
                                CuBlasOperation.CUBLAS_OP_N.operation(),
                                CuBlasOperation.CUBLAS_OP_T.operation(),
                                batchSize,
                                refCount,
                                dims,
                                1.0f,
                                queries,
                                batchSize,
                                refs,
                                refCount,
                                0.0f,
                                scores,
                                batchSize);
        plan =
                new TornadoExecutionPlan(
                        graph.task(
                                        "best",
                                        ReferenceJoinEngine::best,
                                        scores,
                                        bestScore,
                                        bestRef,
                                        batchSize,
                                        refCount)
                                .transferToHost(
                                        DataTransferMode.EVERY_EXECUTION, bestScore, bestRef)
                                .snapshot());
    }

    /** Stages one value of the query row being built. */
    public void set(int dim, double value) {
        queries.set(dim * batchSize + staged, (float) value);
    }

    /**
     * Ends a query row. Returns the number of results ready, or zero if the batch is not full.
     *
     * <p>Results are read from {@link #bestScore(int)} and {@link #bestRef(int)} for indices below
     * the returned count, and are only valid until the next row is staged.
     */
    public int rowComplete() {
        staged++;
        rows++;
        if (staged == batchSize) {
            int ready = staged;
            execute();
            return ready;
        }
        return 0;
    }

    /**
     * Runs whatever is staged, zero-filling the tail. Returns the number of valid results.
     *
     * <p>A zero query vector scores zero against every reference, so the padding produces answers
     * -- they are simply not among the first {@code staged} and are never read.
     */
    public int flush() {
        if (staged == 0) {
            return 0;
        }
        for (int d = 0; d < dims; d++) {
            for (int r = staged; r < batchSize; r++) {
                queries.set(d * batchSize + r, 0.0f);
            }
        }
        int ready = staged;
        execute();
        return ready;
    }

    private void execute() {
        long start = System.nanoTime();
        try {
            plan.execute();
        } catch (Exception e) {
            throw new IllegalStateException("reference-join execution failed", e);
        }
        executeMillis += (System.nanoTime() - start) / 1e6;
        executions++;
        staged = 0;
    }

    /**
     * The contraction as a code generator would plausibly emit it: one thread per (query,
     * reference) pair, each walking the shared dimension.
     *
     * <p>Deliberately naive. It tiles nothing and reuses nothing through shared memory, which is
     * exactly what separates it from {@code cublasSgemm} and exactly what a straightforward
     * generator would produce. Reading its result as "GPUs are slower than cuBLAS" would be a
     * mistake; it measures the generator, not the device.
     *
     * <p>Indices follow the same column-major layout as the library call, so the two arms are
     * interchangeable and the maximum below cannot tell which one ran.
     */
    public static void scores(
            FloatArray queries,
            FloatArray refs,
            FloatArray scores,
            int batchSize,
            int refCount,
            int dims) {
        for (@Parallel int i = 0; i < batchSize * refCount; i++) {
            int q = i % batchSize;
            int r = i / batchSize;
            float sum = 0.0f;
            for (int d = 0; d < dims; d++) {
                sum += queries.get(d * batchSize + q) * refs.get(d * refCount + r);
            }
            scores.set(i, sum);
        }
    }

    /**
     * The row-wise maximum, one thread per query.
     *
     * <p>The scan over references is sequential per thread and reads a column of {@code scores}
     * with a stride of {@code queries}, so consecutive threads read consecutive addresses at every
     * step -- coalesced, which is the reason {@code scores} is column-major rather than the layout
     * that would make this loop contiguous.
     */
    public static void best(
            FloatArray scores, FloatArray bestScore, IntArray bestRef, int queries, int refs) {
        for (@Parallel int q = 0; q < queries; q++) {
            float top = scores.get(q);
            int at = 0;
            for (int r = 1; r < refs; r++) {
                float value = scores.get(r * queries + q);
                if (value > top) {
                    top = value;
                    at = r;
                }
            }
            bestScore.set(q, top);
            bestRef.set(q, at);
        }
    }

    public double bestScore(int index) {
        return bestScore.get(index);
    }

    public int bestRef(int index) {
        return bestRef.get(index);
    }

    /** How many times the plan ran -- the number of times {@code refs} was reused. */
    public long executionCount() {
        return executions;
    }

    public long rowCount() {
        return rows;
    }

    /** Wall time inside {@link TornadoExecutionPlan#execute()}, summed over every batch. */
    public double executeMillis() {
        return executeMillis;
    }

    /** Bytes of the invariant operand, which is what the two arms disagree about. */
    public long referenceBytes() {
        return (long) refCount * dims * Float.BYTES;
    }

    @Override
    public void close() throws Exception {
        if (plan != null) {
            plan.close();
            plan = null;
        }
    }
}
