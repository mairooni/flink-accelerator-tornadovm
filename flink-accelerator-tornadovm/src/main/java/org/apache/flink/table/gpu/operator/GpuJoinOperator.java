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

import org.apache.flink.metrics.Gauge;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedMultiInput;
import org.apache.flink.streaming.api.operators.InputSelectable;
import org.apache.flink.streaming.api.operators.InputSelection;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.writer.BinaryRowWriter;
import org.apache.flink.table.gpu.codegen.GpuJoinSpec;
import org.apache.flink.table.gpu.metrics.OffloadMetrics;
import org.apache.flink.table.runtime.gpu.FallbackToCpuTwoInputOperator;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cudf.Cudf;

import javax.annotation.Nullable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Inner-joins two inputs on one {@code INT} equality: cuDF orders the build keys, a generated
 * kernel binary-searches them, and the host gathers the matched pairs.
 *
 * <h2>The split, and why it is this one</h2>
 *
 * <p>§T5 measured both halves separately and they answered differently. A <b>sort</b> is not
 * expressible as a per-row map, so the ordering is bound: {@code Cudf.sortedOrder}, the same
 * primitive M5.4 uses. A <b>probe</b> is a per-row map — one row in, one binary search, one row out
 * — and a kernel written in this project's own idiom landed within 5% of {@code
 * thrust::lower_bound}. So the probe is {@link #probe}, forty lines of Java that TornadoVM
 * compiles, and not a second library call. Binding both would be slower and larger, and would put a
 * hash table on the device that nothing else here needs.
 *
 * <h2>Indices, not rows</h2>
 *
 * <p>The kernel writes two numbers per probe row: where its key starts in the ordered build side,
 * and how many build rows share it. Nothing about the payload crosses the interconnect in either
 * direction — the same property that lets the sort carry a {@code BIGINT}, for the same reason.
 * Gathering a matched pair into an output row is a host loop over staged columns.
 *
 * <h2>Duplicates are the reason there are two numbers and not one</h2>
 *
 * <p>An equi-join against a build side with repeated keys emits one row per pair, so a probe row
 * matching three build rows produces three. A single {@code lower_bound} cannot say that; a lower
 * and an upper bound can, and the count between them is exactly the fan-out. It costs a second
 * binary search per probe row and it is the difference between a join and a lookup.
 *
 * <h2>The build side is bounded, and declined rather than spilled</h2>
 *
 * <p>Exactly as the sort is, and for the same reason: the build side has to be resident to probe
 * against, and {@code HashJoinOperator} underneath already knows how to spill. A build side larger
 * than the staging is refused in {@link #open()}, which is the last moment {@link
 * FallbackToCpuTwoInputOperator} can hand the task back. If the estimate was wrong and the rows
 * arrive anyway, this finishes on the host with what it is holding — correct, much slower, and
 * reported.
 */
public class GpuJoinOperator extends AbstractStreamOperator<RowData>
        implements TwoInputStreamOperator<RowData, RowData, RowData>,
                BoundedMultiInput,
                InputSelectable,
                FallbackToCpuTwoInputOperator.SelfReportingAccelerator {

    private static final long serialVersionUID = 1L;

    private final GpuJoinSpec spec;
    private final int buildCapacity;
    private final int probeBatchSize;
    private final transient GeneratedKernelEngine.Staging staging;

    private transient StagedColumns build;
    private transient StagedColumns probe;

    /** The build keys in ascending order, which is what the probe searches. */
    private transient IntArray orderedKeys;

    /** For each ordered position, which staged build row it came from. */
    private transient IntArray order;

    private transient IntArray probeKeys;
    private transient IntArray matchStart;
    private transient IntArray matchCount;

    /** {@code [buildRows, probeRows]}, read by the kernel so a short batch launches short. */
    private transient IntArray dims;

    private transient TornadoExecutionPlan probePlan;
    private transient WorkerGrid1D probeGrid;

    private transient int buildRows;
    private transient int probeBuffered;
    private transient boolean buildEnded;

    private transient BinaryRowData outRow;
    private transient BinaryRowWriter outWriter;
    private transient StreamRecord<RowData> outElement;
    private transient OffloadMetrics metrics;

    /** Non-null once this has stopped using the device: every build row, keyed. */
    private transient @Nullable Map<Integer, List<BinaryRowData>> onHost;

    public GpuJoinOperator(
            GpuJoinSpec spec,
            int buildCapacity,
            int probeBatchSize,
            @Nullable GeneratedKernelEngine.Staging staging) {
        this.spec = spec;
        this.buildCapacity = buildCapacity;
        this.probeBatchSize = probeBatchSize;
        this.staging = staging;
    }

    @Override
    public void open() throws Exception {
        super.open();
        if (spec.estimatedBuildRows() > buildCapacity) {
            throw new IllegalStateException(
                    "this join would hold "
                            + spec.estimatedBuildRows()
                            + " build rows and the staging offered fits "
                            + buildCapacity
                            + "; declining rather than spilling");
        }
        build =
                StagedColumns.allocate(
                        spec.buildType(), spec.buildKeyField(), buildCapacity, staging);
        probe =
                StagedColumns.allocate(
                        spec.probeType(), spec.probeKeyField(), probeBatchSize, staging);
        orderedKeys = StagedColumns.ints(buildCapacity, staging);
        order = StagedColumns.ints(buildCapacity, staging);
        probeKeys = StagedColumns.ints(probeBatchSize, staging);
        matchStart = StagedColumns.ints(probeBatchSize, staging);
        matchCount = StagedColumns.ints(probeBatchSize, staging);
        dims = StagedColumns.ints(2, staging);

        outRow = new BinaryRowData(spec.outputType().getFieldCount());
        outWriter = new BinaryRowWriter(outRow);
        outElement = new StreamRecord<>(null);
        metrics = new OffloadMetrics();
        registerMetrics();
        buildRows = 0;
        probeBuffered = 0;
        buildEnded = false;
        onHost = null;
    }

    /**
     * Build side first, which is the contract {@code HashJoinOperator} states and the plan uses.
     */
    @Override
    public InputSelection nextSelection() {
        return buildEnded ? InputSelection.SECOND : InputSelection.FIRST;
    }

    @Override
    public void processElement1(StreamRecord<RowData> element) {
        RowData row = element.getValue();
        if (onHost != null) {
            addToHostTable(row);
            return;
        }
        if (buildRows == buildCapacity) {
            degrade(
                    "more build rows arrived than the planner estimated ("
                            + spec.estimatedBuildRows()
                            + "), and the staging holds "
                            + buildCapacity);
            addToHostTable(row);
            return;
        }
        build.stage(row, buildRows++);
    }

    @Override
    public void processElement2(StreamRecord<RowData> element) throws Exception {
        RowData row = element.getValue();
        if (onHost != null) {
            probeOnHost(row);
            return;
        }
        probe.stage(row, probeBuffered++);
        if (probeBuffered == probeBatchSize) {
            flush();
        }
    }

    @Override
    public void endInput(int inputId) throws Exception {
        if (inputId == 1) {
            buildEnded = true;
            if (onHost == null) {
                try {
                    orderBuildSide();
                } catch (Throwable t) {
                    LOG.warn("ordering the build side on the device failed", t);
                    degrade("the device failed to order the build side");
                }
            }
            return;
        }
        if (onHost == null) {
            flush();
        }
    }

    /**
     * Orders the build keys once, at the end of the build side.
     *
     * <p>The permutation comes back rather than sorted data, so the ordered key column is built
     * here from it. That host pass is one sequential write per build row and it is what lets the
     * probe kernel read a sorted array without cuDF having had to move the payload.
     */
    private void orderBuildSide() throws Exception {
        if (buildRows == 0) {
            return;
        }
        long start = System.nanoTime();
        TaskGraph graph =
                new TaskGraph("join-order")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, build.keys())
                        // The key is refused unless it is NOT NULL, so where absent keys go cannot
                        // arise; stated as 0 for the same reason the sort states it.
                        .libraryTask("order", Cudf::sortedOrder, buildRows, build.keys(), 0, order)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, order);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.execute();
        }
        for (int i = 0; i < buildRows; i++) {
            orderedKeys.set(i, build.keyAt(order.get(i)));
        }
        dims.set(0, buildRows);
        buildProbePlan();
        metrics.recordBatch(buildRows, 0, System.nanoTime() - start, 0L, 0L, null);
    }

    /**
     * The probe, over one batch of probe rows.
     *
     * <p>Built once, at the end of the build side, because the ordered key column and its length
     * are fixed from then on; the batch size varies and is narrowed by the grid rather than by
     * rebuilding the graph.
     */
    private void buildProbePlan() {
        TaskGraph graph =
                new TaskGraph("join-probe")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, orderedKeys)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, probeKeys, dims)
                        .task(
                                "probe",
                                GpuJoinOperator::probe,
                                orderedKeys,
                                probeKeys,
                                matchStart,
                                matchCount,
                                dims)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, matchStart, matchCount);
        // Stated rather than inferred, for the reason recorded in GeneratedKernelEngine: the loop
        // bound is read from a buffer, so inference gives up and emits a sequential loop --
        // silently, with correct results about a thousand times slower.
        probeGrid = new WorkerGrid1D(probeBatchSize);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("join-probe.probe", probeGrid);
        probePlan = new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(scheduler);
    }

    /**
     * One binary search per probe row, twice, for the bounds of its key in the ordered build side.
     *
     * <p>A per-row map with no cross-row cooperation, which is exactly the shape §T5 found a
     * generated kernel serves as well as a tuned library: every lane walks its own {@code log2(n)}
     * steps over a shared, read-only array.
     */
    public static void probe(
            IntArray orderedKeys,
            IntArray probeKeys,
            IntArray matchStart,
            IntArray matchCount,
            IntArray dims) {
        int buildRows = dims.get(0);
        int probeRows = dims.get(1);
        for (@Parallel int i = 0; i < probeRows; i++) {
            int key = probeKeys.get(i);
            int lo = 0;
            int hi = buildRows;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (orderedKeys.get(mid) < key) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            int lower = lo;
            hi = buildRows;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (orderedKeys.get(mid) <= key) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            matchStart.set(i, lower);
            matchCount.set(i, lo - lower);
        }
    }

    private void flush() throws Exception {
        if (probeBuffered == 0) {
            return;
        }
        int count = probeBuffered;
        probeBuffered = 0;
        if (buildRows == 0) {
            // An inner join against nothing emits nothing, and there is no plan to run.
            return;
        }
        for (int i = 0; i < count; i++) {
            probeKeys.set(i, probe.keyAt(i));
        }
        dims.set(1, count);
        long start = System.nanoTime();
        try {
            probeGrid.setGlobalWork(count, 1, 1);
            probePlan.execute();
        } catch (Throwable t) {
            LOG.warn("probing {} rows on the device failed", count, t);
            degrade("the device failed to probe");
            for (int i = 0; i < count; i++) {
                probeOnHostStaged(i);
            }
            return;
        }
        long executeNanos = System.nanoTime() - start;

        long drainStart = System.nanoTime();
        int emitted = 0;
        for (int i = 0; i < count; i++) {
            int start0 = matchStart.get(i);
            int matches = matchCount.get(i);
            for (int m = 0; m < matches; m++) {
                emit(order.get(start0 + m), i);
                emitted++;
            }
        }
        metrics.recordBatch(count, emitted, 0L, executeNanos, System.nanoTime() - drainStart, null);
    }

    /** Build fields then probe fields, or the other way round — whichever the query wrote. */
    private void emit(int buildPosition, int probePosition) {
        outWriter.reset();
        if (spec.buildIsLeft()) {
            build.writeInto(outWriter, 0, buildPosition);
            probe.writeInto(outWriter, spec.buildType().getFieldCount(), probePosition);
        } else {
            probe.writeInto(outWriter, 0, probePosition);
            build.writeInto(outWriter, spec.probeType().getFieldCount(), buildPosition);
        }
        outWriter.complete();
        output.collect(outElement.replace(outRow));
    }

    // ------------------------------------------------------------------------------------------
    // The host path, for when the estimate was wrong or the device stopped answering
    // ------------------------------------------------------------------------------------------

    /** Stops using the device, keeping every build row that has arrived so far. */
    private void degrade(String why) {
        LOG.warn("{}; this join finishes on the host, which does not spill", why);
        Map<Integer, List<BinaryRowData>> table = new HashMap<>();
        for (int i = 0; i < buildRows; i++) {
            BinaryRowData copy = new BinaryRowData(spec.buildType().getFieldCount());
            BinaryRowWriter writer = new BinaryRowWriter(copy);
            writer.reset();
            build.writeInto(writer, 0, i);
            writer.complete();
            table.computeIfAbsent(copy.getInt(spec.buildKeyField()), k -> new ArrayList<>())
                    .add(copy);
        }
        buildRows = 0;
        onHost = table;
    }

    private void addToHostTable(RowData row) {
        BinaryRowData copy = new BinaryRowData(spec.buildType().getFieldCount());
        BinaryRowWriter writer = new BinaryRowWriter(copy);
        writer.reset();
        StagedColumns.materialise(row, spec.buildType(), writer, 0);
        writer.complete();
        onHost.computeIfAbsent(row.getInt(spec.buildKeyField()), k -> new ArrayList<>()).add(copy);
    }

    private void probeOnHost(RowData row) {
        List<BinaryRowData> matches = onHost.get(row.getInt(spec.probeKeyField()));
        if (matches == null) {
            return;
        }
        for (BinaryRowData buildRow : matches) {
            outWriter.reset();
            if (spec.buildIsLeft()) {
                StagedColumns.materialise(buildRow, spec.buildType(), outWriter, 0);
                StagedColumns.materialise(
                        row, spec.probeType(), outWriter, spec.buildType().getFieldCount());
            } else {
                StagedColumns.materialise(row, spec.probeType(), outWriter, 0);
                StagedColumns.materialise(
                        buildRow, spec.buildType(), outWriter, spec.probeType().getFieldCount());
            }
            outWriter.complete();
            output.collect(outElement.replace(outRow));
        }
    }

    /** A probe row already staged when the device gave out. */
    private void probeOnHostStaged(int position) {
        BinaryRowData row = new BinaryRowData(spec.probeType().getFieldCount());
        BinaryRowWriter writer = new BinaryRowWriter(row);
        writer.reset();
        probe.writeInto(writer, 0, position);
        writer.complete();
        probeOnHost(row);
    }

    @Override
    public boolean stillOnDevice() {
        return onHost == null;
    }

    @Override
    public void close() throws Exception {
        if (probePlan != null) {
            probePlan.close();
            probePlan = null;
        }
        if (metrics != null && metrics.getBatches() > 0) {
            LOG.info(metrics.report("GpuJoinOperator " + spec));
        }
        super.close();
    }

    /** The operator's own numbers, for a test that needs to know a device actually ran. */
    public OffloadMetrics metrics() {
        return metrics;
    }

    private void registerMetrics() {
        getMetricGroup().gauge("acceleratorBatches", (Gauge<Long>) metrics::getBatches);
        getMetricGroup().gauge("acceleratorRowsIn", (Gauge<Long>) metrics::getRowsIn);
        getMetricGroup().gauge("acceleratorRowsOut", (Gauge<Long>) metrics::getRowsOut);
        getMetricGroup().gauge("acceleratorKernelNanos", (Gauge<Long>) metrics::getKernelNanos);
        getMetricGroup().gauge("acceleratorDrainNanos", (Gauge<Long>) metrics::getDrainNanos);
    }
}
