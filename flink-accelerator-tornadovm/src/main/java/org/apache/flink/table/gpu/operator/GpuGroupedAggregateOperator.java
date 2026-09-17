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
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.codegen.GpuAggregateSpec;
import org.apache.flink.table.gpu.codegen.GpuCalcSpec;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.gather.RowGather;
import org.apache.flink.table.gpu.metrics.OffloadMetrics;

/**
 * Executes an offloaded local {@code GROUP BY}: buffers rows, runs the projection kernel and cuDF's
 * group-by over the batch in one task graph, emits one row per group.
 *
 * <p>Almost all of this is {@link GpuCalcOperator}: the same staging, the same gather chosen from
 * the first record, the same deferral of output by up to one batch. Two things differ, and both
 * follow from what the device returns.
 *
 * <h2>The drain is per group, not per row</h2>
 *
 * <p>A Calc copies out a column per row and spends most of its time there; this copies out one row
 * per distinct key. On the query this was first measured on that is the difference between 2M rows
 * crossing the interconnect and 64 — which is the argument for grouping on the device rather than
 * projecting on it and grouping after.
 *
 * <h2>It is a partial aggregate, so a key may be emitted more than once</h2>
 *
 * <p>Flink splits {@code GROUP BY} into a local aggregate, a shuffle and a merge, and only the
 * local half reaches here. Emitting each batch's groups as they are produced is therefore already
 * correct — the merge stage downstream combines them — and it is what keeps this operator stateless
 * across batches, with no device-side hash table spanning the partition.
 */
public class GpuGroupedAggregateOperator extends AbstractStreamOperator<RowData>
        implements OneInputStreamOperator<RowData, RowData>, BoundedOneInput {

    private static final long serialVersionUID = 1L;

    private final GpuCalcSpec spec;
    private final GpuAggregateSpec aggregate;

    private final transient GeneratedKernelEngine.Staging staging;
    private final int batchSize;
    private final boolean profile;

    private transient GeneratedKernelEngine engine;
    private transient RowGather[] gathers;
    private transient GenericRowData outRow;
    private transient StreamRecord<RowData> outElement;
    private transient int buffered;

    public GpuGroupedAggregateOperator(
            GpuCalcSpec spec,
            GpuAggregateSpec aggregate,
            int batchSize,
            boolean profile,
            GeneratedKernelEngine.Staging staging) {
        this.spec = spec;
        this.aggregate = aggregate;
        this.batchSize = batchSize;
        this.profile = profile;
        this.staging = staging;
    }

    @Override
    public void open() throws Exception {
        super.open();
        engine = new GeneratedKernelEngine(spec, aggregate, profile, staging);
        engine.open();
        registerMetrics();
        gathers = new RowGather[spec.kernel().inputFieldIndexes().length];
        outRow = new GenericRowData(2);
        outElement = new StreamRecord<>(null);
        buffered = 0;
    }

    @Override
    public void processElement(StreamRecord<RowData> element) throws Exception {
        RowData row = element.getValue();
        if (gathers[0] == null) {
            // As in GpuCalcOperator: the concrete RowData implementation is not knowable at plan
            // time, so the gather strategy is chosen from the first record actually seen.
            int[] fields = spec.kernel().inputFieldIndexes();
            GpuValueType[] types = spec.kernel().inputTypes();
            for (int c = 0; c < fields.length; c++) {
                gathers[c] =
                        RowGather.forColumn(
                                row,
                                fields[c],
                                types[c],
                                engine.inputColumn(c),
                                engine.inputSegment(c));
            }
        }
        // No validity branch, and none is missing: the provider declines an aggregate whose kernel
        // carries validity at all. A null key would become a group of its own on the device and a
        // null summand would poison one, and neither has an answer cuDF's group-by can give here.
        for (RowGather g : gathers) {
            g.accept(row, buffered);
        }
        if (++buffered == batchSize) {
            flush();
        }
    }

    @Override
    public void endInput() throws Exception {
        flush();
    }

    private void flush() throws Exception {
        if (buffered == 0) {
            return;
        }
        int count = buffered;
        // Reset before emitting, for the same reason the Calc operator does: collect runs the rest
        // of the chain.
        buffered = 0;

        // As in GpuCalcOperator: a columnar gather holds a run until it is told to write it.
        for (RowGather g : gathers) {
            if (g != null) {
                g.flush();
            }
        }

        GeneratedKernelEngine.Execution execution = engine.execute(count);

        long drainStart = System.nanoTime();
        int groups = engine.groups();
        for (int g = 0; g < groups; g++) {
            outRow.setField(0, engine.groupKey(g));
            outRow.setField(1, engine.groupSum(g));
            output.collect(outElement.replace(outRow));
        }
        engine.recordBatch(count, 0, execution, groups, System.nanoTime() - drainStart);
    }

    /**
     * What each staged column's gather actually did, once the batches have run.
     *
     * <p>Worth reporting rather than inferring: which tier applies is decided from the first record
     * seen, so it is a property of what the plan put upstream and not of anything in the plan, and
     * the bulk tier can silently degrade to per-row access for a column that turns out to be
     * dictionary-encoded or nullable. A correct result says nothing about which of those happened.
     */
    public String[] gatherTiers() {
        if (gathers == null) {
            return new String[0];
        }
        String[] tiers = new String[gathers.length];
        for (int i = 0; i < gathers.length; i++) {
            tiers[i] = gathers[i] == null ? "unbound" : gathers[i].tier();
        }
        return tiers;
    }

    @Override
    public void close() throws Exception {
        if (engine != null) {
            OffloadMetrics metrics = engine.metrics();
            if (metrics.getBatches() > 0) {
                LOG.info(
                        metrics.report("GpuGroupedAggregateOperator " + aggregate)
                                + "gather tiers: "
                                + String.join(", ", gatherTiers())
                                + System.lineSeparator());
            }
            engine.close();
            engine = null;
        }
        super.close();
    }

    /**
     * The engine's own numbers, for a test that needs to know a device actually ran rather than
     * that an operator was built. Same accessor {@link GpuCalcOperator} exposes and for the same
     * reason.
     */
    public OffloadMetrics metrics() {
        return engine.metrics();
    }

    /** The same gauges the Calc operator publishes; {@code rowsOut} is groups here. */
    private void registerMetrics() {
        OffloadMetrics m = engine.metrics();
        getMetricGroup().gauge("acceleratorBatches", (Gauge<Long>) m::getBatches);
        getMetricGroup().gauge("acceleratorRowsIn", (Gauge<Long>) m::getRowsIn);
        getMetricGroup().gauge("acceleratorRowsOut", (Gauge<Long>) m::getRowsOut);
        getMetricGroup().gauge("acceleratorGatherNanos", (Gauge<Long>) m::getGatherNanos);
        getMetricGroup().gauge("acceleratorCopyInNanos", (Gauge<Long>) m::getCopyInNanos);
        getMetricGroup().gauge("acceleratorKernelNanos", (Gauge<Long>) m::getKernelNanos);
        getMetricGroup().gauge("acceleratorCopyOutNanos", (Gauge<Long>) m::getCopyOutNanos);
        getMetricGroup().gauge("acceleratorDrainNanos", (Gauge<Long>) m::getDrainNanos);
    }
}
