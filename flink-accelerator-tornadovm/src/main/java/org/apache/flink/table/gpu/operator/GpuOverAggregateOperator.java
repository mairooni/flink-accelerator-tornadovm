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
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.writer.BinaryRowWriter;
import org.apache.flink.table.gpu.codegen.GpuOverAggregateSpec;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.metrics.OffloadMetrics;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.cudf.Cudf;

import javax.annotation.Nullable;

/**
 * A running total over an ordered input, scanned on the device a batch at a time.
 *
 * <h2>Why this one is not like the sort or the join</h2>
 *
 * <p>Because a prefix sum composes across batches and neither of those does. Scanning a batch gives
 * that batch's running totals from zero; adding the total carried out of the previous batch gives
 * the totals from the start of the input. So this operator holds one {@code double} between batches
 * and nothing else — no arena sized from an estimate, no refusal on size, and none of the open()
 * machinery M5.4 and M5.5 needed. It is the Calc's shape wearing a cross-row node's name.
 *
 * <h2>What it is competing against</h2>
 *
 * <p>Worth stating in the code because it decides how to read any number this produces. For an
 * {@code UNBOUNDED PRECEDING}/{@code CURRENT ROW} frame Flink's {@code needBufferData()} is false,
 * so the operator being replaced is {@code NonBufferOverWindowOperator}: a streaming accumulator
 * that buffers nothing, reserves no managed memory and spills never. M5.4 beat a sorter that
 * serialises every row into managed memory; there is no equivalent advantage to take here.
 *
 * <h2>The carry is the whole correctness story</h2>
 *
 * <p>An operator that forgot it would emit totals that are monotonic within each batch, correct in
 * the first, and wrong in every one after — with the right number of rows and the right shape. That
 * is the failure this is built around and the one both tests are pointed at.
 */
public class GpuOverAggregateOperator extends AbstractStreamOperator<RowData>
        implements OneInputStreamOperator<RowData, RowData>, BoundedOneInput {

    private static final long serialVersionUID = 1L;

    private final GpuOverAggregateSpec spec;
    private final int batchSize;
    private final transient GeneratedKernelEngine.Staging staging;

    /** The values to scan, and where the scan writes. */
    private transient DoubleArray values;

    private transient DoubleArray totals;

    /** The pass-through columns, so a staged row can be emitted after its total is known. */
    private transient StagedColumns rows;

    private transient int buffered;

    /** The running total at the end of the last batch. The one thing carried across them. */
    private transient double carried;

    private transient BinaryRowData outRow;
    private transient BinaryRowWriter outWriter;
    private transient StreamRecord<RowData> outElement;
    private transient OffloadMetrics metrics;
    private transient int inputFields;

    public GpuOverAggregateOperator(
            GpuOverAggregateSpec spec,
            int batchSize,
            @Nullable GeneratedKernelEngine.Staging staging) {
        this.spec = spec;
        this.batchSize = batchSize;
        this.staging = staging;
    }

    @Override
    public void open() throws Exception {
        super.open();
        inputFields = spec.inputType().getFieldCount();
        // No key column: nothing here goes to the device except the values, which are their own
        // buffer. -1 means every field is held host-side.
        rows = StagedColumns.allocate(spec.inputType(), -1, batchSize, staging);
        values = allocateDoubles(batchSize);
        totals = allocateDoubles(batchSize);

        outRow = new BinaryRowData(spec.outputType().getFieldCount());
        outWriter = new BinaryRowWriter(outRow);
        outElement = new StreamRecord<>(null);
        metrics = new OffloadMetrics();
        registerMetrics();
        buffered = 0;
        carried = 0.0;
    }

    @Override
    public void processElement(StreamRecord<RowData> element) throws Exception {
        RowData row = element.getValue();
        rows.stage(row, buffered);
        values.set(buffered, row.getDouble(spec.valueField()));
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
        buffered = 0;

        long start = System.nanoTime();
        try {
            scan(count);
        } catch (Throwable t) {
            LOG.warn(
                    "scanning {} rows on the device failed; this batch runs on the host", count, t);
            // Recoverable where the sort's and the join's failures are not: this operator holds
            // the batch it was about to scan and nothing else depends on the device having run,
            // so the host can produce exactly the same numbers and the next batch can try again.
            scanOnHost(count);
        }
        long executeNanos = System.nanoTime() - start;

        long drainStart = System.nanoTime();
        for (int i = 0; i < count; i++) {
            outWriter.reset();
            rows.writeInto(outWriter, 0, i);
            // The carry, applied here rather than on the device: adding a scalar to every element
            // would be a second launch over the same buffer to save an add the drain is doing
            // anyway.
            outWriter.writeDouble(inputFields, carried + totals.get(i));
            outWriter.complete();
            output.collect(outElement.replace(outRow));
        }
        carried += totals.get(count - 1);
        metrics.recordBatch(count, count, 0L, executeNanos, System.nanoTime() - drainStart, null);
    }

    /**
     * The device half, and all of it: one library task, no generated kernel.
     *
     * <p>Built per batch rather than once, because {@code Cudf.runningSum} captures its row count
     * when the graph is built — the same constraint the grouped aggregate and the sort are under. A
     * full batch rebuilds an identical graph; a partition's last batch is short and needs a
     * different one.
     */
    private void scan(int count) throws Exception {
        TaskGraph graph =
                new TaskGraph("over-scan")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, values)
                        .libraryTask("scan", Cudf::runningSum, count, values, totals)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, totals);
        try (TornadoExecutionPlan batch = new TornadoExecutionPlan(graph.snapshot())) {
            batch.execute();
        }
    }

    /** The same numbers, computed here, for a batch the device could not take. */
    private void scanOnHost(int count) {
        double total = 0.0;
        for (int i = 0; i < count; i++) {
            total += values.get(i);
            totals.set(i, total);
        }
    }

    private DoubleArray allocateDoubles(int capacity) {
        if (staging == null) {
            DoubleArray array = new DoubleArray(capacity);
            array.init(0.0);
            return array;
        }
        return (DoubleArray)
                GeneratedKernel.allocateOn(
                        GpuValueType.DOUBLE,
                        capacity,
                        staging.allocate(GeneratedKernel.sizeOf(GpuValueType.DOUBLE, capacity)));
    }

    @Override
    public void close() throws Exception {
        if (metrics != null && metrics.getBatches() > 0) {
            LOG.info(metrics.report("GpuOverAggregateOperator " + spec));
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
        getMetricGroup().gauge("acceleratorDrainNanos", (Gauge<Long>) metrics::getDrainNanos);
    }
}
