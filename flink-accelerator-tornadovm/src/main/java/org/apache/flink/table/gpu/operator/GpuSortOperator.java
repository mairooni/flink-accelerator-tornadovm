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
import org.apache.flink.table.gpu.codegen.GpuSortSpec;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.metrics.OffloadMetrics;
import org.apache.flink.table.types.logical.LogicalType;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cudf.Cudf;

import javax.annotation.Nullable;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Orders a whole partition by one {@code INT} key: cuDF returns the permutation, the host emits its
 * staged rows in that order.
 *
 * <h2>Only the key crosses the interconnect</h2>
 *
 * <p>{@code Cudf.sortedOrder} takes the key column and hands back row positions. So the transfer is
 * four bytes a row out and four back, whatever the row is worth, and the payload columns sit in
 * host staging from the moment they arrive until they are emitted. That is what lets this sort a
 * row carrying types no kernel could express — a {@code BIGINT}, which the generator refuses
 * outright — because nothing but the key is ever asked to be device-expressible.
 *
 * <h2>Whole-partition, so the sizing decision is made before the first row</h2>
 *
 * <p>Unlike everything else this provider runs, a sort cannot work a batch at a time: a sorted
 * batch is a sorted run, and runs have to be merged. This operator therefore undertakes to hold
 * every row it is given, and the undertaking has to be made at {@code open()} — after that the rows
 * are already arriving and there is nowhere to put them back. {@link #open()} refuses when the
 * staging it was given is smaller than the planner's estimate, which is the last moment {@code
 * FallbackToCpuOperator} can hand the task to {@code SortOperator} instead; {@code SortOperator}
 * spills and this does not, so that is the right answer rather than a defeat.
 *
 * <h2>When the estimate was wrong</h2>
 *
 * <p>An estimate is a statistic and statistics are wrong. If more rows arrive than the staging
 * holds, this stops using the device and finishes on the host: the staged rows are materialised
 * into a list, the overflow joins them, and the ordering is {@code List.sort}. Slower — much slower
 * — and correct, which is the trade a sort has to make, because by then the rows have been consumed
 * and there is no operator left to give them to. It is reported at {@code WARN} and the device
 * gauge goes to zero, so a job doing this is visible rather than merely disappointing.
 *
 * <p>Deliberately not spilling. The heap is the bound on that path, where Flink's own sorter would
 * go to disk. Getting the estimate right is what keeps it rare, and it is why {@code BatchExecSort}
 * was made to carry one.
 *
 * <h2>Stability</h2>
 *
 * <p>Required, not a nicety: SQL's {@code ORDER BY} is stable over equal keys and a query that
 * sorts twice depends on it. The device half is {@code cudf::stable_sorted_order} and the host
 * fallback is {@code List.sort}, which is a merge sort. Both halves are stable and neither is
 * stable by accident.
 */
public class GpuSortOperator extends AbstractStreamOperator<RowData>
        implements OneInputStreamOperator<RowData, RowData>, BoundedOneInput {

    private static final long serialVersionUID = 1L;

    private final GpuSortSpec spec;
    private final int capacity;
    private final transient GeneratedKernelEngine.Staging staging;

    /** The key column cuDF reads, and the permutation it writes back. */
    private transient IntArray keys;

    private transient IntArray order;

    /** One buffer per field, at the field's declared width; null at the key's own field. */
    private transient ByteBuffer[] columns;

    /** A bit a column a row, or null when no field of this row is nullable. */
    private transient @Nullable ByteBuffer validity;

    private transient int[] widths;
    private transient LogicalType[] types;
    private transient int count;

    /** Non-null once this has stopped using the device. Holds every row, staged ones included. */
    private transient @Nullable List<RowData> onHost;

    private transient BinaryRowData outRow;
    private transient BinaryRowWriter outWriter;
    private transient StreamRecord<RowData> outElement;
    private transient OffloadMetrics metrics;

    public GpuSortOperator(
            GpuSortSpec spec, int capacity, @Nullable GeneratedKernelEngine.Staging staging) {
        this.spec = spec;
        this.capacity = capacity;
        this.staging = staging;
    }

    @Override
    public void open() throws Exception {
        super.open();
        if (spec.estimatedRows() > capacity) {
            // The one refusal that has to happen here. See the class comment: this is the last
            // moment the task can be handed to the operator that knows how to spill.
            throw new IllegalStateException(
                    "this sort would hold "
                            + spec.estimatedRows()
                            + " rows and the staging offered fits "
                            + capacity
                            + "; declining rather than spilling");
        }
        if ((long) capacity * Long.BYTES > Integer.MAX_VALUE - 64) {
            // Every buffer here is indexed by an int, so a capacity this large would overflow the
            // offset arithmetic rather than fail to allocate. Refused where open()'s refusals are
            // recoverable, not discovered later as a corrupt read.
            throw new IllegalStateException(
                    "a staging capacity of " + capacity + " rows overflows an int-indexed buffer");
        }
        int fields = spec.rowType().getFieldCount();
        widths = new int[fields];
        types = new LogicalType[fields];
        columns = new ByteBuffer[fields];
        boolean anyNullable = false;
        for (int i = 0; i < fields; i++) {
            types[i] = spec.rowType().getTypeAt(i);
            widths[i] = GpuSortSpec.widthOf(types[i]);
            anyNullable |= types[i].isNullable();
            if (i != spec.sortField()) {
                columns[i] = buffer(widths[i] * capacity);
            }
        }
        keys = allocateInts();
        order = allocateInts();
        validity = anyNullable ? buffer(4 * capacity) : null;

        outRow = new BinaryRowData(fields);
        outWriter = new BinaryRowWriter(outRow);
        outElement = new StreamRecord<>(null);
        metrics = new OffloadMetrics();
        registerMetrics();
        count = 0;
        onHost = null;
    }

    @Override
    public void processElement(StreamRecord<RowData> element) {
        RowData row = element.getValue();
        if (onHost != null) {
            // Copied rather than retained, as everywhere else here: batch mode reuses the instance.
            onHost.add(materialise(row));
            return;
        }
        if (count == capacity) {
            degrade(
                    "more rows arrived than the planner estimated ("
                            + spec.estimatedRows()
                            + "), and the staging holds "
                            + capacity);
            onHost.add(materialise(row));
            return;
        }
        stage(row, count++);
    }

    @Override
    public void endInput() throws Exception {
        if (onHost == null) {
            int n = count;
            if (n == 0) {
                return;
            }
            long start = System.nanoTime();
            int[] permutation;
            try {
                permutation = orderOnDevice(n);
            } catch (Throwable t) {
                LOG.warn(
                        "ordering {} rows on the device failed; this subtask finishes on the host",
                        n,
                        t);
                degrade("the device failed to order the partition");
                sortOnHost();
                return;
            }
            long executeNanos = System.nanoTime() - start;
            long drainStart = System.nanoTime();
            for (int i = 0; i < n; i++) {
                writeStaged(outWriter, permutation[i]);
                output.collect(outElement.replace(outRow));
            }
            count = 0;
            // Gather is reported as zero rather than estimated. Staging here is one write per
            // field on the arrival path and timing it would cost a clock read a row, on the one
            // operator where the row count is the whole partition; the two figures that do get
            // measured -- ordering and draining -- are the two this operator can act on.
            metrics.recordBatch(n, n, 0L, executeNanos, System.nanoTime() - drainStart, null);
            return;
        }
        sortOnHost();
    }

    /**
     * The device half, and all of it: one library task, no generated kernel.
     *
     * <p>The row count is captured when the graph is built rather than read from a buffer, which is
     * why the graph is built here and not in {@code open()} — the same constraint the grouped
     * aggregate's cuDF stage is under. A sort executes once per partition, so there is nothing to
     * amortise a reusable plan over.
     */
    private int[] orderOnDevice(int n) throws Exception {
        TaskGraph graph =
                new TaskGraph("sort")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, keys)
                        // Nulls first is stated as 0 and could be either: the provider accepts only
                        // a NOT NULL key, because the shim builds a column with no null mask and a
                        // null key would silently order as whatever its bits happen to be.
                        .libraryTask("order", Cudf::sortedOrder, n, keys, 0, order)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, order);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.execute();
        }
        int[] permutation = new int[n];
        for (int i = 0; i < n; i++) {
            permutation[i] = order.get(i);
        }
        return permutation;
    }

    /** Stops using the device, keeping every row that has arrived so far. */
    private void degrade(String why) {
        LOG.warn("{}; this sort finishes on the host, which does not spill", why);
        List<RowData> rows = new ArrayList<>(count + 1024);
        for (int i = 0; i < count; i++) {
            BinaryRowData copy = new BinaryRowData(widths.length);
            writeStaged(new BinaryRowWriter(copy), i);
            rows.add(copy);
        }
        count = 0;
        onHost = rows;
    }

    private void sortOnHost() {
        // TimSort, so stable, which ORDER BY requires over equal keys.
        onHost.sort(Comparator.comparingInt(row -> row.getInt(spec.sortField())));
        for (RowData row : onHost) {
            output.collect(outElement.replace(row));
        }
        onHost.clear();
    }

    // ------------------------------------------------------------------------------------------
    // Staging
    // ------------------------------------------------------------------------------------------

    private void stage(RowData row, int position) {
        if (validity != null) {
            int bits = 0;
            for (int f = 0; f < widths.length; f++) {
                if (row.isNullAt(f)) {
                    bits |= 1 << f;
                }
            }
            validity.putInt(position * 4, bits);
        }
        for (int f = 0; f < widths.length; f++) {
            if (row.isNullAt(f)) {
                continue;
            }
            if (f == spec.sortField()) {
                keys.set(position, row.getInt(f));
                continue;
            }
            int at = position * widths[f];
            switch (types[f].getTypeRoot()) {
                case INTEGER:
                    columns[f].putInt(at, row.getInt(f));
                    break;
                case BIGINT:
                    columns[f].putLong(at, row.getLong(f));
                    break;
                case FLOAT:
                    columns[f].putFloat(at, row.getFloat(f));
                    break;
                default:
                    columns[f].putDouble(at, row.getDouble(f));
                    break;
            }
        }
    }

    /**
     * Writes the staged row at {@code position} into a binary row.
     *
     * <p>{@link BinaryRowData} rather than {@code GenericRowData} because every field here is a
     * primitive and a {@code GenericRowData} would box all of them. Sorting is the one operator
     * where that is not a rounding error: it is the whole partition, once, and the only work this
     * operator does on the host.
     */
    private void writeStaged(BinaryRowWriter writer, int position) {
        writer.reset();
        int bits = validity == null ? 0 : validity.getInt(position * 4);
        for (int f = 0; f < widths.length; f++) {
            if ((bits & (1 << f)) != 0) {
                writer.setNullAt(f);
                continue;
            }
            if (f == spec.sortField()) {
                writer.writeInt(f, keys.get(position));
                continue;
            }
            int at = position * widths[f];
            switch (types[f].getTypeRoot()) {
                case INTEGER:
                    writer.writeInt(f, columns[f].getInt(at));
                    break;
                case BIGINT:
                    writer.writeLong(f, columns[f].getLong(at));
                    break;
                case FLOAT:
                    writer.writeFloat(f, columns[f].getFloat(at));
                    break;
                default:
                    writer.writeDouble(f, columns[f].getDouble(at));
                    break;
            }
        }
        writer.complete();
    }

    /** A live row copied out of whatever the upstream operator reuses. */
    private BinaryRowData materialise(RowData row) {
        BinaryRowData target = new BinaryRowData(widths.length);
        BinaryRowWriter writer = new BinaryRowWriter(target);
        writer.reset();
        for (int f = 0; f < widths.length; f++) {
            if (row.isNullAt(f)) {
                writer.setNullAt(f);
                continue;
            }
            switch (types[f].getTypeRoot()) {
                case INTEGER:
                    writer.writeInt(f, row.getInt(f));
                    break;
                case BIGINT:
                    writer.writeLong(f, row.getLong(f));
                    break;
                case FLOAT:
                    writer.writeFloat(f, row.getFloat(f));
                    break;
                default:
                    writer.writeDouble(f, row.getDouble(f));
                    break;
            }
        }
        writer.complete();
        return target;
    }

    private ByteBuffer buffer(int bytes) {
        // A heap buffer where there is no arena, rather than a direct one: a direct buffer counts
        // against MaxDirectMemorySize, which is the accounting M3.1 exists to stop depending on.
        return staging == null ? ByteBuffer.allocate(bytes) : staging.allocate(bytes);
    }

    private IntArray allocateInts() {
        if (staging == null) {
            return (IntArray) GeneratedKernel.allocate(GpuValueType.INT, capacity);
        }
        return (IntArray)
                GeneratedKernel.allocateOn(
                        GpuValueType.INT,
                        capacity,
                        staging.allocate(GeneratedKernel.sizeOf(GpuValueType.INT, capacity)));
    }

    @Override
    public void close() throws Exception {
        if (metrics != null && metrics.getBatches() > 0) {
            LOG.info(metrics.report("GpuSortOperator " + spec));
        }
        super.close();
    }

    /** The engine's numbers are not available here — there is no kernel — so these are its own. */
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
