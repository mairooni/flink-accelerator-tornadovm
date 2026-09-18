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
import org.apache.flink.table.gpu.metrics.OffloadMetrics;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cudf.Cudf;

import javax.annotation.Nullable;

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

    /** The rows held, key column included; {@link StagedColumns} is shared with the join. */
    private transient StagedColumns rows;

    /** The permutation cuDF writes back. */
    private transient IntArray order;

    private transient int fields;
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
        fields = spec.rowType().getFieldCount();
        rows = StagedColumns.allocate(spec.rowType(), spec.sortField(), capacity, staging);
        order = StagedColumns.ints(capacity, staging);

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
            onHost.add(copyOf(row));
            return;
        }
        if (count == capacity) {
            degrade(
                    "more rows arrived than the planner estimated ("
                            + spec.estimatedRows()
                            + "), and the staging holds "
                            + capacity);
            onHost.add(copyOf(row));
            return;
        }
        rows.stage(row, count++);
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
                outWriter.reset();
                rows.writeInto(outWriter, 0, permutation[i]);
                outWriter.complete();
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
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, rows.keys())
                        // Nulls first is stated as 0 and could be either: the provider accepts only
                        // a NOT NULL key, because the shim builds a column with no null mask and a
                        // null key would silently order as whatever its bits happen to be.
                        .libraryTask("order", Cudf::sortedOrder, n, rows.keys(), 0, order)
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
        List<RowData> copies = new ArrayList<>(count + 1024);
        for (int i = 0; i < count; i++) {
            BinaryRowData copy = new BinaryRowData(fields);
            BinaryRowWriter writer = new BinaryRowWriter(copy);
            writer.reset();
            rows.writeInto(writer, 0, i);
            writer.complete();
            copies.add(copy);
        }
        count = 0;
        onHost = copies;
    }

    private void sortOnHost() {
        // TimSort, so stable, which ORDER BY requires over equal keys.
        onHost.sort(Comparator.comparingInt(row -> row.getInt(spec.sortField())));
        for (RowData row : onHost) {
            output.collect(outElement.replace(row));
        }
        onHost.clear();
    }

    /** A live row copied out of whatever the upstream operator reuses. */
    private BinaryRowData copyOf(RowData row) {
        BinaryRowData target = new BinaryRowData(fields);
        BinaryRowWriter writer = new BinaryRowWriter(target);
        writer.reset();
        StagedColumns.materialise(row, spec.rowType(), writer, 0);
        writer.complete();
        return target;
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
