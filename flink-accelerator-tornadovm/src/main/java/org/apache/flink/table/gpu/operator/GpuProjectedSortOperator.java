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

import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.writer.BinaryRowWriter;
import org.apache.flink.table.gpu.codegen.GpuCalcSpec;
import org.apache.flink.table.gpu.codegen.GpuSortSpec;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.gather.RowGather;
import org.apache.flink.table.gpu.metrics.OffloadMetrics;
import org.apache.flink.table.types.logical.LogicalType;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * A projection and the sort above it, as one operator over one task graph.
 *
 * <h2>What this exists to remove</h2>
 *
 * <p>Unfused, a {@code Calc} feeding a {@code Sort} is two operators, two task graphs and two round
 * trips: the projection stages its input, runs a kernel, copies the result back and builds a {@code
 * BinaryRowData} for every row; the sort immediately takes those rows apart again, stages them back
 * into device buffers, orders them, and builds a {@code BinaryRowData} for every row a second time.
 *
 * <p>Both of the expensive halves are the host's. §T12 attributed 616 ms of a 954 ms offloaded sort
 * to building four million binary rows, against 15.7 ms for the ordering itself, and §T13 found the
 * same shape in the join. Doing it twice for one logical pipeline is most of what a fused pair
 * saves.
 *
 * <p>Fused, the kernel writes its columns, cuDF orders the key column where it lies, and one drain
 * takes the ordered rows back. One stage in, one drain out, whatever the chain's length.
 *
 * <h2>Why the whole partition is one batch</h2>
 *
 * <p>A projection is per-row work that can be done a batch at a time; a sort cannot emit anything
 * until it has seen everything. So this operator's batch <em>is</em> the partition, and the staging
 * it undertakes to hold is sized from the planner's estimate the same way {@link GpuSortOperator}'s
 * is. That is also why the graph is built at flush rather than at {@code open()}: {@code
 * Cudf.sortedOrder} captures its row count when the graph is built, and the count is not known
 * until the last row has arrived.
 *
 * <h2>When the estimate was wrong</h2>
 *
 * <p>Same contract as {@link GpuSortOperator}: more rows than the staging undertook to hold fails
 * the task rather than growing without bound, and batch failover re-runs it on the operator that
 * knows how to spill. A correct answer from an unbounded buffer is the failure mode that takes a
 * TaskManager down.
 */
public class GpuProjectedSortOperator extends AbstractStreamOperator<RowData>
        implements OneInputStreamOperator<RowData, RowData>, BoundedOneInput {

    private static final long serialVersionUID = 1L;

    private static final Logger LOG = LoggerFactory.getLogger(GpuProjectedSortOperator.class);

    private final GpuCalcSpec spec;
    private final GpuSortSpec sort;
    private final int capacity;
    private final boolean profile;
    private final transient GeneratedKernelEngine.Staging staging;

    private transient GeneratedKernelEngine engine;
    private transient RowGather[] gathers;
    private transient BinaryRowData outRow;
    private transient BinaryRowWriter outWriter;
    private transient StreamRecord<RowData> outElement;
    private transient OffloadMetrics metrics;
    private transient int buffered;

    public GpuProjectedSortOperator(
            GpuCalcSpec spec,
            GpuSortSpec sort,
            int capacity,
            boolean profile,
            GeneratedKernelEngine.Staging staging) {
        this.spec = spec;
        this.sort = sort;
        this.capacity = capacity;
        this.profile = profile;
        this.staging = staging;
    }

    @Override
    public void open() throws Exception {
        super.open();
        if (sort.estimatedRows() > capacity) {
            // The last moment this can be handed to the operator that knows how to spill.
            throw new IllegalStateException(
                    "this fused sort would hold "
                            + sort.estimatedRows()
                            + " rows and the staging offered fits "
                            + capacity
                            + "; declining rather than spilling");
        }
        engine = new GeneratedKernelEngine(spec, null, sort, profile, staging);
        engine.open();
        gathers = new RowGather[spec.kernel().inputFieldIndexes().length];
        int fields = sort.rowType().getFieldCount();
        outRow = new BinaryRowData(fields);
        outWriter = new BinaryRowWriter(outRow);
        outElement = new StreamRecord<>(null);
        metrics = new OffloadMetrics();
        buffered = 0;
    }

    @Override
    public void processElement(StreamRecord<RowData> element) throws Exception {
        RowData row = element.getValue();
        if (gathers[0] == null) {
            // The concrete RowData implementation is not knowable at plan time, so the gather
            // strategy is chosen from the first record actually seen, as everywhere else here.
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
        if (buffered == capacity) {
            throw new StagingCapacityExceededException(
                    "this fused sort was given more rows than it undertook to hold",
                    sort.estimatedRows(),
                    capacity);
        }
        for (RowGather g : gathers) {
            g.accept(row, buffered);
        }
        buffered++;
    }

    @Override
    public void endInput() throws Exception {
        if (buffered == 0) {
            return;
        }
        for (RowGather g : gathers) {
            if (g != null) {
                g.flush();
            }
        }
        final int count = buffered;
        buffered = 0;

        final long start = System.nanoTime();
        final int[] permutation = engine.projectAndOrder(count);
        final long executeNanos = System.nanoTime() - start;

        final long drainStart = System.nanoTime();
        final int fields = sort.rowType().getFieldCount();
        for (int i = 0; i < count; i++) {
            final int source = permutation[i];
            outWriter.reset();
            for (int f = 0; f < fields; f++) {
                write(f, source);
            }
            outWriter.complete();
            output.collect(outElement.replace(outRow));
        }
        metrics.recordBatch(count, count, 0L, executeNanos, System.nanoTime() - drainStart, null);
    }

    /** One field of the projection's output, read at the position the ordering chose. */
    private void write(int field, int position) {
        final Object buffer = engine.outputBuffer(field);
        final LogicalType type = sort.rowType().getTypeAt(field);
        switch (type.getTypeRoot()) {
            case INTEGER:
                outWriter.writeInt(field, ((IntArray) buffer).get(position));
                break;
            case FLOAT:
                outWriter.writeFloat(field, ((FloatArray) buffer).get(position));
                break;
            default:
                outWriter.writeDouble(field, ((DoubleArray) buffer).get(position));
                break;
        }
    }

    @Override
    public void close() throws Exception {
        if (metrics != null && metrics.getBatches() > 0) {
            LOG.info(metrics.report("GpuProjectedSortOperator " + spec));
        }
        if (engine != null) {
            engine.close();
        }
        super.close();
    }
}
