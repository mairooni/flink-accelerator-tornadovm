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
import org.apache.flink.table.gpu.codegen.GpuCalcSpec;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.gather.RowGather;
import org.apache.flink.table.gpu.metrics.OffloadMetrics;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;

import java.util.List;

/**
 * Executes an offloaded Calc: buffers rows, runs one kernel over the batch, emits the survivors.
 *
 * <p>Batching is not a tuning choice. TornadoVM needs a sized, contiguous buffer, so there is no
 * offload without accumulating N records first — which is why this operator exists at all rather
 * than a per-record one.
 *
 * <p><b>Output is deferred by up to one batch.</b> Records are emitted when the buffer fills, and
 * the remainder on {@link #endInput()}. That is correct for bounded batch execution; it is one of
 * the reasons streaming is out of scope.
 *
 * <p><b>Input rows are never retained.</b> The Table planner force-enables object reuse in batch
 * mode, so the {@code RowData} handed to {@link #processElement} is very often the same instance
 * every time with different contents. Every field this operator needs is copied into a staging
 * array on arrival.
 */
public class GpuCalcOperator extends AbstractStreamOperator<RowData>
        implements OneInputStreamOperator<RowData, RowData>, BoundedOneInput {

    private static final long serialVersionUID = 1L;

    private final GpuCalcSpec spec;

    private final transient GeneratedKernelEngine.Staging staging;
    private final int batchSize;
    private final boolean profile;

    private transient GeneratedKernelEngine engine;
    private transient RowGather[] gathers;
    private transient PassThroughBuffer[] passThrough;
    private transient GenericRowData outRow;
    private transient StreamRecord<RowData> outElement;
    private transient int buffered;

    public GpuCalcOperator(GpuCalcSpec spec, int batchSize, boolean profile) {
        this(spec, batchSize, profile, null);
    }

    /**
     * @param staging where staging buffers come from, or null to allocate privately. See {@link
     *     GeneratedKernelEngine.Staging}.
     */
    public GpuCalcOperator(
            GpuCalcSpec spec,
            int batchSize,
            boolean profile,
            GeneratedKernelEngine.Staging staging) {
        this.spec = spec;
        this.batchSize = batchSize;
        this.profile = profile;
        this.staging = staging;
        // Emitting downstream from inside processElement is the normal chained path; no timers or
        // state are used, so the default chaining strategy is fine.
    }

    @Override
    public void open() throws Exception {
        super.open();
        engine = new GeneratedKernelEngine(spec, profile, staging);
        engine.open();
        registerMetrics();
        gathers = new RowGather[spec.kernel().inputFieldIndexes().length];

        int[] layout = spec.outputLayout();
        passThrough = new PassThroughBuffer[layout.length];
        List<LogicalType> fields = spec.outputType().getChildren();
        for (int i = 0; i < layout.length; i++) {
            if (layout[i] != GpuCalcSpec.COMPUTED) {
                passThrough[i] = PassThroughBuffer.create(fields.get(i), layout[i], batchSize);
            }
        }

        outRow = new GenericRowData(layout.length);
        outElement = new StreamRecord<>(null);
        buffered = 0;
    }

    @Override
    public void processElement(StreamRecord<RowData> element) throws Exception {
        RowData row = element.getValue();
        if (gathers[0] == null) {
            // The concrete RowData implementation is not knowable at plan time -- the planner
            // declares only InternalTypeInfo<RowData> and the connector picks the class -- so the
            // gather strategy is chosen from the first record actually seen.
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
        if (engine.carriesValidity()) {
            // Cleared first, because the buffer is reused across batches and a bit left set from a
            // previous row would make a present value read as absent.
            engine.clearInputNulls(buffered);
            int[] fields = spec.kernel().inputFieldIndexes();
            for (int c = 0; c < fields.length; c++) {
                if (row.isNullAt(fields[c])) {
                    engine.setInputNull(c, buffered);
                    // The gather still writes a value, and it has to be a benign one: the kernel
                    // computes every row whatever its validity -- branching to skip would cost
                    // divergence and buy nothing -- so an absent slot must not hold something that
                    // turns into an INF, a NAN or a denormal stall on the way through.
                    engine.inputColumn(c).set(buffered, 1.0);
                    continue;
                }
                gathers[c].accept(row, buffered);
            }
        } else {
            for (RowGather g : gathers) {
                g.accept(row, buffered);
            }
        }
        for (PassThroughBuffer buffer : passThrough) {
            if (buffer != null) {
                buffer.accept(row, buffered);
            }
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
        // Reset before emitting: output.collect runs the rest of the chain, which for a
        // self-referential plan could re-enter this operator.
        buffered = 0;

        GeneratedKernelEngine.Execution execution = engine.execute(count);

        long drainStart = System.nanoTime();
        int emitted = 0;
        int[] layout = spec.outputLayout();
        for (int i = 0; i < count; i++) {
            if (!engine.selected(i)) {
                continue;
            }
            int computed = 0;
            for (int field = 0; field < layout.length; field++) {
                if (layout[field] == GpuCalcSpec.COMPUTED) {
                    int column = computed++;
                    outRow.setField(
                            field,
                            engine.outputIsNull(column, i) ? null : engine.output(column, i));
                } else {
                    passThrough[field].writeInto(outRow, field, i);
                }
            }
            output.collect(outElement.replace(outRow));
            emitted++;
        }
        engine.recordBatch(count, 0, execution, emitted, System.nanoTime() - drainStart);
    }

    @Override
    public void close() throws Exception {
        if (engine != null) {
            OffloadMetrics metrics = engine.metrics();
            if (metrics.getBatches() > 0) {
                LOG.info(metrics.report("GpuCalcOperator " + spec));
            }
            engine.close();
            engine = null;
        }
        super.close();
    }

    /**
     * Stages one pass-through column without boxing.
     *
     * <p>Only fixed-width numeric types are handled, which is all the matcher admits. Anything else
     * would have to be either boxed or copied through a serializer, and both are expensive enough
     * that a Calc needing one is better left on the CPU.
     */
    /**
     * One column carried past the kernel, staged host-side at its declared width.
     *
     * <h2>Validity</h2>
     *
     * <p>A {@code long[]} has no null; it has a zero. The first version of this class wrote {@code
     * values[position] = row.getLong(field)} with no check, so a NULL arrived downstream as 0 —
     * silently, in a path whose whole premise is that the user cannot tell a device was used, and
     * for a column the device never even touches. Nothing about the query said so and no test could
     * see it, because the only tests that existed ran against a host-side provider that evaluates
     * with object semantics and gets nulls right for free.
     *
     * <p>So each buffer carries a validity bit alongside its values. Not a bitmap: a {@code
     * boolean[]} costs a byte a row against the eight a {@code long} already costs, and the staging
     * is not where this operator spends its time — the measured breakdown puts 90% in gather and
     * drain, which this does not change the shape of.
     *
     * <p>This is a pass-through concern only, and deliberately. Nullable operands of a
     * <em>call</em> are refused by the planner and should stay refused: a kernel has no null to
     * compute with. A column merely carried past it needs no null semantics at all, only somewhere
     * to record that it was one.
     */
    private abstract static class PassThroughBuffer {

        final int inputField;

        /** Whether the row at this position had a value. Parallel to the value array. */
        final boolean[] present;

        PassThroughBuffer(int inputField, int capacity) {
            this.inputField = inputField;
            this.present = new boolean[capacity];
        }

        /** Stages one row, recording a null as an absence rather than as a zero. */
        final void accept(RowData row, int position) {
            if (row.isNullAt(inputField)) {
                present[position] = false;
                return;
            }
            present[position] = true;
            read(row, position);
        }

        final void writeInto(GenericRowData out, int field, int position) {
            if (!present[position]) {
                out.setField(field, null);
                return;
            }
            write(out, field, position);
        }

        abstract void read(RowData row, int position);

        abstract void write(GenericRowData out, int field, int position);

        static PassThroughBuffer create(LogicalType type, int inputField, int capacity) {
            LogicalTypeRoot root = type.getTypeRoot();
            if (root == LogicalTypeRoot.BIGINT) {
                long[] values = new long[capacity];
                return new PassThroughBuffer(inputField, capacity) {
                    @Override
                    void read(RowData row, int position) {
                        values[position] = row.getLong(inputField);
                    }

                    @Override
                    void write(GenericRowData out, int field, int position) {
                        out.setField(field, values[position]);
                    }
                };
            }
            if (root == LogicalTypeRoot.INTEGER) {
                int[] values = new int[capacity];
                return new PassThroughBuffer(inputField, capacity) {
                    @Override
                    void read(RowData row, int position) {
                        values[position] = row.getInt(inputField);
                    }

                    @Override
                    void write(GenericRowData out, int field, int position) {
                        out.setField(field, values[position]);
                    }
                };
            }
            if (root == LogicalTypeRoot.FLOAT) {
                float[] values = new float[capacity];
                return new PassThroughBuffer(inputField, capacity) {
                    @Override
                    void read(RowData row, int position) {
                        values[position] = row.getFloat(inputField);
                    }

                    @Override
                    void write(GenericRowData out, int field, int position) {
                        out.setField(field, values[position]);
                    }
                };
            }
            if (root == LogicalTypeRoot.DOUBLE) {
                double[] values = new double[capacity];
                return new PassThroughBuffer(inputField, capacity) {
                    @Override
                    void read(RowData row, int position) {
                        values[position] = row.getDouble(inputField);
                    }

                    @Override
                    void write(GenericRowData out, int field, int position) {
                        out.setField(field, values[position]);
                    }
                };
            }
            throw new UnsupportedOperationException(
                    "no pass-through buffer for "
                            + type
                            + "; the matcher should have refused this "
                            + "Calc before the operator was built");
        }
    }

    /**
     * Publishes where the time actually goes, per subtask.
     *
     * <p>Flink publishes whether a device served this subtask (M3.2); this is the breakdown behind
     * that answer, and it is the provider's to give because only the provider knows what the parts
     * are. The measured shape is the reason it is worth having: on the first hardware this ran on,
     * the kernel was 0.4% of the time and host-side gather and drain were 90%, which is the
     * opposite of what the design assumed and would not have been visible without it.
     *
     * <p>Gauges rather than counters: they are cumulative totals read from the engine, not events
     * to be summed by a reporter.
     */
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

    /** Metrics for the batches this operator has run so far; exposed for tests. */
    public OffloadMetrics metrics() {
        return engine.metrics();
    }
}
