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
import org.apache.flink.table.gpu.codegen.GpuGramSpec;
import org.apache.flink.table.gpu.codegen.GpuKernelSource;
import org.apache.flink.table.gpu.codegen.GpuValueType;

/**
 * Computes a Gram matrix on the device and emits it as one row of upper-triangle sums.
 *
 * <h2>One row, at end of input</h2>
 *
 * <p>This serves the <em>local</em> half of an ungrouped aggregate, so what it emits is a partial
 * result that Flink's global half sums. It could therefore emit one row per batch, as the grouped
 * aggregate does, and the answer would be the same; it emits one row per subtask instead because
 * the accumulation is a {@code d x d} array of doubles rather than device state, and folding each
 * batch into it costs nothing worth saving.
 *
 * <h2>What crosses the interconnect</h2>
 *
 * <p>Per batch: the input columns in, and {@code d x d} doubles back. Not {@code d(d+1)/2} per
 * <em>row</em> — which is what the query asks for arithmetically and what the CPU plan materialises
 * — and that difference is the whole of §T15. At sixty-four features the projection Flink would
 * otherwise compute is 2,080 columns a row; this moves sixty-four.
 */
public class GpuGramOperator extends AbstractStreamOperator<RowData>
        implements OneInputStreamOperator<RowData, RowData>, BoundedOneInput {

    private static final long serialVersionUID = 1L;

    private final GpuGramSpec spec;
    private final GpuKernelSource kernel;
    private final int batchSize;

    private transient GpuGramEngine engine;
    private transient int[] inputFields;
    private transient GpuValueType[] inputTypes;
    private transient int buffered;
    private transient long executeNanos;

    public GpuGramOperator(GpuGramSpec spec, GpuKernelSource kernel, int batchSize) {
        this.spec = spec;
        this.kernel = kernel;
        this.batchSize = batchSize;
    }

    @Override
    public void open() throws Exception {
        super.open();
        engine = new GpuGramEngine(kernel, spec.featureCount(), batchSize);
        engine.open();
        inputFields = kernel.inputFieldIndexes();
        inputTypes = kernel.inputTypes();
        registerMetrics();
        buffered = 0;
        executeNanos = 0L;
    }

    @Override
    public void processElement(StreamRecord<RowData> element) throws Exception {
        RowData row = element.getValue();
        for (int c = 0; c < inputFields.length; c++) {
            engine.stage(c, buffered, read(row, inputFields[c], inputTypes[c]));
        }
        if (++buffered == batchSize) {
            flush();
        }
    }

    @Override
    public void endInput() throws Exception {
        flush();
        emit();
    }

    private void flush() throws Exception {
        if (buffered == 0) {
            return;
        }
        int count = buffered;
        buffered = 0;
        long start = System.nanoTime();
        engine.execute(count);
        executeNanos += System.nanoTime() - start;
    }

    /**
     * The upper triangle, row-major, as one row.
     *
     * <p>The order has to be the one {@link GpuGramSpec} recognised and not merely <em>an</em>
     * order: the sums downstream are positional, so a transposed read produces a complete Gram
     * matrix with entries in the wrong cells and no count that could notice.
     */
    private void emit() {
        int d = spec.featureCount();
        GenericRowData out = new GenericRowData(spec.outputType().getFieldCount());
        int at = 0;
        for (int i = 0; i < d; i++) {
            for (int j = i; j < d; j++) {
                out.setField(at++, engine.entry(i, j));
            }
        }
        output.collect(new StreamRecord<>(out));
        LOG.info(
                "GpuGramOperator {}: {} rows in {} batches, {} ms on the device, {} sums out",
                spec,
                engine.rowsSeen(),
                engine.batchCount(),
                executeNanos / 1_000_000,
                at);
    }

    private static double read(RowData row, int field, GpuValueType type) {
        switch (type) {
            case INT:
                return row.getInt(field);
            case FLOAT:
                return row.getFloat(field);
            default:
                return row.getDouble(field);
        }
    }

    @Override
    public void close() throws Exception {
        if (engine != null) {
            engine.close();
            engine = null;
        }
        super.close();
    }

    /** Batches and rows this operator has run, for a test that needs a device to have run. */
    public long batchCount() {
        return engine == null ? 0 : engine.batchCount();
    }

    public long rowsSeen() {
        return engine == null ? 0 : engine.rowsSeen();
    }

    private void registerMetrics() {
        getMetricGroup().gauge("acceleratorBatches", (Gauge<Long>) this::batchCount);
        getMetricGroup().gauge("acceleratorRowsIn", (Gauge<Long>) this::rowsSeen);
        getMetricGroup().gauge("acceleratorKernelNanos", (Gauge<Long>) () -> executeNanos);
    }
}
