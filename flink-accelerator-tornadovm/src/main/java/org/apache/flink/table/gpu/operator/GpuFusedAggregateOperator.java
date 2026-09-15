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
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.codegen.GpuAggregateSpec;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.gather.RowGather;
import org.apache.flink.table.types.logical.LogicalTypeRoot;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Runs a {@code Calc} and the ungrouped {@code SUM}s above it on a device, as one operator.
 *
 * <p>Emits nothing until the input ends, which is what an ungrouped aggregate does anyway: one row,
 * carrying one accumulated value per output field. Between those it stages rows into device
 * buffers, and each full batch is a projection kernel followed by the contractions that read the
 * buffers it wrote.
 *
 * <h2>Null and the empty input</h2>
 *
 * <p>{@code SUM} over no rows is {@code NULL}, not zero, and a filter in the fused {@code Calc}
 * makes that reachable from a non-empty table. The operator therefore counts rows that survived the
 * condition and emits nulls when none did, so the answer matches the plan Flink would otherwise
 * have run rather than differing on the one input nobody tests.
 */
public class GpuFusedAggregateOperator extends AbstractStreamOperator<RowData>
        implements OneInputStreamOperator<RowData, RowData>, BoundedOneInput {

    private static final long serialVersionUID = 1L;

    private static final Logger LOG = LoggerFactory.getLogger(GpuFusedAggregateOperator.class);

    private final GpuAggregateSpec spec;

    private transient FusedAggregateEngine engine;
    private transient RowGather[] gathers;

    public GpuFusedAggregateOperator(GpuAggregateSpec spec) {
        this.spec = spec;
    }

    @Override
    public void open() throws Exception {
        super.open();
        engine = new FusedAggregateEngine(spec);
        engine.open();
        gathers = new RowGather[spec.kernel().inputFieldIndexes().length];
    }

    @Override
    public void processElement(StreamRecord<RowData> element) throws Exception {
        RowData row = element.getValue();
        if (gathers[0] == null) {
            // The concrete RowData implementation is not knowable at plan time, so the gather
            // strategy is chosen from the first record actually seen -- as in the Calc operator.
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
        // The position, not zero. A gather writes by index into the staging buffer, so a constant
        // here stacks every row onto the first slot and leaves the rest of the batch at zero.
        int position = engine.position();
        for (RowGather gather : gathers) {
            gather.accept(row, position);
        }
        engine.rowComplete();
    }

    /** One row, once, carrying every accumulated value. */
    @Override
    public void endInput() throws Exception {
        engine.flush();

        int[] sources = spec.sumSources();
        GenericRowData out = new GenericRowData(sources.length);
        int contracted = 0;
        for (int field = 0; field < sources.length; field++) {
            if (sources[field] == GpuAggregateSpec.COUNT_STAR) {
                out.setField(field, engine.contributingRows());
            } else if (engine.contributingRows() == 0) {
                // SUM over nothing is NULL. Reached whenever a fused filter rejects every row.
                out.setField(field, null);
            } else {
                out.setField(field, boxed(field, engine.total(contracted)));
            }
            if (sources[field] != GpuAggregateSpec.COUNT_STAR) {
                contracted++;
            }
        }

        LOG.info(
                "fused aggregate: {} rows staged, {} contributed, {} contracted columns",
                engine.stagedRows(),
                engine.contributingRows(),
                contracted);
        output.collect(new StreamRecord<>(out));
    }

    /** The accumulator as the partial aggregate's row type declares it. */
    private Object boxed(int field, double value) {
        LogicalTypeRoot root = spec.outputType().getChildren().get(field).getTypeRoot();
        switch (root) {
            case FLOAT:
                return (float) value;
            case INTEGER:
                return (int) value;
            case BIGINT:
                return (long) value;
            default:
                return value;
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
}
