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

package org.apache.flink.table.gpu.device;

import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.util.OneInputStreamOperatorTestHarness;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelOverAggregate;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.operator.GpuOverAggregateOperator;
import org.apache.flink.table.runtime.accelerator.AcceleratorContext;
import org.apache.flink.table.runtime.accelerator.AcceleratorPlan;
import org.apache.flink.table.runtime.typeutils.RowDataSerializer;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;

/**
 * A running total scanned by {@code Cudf.runningSum}, checked across batch boundaries.
 *
 * <p>The batch size is far below the row count on purpose, because the boundary is where this
 * operator can be wrong while looking right. A scan that started each batch from zero returns the
 * right number of rows, monotonic within every batch, correct in the first — and wrong from the
 * second onwards. Nothing but a row-for-row comparison against a host-computed prefix sum catches
 * it, so that is the assertion.
 *
 * <p>Skips without a device, and separately without the cuDF shim. See {@link
 * DeviceAssumptions#requireCudf()}.
 */
class AcceleratedOverAggregateDeviceIT {

    private static final int ROWS = 40_000;

    /** Nine and a bit batches, and the last one short. */
    private static final int BATCH = 4_096;

    private static final LogicalType INT = new IntType(false);
    private static final LogicalType BIGINT = new BigIntType(false);
    private static final LogicalType DOUBLE = new DoubleType(false);

    /** (ordinal, tag, value). The BIGINT is a column no kernel could compute with. */
    private static final RowType INPUT = RowType.of(INT, BIGINT, DOUBLE);

    /** The input row with the running total appended. */
    private static final RowType OUTPUT = RowType.of(INT, BIGINT, DOUBLE, DOUBLE);

    private static final AcceleratorContext CONTEXT =
            new AcceleratorContext() {
                @Override
                public RowType outputType() {
                    return OUTPUT;
                }

                @Override
                public int maxBatchSize() {
                    return BATCH;
                }

                @Override
                public ClassLoader userCodeClassLoader() {
                    return AcceleratedOverAggregateDeviceIT.class.getClassLoader();
                }

                @Override
                public boolean providesOffHeap() {
                    return false;
                }

                @Override
                public ByteBuffer allocateOffHeap(int bytes) {
                    throw new IllegalStateException("no managed memory in this harness");
                }
            };

    @BeforeEach
    void requireCudf() {
        DeviceAssumptions.requireCudf();
    }

    @Test
    @DisplayName("the running total is right across every batch boundary")
    void theTotalCrossesBatches() throws Exception {
        List<RowData> out = run();

        assertThat(out).hasSize(ROWS);
        double expected = 0.0;
        for (int i = 0; i < ROWS; i++) {
            expected += value(i);
            RowData row = out.get(i);
            assertThat(row.getInt(0)).as("row " + i + " is out of order").isEqualTo(i);
            assertThat(row.getLong(1)).isEqualTo(1000L + i);
            // A tolerance, and only for the reason §T9 records: the device reassociates the sum
            // within a batch while the host adds in row order. Every input is exact; the order of
            // the additions is not the same one.
            assertThat(row.getDouble(3))
                    .as("running total at row " + i)
                    .isCloseTo(expected, offset(1e-6));
        }
    }

    @Test
    @DisplayName("the payload rides through untouched")
    void thePayloadIsCarried() throws Exception {
        List<RowData> out = run();
        for (int i = 0; i < ROWS; i++) {
            // Nothing but the value column reaches the device; a BIGINT the kernel generator would
            // refuse outright must come back exactly as it went in.
            assertThat(out.get(i).getLong(1)).isEqualTo(1000L + i);
            assertThat(out.get(i).getDouble(2)).isEqualTo(value(i));
        }
    }

    @Test
    @DisplayName("a nullable value is declined rather than summed as a zero")
    void aNullableValueIsDeclined() {
        RowType nullable = RowType.of(INT, BIGINT, new DoubleType(true));
        AccelOverAggregate over =
                new AccelOverAggregate(
                        2,
                        new AccelInput(nullable),
                        RowType.of(INT, BIGINT, new DoubleType(true), DOUBLE));
        assertThat(DeviceAssumptions.provider().accept(over, work())).isEmpty();
    }

    private static List<RowData> run() throws Exception {
        AccelOverAggregate over = new AccelOverAggregate(2, new AccelInput(INPUT), OUTPUT);
        Optional<AcceleratorPlan> offered = DeviceAssumptions.provider().accept(over, work());
        assertThat(offered).as("the provider declined a shape it is meant to serve").isPresent();

        StreamOperatorFactory<RowData> factory =
                DeviceAssumptions.provider().createOperator(offered.get(), CONTEXT);

        List<RowData> out = new ArrayList<>();
        try (OneInputStreamOperatorTestHarness<RowData, RowData> harness =
                new OneInputStreamOperatorTestHarness<>(factory, 1, 1, 0)) {
            harness.setup(new RowDataSerializer(OUTPUT));
            harness.open();
            for (int i = 0; i < ROWS; i++) {
                GenericRowData row = new GenericRowData(3);
                row.setField(0, i);
                row.setField(1, 1000L + i);
                row.setField(2, value(i));
                harness.processElement(new StreamRecord<>(row));
            }
            GpuOverAggregateOperator operator =
                    (GpuOverAggregateOperator) harness.getOneInputOperator();
            operator.endInput();
            assertThat(operator.metrics().getBatches())
                    .as("the input was not scanned on the device")
                    .isEqualTo(ROWS / BATCH + 1);

            for (Object o : harness.getOutput()) {
                @SuppressWarnings("unchecked")
                RowData emitted = ((StreamRecord<RowData>) o).getValue();
                out.add(emitted);
            }
        }
        return out;
    }

    /** Small and varied, so a lost carry shows as a large error rather than a rounding one. */
    private static double value(int i) {
        return 1.0 + (i % 7) * 0.25;
    }

    private static AccelWorkProfile work() {
        int[] ops = new int[AccelFunction.values().length];
        ops[AccelFunction.PLUS.ordinal()] = 1;
        return new AccelWorkProfile(ops, 20, 28, ROWS);
    }
}
