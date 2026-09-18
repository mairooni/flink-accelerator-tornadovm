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
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelSort;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.operator.GpuSortOperator;
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

/**
 * A whole partition ordered by {@code Cudf.sortedOrder}, checked against the host's own ordering.
 *
 * <p>Two things here cannot be seen without a device, and the second is the one worth the test.
 * That the rows come back sorted is the obvious assertion. That the permutation is <em>applied</em>
 * is not: an operator that ordered nothing — because the transfer silently gave it the identity, or
 * because the permutation was read as sorted data — returns every input row exactly once, and only
 * an order-sensitive assertion over an input that was not already sorted can tell the difference.
 * So the input arrives deliberately unsorted and the comparison is row for row.
 *
 * <p>Stability is asserted separately, over an input with far more rows than distinct keys so that
 * nearly every comparison is a tie. A carried {@code BIGINT} records the arrival order, which is
 * also the second thing this proves: a payload column the kernel generator would refuse outright
 * rides through a sort untouched, because nothing but the key reaches the device.
 *
 * <p>Skips without a device, and separately without the cuDF shim. See {@link
 * DeviceAssumptions#requireCudf()}.
 */
class AcceleratedSortDeviceIT {

    private static final int ROWS = 40_000;

    /** Few enough that almost every comparison in the stability case is a tie. */
    private static final int DISTINCT_KEYS = 16;

    private static final LogicalType INT = new IntType(false);
    private static final LogicalType BIGINT = new BigIntType(false);
    private static final LogicalType DOUBLE = new DoubleType(false);

    /** (key, arrival ordinal, payload). The BIGINT is not something a kernel could compute with. */
    private static final RowType ROW = RowType.of(INT, BIGINT, DOUBLE);

    private static final AcceleratorContext CONTEXT =
            new AcceleratorContext() {
                @Override
                public RowType outputType() {
                    return ROW;
                }

                @Override
                public int maxBatchSize() {
                    // Deliberately far below the partition: a sort that sized itself from this
                    // would refuse, which is the thing stagingBytes() exists to stop.
                    return 1024;
                }

                @Override
                public ClassLoader userCodeClassLoader() {
                    return AcceleratedSortDeviceIT.class.getClassLoader();
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
    @DisplayName("the device's order is the host's, row for row")
    void theDeviceOrdersLikeTheHost() throws Exception {
        List<int[]> scrambled = new ArrayList<>();
        for (int i = 0; i < ROWS; i++) {
            // A multiplicative shuffle: every key appears once, in an order nothing sorted.
            scrambled.add(new int[] {(int) (((long) i * 2654435761L) % ROWS), i});
        }

        List<RowData> out = run(scrambled, ROWS);

        assertThat(out).hasSize(ROWS);
        for (int i = 0; i < ROWS; i++) {
            assertThat(out.get(i).getInt(0)).as("row " + i + "'s key").isEqualTo(i);
        }
        // The payload rode along: field 2 is derived from the ordinal, so a row whose key moved
        // without its payload would fail here and pass the assertion above.
        for (RowData row : out) {
            long ordinal = row.getLong(1);
            assertThat(row.getDouble(2)).isEqualTo(payload(ordinal));
        }
    }

    @Test
    @DisplayName("equal keys keep their arrival order")
    void theSortIsStable() throws Exception {
        List<int[]> ties = new ArrayList<>();
        for (int i = 0; i < ROWS; i++) {
            ties.add(new int[] {i % DISTINCT_KEYS, i});
        }

        List<RowData> out = run(ties, ROWS);

        assertThat(out).hasSize(ROWS);
        boolean first = true;
        int previousKey = 0;
        long previousOrdinal = 0;
        for (RowData row : out) {
            int key = row.getInt(0);
            long ordinal = row.getLong(1);
            if (!first) {
                if (key == previousKey) {
                    assertThat(ordinal)
                            .as("equal keys were reordered, so the sort is not stable")
                            .isGreaterThan(previousOrdinal);
                } else {
                    assertThat(key).as("keys are not ascending").isGreaterThan(previousKey);
                }
            }
            first = false;
            previousKey = key;
            previousOrdinal = ordinal;
        }
    }

    @Test
    @DisplayName("a descending sort is declined rather than approximated")
    void descendingIsDeclined() {
        AccelSort descending = new AccelSort(0, false, true, new AccelInput(ROW), ROW);
        assertThat(DeviceAssumptions.provider().accept(descending, work(ROWS))).isEmpty();
    }

    /**
     * Runs one partition through the device operator and returns what it emitted.
     *
     * <p>Asserts on the way through that a device actually served it: an operator that fell back to
     * the host sorts correctly and would pass every ordering assertion above in silence.
     */
    private static List<RowData> run(List<int[]> input, long estimatedRows) throws Exception {
        AccelNode subtree = new AccelSort(0, true, true, new AccelInput(ROW), ROW);
        Optional<AcceleratorPlan> offered =
                DeviceAssumptions.provider().accept(subtree, work(estimatedRows));
        assertThat(offered).as("the provider declined a shape it is meant to serve").isPresent();

        StreamOperatorFactory<RowData> factory =
                DeviceAssumptions.provider().createOperator(offered.get(), CONTEXT);

        List<RowData> out = new ArrayList<>();
        try (OneInputStreamOperatorTestHarness<RowData, RowData> harness =
                new OneInputStreamOperatorTestHarness<>(factory, 1, 1, 0)) {
            // An explicit serializer, because this operator emits a reused BinaryRowData and the
            // harness copies with whatever TypeExtractor infers when it is not told -- which for a
            // binary row means Kryo over its MemorySegments. Naming the row's own serializer keeps
            // the test measuring the operator rather than that inference.
            harness.setup(new RowDataSerializer(ROW));
            harness.open();
            for (int[] row : input) {
                GenericRowData record = new GenericRowData(3);
                record.setField(0, row[0]);
                record.setField(1, (long) row[1]);
                record.setField(2, payload(row[1]));
                harness.processElement(new StreamRecord<>(record));
            }
            GpuSortOperator operator = (GpuSortOperator) harness.getOneInputOperator();
            operator.endInput();
            assertThat(operator.metrics().getBatches())
                    .as("the partition was not ordered on the device")
                    .isEqualTo(1);

            for (Object o : harness.getOutput()) {
                @SuppressWarnings("unchecked")
                RowData emitted = ((StreamRecord<RowData>) o).getValue();
                out.add(emitted);
            }
        }
        return out;
    }

    private static double payload(long ordinal) {
        return ordinal * 0.5;
    }

    private static AccelWorkProfile work(long rows) {
        int[] ops = new int[AccelFunction.values().length];
        ops[AccelFunction.LESS_THAN.ordinal()] = 16;
        int width = 4 + 8 + 8;
        return new AccelWorkProfile(ops, width, width, rows);
    }
}
