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
import org.apache.flink.table.accelerator.AccelCall;
import org.apache.flink.table.accelerator.AccelExpression;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelLiteral;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.columnar.ColumnarRowData;
import org.apache.flink.table.data.columnar.vector.ColumnVector;
import org.apache.flink.table.data.columnar.vector.VectorizedColumnBatch;
import org.apache.flink.table.data.columnar.vector.heap.HeapDoubleVector;
import org.apache.flink.table.gpu.operator.GpuCalcOperator;
import org.apache.flink.table.runtime.accelerator.AcceleratorContext;
import org.apache.flink.table.runtime.accelerator.AcceleratorPlan;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * M5.1 — a vectorized batch staged by the column, all the way onto the device.
 *
 * <p>{@link org.apache.flink.table.gpu.gather.BulkColumnarGatherTest} covers the copy itself on the
 * host. What this adds is the rest of the path: that the values the kernel reads are the ones the
 * batch held, that a run spanning several source batches lands in the right places, and that a
 * column the bulk path refuses still produces the right answer. A staging bug here is silent — the
 * kernel computes whatever is in the buffer — so it has to be asserted against a device, not
 * against the staging buffer.
 *
 * <p>The rows are driven the way a reader drives them: one {@link ColumnarRowData} per source
 * batch, moved by {@code setRowId}. Tests that make a row object per value pass against staging
 * code that is wrong in production, which is how the previous design of this path survived review.
 */
class ColumnarSourceDeviceIT {

    /** Several source batches per staging batch, as a real reader produces. */
    private static final int BATCH_ROWS = 2048;

    private static final int SOURCE_BATCHES = 12;

    private static final int ROWS = BATCH_ROWS * SOURCE_BATCHES;

    private static final LogicalType DOUBLE = new DoubleType(false);

    private static final AcceleratorContext CONTEXT =
            new AcceleratorContext() {
                @Override
                public RowType outputType() {
                    return RowType.of(new DoubleType(false));
                }

                @Override
                public int maxBatchSize() {
                    return 16_384;
                }

                @Override
                public ClassLoader userCodeClassLoader() {
                    return ColumnarSourceDeviceIT.class.getClassLoader();
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
    void requireDevice() {
        DeviceAssumptions.requireDevice();
    }

    @Test
    @DisplayName("a column staged in bulk computes what the same column staged per row does")
    void bulkStagedColumnAgreesWithTheHost() throws Exception {
        List<RowData> emitted = run(false);

        assertThat(emitted).hasSize(ROWS);
        for (int i = 0; i < ROWS; i++) {
            // Exact. The staging path changed; the arithmetic did not, and if a user could tell
            // which gather ran, the whole premise of the tier system would be gone.
            assertThat(emitted.get(i).getDouble(0)).as("row " + i).isEqualTo(onHost(value(i)));
        }
    }

    /**
     * The alignment case, and the one worth having a device for.
     *
     * <p>A run ends at the end of every source batch, and a staging batch holds several of them, so
     * the copies land at 2048-row offsets into the buffer. An off-by-one in the run's start
     * position or its length shifts a whole source batch and every value after it stays plausible.
     */
    @Test
    @DisplayName("runs from several source batches land at the right offsets")
    void severalSourceBatchesStageInOrder() throws Exception {
        List<RowData> emitted = run(false);

        assertThat(emitted).hasSize(ROWS);
        for (int batch = 0; batch < SOURCE_BATCHES; batch++) {
            int first = batch * BATCH_ROWS;
            int last = first + BATCH_ROWS - 1;
            assertThat(emitted.get(first).getDouble(0))
                    .as("first row of source batch " + batch)
                    .isEqualTo(onHost(value(first)));
            assertThat(emitted.get(last).getDouble(0))
                    .as("last row of source batch " + batch)
                    .isEqualTo(onHost(value(last)));
        }
    }

    /** A column the bulk path refuses is still staged, just a slot at a time. */
    @Test
    @DisplayName("a dictionary-encoded column falls back and still computes correctly")
    void dictionaryColumnFallsBackAndStillAgrees() throws Exception {
        List<RowData> emitted = run(true);

        assertThat(tiers)
                .as("a dictionary-encoded vector's array holds ids; none of it may be copied")
                .containsExactly("tier1-columnar-bulk(0.0% bulk)");
        assertThat(emitted).hasSize(ROWS);
        for (int i = 0; i < ROWS; i++) {
            assertThat(emitted.get(i).getDouble(0)).as("row " + i).isEqualTo(onHost(value(i)));
        }
    }

    private String[] tiers;

    @SuppressWarnings("unchecked")
    private List<RowData> run(boolean dictionary) throws Exception {
        AccelNode subtree = plan();
        Optional<AcceleratorPlan> offered =
                DeviceAssumptions.provider().accept(subtree, work(subtree));
        assertThat(offered).isPresent();

        StreamOperatorFactory<RowData> factory =
                DeviceAssumptions.provider().createOperator(offered.get(), CONTEXT);

        List<RowData> emitted = new ArrayList<>();
        try (OneInputStreamOperatorTestHarness<RowData, RowData> harness =
                new OneInputStreamOperatorTestHarness<>(factory, 1, 1, 0)) {
            harness.setup();
            harness.open();
            GpuCalcOperator operator = (GpuCalcOperator) harness.getOneInputOperator();

            for (int b = 0; b < SOURCE_BATCHES; b++) {
                // One row object per source batch, moved across it -- which is what
                // ColumnarRowIterator.next() does.
                ColumnarRowData row = new ColumnarRowData(batch(b, dictionary));
                for (int i = 0; i < BATCH_ROWS; i++) {
                    row.setRowId(i);
                    harness.processElement(new StreamRecord<>(row));
                }
            }
            operator.endInput();

            assertThat(operator.metrics().getBatches())
                    .as("no batch ran, so nothing reached the device")
                    .isPositive();
            tiers = operator.gatherTiers();

            harness.getOutput().stream()
                    .map(o -> ((StreamRecord<RowData>) o).getValue())
                    .forEach(emitted::add);
        }
        return emitted;
    }

    /** One double column holding the values for source batch {@code b}. */
    private static VectorizedColumnBatch batch(int b, boolean dictionary) {
        HeapDoubleVector vector = new HeapDoubleVector(BATCH_ROWS);
        if (dictionary) {
            vector.setDictionary(new ValueDictionary(b));
            org.apache.flink.table.data.columnar.vector.heap.HeapIntVector ids =
                    vector.reserveDictionaryIds(BATCH_ROWS);
            for (int i = 0; i < BATCH_ROWS; i++) {
                ids.vector[i] = i;
            }
        } else {
            for (int i = 0; i < BATCH_ROWS; i++) {
                vector.vector[i] = value(b * BATCH_ROWS + i);
            }
        }
        return new VectorizedColumnBatch(new ColumnVector[] {vector});
    }

    /** Decodes an id back to the value the row would have held. */
    private static final class ValueDictionary
            implements org.apache.flink.table.data.columnar.vector.Dictionary {

        private final int sourceBatch;

        private ValueDictionary(int sourceBatch) {
            this.sourceBatch = sourceBatch;
        }

        @Override
        public double decodeToDouble(int id) {
            return value(sourceBatch * BATCH_ROWS + id);
        }

        @Override
        public int decodeToInt(int id) {
            return id;
        }

        @Override
        public long decodeToLong(int id) {
            return id;
        }

        @Override
        public float decodeToFloat(int id) {
            return (float) decodeToDouble(id);
        }

        @Override
        public org.apache.flink.table.data.TimestampData decodeToTimestamp(int id) {
            throw new UnsupportedOperationException();
        }

        @Override
        public byte[] decodeToBinary(int id) {
            throw new UnsupportedOperationException();
        }
    }

    private static double value(int i) {
        return 1.0 + (i % 977) * 0.0625;
    }

    /** val * 2.0 + 1.0, enough arithmetic to be a real kernel and exact in both processors. */
    private static double onHost(double v) {
        return v * 2.0 + 1.0;
    }

    private static AccelNode plan() {
        AccelExpression projection =
                new AccelCall(
                        AccelFunction.PLUS,
                        Arrays.asList(
                                new AccelCall(
                                        AccelFunction.TIMES,
                                        Arrays.asList(
                                                new AccelInputRef(0, DOUBLE),
                                                new AccelLiteral(2.0, DOUBLE)),
                                        DOUBLE),
                                new AccelLiteral(1.0, DOUBLE)),
                        DOUBLE);
        return new AccelProject(
                Collections.singletonList(projection),
                new AccelInput(RowType.of(DOUBLE)),
                RowType.of(DOUBLE));
    }

    private static AccelWorkProfile work(AccelNode subtree) {
        int[] ops = new int[AccelFunction.values().length];
        ops[AccelFunction.PLUS.ordinal()] = 1;
        ops[AccelFunction.TIMES.ordinal()] = 1;
        return new AccelWorkProfile(ops, 8, 8, ROWS);
    }
}
