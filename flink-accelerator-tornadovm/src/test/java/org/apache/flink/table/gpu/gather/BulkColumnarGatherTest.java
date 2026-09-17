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

package org.apache.flink.table.gpu.gather;

import org.apache.flink.table.data.columnar.ColumnarRowData;
import org.apache.flink.table.data.columnar.vector.ColumnVector;
import org.apache.flink.table.data.columnar.vector.VectorizedColumnBatch;
import org.apache.flink.table.data.columnar.vector.heap.HeapDoubleVector;
import org.apache.flink.table.gpu.codegen.GpuValueType;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Staging a vectorized batch by the column rather than by the row.
 *
 * <p>Every case here drives the gather the way the operator does — one {@link ColumnarRowData}
 * whose row id moves, because that is what Flink's readers hand out and what the previous design of
 * this path could not survive. A test that made a row per value would pass against code that is
 * wrong in production.
 */
class BulkColumnarGatherTest {

    private static final int ROWS = 8;

    /** A batch of one double column, values 100.0, 101.0, ... */
    private static VectorizedColumnBatch batch(boolean nulls, boolean dictionary) {
        HeapDoubleVector vector = new HeapDoubleVector(ROWS);
        for (int i = 0; i < ROWS; i++) {
            vector.vector[i] = 100.0 + i;
        }
        if (nulls) {
            vector.setNullAt(3);
        }
        if (dictionary) {
            // A real dictionary-encoded vector holds ids in its backing array and the values in
            // the dictionary; setting only the flag would make the fallback path read ids it has
            // nowhere to decode.
            org.apache.flink.table.data.columnar.vector.heap.HeapIntVector ids =
                    vector.reserveDictionaryIds(ROWS);
            for (int i = 0; i < ROWS; i++) {
                ids.vector[i] = i;
                vector.vector[i] = i;
            }
            // Only the flag matters here: a gather that copies the backing array of a
            // dictionary-encoded vector copies ids, and the point is that it declines to.
            vector.setDictionary(
                    new org.apache.flink.table.data.columnar.vector.Dictionary() {
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
                            return id;
                        }

                        @Override
                        public double decodeToDouble(int id) {
                            return 100.0 + id;
                        }

                        @Override
                        public org.apache.flink.table.data.TimestampData decodeToTimestamp(int id) {
                            throw new UnsupportedOperationException();
                        }

                        @Override
                        public byte[] decodeToBinary(int id) {
                            throw new UnsupportedOperationException();
                        }
                    });
        }
        VectorizedColumnBatch built = new VectorizedColumnBatch(new ColumnVector[] {vector});
        built.setNumRows(ROWS);
        return built;
    }

    /** Runs the whole batch through one gather, moving a single row object as a reader does. */
    private static double[] stage(VectorizedColumnBatch batch, Arena arena) {
        MemorySegment segment = arena.allocate((long) ROWS * Double.BYTES);
        ColumnarRowData row = new ColumnarRowData(batch);
        RowGather gather =
                RowGather.forColumn(
                        row,
                        0,
                        GpuValueType.DOUBLE,
                        (position, value) ->
                                segment.setAtIndex(ValueLayout.JAVA_DOUBLE, position, value),
                        segment);
        for (int i = 0; i < ROWS; i++) {
            row.setRowId(i);
            gather.accept(row, i);
        }
        gather.flush();

        double[] staged = new double[ROWS];
        for (int i = 0; i < ROWS; i++) {
            staged[i] = segment.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
        assertThat(gather.tier()).as("the tier string reports what actually happened").isNotNull();
        return staged;
    }

    @Test
    void aWholeBatchArrivesInOrder() {
        try (Arena arena = Arena.ofConfined()) {
            assertThat(stage(batch(false, false), arena))
                    .containsExactly(100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0);
        }
    }

    /**
     * The regression the redesign exists for.
     *
     * <p>The reader moves one row object, so a gather that read the run's start id after the fact
     * would read 7 and copy from there — eight values of which seven are past the end. Asserting
     * the tier is fully bulk is what says the run was actually taken rather than quietly falling
     * back to the per-row path, which would also produce the right numbers.
     */
    @Test
    void oneMovingRowStagesAsOneRun() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate((long) ROWS * Double.BYTES);
            ColumnarRowData row = new ColumnarRowData(batch(false, false));
            RowGather gather =
                    RowGather.forColumn(
                            row,
                            0,
                            GpuValueType.DOUBLE,
                            (position, value) ->
                                    segment.setAtIndex(ValueLayout.JAVA_DOUBLE, position, value),
                            segment);
            for (int i = 0; i < ROWS; i++) {
                row.setRowId(i);
                gather.accept(row, i);
            }
            gather.flush();

            assertThat(gather.tier()).isEqualTo("tier1-columnar-bulk(100.0% bulk)");
            assertThat(segment.getAtIndex(ValueLayout.JAVA_DOUBLE, 7)).isEqualTo(107.0);
        }
    }

    /**
     * The regression a differential run on cyclone found and every test here missed.
     *
     * <p>A reader allocates its vectors once and refills them per batch, so this drives ONE
     * HeapDoubleVector through three batches, resetting its contents each time exactly as {@code
     * nextBatch()} does. A gather that holds a run across that boundary copies whichever batch the
     * reader finished with -- here, three times the last batch's values, which are entirely
     * plausible numbers in the right range.
     */
    @Test
    void aRefilledVectorIsCopiedBeforeItIsRefilled() {
        final int batches = 3;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate((long) ROWS * batches * Double.BYTES);
            HeapDoubleVector vector = new HeapDoubleVector(ROWS);
            VectorizedColumnBatch batch = new VectorizedColumnBatch(new ColumnVector[] {vector});
            batch.setNumRows(ROWS);
            ColumnarRowData row = new ColumnarRowData(batch);
            RowGather gather =
                    RowGather.forColumn(
                            row,
                            0,
                            GpuValueType.DOUBLE,
                            (position, value) ->
                                    segment.setAtIndex(ValueLayout.JAVA_DOUBLE, position, value),
                            segment);

            int position = 0;
            for (int b = 0; b < batches; b++) {
                // The refill. Nothing of the previous batch survives it, which is the point.
                for (int i = 0; i < ROWS; i++) {
                    vector.vector[i] = 1000.0 * b + i;
                }
                for (int i = 0; i < ROWS; i++) {
                    row.setRowId(i);
                    gather.accept(row, position++);
                }
            }
            gather.flush();

            for (int b = 0; b < batches; b++) {
                for (int i = 0; i < ROWS; i++) {
                    assertThat(segment.getAtIndex(ValueLayout.JAVA_DOUBLE, b * ROWS + i))
                            .as("batch " + b + " row " + i)
                            .isEqualTo(1000.0 * b + i);
                }
            }
            assertThat(gather.tier()).isEqualTo("tier1-columnar-bulk(100.0% bulk)");
        }
    }

    /** Nothing is in the buffer until flush, which is why the operator has to call it. */
    @Test
    void nothingIsStagedBeforeFlush() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate((long) ROWS * Double.BYTES);
            segment.fill((byte) 0);
            ColumnarRowData row = new ColumnarRowData(batch(false, false));
            RowGather gather =
                    RowGather.forColumn(
                            row,
                            0,
                            GpuValueType.DOUBLE,
                            (position, value) ->
                                    segment.setAtIndex(ValueLayout.JAVA_DOUBLE, position, value),
                            segment);
            row.setRowId(0);
            gather.accept(row, 0);

            assertThat(segment.getAtIndex(ValueLayout.JAVA_DOUBLE, 0))
                    .as("a deferred run must not have been written yet")
                    .isZero();
            gather.flush();
            assertThat(segment.getAtIndex(ValueLayout.JAVA_DOUBLE, 0)).isEqualTo(100.0);
        }
    }

    /**
     * A null slot holds whatever was last written there, so the run is refused and every value
     * comes through the accessor — which is the only thing that knows the slot is absent.
     */
    @Test
    void aNullableColumnFallsBackToPerRow() {
        try (Arena arena = Arena.ofConfined()) {
            VectorizedColumnBatch batch = batch(true, false);
            double[] staged = stage(batch, arena);

            assertThat(staged[0]).isEqualTo(100.0);
            assertThat(staged[7]).isEqualTo(107.0);
        }
    }

    /** A dictionary-encoded vector's backing array holds ids, so copying it copies ids. */
    @Test
    void aDictionaryColumnFallsBackToPerRow() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate((long) ROWS * Double.BYTES);
            ColumnarRowData row = new ColumnarRowData(batch(false, true));
            RowGather gather =
                    RowGather.forColumn(
                            row,
                            0,
                            GpuValueType.DOUBLE,
                            (position, value) ->
                                    segment.setAtIndex(ValueLayout.JAVA_DOUBLE, position, value),
                            segment);
            for (int i = 0; i < ROWS; i++) {
                row.setRowId(i);
                gather.accept(row, i);
            }
            gather.flush();

            assertThat(gather.tier())
                    .as("no row may be copied wholesale out of a dictionary-encoded vector")
                    .isEqualTo("tier1-columnar-bulk(0.0% bulk)");
            // And the decoded values arrive, rather than the ids the array holds.
            assertThat(segment.getAtIndex(ValueLayout.JAVA_DOUBLE, 7)).isEqualTo(107.0);
        }
    }
}
