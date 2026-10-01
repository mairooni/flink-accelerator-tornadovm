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
import org.apache.flink.table.data.columnar.vector.heap.HeapFloatVector;
import org.apache.flink.table.gpu.codegen.GpuValueType;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The same path as {@link BulkColumnarGatherTest}, for a {@code FLOAT} column.
 *
 * <p>Separate rather than parameterised because the two differ in the element layout and nothing
 * else, and a test that abstracted over the layout would stop asserting the one thing that can go
 * wrong with it -- that a float is read at a float's stride and not a double's.
 */
class BulkColumnarFloatGatherTest {

    private static final int ROWS = 8;

    private static VectorizedColumnBatch batch(boolean nulls) {
        HeapFloatVector vector = new HeapFloatVector(ROWS);
        for (int i = 0; i < ROWS; i++) {
            vector.vector[i] = 100.5f + i;
        }
        if (nulls) {
            vector.setNullAt(3);
        }
        VectorizedColumnBatch built = new VectorizedColumnBatch(new ColumnVector[] {vector});
        built.setNumRows(ROWS);
        return built;
    }

    private static RowGather gatherInto(ColumnarRowData row, MemorySegment segment) {
        return RowGather.forColumn(
                row,
                0,
                GpuValueType.FLOAT,
                (position, value) ->
                        segment.setAtIndex(ValueLayout.JAVA_FLOAT, position, (float) value),
                segment);
    }

    private static float[] stage(VectorizedColumnBatch batch, Arena arena, String expectedTier) {
        MemorySegment segment = arena.allocate((long) ROWS * Float.BYTES);
        ColumnarRowData row = new ColumnarRowData(batch);
        RowGather gather = gatherInto(row, segment);
        for (int i = 0; i < ROWS; i++) {
            row.setRowId(i);
            gather.accept(row, i);
        }
        gather.flush();
        float[] staged = new float[ROWS];
        for (int i = 0; i < ROWS; i++) {
            staged[i] = segment.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        if (expectedTier != null) {
            assertThat(gather.tier()).isEqualTo(expectedTier);
        }
        return staged;
    }

    @Test
    @DisplayName("a FLOAT column takes the bulk tier, which it did not before M9.17")
    void floatIsDispatchedToTheBulkGather() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate((long) ROWS * Float.BYTES);
            ColumnarRowData row = new ColumnarRowData(batch(false));
            assertThat(gatherInto(row, segment)).isInstanceOf(BulkColumnarFloatGather.class);
        }
    }

    @Test
    @DisplayName("a whole batch arrives in order, at a float's stride")
    void aWholeBatchArrivesInOrder() {
        try (Arena arena = Arena.ofConfined()) {
            // The values are deliberately not integral: 100.5f is exact in a float, so a value
            // that came back shifted or widened would not merely be imprecise, it would be wrong.
            assertThat(stage(batch(false), arena, "tier1-columnar-bulk(100.0% bulk)"))
                    .containsExactly(
                            100.5f, 101.5f, 102.5f, 103.5f, 104.5f, 105.5f, 106.5f, 107.5f);
        }
    }

    @Test
    @DisplayName("a refilled vector is copied before it is refilled")
    void aRefilledVectorIsCopiedBeforeItIsRefilled() {
        final int batches = 3;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate((long) ROWS * batches * Float.BYTES);
            HeapFloatVector vector = new HeapFloatVector(ROWS);
            VectorizedColumnBatch batch = new VectorizedColumnBatch(new ColumnVector[] {vector});
            batch.setNumRows(ROWS);
            ColumnarRowData row = new ColumnarRowData(batch);
            RowGather gather = gatherInto(row, segment);

            int position = 0;
            for (int b = 0; b < batches; b++) {
                // The refill. Nothing of the previous batch survives it, which is the point.
                for (int i = 0; i < ROWS; i++) {
                    vector.vector[i] = 1000.0f * b + i;
                }
                for (int i = 0; i < ROWS; i++) {
                    row.setRowId(i);
                    gather.accept(row, position++);
                }
            }
            gather.flush();

            for (int b = 0; b < batches; b++) {
                for (int i = 0; i < ROWS; i++) {
                    assertThat(segment.getAtIndex(ValueLayout.JAVA_FLOAT, b * ROWS + i))
                            .as("batch " + b + " row " + i)
                            .isEqualTo(1000.0f * b + i);
                }
            }
            assertThat(gather.tier()).isEqualTo("tier1-columnar-bulk(100.0% bulk)");
        }
    }

    @Test
    @DisplayName("a nullable column falls back to per-row, as a null slot holds anything")
    void aNullableColumnFallsBackToPerRow() {
        try (Arena arena = Arena.ofConfined()) {
            float[] staged = stage(batch(true), arena, "tier1-columnar-bulk(0.0% bulk)");
            // The values still arrive; what changes is that each one is read through the accessor.
            assertThat(staged[0]).isEqualTo(100.5f);
            assertThat(staged[7]).isEqualTo(107.5f);
        }
    }
}
