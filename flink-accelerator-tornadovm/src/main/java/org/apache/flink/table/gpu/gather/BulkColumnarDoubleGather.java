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

import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.columnar.ColumnarRowData;
import org.apache.flink.table.data.columnar.vector.ColumnVector;
import org.apache.flink.table.data.columnar.vector.VectorizedColumnBatch;
import org.apache.flink.table.data.columnar.vector.heap.HeapDoubleVector;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Tier 1, bulk path -- copies runs of rows out of a {@link VectorizedColumnBatch} with one {@link
 * MemorySegment#copy} instead of an accessor call per row.
 *
 * <p>This is the only gather that attacks the host-side staging cost rather than paying it. A
 * vectorized Parquet/ORC source has already produced the column packed and column-major, so there
 * is no transposition to do and the rest is a memcpy from the heap {@code double[]} into the
 * off-heap staging buffer.
 *
 * <h2>Why it buffers instead of being handed a list of rows</h2>
 *
 * <p>Because there is no list to be handed. Flink's vectorized readers hand out one {@link
 * ColumnarRowData} and call {@code setRowId} on it per row, and the Table planner force-enables
 * object reuse in batch mode, so rows collected into a list are one object repeated -- every
 * element reporting whatever row the reader has since reached. An earlier version of this class
 * took {@code (List<RowData>, from, to)} and would have read the last row's id as the run's start:
 * a copy from the wrong offset, silently, with entirely plausible values in it.
 *
 * <p>So {@link #accept} holds the run's bounds and nothing else: while each row continues the run
 * -- same batch, next row id, next staging position -- it records that and returns, and {@link
 * #flush} issues the copy. Per row that is two field reads and three comparisons, against a virtual
 * accessor call with its bounds and null checks.
 *
 * <h2>When the bulk path is not valid</h2>
 *
 * <p>A run is only started for a plain {@link HeapDoubleVector} with no dictionary and no nulls;
 * anything else falls back to per-row access. Both conditions produce silently wrong values rather
 * than failures if ignored: a dictionary-encoded vector holds ids rather than values, and a null
 * slot holds whatever was last written there.
 */
public final class BulkColumnarDoubleGather implements RowGather {

    private final int field;
    private final RowGather.StagingColumn target;
    private final MemorySegment targetSegment;

    private long bulkRows;
    private long perRowRows;

    /** The vector the pending run is copied from, or null when no run is pending. */
    private HeapDoubleVector runVector;

    private int runStartRowId;
    private int runStartPosition;
    private int runLength;

    public BulkColumnarDoubleGather(
            int field, RowGather.StagingColumn target, MemorySegment targetSegment) {
        this.field = field;
        this.target = target;
        this.targetSegment = targetSegment;
    }

    @Override
    public void accept(RowData row, int position) {
        if (!(row instanceof ColumnarRowData columnar)) {
            // Not reachable through forColumn, which picks this gather from the first row's class.
            // Still worth not corrupting the buffer if a plan ever mixes implementations.
            flush();
            perRowRows++;
            target.set(position, row.getDouble(field));
            return;
        }
        VectorizedColumnBatch batch = columnar.getVectorizedColumnBatch();
        int rowId = columnar.getRowId();

        if (runVector != null
                && batch.columns[field] == runVector
                && rowId == runStartRowId + runLength
                && position == runStartPosition + runLength) {
            runLength++;
            return;
        }

        flush();

        ColumnVector column = batch.columns[field];
        if (column instanceof HeapDoubleVector vector && copyable(vector)) {
            runVector = vector;
            runStartRowId = rowId;
            runStartPosition = position;
            runLength = 1;
            return;
        }
        perRowRows++;
        target.set(position, columnar.getDouble(field));
    }

    /**
     * Whether a range of this vector can be copied as it stands.
     *
     * <p>Both questions are asked once per run rather than once per row, which is the point: {@code
     * isNullAt} would answer the second one per slot and that is exactly the per-row cost the copy
     * exists to remove. {@code hasNulls} is the counterpart of {@code hasDictionary} and was added
     * to {@code AbstractWritableVector} for this.
     */
    private boolean copyable(HeapDoubleVector vector) {
        return !vector.hasDictionary() && !vector.hasNulls();
    }

    @Override
    public void flush() {
        if (runVector == null) {
            return;
        }
        MemorySegment.copy(
                runVector.vector,
                runStartRowId,
                targetSegment,
                ValueLayout.JAVA_DOUBLE,
                (long) runStartPosition * Double.BYTES,
                runLength);
        bulkRows += runLength;
        runVector = null;
        runLength = 0;
    }

    @Override
    public String tier() {
        long total = bulkRows + perRowRows;
        double pct = total == 0 ? 0.0 : 100.0 * bulkRows / total;
        return String.format("tier1-columnar-bulk(%.1f%% bulk)", pct);
    }
}
