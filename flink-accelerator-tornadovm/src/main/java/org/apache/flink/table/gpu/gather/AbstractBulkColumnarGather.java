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

import javax.annotation.Nullable;

import java.lang.foreign.MemorySegment;

/**
 * Tier 1, bulk path -- copies runs of rows out of a {@link VectorizedColumnBatch} with one {@link
 * MemorySegment#copy} instead of an accessor call per row.
 *
 * <p>This is the only gather that attacks the host-side staging cost rather than paying it. A
 * vectorized Parquet/ORC source has already produced the column packed and column-major, so there
 * is no transposition to do and the rest is a memcpy from the heap array into the off-heap staging
 * buffer.
 *
 * <p>The run tracking lives here and the element type lives in the subclass. It is split that way
 * because the tracking is the part that is subtle -- see {@link #flushIfBatchEnds} for a bug that
 * produced plausible wrong numbers -- and a second copy of it for a second element type would be a
 * second chance to get it wrong.
 *
 * <h2>Why it buffers instead of being handed a list of rows</h2>
 *
 * <p>Because there is no list to be handed. Flink's vectorized readers hand out one {@link
 * ColumnarRowData} and call {@code setRowId} on it per row, and the Table planner force-enables
 * object reuse in batch mode, so rows collected into a list are one object repeated -- every
 * element reporting whatever row the reader has since reached. An earlier version took {@code
 * (List<RowData>, from, to)} and would have read the last row's id as the run's start: a copy from
 * the wrong offset, silently, with entirely plausible values in it.
 *
 * <p>So {@link #accept} holds the run's bounds and nothing else: while each row continues the run
 * -- same vector, next row id, next staging position -- it records that and returns, and the copy
 * is issued at the end of the source batch. Per row that is two field reads and three comparisons,
 * against a virtual accessor call with its bounds and null checks.
 *
 * <h2>When the bulk path is not valid</h2>
 *
 * <p>A run is only started for a plain heap vector of the expected type with no dictionary and no
 * nulls; anything else falls back to per-row access. Both conditions produce silently wrong values
 * rather than failures if ignored: a dictionary-encoded vector holds ids rather than values, and a
 * null slot holds whatever was last written there.
 */
abstract class AbstractBulkColumnarGather implements RowGather {

    protected final int field;
    protected final RowGather.StagingColumn target;
    protected final MemorySegment targetSegment;

    private long bulkRows;
    private long perRowRows;

    /** The vector the pending run is copied from, or null when no run is pending. */
    private @Nullable ColumnVector runVector;

    private int runStartRowId;
    private int runStartPosition;
    private int runLength;

    AbstractBulkColumnarGather(
            int field, RowGather.StagingColumn target, MemorySegment targetSegment) {
        this.field = field;
        this.target = target;
        this.targetSegment = targetSegment;
    }

    /**
     * The vector, if a range of it can be copied as it stands, or null to take the per-row path.
     *
     * <p>Asked once per run rather than once per row, which is the point: {@code isNullAt} would
     * answer half of it per slot and that is exactly the cost the copy exists to remove.
     */
    protected abstract @Nullable ColumnVector copyableVector(ColumnVector column);

    /** Copies {@code length} values from the vector into the staging buffer at {@code position}. */
    protected abstract void copyRun(ColumnVector vector, int fromRowId, int position, int length);

    /** One value, for the rows the bulk path cannot take. */
    protected abstract double valueAt(RowData row, int field);

    @Override
    public void accept(RowData row, int position) {
        if (!(row instanceof ColumnarRowData columnar)) {
            // Not reachable through forColumn, which picks this gather from the first row's class.
            // Still worth not corrupting the buffer if a plan ever mixes implementations.
            flush();
            perRowRows++;
            target.set(position, valueAt(row, field));
            return;
        }
        VectorizedColumnBatch batch = columnar.getVectorizedColumnBatch();
        int rowId = columnar.getRowId();

        if (runVector != null
                && batch.columns[field] == runVector
                && rowId == runStartRowId + runLength
                && position == runStartPosition + runLength) {
            runLength++;
            flushIfBatchEnds(batch, rowId);
            return;
        }

        flush();

        ColumnVector copyable = copyableVector(batch.columns[field]);
        if (copyable != null) {
            runVector = copyable;
            runStartRowId = rowId;
            runStartPosition = position;
            runLength = 1;
            flushIfBatchEnds(batch, rowId);
            return;
        }
        perRowRows++;
        target.set(position, valueAt(columnar, field));
    }

    /**
     * Copies the pending run at the last row of its source batch, rather than waiting to be told.
     *
     * <p>Not an optimisation -- correctness, and the kind that produces plausible numbers when it
     * is missing. A reader allocates its vectors once and {@code reset()}s and refills them for
     * each batch it reads, so the array a run points into stops holding that run's values the
     * moment the reader moves on. Deferring to the staging batch's flush, a hundred source batches
     * later, copies whichever batch the reader happened to finish with.
     *
     * <p>The end of the source batch is the last moment the data is still there, and it is knowable
     * here: {@code getNumRows} says how many rows the batch holds. Waiting for the next row to
     * arrive and noticing it belongs to a different batch is already too late -- the refill has
     * happened by then.
     *
     * <p>Found by a differential arm on cyclone, where a Parquet-sourced sum came back differing
     * from the CPU's in the seventh significant digit. Neither the host nor the device test caught
     * it: both built a vector per source batch and so never reused one.
     */
    private void flushIfBatchEnds(VectorizedColumnBatch batch, int rowId) {
        if (rowId + 1 >= batch.getNumRows()) {
            flush();
        }
    }

    @Override
    public void flush() {
        if (runVector == null) {
            return;
        }
        copyRun(runVector, runStartRowId, runStartPosition, runLength);
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
