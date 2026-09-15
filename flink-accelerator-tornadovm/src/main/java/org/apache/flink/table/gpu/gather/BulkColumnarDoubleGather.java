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
import org.apache.flink.table.data.columnar.vector.writable.AbstractWritableVector;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.List;

/**
 * Tier 1, bulk path — copies a whole run of rows out of a {@link VectorizedColumnBatch} in one
 * {@link MemorySegment#copy} instead of per-row accessor calls.
 *
 * <p>This is the only gather that attacks the host-side staging cost rather than paying it: a
 * vectorized Parquet/ORC source has already produced the column packed and column-major, so there
 * is no transposition to do. The rest is a memcpy from the heap {@code double[]} into the off-heap
 * staging buffer.
 *
 * <h2>Reaching the batch</h2>
 *
 * <p>{@link ColumnarRowData} exposes {@code setVectorizedColumnBatch} and {@code setRowId} but no
 * getters, so the batch is read reflectively here. That works because {@code flink-table-common} is
 * an ordinary classpath jar and therefore in the unnamed module — no {@code --add-opens} needed.
 * The handles are resolved once, statically, so the per-run cost is a plain field read.
 *
 * <p>Reflection is a stand-in for a five-line accessor patch to {@code flink-table-common}. It is
 * used here so the tier can be measured before committing to the patch, not as the intended
 * long-term mechanism.
 *
 * <h2>When the bulk path is not valid</h2>
 *
 * <p>{@link #acceptBulk} refuses and falls back to per-row access unless the column is a plain
 * {@link HeapDoubleVector} with no dictionary and no nulls. Both conditions produce silently wrong
 * values rather than failures if ignored: a dictionary-encoded vector holds IDs rather than values,
 * and a null slot holds whatever was last written there.
 */
public final class BulkColumnarDoubleGather implements RowGather {

    private static final VarHandle BATCH;
    private static final VarHandle ROW_ID;

    /** {@code noNulls} is protected; {@code hasDictionary()} next to it is public. */
    private static final VarHandle NO_NULLS;

    static {
        try {
            MethodHandles.Lookup lookup =
                    MethodHandles.privateLookupIn(ColumnarRowData.class, MethodHandles.lookup());
            BATCH =
                    lookup.findVarHandle(
                            ColumnarRowData.class,
                            "vectorizedColumnBatch",
                            VectorizedColumnBatch.class);
            ROW_ID = lookup.findVarHandle(ColumnarRowData.class, "rowId", int.class);
            MethodHandles.Lookup vectorLookup =
                    MethodHandles.privateLookupIn(
                            AbstractWritableVector.class, MethodHandles.lookup());
            NO_NULLS =
                    vectorLookup.findVarHandle(
                            AbstractWritableVector.class, "noNulls", boolean.class);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(
                    "ColumnarRowData layout changed; the bulk columnar gather needs updating: "
                            + e);
        }
    }

    private final int field;
    private final RowGather.StagingColumn target;
    private final MemorySegment targetSegment;

    private long bulkRows;
    private long perRowRows;

    public BulkColumnarDoubleGather(
            int field, RowGather.StagingColumn target, MemorySegment targetSegment) {
        this.field = field;
        this.target = target;
        this.targetSegment = targetSegment;
    }

    @Override
    public void accept(RowData row, int position) {
        perRowRows++;
        target.set(position, ((ColumnarRowData) row).getDouble(field));
    }

    /**
     * Copies as many rows from {@code rows[from..to)} as share one column batch and are laid out
     * consecutively within it.
     *
     * @return the number of rows consumed; 0 means the caller should fall back to per-row access
     */
    @Override
    public int acceptBulk(List<RowData> rows, int from, int to, int position) {
        RowData head = rows.get(from);
        if (!(head instanceof ColumnarRowData)) {
            return 0;
        }
        VectorizedColumnBatch batch = (VectorizedColumnBatch) BATCH.get((ColumnarRowData) head);
        ColumnVector column = batch.columns[field];
        if (!(column instanceof HeapDoubleVector vector)) {
            return 0;
        }
        if (vector.hasDictionary() || !((boolean) NO_NULLS.get(vector))) {
            return 0;
        }

        int startRowId = (int) ROW_ID.get((ColumnarRowData) head);

        // Extend the run while rows stay in this batch and stay consecutive. Flink's readers
        // produce rows in order, so this normally runs to the end of the batch in one step.
        int run = 1;
        while (from + run < to) {
            RowData next = rows.get(from + run);
            if (!(next instanceof ColumnarRowData)) {
                break;
            }
            if (BATCH.get((ColumnarRowData) next) != batch) {
                break;
            }
            if ((int) ROW_ID.get((ColumnarRowData) next) != startRowId + run) {
                break;
            }
            run++;
        }

        MemorySegment.copy(
                vector.vector,
                startRowId,
                targetSegment,
                ValueLayout.JAVA_DOUBLE,
                (long) position * Double.BYTES,
                run);

        bulkRows += run;
        return run;
    }

    @Override
    public String tier() {
        long total = bulkRows + perRowRows;
        double pct = total == 0 ? 0.0 : 100.0 * bulkRows / total;
        return String.format("tier1-columnar-bulk(%.1f%% bulk)", pct);
    }
}
