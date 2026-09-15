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
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.columnar.ColumnarRowData;
import org.apache.flink.table.gpu.codegen.GpuValueType;

import javax.annotation.Nullable;

import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * Copies one column out of a stream of {@link RowData} into a packed, column-major staging buffer.
 *
 * <p>This is the boundary where the input's physical layout stops mattering. Which implementation
 * applies is fixed by what is upstream in the plan, and is knowable at plan time in three of the
 * four positions:
 *
 * <table>
 *   <tr><th>Upstream</th><th>RowData impl</th><th>Tier</th></tr>
 *   <tr><td>Vectorized Parquet/ORC source</td><td>{@link ColumnarRowData}</td><td>1</td></tr>
 *   <tr><td>Downstream of a shuffling Exchange</td><td>{@link BinaryRowData}</td><td>2</td></tr>
 *   <tr><td>Chained after a Calc/Expand</td><td>BoxedWrapperRowData</td><td>4</td></tr>
 * </table>
 *
 * <p><b>Gathering is not an extra pass.</b> Buffering is mandatory — TornadoVM needs a sized,
 * contiguous buffer — so every field is read and written once regardless. Writing column-major
 * rather than row-major is the same traffic to a different address. The tiers therefore differ by
 * per-field access cost, not by an added traversal.
 */
public interface RowGather {

    /** Appends {@code row}'s value for the configured column at {@code position}. */
    void accept(RowData row, int position);

    /**
     * Optionally consumes a run of rows in one operation, returning how many were taken.
     *
     * <p>Returning 0 -- the default -- means the caller falls back to {@link #accept} per row. Only
     * a source whose data is already column-major can do better, so this is where tier 1 stops
     * paying the per-row staging cost that dominates every other tier.
     */
    default int acceptBulk(List<RowData> rows, int from, int to, int position) {
        return 0;
    }

    /** Human-readable tier name, for the metrics report. */
    String tier();

    /**
     * Picks the cheapest gather for the concrete {@link RowData} implementation actually observed.
     *
     * <p>This is a runtime decision because the physical row class is not part of the {@code
     * ExecNodeGraph}: the planner declares only {@code InternalTypeInfo<RowData>} and the connector
     * chooses the implementation inside {@code getScanRuntimeProvider}. There is no source ability
     * for negotiating layout.
     */
    static RowGather forColumn(RowData sample, int field, GpuValueType type, StagingColumn target) {
        return forColumn(sample, field, type, target, null);
    }

    /**
     * As {@link #forDouble(RowData, int, DoubleColumn)}, but given the staging buffer's segment a
     * columnar input gets the bulk path instead of per-row access.
     */
    static RowGather forColumn(
            RowData sample,
            int field,
            GpuValueType type,
            StagingColumn target,
            @Nullable MemorySegment targetSegment) {
        if (sample instanceof ColumnarRowData) {
            // The bulk path copies a HeapDoubleVector's backing array wholesale into a
            // double-typed segment, so it has nothing to offer a narrower column; those fall back
            // to the per-row columnar tier rather than getting a second bulk implementation for a
            // path that is not reachable yet anyway (it needs a vectorized source).
            boolean bulk = targetSegment != null && type == GpuValueType.DOUBLE;
            return bulk
                    ? new BulkColumnarDoubleGather(field, target, targetSegment)
                    : new ColumnarGather(field, type, target);
        }
        if (sample instanceof BinaryRowData) {
            return new BinaryGather(field, type, target);
        }
        return new GenericGather(field, type, target);
    }

    /**
     * Write side of a staging column, so gathers do not depend on the buffer implementation.
     *
     * <p>Values cross as {@code double} whatever the column's width, because {@code INT} and {@code
     * FLOAT} are both exact in one; the implementation narrows to the buffer it holds.
     */
    interface StagingColumn {
        void set(int position, double value);
    }
}
