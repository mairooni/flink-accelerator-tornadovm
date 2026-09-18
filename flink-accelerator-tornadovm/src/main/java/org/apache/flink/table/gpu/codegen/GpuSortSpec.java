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

package org.apache.flink.table.gpu.codegen;

import org.apache.flink.table.accelerator.AccelSort;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;
import org.apache.flink.table.types.logical.RowType;

import javax.annotation.Nullable;

import java.io.Serializable;

/**
 * Everything the runtime needs to order one partition on a device.
 *
 * <p>Much smaller than {@link GpuCalcSpec}, and the difference is the whole point of the sort path:
 * there is no kernel. Only the key column crosses the interconnect, cuDF hands back a permutation,
 * and the rows are emitted from host-side staging in that order. So this carries the key, the row
 * being held, and how many of them the planner thinks there will be — nothing about expressions,
 * because a sort has none.
 *
 * <h2>What the payload has to be, and why that is not the same as device-expressible</h2>
 *
 * <p>No payload column ever reaches the device, so none of them has to be something a kernel could
 * compute with. What every column does have to be is something this can <em>hold</em> at a known
 * width for the life of the partition, which rules out anything variable-length: a {@code VARCHAR}
 * would have to be either boxed or serialised, and a sort that boxes every row has already lost to
 * {@code SortOperator}, which keeps binary rows in managed memory with normalized keys.
 */
public final class GpuSortSpec implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Fields a row may have.
     *
     * <p>Validity is one {@code int} a row, a bit a column — the same layout the kernel path uses
     * for its input nulls, and for the same reason: four bytes a row whatever the column count,
     * against four bytes per column the obvious way. Thirty-two is what an {@code int} holds.
     */
    public static final int MAX_FIELDS = 32;

    private final int sortField;
    private final RowType rowType;
    private final long estimatedRows;

    public GpuSortSpec(int sortField, RowType rowType, long estimatedRows) {
        this.sortField = sortField;
        this.rowType = rowType;
        this.estimatedRows = estimatedRows;
    }

    /** The field to order by. An {@code INT NOT NULL}; see {@link #canHold}. */
    public int sortField() {
        return sortField;
    }

    /** The row held, which is both the input row and the output row: a sort does not change it. */
    public RowType rowType() {
        return rowType;
    }

    /** What the planner expected this subtask to see. The operator sizes itself from it. */
    public long estimatedRows() {
        return estimatedRows;
    }

    /**
     * Why this sort cannot be served here, or null when it can.
     *
     * <p>Every reason is the binding's shape rather than a device's. cuDF orders far more than
     * this; {@code Cudf.sortedOrder} is one entry point over an {@code INT32} column with no null
     * mask, and widening it is more shim rather than a different design. What is deliberately
     * <em>not</em> a reason is the payload's device-expressibility: no column but the key ever
     * reaches the device, so a row carrying a {@code BIGINT} — which the kernel generator refuses
     * outright — is sorted here perfectly well.
     *
     * <p>A method rather than a sequence of {@code if}s inside the provider because that is what
     * makes it testable: the provider declines everything before this on a host with no cuDF shim,
     * so refusals checked only there would be checked nowhere.
     *
     * @param estimatedRows the planner's cardinality estimate, or {@link
     *     org.apache.flink.table.accelerator.AccelWorkProfile#UNKNOWN_ROWS}
     * @param maxBytes the largest partition this provider will undertake to hold
     */
    public static @Nullable String refuse(AccelSort sort, long estimatedRows, long maxBytes) {
        if (!sort.ascending()) {
            // Reversing the permutation on the host is not the fix it looks like: it reverses runs
            // of equal keys too, and a stable sort that loses stability is a wrong answer for a
            // query that sorts twice.
            return "the binding orders ascending only";
        }
        RowType row = sort.outputType();
        LogicalType keyType = row.getTypeAt(sort.sortField());
        if (keyType.getTypeRoot() != LogicalTypeRoot.INTEGER) {
            return "the key is " + keyType + ", and the binding orders INT";
        }
        if (keyType.isNullable()) {
            // The shim builds a column view with no null mask, so a null key would order as
            // whatever its bits happen to be -- silently. NOT NULL in the DDL is the sanctioned
            // answer here as it is for a Calc, and AccelSort.nullsLast() is not consulted.
            return "a nullable key has no null mask in the binding";
        }
        if (!canHold(row)) {
            return row + " cannot be staged for a whole partition";
        }
        if (estimatedRows <= 0) {
            return "no cardinality estimate, and a sort holds its whole input";
        }
        long bytes = estimatedRows * bytesPerRow(row);
        if (bytes > maxBytes) {
            return estimatedRows
                    + " rows is "
                    + bytes
                    + " bytes of staging, over the "
                    + maxBytes
                    + " this provider will hold";
        }
        return null;
    }

    /**
     * Whether every column of this row can be staged for the life of a partition.
     *
     * <p>Fixed-width numerics only. The refusal is deliberately about storage rather than about
     * arithmetic — {@code BIGINT} is here and is not something the kernel generator admits.
     */
    public static boolean canHold(RowType rowType) {
        if (rowType.getFieldCount() == 0 || rowType.getFieldCount() > MAX_FIELDS) {
            return false;
        }
        for (LogicalType field : rowType.getChildren()) {
            if (widthOf(field) == 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Bytes one row costs in this operator's staging, everything included.
     *
     * <p>The key is counted twice on purpose: once as the column cuDF reads and once as the
     * permutation it writes back, both four bytes a row and both resident for the whole partition.
     * A sizing decision made without the permutation would be a fifth short on a two-column row.
     */
    public static int bytesPerRow(RowType rowType) {
        int bytes = 4; // the permutation
        boolean anyNullable = false;
        for (LogicalType field : rowType.getChildren()) {
            bytes += widthOf(field);
            anyNullable |= field.isNullable();
        }
        return bytes + (anyNullable ? 4 : 0);
    }

    /** Bytes one value of this type occupies, or 0 for a type this operator cannot hold. */
    public static int widthOf(LogicalType type) {
        LogicalTypeRoot root = type.getTypeRoot();
        if (root == LogicalTypeRoot.INTEGER || root == LogicalTypeRoot.FLOAT) {
            return 4;
        }
        if (root == LogicalTypeRoot.BIGINT || root == LogicalTypeRoot.DOUBLE) {
            return 8;
        }
        return 0;
    }

    @Override
    public String toString() {
        return "GpuSortSpec[f" + sortField + " ASC, " + rowType + ", ~" + estimatedRows + " rows]";
    }
}
