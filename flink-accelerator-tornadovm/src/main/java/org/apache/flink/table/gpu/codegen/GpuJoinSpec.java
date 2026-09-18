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

import org.apache.flink.table.accelerator.AccelJoin;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;
import org.apache.flink.table.types.logical.RowType;

import javax.annotation.Nullable;

import java.io.Serializable;

/**
 * Everything the runtime needs to join two inputs on a device.
 *
 * <p>{@link GpuSortSpec} with a second side. The refusals sit here for the same reason they do
 * there: the provider declines before reaching any of them on a host with no cuDF shim, which is
 * nearly every host, so a refusal checked only in {@code accept} would be checked nowhere.
 */
public final class GpuJoinSpec implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Fields a row may have. Validity is one {@code int} a row, a bit a column. */
    public static final int MAX_FIELDS = 32;

    private final int buildKeyField;
    private final int probeKeyField;
    private final boolean buildIsLeft;
    private final long estimatedBuildRows;
    private final RowType buildType;
    private final RowType probeType;
    private final RowType outputType;

    public GpuJoinSpec(
            int buildKeyField,
            int probeKeyField,
            boolean buildIsLeft,
            long estimatedBuildRows,
            RowType buildType,
            RowType probeType,
            RowType outputType) {
        this.buildKeyField = buildKeyField;
        this.probeKeyField = probeKeyField;
        this.buildIsLeft = buildIsLeft;
        this.estimatedBuildRows = estimatedBuildRows;
        this.buildType = buildType;
        this.probeType = probeType;
        this.outputType = outputType;
    }

    public int buildKeyField() {
        return buildKeyField;
    }

    public int probeKeyField() {
        return probeKeyField;
    }

    /** Whether the build side's fields come first in the output row. */
    public boolean buildIsLeft() {
        return buildIsLeft;
    }

    /** What the planner expected the build side to hold. The operator sizes itself from it. */
    public long estimatedBuildRows() {
        return estimatedBuildRows;
    }

    public RowType buildType() {
        return buildType;
    }

    public RowType probeType() {
        return probeType;
    }

    public RowType outputType() {
        return outputType;
    }

    /**
     * Why this join cannot be served here, or null when it can.
     *
     * <p>Two of these are about the binding and two are about holding the build side:
     *
     * <ul>
     *   <li><b>{@code NOT NULL INT} keys on both sides.</b> The shim's ordering takes an {@code
     *       INT32} column with no null mask. A null key is not merely unrepresentable — SQL says it
     *       matches nothing, and a column with no mask would match it to every other null.
     *   <li><b>Rows this can hold at a fixed width.</b> Both sides, because both are staged: the
     *       build side for the whole join and the probe side for a batch.
     *   <li><b>A build cardinality within the ceiling.</b> The build side has to be resident, so an
     *       unknown one is a commitment that cannot be made.
     * </ul>
     *
     * @param maxBytes the largest build side this provider will undertake to hold
     */
    public static @Nullable String refuse(AccelJoin join, long maxBytes) {
        RowType buildType = join.inputs().get(0).outputType();
        RowType probeType = join.inputs().get(1).outputType();
        LogicalType buildKey = buildType.getTypeAt(join.buildKeyField());
        LogicalType probeKey = probeType.getTypeAt(join.probeKeyField());
        if (buildKey.getTypeRoot() != LogicalTypeRoot.INTEGER
                || probeKey.getTypeRoot() != LogicalTypeRoot.INTEGER) {
            return "the keys are " + buildKey + " and " + probeKey + ", and the binding joins INT";
        }
        if (buildKey.isNullable() || probeKey.isNullable()) {
            return "a nullable join key has no null mask in the binding";
        }
        if (!canHold(buildType)) {
            return buildType + " cannot be staged for a whole build side";
        }
        if (!canHold(probeType)) {
            return probeType + " cannot be staged for a probe batch";
        }
        long rows = join.estimatedBuildRows();
        if (rows <= 0) {
            return "no build-side estimate, and a join holds its whole build side";
        }
        long bytes = rows * bytesPerRow(buildType);
        if (bytes > maxBytes) {
            return rows
                    + " build rows is "
                    + bytes
                    + " bytes of staging, over the "
                    + maxBytes
                    + " this provider will hold";
        }
        return null;
    }

    /** Whether every column of this row can be staged at a fixed width. */
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
     * Bytes one row costs in staging, everything included.
     *
     * <p>The build side also carries the ordered key column and the permutation, four bytes each,
     * which is why this counts eight more than the row itself.
     */
    public static int bytesPerRow(RowType rowType) {
        int bytes = 8; // the ordered keys and the permutation
        boolean anyNullable = false;
        for (LogicalType field : rowType.getChildren()) {
            bytes += widthOf(field);
            anyNullable |= field.isNullable();
        }
        return bytes + (anyNullable ? 4 : 0);
    }

    /** Bytes one value of this type occupies, or 0 for a type this cannot hold. */
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
        return "GpuJoinSpec[build.f"
                + buildKeyField
                + " = probe.f"
                + probeKeyField
                + ", build="
                + (buildIsLeft ? "left" : "right")
                + ", ~"
                + estimatedBuildRows
                + " build rows]";
    }
}
