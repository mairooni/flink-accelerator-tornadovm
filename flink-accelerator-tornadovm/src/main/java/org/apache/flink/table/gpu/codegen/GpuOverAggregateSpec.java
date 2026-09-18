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

import org.apache.flink.table.accelerator.AccelOverAggregate;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;
import org.apache.flink.table.types.logical.RowType;

import javax.annotation.Nullable;

import java.io.Serializable;

/**
 * Everything the runtime needs to run one running total on a device.
 *
 * <p>The smallest spec here, and it has no cardinality in it at all. {@link GpuSortSpec} and {@link
 * GpuJoinSpec} both carry an estimate because both undertake to hold something; a prefix sum holds
 * one {@code double}, so there is no size it stops fitting at and nothing to refuse on.
 */
public final class GpuOverAggregateSpec implements Serializable {

    private static final long serialVersionUID = 1L;

    private final int valueField;
    private final RowType inputType;
    private final RowType outputType;

    public GpuOverAggregateSpec(int valueField, RowType inputType, RowType outputType) {
        this.valueField = valueField;
        this.inputType = inputType;
        this.outputType = outputType;
    }

    /** The input field being accumulated. */
    public int valueField() {
        return valueField;
    }

    public RowType inputType() {
        return inputType;
    }

    /** The input row with the running total appended. */
    public RowType outputType() {
        return outputType;
    }

    /**
     * Why this window cannot be served here, or null when it can.
     *
     * <p>Short, because the planner has already refused every frame, partition and function this
     * cannot express — what is left is whether the columns can be staged and moved. Checked here
     * anyway, and testably, for the reason {@link GpuSortSpec#refuse} is: the provider declines
     * everything before this on a host with no cuDF shim.
     */
    public static @Nullable String refuse(AccelOverAggregate over) {
        RowType inputType = over.inputs().get(0).outputType();
        LogicalType valueType = inputType.getTypeAt(over.valueField());
        if (valueType.getTypeRoot() != LogicalTypeRoot.DOUBLE) {
            return "the value is " + valueType + ", and the binding sums DOUBLE";
        }
        if (valueType.isNullable()) {
            // A null in a running total poisons every row after it, which is worse than a wrong
            // row: it is a wrong suffix. NOT NULL in the DDL is the sanctioned answer.
            return "a nullable value has no null mask in the binding";
        }
        if (!canHold(inputType)) {
            return inputType + " cannot be staged for a batch";
        }
        if (over.outputType().getFieldCount() != inputType.getFieldCount() + 1) {
            // The node's contract is the input row plus one appended column. A planner that built
            // something else would produce rows this operator would fill wrongly rather than fail.
            return "the output row is not the input row plus one running total";
        }
        LogicalType appended = over.outputType().getTypeAt(over.outputType().getFieldCount() - 1);
        if (appended.getTypeRoot() != LogicalTypeRoot.DOUBLE) {
            return "the running total column is " + appended + ", not DOUBLE";
        }
        return null;
    }

    /** Whether every column of this row can be staged at a fixed width. */
    public static boolean canHold(RowType rowType) {
        if (rowType.getFieldCount() == 0 || rowType.getFieldCount() > 32) {
            return false;
        }
        for (LogicalType field : rowType.getChildren()) {
            if (widthOf(field) == 0) {
                return false;
            }
        }
        return true;
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
        return "GpuOverAggregateSpec[SUM(f" + valueField + ") ROWS UNBOUNDED PRECEDING]";
    }
}
