/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding ownership.  The ASF licenses this file
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

import org.apache.flink.table.types.logical.RowType;

import java.io.Serializable;

/**
 * A {@code SUM} grouped by one key, run by cuDF over the columns a generated kernel just wrote.
 *
 * <p>Two field indices and a row type is all this needs to be, because the projection underneath is
 * already described by a {@link GpuCalcSpec}: {@link #keyField()} and {@link #valueField()} name
 * columns of <em>that</em> spec's output row, and the buffers holding them are the kernel's own. No
 * copy sits between the two stages -- the whole point of putting them in one task graph -- so the
 * aggregate is addressed by field index rather than by a buffer it would otherwise have to own.
 *
 * <h2>Why one key and one sum</h2>
 *
 * <p>Because that is what the cuDF binding exposes ({@code Cudf.groupSum}), and widening it is a
 * matter of more shim entry points rather than a different design here. A query with two grouping
 * columns or a second aggregate is declined by the provider and runs on the CPU, which is the same
 * answer it gets today.
 *
 * <h2>Why the operator may emit several rows per key</h2>
 *
 * <p>It is a <em>partial</em> aggregate. Flink splits {@code GROUP BY} into a local aggregate, a
 * shuffle, and a merge, and only the local half is offered here; the merge stage downstream
 * combines whatever this emits. That is what makes the operator stateless across batches -- stage a
 * batch, group it, emit its groups, forget it -- and it is the reason a device-side hash table
 * spanning the whole partition is not needed to be correct.
 */
public final class GpuAggregateSpec implements Serializable {

    private static final long serialVersionUID = 1L;

    private final int keyField;
    private final int valueField;
    private final RowType projectionType;
    private final RowType outputType;

    public GpuAggregateSpec(
            int keyField, int valueField, RowType projectionType, RowType outputType) {
        if (outputType.getFieldCount() != 2) {
            throw new IllegalArgumentException(
                    "a grouped aggregate emits (key, sum), not " + outputType);
        }
        this.keyField = keyField;
        this.valueField = valueField;
        this.projectionType = projectionType;
        this.outputType = outputType;
    }

    /**
     * The row the projection produces, which is this aggregate's input and not what it emits.
     *
     * <p>Carried because the operator builds its {@link GpuCalcSpec} on the TaskManager, where
     * Flink offers the node's <em>output</em> type -- one row a group -- and the staging needs the
     * other one.
     */
    public RowType projectionType() {
        return projectionType;
    }

    /** The grouping column, as a field of the projection's output row. */
    public int keyField() {
        return keyField;
    }

    /** The summed column, as a field of the projection's output row. */
    public int valueField() {
        return valueField;
    }

    /** What this operator emits: the key, then the partial sum. */
    public RowType outputType() {
        return outputType;
    }

    @Override
    public String toString() {
        return "GpuAggregateSpec{SUM(f" + valueField + ") GROUP BY f" + keyField + "}";
    }
}
