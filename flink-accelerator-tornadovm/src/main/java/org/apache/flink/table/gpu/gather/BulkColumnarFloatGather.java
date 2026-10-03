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
import org.apache.flink.table.data.columnar.vector.ColumnVector;
import org.apache.flink.table.data.columnar.vector.heap.HeapFloatVector;

import javax.annotation.Nullable;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * The bulk columnar gather for a {@code FLOAT} column. See {@link AbstractBulkColumnarGather} for
 * the run tracking, which is where everything subtle lives.
 */
public final class BulkColumnarFloatGather extends AbstractBulkColumnarGather {

    public BulkColumnarFloatGather(
            int field, RowGather.StagingColumn target, MemorySegment targetSegment) {
        super(field, target, targetSegment);
    }

    @Override
    protected @Nullable ColumnVector copyableVector(ColumnVector column) {
        return column instanceof HeapFloatVector vector
                        && !vector.hasDictionary()
                        && !vector.hasNulls()
                ? vector
                : null;
    }

    @Override
    protected void copyRun(ColumnVector vector, int fromRowId, int position, int length) {
        MemorySegment.copy(
                ((HeapFloatVector) vector).vector,
                fromRowId,
                targetSegment,
                ValueLayout.JAVA_FLOAT,
                (long) position * Float.BYTES,
                length);
    }

    @Override
    protected double valueAt(RowData row, int field) {
        return row.getFloat(field);
    }
}
