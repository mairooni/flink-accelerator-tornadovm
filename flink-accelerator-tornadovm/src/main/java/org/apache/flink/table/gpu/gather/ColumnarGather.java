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
import org.apache.flink.table.gpu.codegen.GpuValueType;

/**
 * Vectorized batches, read one row at a time; the bulk path is {@link BulkColumnarDoubleGather}.
 *
 * <p>The accessor is chosen by the column's declared type. Reading a four-byte field with {@code
 * getDouble} does not fail, it returns eight bytes of whatever is there, so this switch is the
 * difference between a value and garbage.
 */
final class ColumnarGather implements RowGather {

    private final int field;
    private final GpuValueType type;
    private final StagingColumn target;

    ColumnarGather(int field, GpuValueType type, StagingColumn target) {
        this.field = field;
        this.type = type;
        this.target = target;
    }

    @Override
    public void accept(RowData row, int position) {
        switch (type) {
            case INT:
                target.set(position, ((ColumnarRowData) row).getInt(field));
                break;
            case FLOAT:
                target.set(position, ((ColumnarRowData) row).getFloat(field));
                break;
            default:
                target.set(position, ((ColumnarRowData) row).getDouble(field));
                break;
        }
    }

    @Override
    public String tier() {
        return "tier1-columnar(per-row; bulk path needs accessor patch)";
    }
}
