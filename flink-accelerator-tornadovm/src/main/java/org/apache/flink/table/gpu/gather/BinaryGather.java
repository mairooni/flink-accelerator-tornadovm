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
import org.apache.flink.table.gpu.codegen.GpuValueType;

/**
 * Typed access rather than the {@link RowData} interface: identical result, one less virtual call.
 *
 * <p>The accessor is chosen by the column's declared type. Reading a four-byte field with {@code
 * getDouble} does not fail, it returns eight bytes of whatever is there, so this switch is the
 * difference between a value and garbage.
 */
final class BinaryGather implements RowGather {

    private final int field;
    private final GpuValueType type;
    private final StagingColumn target;

    BinaryGather(int field, GpuValueType type, StagingColumn target) {
        this.field = field;
        this.type = type;
        this.target = target;
    }

    @Override
    public void accept(RowData row, int position) {
        switch (type) {
            case INT:
                target.set(position, ((BinaryRowData) row).getInt(field));
                break;
            case FLOAT:
                target.set(position, ((BinaryRowData) row).getFloat(field));
                break;
            default:
                target.set(position, ((BinaryRowData) row).getDouble(field));
                break;
        }
    }

    @Override
    public String tier() {
        return "tier2-binary";
    }
}
