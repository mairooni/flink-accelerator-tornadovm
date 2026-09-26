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
import org.apache.flink.table.gpu.codegen.GpuValueType;

/**
 * Any {@link RowData}: the interface accessors, which for a boxed row unwrap an Object.
 *
 * <p>The accessor is chosen by the column's declared type. Reading a four-byte field with {@code
 * getDouble} does not fail, it returns eight bytes of whatever is there, so this switch is the
 * difference between a value and garbage.
 */
final class GenericGather implements RowGather {

    /** The concrete row class, recorded once so the tier string can name it. */
    private String seen;

    private final int field;
    private final GpuValueType type;
    private final StagingColumn target;

    GenericGather(int field, GpuValueType type, StagingColumn target) {
        this.field = field;
        this.type = type;
        this.target = target;
    }

    @Override
    public void accept(RowData row, int position) {
        if (seen == null) {
            seen = row.getClass().getSimpleName();
        }
        switch (type) {
            case INT:
                target.set(position, row.getInt(field));
                break;
            case FLOAT:
                target.set(position, row.getFloat(field));
                break;
            default:
                target.set(position, row.getDouble(field));
                break;
        }
    }

    @Override
    public String tier() {
        // Names the class, because "generic" only says which gather was chosen and the useful
        // question is what the plan put upstream. Whether a faster tier could apply is a property
        // of that class, and guessing it from the plan has been wrong before.
        return "tier4-generic(" + (seen == null ? "?" : seen) + ")";
    }
}
