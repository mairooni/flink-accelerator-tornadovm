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

import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.writer.BinaryRowWriter;
import org.apache.flink.table.gpu.codegen.GpuValueType;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * What a gather reads out of a column that is not {@code DOUBLE}.
 *
 * <p>{@link org.apache.flink.table.planner.plan.gpu.GpuKernelGenerator} admits {@code INTEGER},
 * {@code FLOAT} and {@code REAL} columns as kernel inputs on the grounds that each is exact in a
 * double. That is true of the values; it says nothing about how they are fetched.
 */
class RowGatherTypeTest {

    private static double gather(RowData row, int field, GpuValueType type) {
        double[] landed = new double[1];
        RowGather.forColumn(row, field, type, (position, value) -> landed[position] = value)
                .accept(row, 0);
        return landed[0];
    }

    private static BinaryRowData binaryRow() {
        BinaryRowData row = new BinaryRowData(3);
        BinaryRowWriter writer = new BinaryRowWriter(row);
        writer.writeInt(0, 42);
        writer.writeFloat(1, 1.5f);
        writer.writeDouble(2, 3.25);
        writer.complete();
        return row;
    }

    @Test
    void readsADoubleColumn() {
        assertThat(gather(binaryRow(), 2, GpuValueType.DOUBLE)).isEqualTo(3.25);
        assertThat(gather(GenericRowData.of(42, 1.5f, 3.25), 2, GpuValueType.DOUBLE))
                .isEqualTo(3.25);
    }

    /** Staged as a double before this was typed, which read eight bytes and returned 2.08e-322. */
    @Test
    void readsAnIntegerColumn() {
        assertThat(gather(binaryRow(), 0, GpuValueType.INT)).isEqualTo(42.0);
        assertThat(gather(GenericRowData.of(42, 1.5f, 3.25), 0, GpuValueType.INT)).isEqualTo(42.0);
    }

    /** Likewise: 1.5f came back as 5.28e-315. */
    @Test
    void readsAFloatColumn() {
        assertThat(gather(binaryRow(), 1, GpuValueType.FLOAT)).isEqualTo(1.5);
        assertThat(gather(GenericRowData.of(42, 1.5f, 3.25), 1, GpuValueType.FLOAT)).isEqualTo(1.5);
    }

    /** A binary row is what arrives downstream of a shuffle; a generic one when chained. */
    @Test
    void picksATierFromTheRowImplementation() {
        assertThat(
                        RowGather.forColumn(
                                        binaryRow(), 0, GpuValueType.INT, (position, value) -> {})
                                .tier())
                .isEqualTo("tier2-binary");
        assertThat(
                        RowGather.forColumn(
                                        GenericRowData.of(42),
                                        0,
                                        GpuValueType.INT,
                                        (position, value) -> {})
                                .tier())
                .isEqualTo("tier4-generic");
    }
}
