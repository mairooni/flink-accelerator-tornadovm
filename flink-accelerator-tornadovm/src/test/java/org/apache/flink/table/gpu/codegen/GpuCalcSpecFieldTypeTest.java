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

import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Which device buffer holds which output field.
 *
 * <p>Worth its own test because the answer is silent when it is wrong. A later stage in the same
 * task graph -- cuDF's group-by is the first -- reads an output field by asking this, and an
 * off-by-one between the computed slots and the staged columns names a real buffer holding real
 * numbers. Nothing throws; the sums are just wrong.
 */
class GpuCalcSpecFieldTypeTest {

    private static final int OUT = GpuCalcSpec.COMPUTED;

    /**
     * A kernel staging input fields 3 and 7, and computing two columns.
     *
     * <p>The layout interleaves them -- computed, copied, computed, copied -- because that is the
     * arrangement a naive "count the fields before me" gets wrong in both directions.
     */
    private static GpuCalcSpec spec() {
        GpuKernelSource kernel =
                new GpuKernelSource(
                        "K",
                        "evaluate",
                        "",
                        new int[] {3, 7},
                        new GpuValueType[] {GpuValueType.INT, GpuValueType.DOUBLE},
                        new GpuValueType[] {GpuValueType.DOUBLE, GpuValueType.INT},
                        false,
                        new int[] {OUT, 3, OUT, 7});
        return new GpuCalcSpec(
                kernel,
                new int[] {OUT, 3, OUT, 7},
                RowType.of(new DoubleType(), new IntType(), new IntType(), new DoubleType()),
                1024);
    }

    @Test
    @DisplayName("computed fields resolve to the kernel outputs in order")
    void computedFieldsTakeTheirSlotInOrder() {
        assertThat(spec().computedSlot(0)).isZero();
        assertThat(spec().computedSlot(2)).isEqualTo(1);
        assertThat(spec().fieldType(0)).isEqualTo(GpuValueType.DOUBLE);
        assertThat(spec().fieldType(2)).isEqualTo(GpuValueType.INT);
    }

    @Test
    @DisplayName("a copied field resolves to the staged column, not to the input field index")
    void copiedFieldsResolveToTheStagedColumn() {
        assertThat(spec().computedSlot(1)).isEqualTo(-1);
        // Input field 3 is staged column 0 and input field 7 is staged column 1. Reading the
        // layout value as a column index would give buffers 3 and 7, which do not exist.
        assertThat(spec().stagedColumn(1)).isZero();
        assertThat(spec().stagedColumn(3)).isEqualTo(1);
        assertThat(spec().fieldType(1)).isEqualTo(GpuValueType.INT);
        assertThat(spec().fieldType(3)).isEqualTo(GpuValueType.DOUBLE);
    }

    @Test
    @DisplayName("a field copied from a column the kernel never stages is an error, not a guess")
    void anUnstagedSourceIsRejected() {
        GpuKernelSource kernel =
                new GpuKernelSource(
                        "K",
                        "evaluate",
                        "",
                        new int[] {3},
                        new GpuValueType[] {GpuValueType.DOUBLE},
                        new GpuValueType[] {GpuValueType.DOUBLE},
                        false,
                        new int[] {OUT, 9});
        GpuCalcSpec spec =
                new GpuCalcSpec(
                        kernel,
                        new int[] {OUT, 9},
                        RowType.of(new DoubleType(), new DoubleType()),
                        1024);

        assertThatThrownBy(() -> spec.fieldType(1))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("does not stage");
    }
}
