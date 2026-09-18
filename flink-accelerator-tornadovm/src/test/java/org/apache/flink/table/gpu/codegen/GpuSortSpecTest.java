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

import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelSort;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.table.types.logical.VarCharType;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * What this provider will and will not order, checked where it can be.
 *
 * <p>Not through {@code TornadoVmAcceleratorProvider.accept}, deliberately: that returns empty
 * before any of this is reached on a host with no {@code libtornado-cudf.so}, which is nearly every
 * host including this one. A refusal checked only there would be checked nowhere.
 */
class GpuSortSpecTest {

    private static final long ONE_GIB = 1L << 30;

    private static RowType row(LogicalType... fields) {
        return RowType.of(fields);
    }

    private static AccelSort sort(int field, boolean ascending, RowType type) {
        return new AccelSort(field, ascending, true, new AccelInput(type), type);
    }

    @Test
    void acceptsAnIntKeyOverFixedWidthPayload() {
        RowType type = row(new IntType(false), new DoubleType(false), new BigIntType(false));
        assertThat(GpuSortSpec.refuse(sort(0, true, type), 1_000_000, ONE_GIB)).isNull();
    }

    @Test
    void aBigintPayloadIsHeldThoughNoKernelCouldComputeIt() {
        // The point of the permutation design: expressibility is a question about the key only.
        RowType type = row(new IntType(false), new BigIntType(false));
        assertThat(GpuSortSpec.canHold(type)).isTrue();
    }

    @Test
    void refusesDescending() {
        RowType type = row(new IntType(false));
        assertThat(GpuSortSpec.refuse(sort(0, false, type), 1000, ONE_GIB))
                .isEqualTo("the binding orders ascending only");
    }

    @Test
    void refusesANonIntKey() {
        RowType type = row(new BigIntType(false), new IntType(false));
        assertThat(GpuSortSpec.refuse(sort(0, true, type), 1000, ONE_GIB))
                .contains("the binding orders INT");
    }

    @Test
    void refusesANullableKey() {
        RowType type = row(new IntType(true));
        assertThat(GpuSortSpec.refuse(sort(0, true, type), 1000, ONE_GIB))
                .isEqualTo("a nullable key has no null mask in the binding");
    }

    @Test
    void refusesAPayloadItCannotHold() {
        RowType type = row(new IntType(false), new VarCharType(false, 32));
        assertThat(GpuSortSpec.refuse(sort(0, true, type), 1000, ONE_GIB))
                .contains("cannot be staged for a whole partition");
    }

    @Test
    void refusesWhenNothingEstimatedTheInput() {
        RowType type = row(new IntType(false));
        assertThat(GpuSortSpec.refuse(sort(0, true, type), -1L, ONE_GIB))
                .isEqualTo("no cardinality estimate, and a sort holds its whole input");
    }

    @Test
    void refusesAPartitionLargerThanItWillHold() {
        RowType type = row(new IntType(false), new DoubleType(false));
        // 16 bytes a row: key, permutation and one double.
        assertThat(GpuSortSpec.bytesPerRow(type)).isEqualTo(16);
        assertThat(GpuSortSpec.refuse(sort(0, true, type), 100, 1000L))
                .contains("over the 1000 this provider will hold");
        assertThat(GpuSortSpec.refuse(sort(0, true, type), 60, 1000L)).isNull();
    }

    @Test
    void countsTheValidityWordOnlyWhenSomethingIsNullable() {
        assertThat(GpuSortSpec.bytesPerRow(row(new IntType(false), new IntType(false))))
                .isEqualTo(12);
        assertThat(GpuSortSpec.bytesPerRow(row(new IntType(false), new IntType(true))))
                .isEqualTo(16);
    }

    @Test
    void refusesARowWiderThanTheValidityWord() {
        LogicalType[] fields = new LogicalType[GpuSortSpec.MAX_FIELDS + 1];
        for (int i = 0; i < fields.length; i++) {
            fields[i] = new IntType(false);
        }
        assertThat(GpuSortSpec.canHold(row(fields))).isFalse();
    }
}
