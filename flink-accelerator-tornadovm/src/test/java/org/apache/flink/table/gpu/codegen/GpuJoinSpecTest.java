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
import org.apache.flink.table.accelerator.AccelJoin;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.table.types.logical.VarCharType;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * What this provider will and will not join, checked where it can be.
 *
 * <p>Not through {@code TornadoVmAcceleratorProvider.accept}, for the reason {@link
 * GpuSortSpecTest} records: that returns empty before any of this is reached on a host with no
 * {@code libtornado-cudf.so}.
 */
class GpuJoinSpecTest {

    private static final long ONE_GIB = 1L << 30;

    private static RowType row(LogicalType... fields) {
        return RowType.of(fields);
    }

    private static AccelJoin join(RowType buildType, RowType probeType, long buildRows) {
        RowType output =
                RowType.of(
                        java.util.stream.Stream.concat(
                                        buildType.getChildren().stream(),
                                        probeType.getChildren().stream())
                                .toArray(LogicalType[]::new));
        return new AccelJoin(
                0,
                0,
                true,
                buildRows,
                new AccelInput(buildType),
                new AccelInput(probeType),
                output);
    }

    @Test
    void acceptsIntKeysOverFixedWidthPayload() {
        RowType build = row(new IntType(false), new BigIntType(false));
        RowType probe = row(new IntType(false), new DoubleType(false));
        assertThat(GpuJoinSpec.refuse(join(build, probe, 1_000_000), ONE_GIB)).isNull();
    }

    @Test
    void refusesANonIntKey() {
        RowType build = row(new BigIntType(false), new IntType(false));
        RowType probe = row(new IntType(false));
        assertThat(GpuJoinSpec.refuse(join(build, probe, 1000), ONE_GIB))
                .contains("the binding joins INT");
    }

    @Test
    void refusesANullableKey() {
        // SQL says a null key matches nothing; a column with no null mask would match it to every
        // other null, which is a wrong answer rather than a slow one.
        RowType build = row(new IntType(true));
        RowType probe = row(new IntType(false));
        assertThat(GpuJoinSpec.refuse(join(build, probe, 1000), ONE_GIB))
                .isEqualTo("a nullable join key has no null mask in the binding");
    }

    @Test
    void refusesABuildRowItCannotHold() {
        RowType build = row(new IntType(false), new VarCharType(false, 32));
        RowType probe = row(new IntType(false));
        assertThat(GpuJoinSpec.refuse(join(build, probe, 1000), ONE_GIB))
                .contains("cannot be staged for a whole build side");
    }

    @Test
    void refusesAProbeRowItCannotHold() {
        RowType build = row(new IntType(false));
        RowType probe = row(new IntType(false), new VarCharType(false, 32));
        assertThat(GpuJoinSpec.refuse(join(build, probe, 1000), ONE_GIB))
                .contains("cannot be staged for a probe batch");
    }

    @Test
    void refusesWhenNothingEstimatedTheBuildSide() {
        RowType build = row(new IntType(false));
        RowType probe = row(new IntType(false));
        assertThat(GpuJoinSpec.refuse(join(build, probe, AccelJoin.UNKNOWN_ROWS), ONE_GIB))
                .isEqualTo("no build-side estimate, and a join holds its whole build side");
    }

    @Test
    void refusesABuildSideLargerThanItWillHold() {
        RowType build = row(new IntType(false), new DoubleType(false));
        RowType probe = row(new IntType(false));
        // 20 bytes a row: the key, the ordered keys, the permutation and one double.
        assertThat(GpuJoinSpec.bytesPerRow(build)).isEqualTo(20);
        assertThat(GpuJoinSpec.refuse(join(build, probe, 100), 1000L))
                .contains("over the 1000 this provider will hold");
        assertThat(GpuJoinSpec.refuse(join(build, probe, 40), 1000L)).isNull();
    }

    @Test
    void countsTheValidityWordOnlyWhenSomethingIsNullable() {
        assertThat(GpuJoinSpec.bytesPerRow(row(new IntType(false), new IntType(false))))
                .isEqualTo(16);
        assertThat(GpuJoinSpec.bytesPerRow(row(new IntType(false), new IntType(true))))
                .isEqualTo(20);
    }

    @Test
    void refusesARowWiderThanTheValidityWord() {
        LogicalType[] fields = new LogicalType[GpuJoinSpec.MAX_FIELDS + 1];
        Arrays.fill(fields, new IntType(false));
        assertThat(GpuJoinSpec.canHold(row(fields))).isFalse();
    }
}
