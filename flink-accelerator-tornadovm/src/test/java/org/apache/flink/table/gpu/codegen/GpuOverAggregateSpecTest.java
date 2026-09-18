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
import org.apache.flink.table.accelerator.AccelOverAggregate;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.table.types.logical.VarCharType;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * What this provider will and will not accumulate, checked where it can be.
 *
 * <p>Not through {@code accept}, for the reason {@link GpuSortSpecTest} records: that returns empty
 * before any of this is reached on a host with no {@code libtornado-cudf.so}.
 */
class GpuOverAggregateSpecTest {

    private static AccelOverAggregate over(int valueField, RowType input, RowType output) {
        return new AccelOverAggregate(valueField, new AccelInput(input), output);
    }

    private static RowType row(LogicalType... fields) {
        return RowType.of(fields);
    }

    @Test
    void acceptsADoubleValueOverFixedWidthColumns() {
        RowType in = row(new IntType(false), new BigIntType(false), new DoubleType(false));
        RowType out =
                row(
                        new IntType(false),
                        new BigIntType(false),
                        new DoubleType(false),
                        new DoubleType(false));
        assertThat(GpuOverAggregateSpec.refuse(over(2, in, out))).isNull();
    }

    @Test
    void refusesANonDoubleValue() {
        RowType in = row(new BigIntType(false));
        RowType out = row(new BigIntType(false), new DoubleType(false));
        assertThat(GpuOverAggregateSpec.refuse(over(0, in, out)))
                .contains("the binding sums DOUBLE");
    }

    @Test
    void refusesANullableValue() {
        // A null does not give one wrong row here, it gives a wrong suffix.
        RowType in = row(new DoubleType(true));
        RowType out = row(new DoubleType(true), new DoubleType(false));
        assertThat(GpuOverAggregateSpec.refuse(over(0, in, out)))
                .isEqualTo("a nullable value has no null mask in the binding");
    }

    @Test
    void refusesARowItCannotStage() {
        RowType in = row(new DoubleType(false), new VarCharType(false, 32));
        RowType out = row(new DoubleType(false), new VarCharType(false, 32), new DoubleType(false));
        assertThat(GpuOverAggregateSpec.refuse(over(0, in, out)))
                .contains("cannot be staged for a batch");
    }

    @Test
    void refusesAnOutputThatIsNotTheInputPlusOne() {
        RowType in = row(new DoubleType(false));
        assertThat(GpuOverAggregateSpec.refuse(over(0, in, row(new DoubleType(false)))))
                .isEqualTo("the output row is not the input row plus one running total");
    }

    @Test
    void refusesARunningTotalColumnThatIsNotADouble() {
        RowType in = row(new DoubleType(false));
        RowType out = row(new DoubleType(false), new BigIntType(false));
        assertThat(GpuOverAggregateSpec.refuse(over(0, in, out)))
                .contains("the running total column is");
    }
}
