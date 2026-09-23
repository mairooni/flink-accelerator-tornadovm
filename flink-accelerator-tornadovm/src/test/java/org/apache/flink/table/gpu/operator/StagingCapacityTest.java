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

package org.apache.flink.table.gpu.operator;

import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.util.OneInputStreamOperatorTestHarness;
import org.apache.flink.streaming.util.TwoInputStreamOperatorTestHarness;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.codegen.GpuJoinSpec;
import org.apache.flink.table.gpu.codegen.GpuSortSpec;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.fail;

/**
 * What happens when the planner's estimate was low, which is the case these operators exist inside.
 *
 * <p>A sort and a join each undertake at {@code open()} to hold a whole input, and size that
 * undertaking against a staging reservation. Nothing enforces the estimate they sized it from. The
 * behaviour under a miss used to be: materialise what is staged onto the ordinary heap and keep
 * going, which is correct and bounded by nothing — a miss large enough became a TaskManager OOM
 * rather than a slow query.
 *
 * <p>These pin the replacement. Overflow fails the task with a type that names the cause, so the
 * bound holds at every point. The retry runs on Flink's own operator, which spills, because Flink
 * does not offer a retried attempt to an accelerator.
 *
 * <p>No device is involved and none is needed: staging is host memory and the failure happens on
 * the arrival path, before anything is dispatched. That is deliberate — a test that could only run
 * on a machine with a card would not run in the CI that is meant to protect this.
 */
class StagingCapacityTest {

    private static final RowType TWO_COLUMNS =
            RowType.of(new IntType(false), new DoubleType(false));

    private static final RowType FOUR_COLUMNS =
            RowType.of(
                    new IntType(false),
                    new DoubleType(false),
                    new IntType(false),
                    new DoubleType(false));

    private static RowData row(int key, double payload) {
        return GenericRowData.of(key, payload);
    }

    private static GpuSortOperator sort(long estimatedRows, int capacity) {
        return new GpuSortOperator(new GpuSortSpec(0, TWO_COLUMNS, estimatedRows), capacity, null);
    }

    private static GpuJoinOperator join(long estimatedBuildRows, int buildCapacity) {
        return new GpuJoinOperator(
                new GpuJoinSpec(
                        0, 0, true, estimatedBuildRows, TWO_COLUMNS, TWO_COLUMNS, FOUR_COLUMNS),
                buildCapacity,
                16,
                null);
    }

    @Test
    @DisplayName("a sort given more rows than it undertook to hold fails, and says so")
    void sortOverflowFailsRatherThanBuffering() throws Exception {
        // Estimated four, staging holds four, five arrive. The estimate is what open() checks
        // against, so this is the "statistic was wrong" case and not a misconfiguration.
        try (OneInputStreamOperatorTestHarness<RowData, RowData> harness =
                new OneInputStreamOperatorTestHarness<>(sort(4L, 4))) {
            harness.setup();
            harness.open();
            for (int i = 0; i < 4; i++) {
                harness.processElement(new StreamRecord<>(row(i, i)));
            }
            assertThatThrownBy(() -> harness.processElement(new StreamRecord<>(row(4, 4.0))))
                    .isInstanceOf(StagingCapacityExceededException.class)
                    .hasMessageContaining("more rows than it undertook to hold")
                    .hasMessageContaining("will run on the operator that spills");
        }
    }

    @Test
    @DisplayName("a sort filled exactly to its capacity is not an overflow")
    void sortAtCapacityIsFine() throws Exception {
        try (OneInputStreamOperatorTestHarness<RowData, RowData> harness =
                new OneInputStreamOperatorTestHarness<>(sort(4L, 4))) {
            harness.setup();
            harness.open();
            for (int i = 0; i < 4; i++) {
                harness.processElement(new StreamRecord<>(row(i, i)));
            }
            // Not drained: draining orders on the device, which this test deliberately cannot do.
            // The boundary under test is the last row that fits, not what becomes of it.
        }
    }

    @Test
    @DisplayName("the failure carries the two numbers that disagreed")
    void theFailureNamesTheEstimateAndTheCapacity() throws Exception {
        try (OneInputStreamOperatorTestHarness<RowData, RowData> harness =
                new OneInputStreamOperatorTestHarness<>(sort(2L, 2))) {
            harness.setup();
            harness.open();
            harness.processElement(new StreamRecord<>(row(0, 0.0)));
            harness.processElement(new StreamRecord<>(row(1, 1.0)));
            try {
                harness.processElement(new StreamRecord<>(row(2, 2.0)));
                fail("the third row should not have been accepted");
            } catch (StagingCapacityExceededException expected) {
                assertThat(expected.estimatedRows()).isEqualTo(2L);
                assertThat(expected.capacity()).isEqualTo(2L);
            }
        }
    }

    @Test
    @DisplayName("a sort refuses at open() when the estimate alone already exceeds the staging")
    void sortRefusesBeforeConsumingAnything() throws Exception {
        // The recoverable refusal, and the one that must keep working: FallbackToCpuOperator turns
        // a failure in open() into the generated operator, because nothing has been consumed yet.
        try (OneInputStreamOperatorTestHarness<RowData, RowData> harness =
                new OneInputStreamOperatorTestHarness<>(sort(1_000L, 10))) {
            harness.setup();
            assertThatThrownBy(harness::open)
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("declining rather than spilling");
        }
    }

    @Test
    @DisplayName("a join given a larger build side than it undertook to hold fails, and says so")
    void joinBuildOverflowFailsRatherThanBuffering() throws Exception {
        try (TwoInputStreamOperatorTestHarness<RowData, RowData, RowData> harness =
                new TwoInputStreamOperatorTestHarness<>(join(4L, 4))) {
            harness.setup();
            harness.open();
            for (int i = 0; i < 4; i++) {
                harness.processElement1(new StreamRecord<>(row(i, i)));
            }
            assertThatThrownBy(() -> harness.processElement1(new StreamRecord<>(row(4, 4.0))))
                    .isInstanceOf(StagingCapacityExceededException.class)
                    .hasMessageContaining("larger build side than it undertook to hold");
        }
    }

    @Test
    @DisplayName("a join refuses at open() when the estimated build side already exceeds it")
    void joinRefusesBeforeConsumingAnything() throws Exception {
        try (TwoInputStreamOperatorTestHarness<RowData, RowData, RowData> harness =
                new TwoInputStreamOperatorTestHarness<>(join(1_000L, 10))) {
            harness.setup();
            assertThatThrownBy(harness::open)
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("declining rather than spilling");
        }
    }
}
