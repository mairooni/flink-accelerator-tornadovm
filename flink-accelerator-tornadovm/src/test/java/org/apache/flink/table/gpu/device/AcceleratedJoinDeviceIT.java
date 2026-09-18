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

package org.apache.flink.table.gpu.device;

import org.apache.flink.streaming.api.operators.SimpleOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.util.TwoInputStreamOperatorTestHarness;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelJoin;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.operator.GpuJoinOperator;
import org.apache.flink.table.runtime.accelerator.AcceleratorContext;
import org.apache.flink.table.runtime.accelerator.AcceleratorPlan;
import org.apache.flink.table.runtime.typeutils.RowDataSerializer;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * An inner equi-join run by cuDF's ordering and a generated binary-search probe.
 *
 * <p>Three things here cannot be seen without a device and the last is the one worth the test. That
 * matched rows come back is the obvious assertion. That the <em>fan-out</em> is right is not: a
 * probe that returned one match per probe row instead of all of them produces a plausible, smaller
 * answer, and only a build side with deliberately repeated keys can tell. And that probe rows with
 * no match produce nothing — a binary search that failed to distinguish "not present" from "insert
 * here" would emit a neighbouring row and be wrong in a way no count would show.
 *
 * <p>Skips without a device, and separately without the cuDF shim. See {@link
 * DeviceAssumptions#requireCudf()}.
 */
class AcceleratedJoinDeviceIT {

    private static final int BUILD_ROWS = 20_000;
    private static final int PROBE_ROWS = 40_000;

    /** Build keys repeat this many times, so every match is a fan-out rather than a lookup. */
    private static final int DUPLICATES = 4;

    private static final LogicalType INT = new IntType(false);
    private static final LogicalType BIGINT = new BigIntType(false);
    private static final LogicalType DOUBLE = new DoubleType(false);

    /** Build: (key, label). The BIGINT is not something a kernel could compute with. */
    private static final RowType BUILD = RowType.of(INT, BIGINT);

    /** Probe: (key, value). */
    private static final RowType PROBE = RowType.of(INT, DOUBLE);

    /** Build fields then probe fields. */
    private static final RowType OUTPUT = RowType.of(INT, BIGINT, INT, DOUBLE);

    private static final AcceleratorContext CONTEXT =
            new AcceleratorContext() {
                @Override
                public RowType outputType() {
                    return OUTPUT;
                }

                @Override
                public int maxBatchSize() {
                    // Smaller than the probe side on purpose: the probe runs a batch at a time and
                    // a join that only worked when everything fitted in one would pass a test that
                    // did not say so.
                    return 4096;
                }

                @Override
                public ClassLoader userCodeClassLoader() {
                    return AcceleratedJoinDeviceIT.class.getClassLoader();
                }

                @Override
                public boolean providesOffHeap() {
                    return false;
                }

                @Override
                public ByteBuffer allocateOffHeap(int bytes) {
                    throw new IllegalStateException("no managed memory in this harness");
                }
            };

    @BeforeEach
    void requireCudf() {
        DeviceAssumptions.requireCudf();
    }

    @Test
    @DisplayName("the device's join is the host's, pair for pair")
    void theDeviceJoinsLikeTheHost() throws Exception {
        List<RowData> out = run();

        // Every build key k appears DUPLICATES times; a probe row matches when its key is one of
        // them, so the expected size is exactly the fan-out times the matching probe rows.
        Map<Integer, Integer> expected = new HashMap<>();
        int distinctBuildKeys = BUILD_ROWS / DUPLICATES;
        for (int i = 0; i < PROBE_ROWS; i++) {
            int key = probeKey(i);
            if (key < distinctBuildKeys) {
                expected.merge(key, DUPLICATES, Integer::sum);
            }
        }
        int expectedRows = expected.values().stream().mapToInt(Integer::intValue).sum();
        assertThat(out)
                .as("a probe returning one match per row instead of all of them lands here")
                .hasSize(expectedRows);

        Map<Integer, Integer> actual = new HashMap<>();
        for (RowData row : out) {
            int buildKey = row.getInt(0);
            int probeKey = row.getInt(2);
            assertThat(buildKey)
                    .as("a pair whose two halves disagree about the key")
                    .isEqualTo(probeKey);
            // label = 1000 + key and value = key * 0.5, so a row assembled from two different
            // pairs fails here while still having matching keys.
            assertThat(row.getLong(1)).isEqualTo(1000L + buildKey);
            assertThat(row.getDouble(3)).isEqualTo(buildKey * 0.5);
            actual.merge(buildKey, 1, Integer::sum);
        }
        assertThat(actual).isEqualTo(expected);
    }

    @Test
    @DisplayName("a probe key absent from the build side matches nothing")
    void anAbsentKeyMatchesNothing() throws Exception {
        List<RowData> out = run();
        int distinctBuildKeys = BUILD_ROWS / DUPLICATES;
        for (RowData row : out) {
            assertThat(row.getInt(0))
                    .as("a binary search that returned an insertion point rather than a match")
                    .isLessThan(distinctBuildKeys);
        }
    }

    @Test
    @DisplayName("a nullable key is declined rather than matched to other nulls")
    void aNullableKeyIsDeclined() {
        RowType nullableBuild = RowType.of(new IntType(true), BIGINT);
        AccelJoin join =
                new AccelJoin(
                        0,
                        0,
                        true,
                        BUILD_ROWS,
                        new AccelInput(nullableBuild),
                        new AccelInput(PROBE),
                        OUTPUT);
        assertThat(DeviceAssumptions.provider().accept(join, work())).isEmpty();
    }

    /** Runs one join through the device operator and returns what it emitted. */
    private static List<RowData> run() throws Exception {
        AccelJoin join =
                new AccelJoin(
                        0,
                        0,
                        true,
                        BUILD_ROWS,
                        new AccelInput(BUILD),
                        new AccelInput(PROBE),
                        OUTPUT);
        Optional<AcceleratorPlan> offered = DeviceAssumptions.provider().accept(join, work());
        assertThat(offered).as("the provider declined a shape it is meant to serve").isPresent();

        StreamOperatorFactory<RowData> factory =
                DeviceAssumptions.provider().createOperator(offered.get(), CONTEXT);
        // The two-input harness takes an operator rather than a factory, unlike the one-input one.
        GpuJoinOperator operator =
                (GpuJoinOperator) ((SimpleOperatorFactory<RowData>) factory).getOperator();

        List<RowData> out = new ArrayList<>();
        try (TwoInputStreamOperatorTestHarness<RowData, RowData, RowData> harness =
                new TwoInputStreamOperatorTestHarness<>(operator, 1, 1, 0)) {
            // An explicit serializer, because this operator emits a reused BinaryRowData and the
            // harness copies with whatever TypeExtractor infers when it is not told.
            harness.setup(new RowDataSerializer(OUTPUT));
            harness.open();

            int distinctBuildKeys = BUILD_ROWS / DUPLICATES;
            for (int i = 0; i < BUILD_ROWS; i++) {
                int key = i % distinctBuildKeys;
                GenericRowData row = new GenericRowData(2);
                row.setField(0, key);
                row.setField(1, 1000L + key);
                harness.processElement1(new StreamRecord<>(row));
            }
            operator.endInput(1);

            for (int i = 0; i < PROBE_ROWS; i++) {
                int key = probeKey(i);
                GenericRowData row = new GenericRowData(2);
                row.setField(0, key);
                row.setField(1, key * 0.5);
                harness.processElement2(new StreamRecord<>(row));
            }
            operator.endInput(2);

            assertThat(operator.metrics().getBatches())
                    .as("no batch ran, so nothing reached the device")
                    .isPositive();

            for (Object o : harness.getOutput()) {
                @SuppressWarnings("unchecked")
                RowData emitted = ((StreamRecord<RowData>) o).getValue();
                out.add(emitted);
            }
        }
        return out;
    }

    /** Half the probe keys land in the build side's range and half do not. */
    private static int probeKey(int i) {
        return i % (2 * (BUILD_ROWS / DUPLICATES));
    }

    private static AccelWorkProfile work() {
        int[] ops = new int[AccelFunction.values().length];
        ops[AccelFunction.LESS_THAN.ordinal()] = 15;
        return new AccelWorkProfile(ops, 12, 24, PROBE_ROWS);
    }
}
