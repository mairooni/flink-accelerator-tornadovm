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

import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.util.OneInputStreamOperatorTestHarness;
import org.apache.flink.table.accelerator.AccelAggCall;
import org.apache.flink.table.accelerator.AccelAggFunction;
import org.apache.flink.table.accelerator.AccelAggregate;
import org.apache.flink.table.accelerator.AccelCall;
import org.apache.flink.table.accelerator.AccelExpression;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.operator.GpuGramOperator;
import org.apache.flink.table.runtime.accelerator.AcceleratorContext;
import org.apache.flink.table.runtime.accelerator.AcceleratorPlan;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;

/**
 * A Gram matrix computed by a generated feature kernel and {@code cublasDgemm} in one task graph.
 *
 * <p>Two things here need a device and the second is the one worth the test. That the numbers come
 * back is the obvious assertion. That they come back in the <b>right cells</b> is not: a Gram
 * matrix is symmetric, so an operator that transposed its read, or that walked the triangle in
 * column-major order, returns a complete and plausible answer with entries swapped. The features
 * here are therefore deliberately asymmetric in magnitude — feature {@code i} is scaled by {@code
 * 10^i} — so that every off-diagonal entry is distinguishable from its neighbours by orders of
 * magnitude rather than by rounding.
 *
 * <p>Skips without a device, and separately without the cuBLAS shim.
 */
class AcceleratedGramDeviceIT {

    private static final int ROWS = 20_000;

    /** Below the row count, so the contraction crosses batches and the last one is short. */
    private static final int BATCH = 4_096;

    private static final int FEATURES = 8;

    private static final LogicalType DOUBLE = new DoubleType(false);

    private static RowType row(int n) {
        LogicalType[] fields = new LogicalType[n];
        Arrays.fill(fields, DOUBLE);
        return RowType.of(fields);
    }

    /** Feature i is {@code ci * 10^i}: expressible, and asymmetric enough to catch a transpose. */
    private static AccelExpression feature(int i) {
        return new AccelCall(
                AccelFunction.TIMES,
                Arrays.asList(
                        new AccelInputRef(i, DOUBLE),
                        new org.apache.flink.table.accelerator.AccelLiteral(
                                Math.pow(10.0, i), DOUBLE)),
                DOUBLE);
    }

    private static AccelAggregate gram() {
        List<AccelExpression> products = new ArrayList<>();
        for (int i = 0; i < FEATURES; i++) {
            for (int j = i; j < FEATURES; j++) {
                products.add(
                        new AccelCall(
                                AccelFunction.TIMES,
                                Arrays.asList(feature(i), feature(j)),
                                DOUBLE));
            }
        }
        RowType projType = row(products.size());
        AccelProject projection =
                new AccelProject(products, new AccelInput(row(FEATURES)), projType);
        List<AccelAggCall> calls = new ArrayList<>();
        for (int i = 0; i < products.size(); i++) {
            calls.add(new AccelAggCall(AccelAggFunction.SUM, i, DOUBLE));
        }
        return new AccelAggregate(new int[0], calls, projection, projType);
    }

    private static final AcceleratorContext CONTEXT =
            new AcceleratorContext() {
                @Override
                public RowType outputType() {
                    return row(FEATURES * (FEATURES + 1) / 2);
                }

                @Override
                public int maxBatchSize() {
                    return BATCH;
                }

                @Override
                public ClassLoader userCodeClassLoader() {
                    return AcceleratedGramDeviceIT.class.getClassLoader();
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
    void requireDevice() {
        DeviceAssumptions.requireDevice();
    }

    @Test
    @DisplayName("the device's Gram matrix is the host's, cell for cell")
    void theGramMatrixAgreesWithTheHost() throws Exception {
        AccelAggregate subtree = gram();
        Optional<AcceleratorPlan> offered = DeviceAssumptions.provider().accept(subtree, work());
        org.junit.jupiter.api.Assumptions.assumeTrue(
                offered.isPresent(), "the cuBLAS binding is not usable on this host");

        StreamOperatorFactory<RowData> factory =
                DeviceAssumptions.provider().createOperator(offered.get(), CONTEXT);

        // The answer, accumulated here in the obvious way.
        double[][] expected = new double[FEATURES][FEATURES];
        List<RowData> out = new ArrayList<>();
        try (OneInputStreamOperatorTestHarness<RowData, RowData> harness =
                new OneInputStreamOperatorTestHarness<>(factory, 1, 1, 0)) {
            harness.setup();
            harness.open();
            for (int r = 0; r < ROWS; r++) {
                GenericRowData record = new GenericRowData(FEATURES);
                double[] f = new double[FEATURES];
                for (int c = 0; c < FEATURES; c++) {
                    record.setField(c, value(r, c));
                    f[c] = value(r, c) * Math.pow(10.0, c);
                }
                for (int i = 0; i < FEATURES; i++) {
                    for (int j = 0; j < FEATURES; j++) {
                        expected[i][j] += f[i] * f[j];
                    }
                }
                harness.processElement(new StreamRecord<>(record));
            }
            GpuGramOperator operator = (GpuGramOperator) harness.getOneInputOperator();
            operator.endInput();
            assertThat(operator.batchCount())
                    .as("the contraction did not run on the device")
                    .isEqualTo(ROWS / BATCH + 1);
            assertThat(operator.rowsSeen()).isEqualTo(ROWS);

            for (Object o : harness.getOutput()) {
                @SuppressWarnings("unchecked")
                RowData emitted = ((StreamRecord<RowData>) o).getValue();
                out.add(emitted);
            }
        }

        // One row: this is the local half of an ungrouped aggregate, and it emits its total once.
        assertThat(out).hasSize(1);
        RowData result = out.get(0);
        int at = 0;
        for (int i = 0; i < FEATURES; i++) {
            for (int j = i; j < FEATURES; j++) {
                // Relative, because the entries span sixteen orders of magnitude by construction;
                // an absolute bound would be meaningless at both ends.
                double want = expected[i][j];
                assertThat(result.getDouble(at))
                        .as("entry (" + i + "," + j + ") at position " + at)
                        .isCloseTo(want, offset(Math.abs(want) * 1e-9));
                at++;
            }
        }
        assertThat(at).isEqualTo(FEATURES * (FEATURES + 1) / 2);
    }

    /** Varied but bounded, so the FP64 contraction has something to get right. */
    private static double value(int row, int column) {
        return 1.0 + ((row + column) % 13) * 0.125;
    }

    private static AccelWorkProfile work() {
        int[] ops = new int[AccelFunction.values().length];
        ops[AccelFunction.TIMES.ordinal()] = FEATURES * (FEATURES + 1) / 2;
        return new AccelWorkProfile(ops, 8 * FEATURES, 8 * FEATURES, ROWS);
    }
}
