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
import org.apache.flink.table.accelerator.AccelCall;
import org.apache.flink.table.accelerator.AccelExpression;
import org.apache.flink.table.accelerator.AccelFilter;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelLiteral;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.metrics.OffloadMetrics;
import org.apache.flink.table.gpu.operator.GpuCalcOperator;
import org.apache.flink.table.runtime.accelerator.AcceleratorContext;
import org.apache.flink.table.runtime.accelerator.AcceleratorPlan;
import org.apache.flink.table.types.logical.BooleanType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * How many bytes the device compaction actually moves, measured rather than assumed.
 *
 * <h2>The claim this exists to check</h2>
 *
 * <p>The compaction was built on the argument that only the surviving rows need cross the
 * interconnect: the projections and the mask stay on the device, cuDF selects the survivors in
 * place, and a gather packs them at the front of a second set of buffers. That argument is about
 * bytes, and bytes are measurable, so asserting it in prose was never good enough.
 *
 * <p>Run with {@code TORNADOVM_ACCELERATOR_COMPACTION=false} the same workload drains on the host
 * instead, which is the comparison that makes the number mean anything. The numbers are printed,
 * not asserted: what the right figure is depends on the batch size and the selectivity, and a test
 * that pinned one would be pinning this machine.
 */
class CompactionTransferDeviceIT {

    private static final DoubleType DOUBLE = new DoubleType(false);
    private static final RowType ROW = RowType.of(new IntType(false), new DoubleType(false));
    private static final int BATCH = 4096;
    private static final int ROWS = 65_536;

    /** Keeps roughly one row in ten, which is the shape a compaction is supposed to be for. */
    private static final double THRESHOLD = 8.0;

    private static final AcceleratorContext CONTEXT =
            new AcceleratorContext() {
                @Override
                public RowType outputType() {
                    return ROW;
                }

                @Override
                public int maxBatchSize() {
                    return BATCH;
                }

                @Override
                public ClassLoader userCodeClassLoader() {
                    return CompactionTransferDeviceIT.class.getClassLoader();
                }

                @Override
                public boolean providesOffHeap() {
                    return false;
                }

                @Override
                public ByteBuffer allocateOffHeap(int bytes) {
                    throw new IllegalStateException("this context provides none");
                }
            };

    @Test
    @DisplayName("report the bytes moved, with the compaction in whichever state it is configured")
    void reportTransferVolume() throws Exception {
        DeviceAssumptions.requireDevice();

        AccelNode subtree = plan();
        int[] ops = new int[AccelFunction.values().length];
        ops[AccelFunction.TIMES.ordinal()] = 1;
        ops[AccelFunction.GREATER_THAN.ordinal()] = 1;

        Optional<AcceleratorPlan> offered =
                DeviceAssumptions.provider()
                        .accept(subtree, new AccelWorkProfile(ops, 12, 12, 4_000_000L));
        assertThat(offered).isPresent();

        StreamOperatorFactory<RowData> factory =
                DeviceAssumptions.provider().createOperator(offered.get(), CONTEXT);

        int emitted = 0;
        OffloadMetrics metrics;
        boolean compacted;
        try (OneInputStreamOperatorTestHarness<RowData, RowData> harness =
                new OneInputStreamOperatorTestHarness<>(factory, 1, 1, 0)) {
            harness.setup();
            harness.open();
            GpuCalcOperator operator = (GpuCalcOperator) harness.getOneInputOperator();
            compacted = operator.compactedOnDevice();

            for (int i = 0; i < ROWS; i++) {
                GenericRowData row = new GenericRowData(2);
                row.setField(0, i);
                row.setField(1, (double) (i % 10));
                harness.processElement(new StreamRecord<>(row));
            }
            operator.endInput();
            metrics = operator.metrics();
            emitted = harness.getOutput().size();
        }

        int expected = 0;
        for (int i = 0; i < ROWS; i++) {
            if ((i % 10) > THRESHOLD) {
                expected++;
            }
        }

        System.out.println();
        System.out.println("=== transfer volume ===");
        System.out.printf("  compaction on device: %s%n", compacted);
        System.out.printf("  rows in:              %d%n", metrics.getRowsIn());
        System.out.printf(
                "  rows out:             %d  (%.1f%% selectivity)%n",
                metrics.getRowsOut(), 100.0 * metrics.getRowsOut() / metrics.getRowsIn());
        System.out.printf("  batches:              %d  of %d rows%n", metrics.getBatches(), BATCH);
        System.out.printf(
                "  bytes copy-in:        %,d  (%.3f MiB)%n",
                metrics.getBytesCopyIn(), metrics.getBytesCopyIn() / 1048576.0);
        System.out.printf(
                "  bytes copy-out:       %,d  (%.3f MiB)%n",
                metrics.getBytesCopyOut(), metrics.getBytesCopyOut() / 1048576.0);
        System.out.printf(
                "  bytes out per row in: %.2f%n",
                (double) metrics.getBytesCopyOut() / Math.max(1, metrics.getRowsIn()));
        System.out.printf(
                "  bytes out per row out:%.2f%n",
                (double) metrics.getBytesCopyOut() / Math.max(1, metrics.getRowsOut()));
        System.out.println(metrics.report("calc"));

        // Correctness is not optional just because the subject is bytes.
        assertThat(emitted)
                .as("the filter must keep the rows it is supposed to")
                .isEqualTo(expected);
        assertThat(metrics.getRowsOut()).isEqualTo(expected);
    }

    /** {@code SELECT id, v * 2.0 FROM t WHERE v > 9.0} — one computed column, one pass-through. */
    private static AccelNode plan() {
        AccelExpression condition =
                new AccelCall(
                        AccelFunction.GREATER_THAN,
                        Arrays.asList(
                                new AccelInputRef(1, DOUBLE), new AccelLiteral(THRESHOLD, DOUBLE)),
                        new BooleanType(false));
        return new AccelProject(
                Arrays.asList(
                        new AccelInputRef(0, new IntType(false)),
                        new AccelCall(
                                AccelFunction.TIMES,
                                Arrays.asList(
                                        new AccelInputRef(1, DOUBLE),
                                        new AccelLiteral(2.0, DOUBLE)),
                                DOUBLE)),
                new AccelFilter(condition, new AccelInput(ROW), ROW),
                ROW);
    }
}
