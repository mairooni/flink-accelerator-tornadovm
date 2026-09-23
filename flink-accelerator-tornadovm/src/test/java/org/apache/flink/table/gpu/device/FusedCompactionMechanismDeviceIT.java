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

import org.apache.flink.table.gpu.operator.DeviceCompaction;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cudf.Cudf;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The mechanism Phase 3 rests on, tested on its own before any query goes near it.
 *
 * <p>Three things have to hold at once and none of them is obvious from the API: two task graphs
 * live in one {@link TornadoExecutionPlan}; a buffer the first graph wrote is visible to the second
 * without a round trip through the host; and a cuDF library task can read that persisted buffer
 * rather than one the runtime has just copied in.
 *
 * <p>If any of the three does not hold, the fused path is not possible and the right answer is to
 * find that out here rather than inside an operator.
 */
class FusedCompactionMechanismDeviceIT {

    private static final int N = 1024;

    @Test
    @DisplayName("a mask written by one graph compacts in the next, without touching the host")
    void twoGraphsOnePlanWithAPersistedMask() throws Exception {
        DeviceAssumptions.requireCudf();

        DoubleArray values = new DoubleArray(N);
        IntArray mask = new IntArray(N);
        for (int i = 0; i < N; i++) {
            values.set(i, i);
            mask.set(i, 0);
        }

        ByteArray maskBytes = new ByteArray(N);
        IntArray indices = new IntArray(N);
        IntArray survivors = new IntArray(1);
        DoubleArray compacted = new DoubleArray(N);
        maskBytes.init((byte) 0);
        indices.init(0);
        survivors.set(0, 0);
        compacted.init(0.0);

        IntArray rows = new IntArray(1);
        rows.set(0, N);

        // Graph 0: the producer. Keeps every third value, writes a 0/1 mask, and hands both on
        // without a transferToHost anywhere in it.
        TaskGraph producer =
                new TaskGraph("calc")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, values, rows)
                        .task(
                                "predicate",
                                FusedCompactionMechanismDeviceIT::everyThird,
                                values,
                                mask,
                                rows)
                        .persistOnDevice(values, mask);

        // Graph 1: the consumer. Reads both where they lie, compacts with cuDF, and copies back
        // only the survivors.
        TaskGraph consumer =
                new TaskGraph("compact")
                        .consumeFromDevice("calc", values, mask)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                maskBytes,
                                indices,
                                survivors,
                                compacted)
                        .task("widen", DeviceCompaction::widenMask, mask, maskBytes, rows)
                        .libraryTask(
                                "select",
                                Cudf::selectedIndices,
                                N,
                                N,
                                maskBytes,
                                indices,
                                survivors)
                        .task(
                                "gather",
                                DeviceCompaction::gatherDoubles,
                                indices,
                                survivors,
                                values,
                                compacted)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, compacted, survivors);

        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("calc.predicate", new WorkerGrid1D(N));
        scheduler.addWorkerGrid("compact.widen", new WorkerGrid1D(N));
        scheduler.addWorkerGrid("compact.gather", new WorkerGrid1D(N));

        try (TornadoExecutionPlan plan =
                new TornadoExecutionPlan(producer.snapshot(), consumer.snapshot())
                        .withGridScheduler(scheduler)) {
            plan.withGraph(0).execute();
            plan.withGraph(1).execute();
        }

        int expected = (N + 2) / 3;
        assertThat(survivors.get(0)).isEqualTo(expected);
        for (int j = 0; j < expected; j++) {
            assertThat(compacted.get(j)).isEqualTo(j * 3.0);
        }
    }

    /** Stands in for a generated predicate: the kernel writes a mask and nothing else leaves. */
    public static void everyThird(DoubleArray values, IntArray mask, IntArray rows) {
        for (@uk.ac.manchester.tornado.api.annotations.Parallel int i = 0;
                i < values.getSize();
                i++) {
            if (i < rows.get(0)) {
                mask.set(i, (((int) values.get(i)) % 3 == 0) ? 1 : 0);
            }
        }
    }
}
