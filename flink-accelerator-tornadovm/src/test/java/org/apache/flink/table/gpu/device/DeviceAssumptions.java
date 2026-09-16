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
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelLiteral;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.provider.TornadoVmAcceleratorProvider;
import org.apache.flink.table.runtime.accelerator.AcceleratorContext;
import org.apache.flink.table.runtime.accelerator.AcceleratorPlan;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.RowType;

import java.util.Arrays;
import java.util.Collections;
import java.util.Optional;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Whether this machine can actually run a kernel, decided once by running one.
 *
 * <p>A missing card is not a broken build. These tests are enabled by a profile the operator turns
 * on deliberately, and even then the machine may not have what they need — a laptop with the
 * profile in its shell history, a CI runner whose GPU was reallocated. Skipping says "not checked
 * here"; failing would say "this change is wrong", which would be a lie.
 *
 * <h2>Why this does not just ask the provider</h2>
 *
 * <p>The obvious probe is {@link TornadoVmAcceleratorProvider}'s own, and the first version of this
 * class used it. It is too weak for the question asked here. That probe loads a class compiled
 * against TornadoVM's off-heap array types without initialising it, which catches the case it was
 * written for — a JVM lacking {@code --enable-preview} cannot load it at all — but says nothing
 * about whether TornadoVM can reach a device. Run without the SDK's JVM arguments, the class loads
 * happily and the failure arrives later, out of {@code new TaskGraph()}:
 *
 * <pre>TornadoAPIException: Tornado API Implementation class not specified.</pre>
 *
 * <p>which surfaced as three errors where three skips were wanted. So the probe here is the real
 * thing: stage one row through one generated kernel and see whether it comes back. It costs a
 * kernel compilation once per JVM, and it cannot drift from what the tests need, because it is what
 * the tests do.
 *
 * <p>Deliberately not fixed by strengthening the provider's own probe. That probe runs on every
 * TaskManager while a task is starting, and making it build a TaskGraph to find out would move a
 * kernel compilation onto the startup path of every task on every machine — including the ones that
 * have no device and are the reason it exists.
 */
final class DeviceAssumptions {

    private DeviceAssumptions() {}

    private static final TornadoVmAcceleratorProvider PROVIDER = new TornadoVmAcceleratorProvider();

    private static Boolean usable;
    private static String reason = "not probed";

    static synchronized boolean usable() {
        if (usable == null) {
            usable = probe();
        }
        return usable;
    }

    /** Skips the calling test, with the reason, when no device is available. */
    static void requireDevice() {
        assumeTrue(usable(), () -> "no usable device on this machine: " + reason);
    }

    static TornadoVmAcceleratorProvider provider() {
        return PROVIDER;
    }

    private static boolean probe() {
        String selfReport = PROVIDER.toString();
        if (!selfReport.startsWith("TornadoVM (")) {
            reason = selfReport;
            return false;
        }
        try {
            runOneRow();
            reason = selfReport;
            return true;
        } catch (Throwable t) {
            reason = "a one-row smoke offload failed (" + t + ")";
            return false;
        }
    }

    /**
     * The smallest thing that exercises everything: generate, compile, transfer, run, read back.
     */
    private static void runOneRow() throws Exception {
        RowType row = RowType.of(new DoubleType(false));
        AccelNode subtree =
                new AccelProject(
                        Collections.singletonList(
                                new AccelCall(
                                        AccelFunction.TIMES,
                                        Arrays.asList(
                                                new AccelInputRef(0, new DoubleType(false)),
                                                new AccelLiteral(2.0, new DoubleType(false))),
                                        new DoubleType(false))),
                        new AccelInput(row),
                        row);

        int[] ops = new int[AccelFunction.values().length];
        ops[AccelFunction.TIMES.ordinal()] = 1;
        Optional<AcceleratorPlan> plan =
                PROVIDER.accept(subtree, new AccelWorkProfile(ops, 8, 8, 1));
        if (!plan.isPresent()) {
            throw new IllegalStateException("the provider declined a single multiply");
        }

        StreamOperatorFactory<RowData> factory =
                PROVIDER.createOperator(
                        plan.get(),
                        new AcceleratorContext() {
                            @Override
                            public RowType outputType() {
                                return row;
                            }

                            @Override
                            public int maxBatchSize() {
                                return 16;
                            }

                            @Override
                            public ClassLoader userCodeClassLoader() {
                                return DeviceAssumptions.class.getClassLoader();
                            }

                            @Override
                            public boolean providesOffHeap() {
                                return false;
                            }

                            @Override
                            public java.nio.ByteBuffer allocateOffHeap(int bytes) {
                                // No slot behind this harness, so the engine allocates privately --
                                // which is
                                // itself worth exercising, since that is what a benchmark and any
                                // pre-M3.1 plan get.
                                throw new IllegalStateException(
                                        "no managed memory in this harness");
                            }
                        });

        try (OneInputStreamOperatorTestHarness<RowData, RowData> harness =
                new OneInputStreamOperatorTestHarness<>(factory, 1, 1, 0)) {
            harness.setup();
            harness.open();
            GenericRowData input = new GenericRowData(1);
            input.setField(0, 1.0);
            harness.processElement(new StreamRecord<>(input));
        }
    }
}
