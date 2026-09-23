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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;

import javax.annotation.Nullable;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Whether this device computes {@code a * b + c} the way Flink's CPU does, established by asking
 * it.
 *
 * <h2>Why this exists</h2>
 *
 * <p>SQL arithmetic on the CPU rounds the multiply, then rounds the add. A GPU will happily do both
 * in one fused operation with a single rounding, which is a <em>more accurate</em> answer and a
 * <em>different</em> one. Different is what matters: a filter over a computed value then selects a
 * different set of rows, and the query returns a different answer depending on where it ran.
 *
 * <p>Measured on this project, 2026-09-23, on a query containing no transcendental at all — only
 * {@code -}, {@code *}, {@code +} and {@code LEAST}: the CPU returned 24 rows and the device 23.
 * Per-row comparison put 37–50% of computed values apart, by up to 12 535 ulp.
 *
 * <h2>Two mechanisms, and why one flag is not enough</h2>
 *
 * <p>TornadoVM's CUDA backend fuses at two independent stages, and disabling either alone leaves
 * the other in place. Measured, all four combinations, on an RTX 4070:
 *
 * <table>
 *   <caption>what the device computed for {@code a*b + (-fl(a*b))}</caption>
 *   <tr><th>{@code tornado.enable.fma}</th><th>{@code tornado.cuda.compile.profile}</th><th>result</th></tr>
 *   <tr><td>true (default)</td><td>default</td><td>fused</td></tr>
 *   <tr><td>true</td><td>repro</td><td>fused</td></tr>
 *   <tr><td>false</td><td>default</td><td>fused</td></tr>
 *   <tr><td>false</td><td>repro</td><td><b>separate — matches the CPU exactly</b></td></tr>
 * </table>
 *
 * <p>The generated CUDA says why. With {@code enable.fma=true} the Graal phase {@code CUDAFMAPhase}
 * emits {@code fma(d_11, d_13, d_15)} as a literal call, and {@code --fmad=false} cannot un-call a
 * function. With it false the source carries {@code d_17 = d_11 * d_13; d_18 = d_17 + d_15;} and
 * NVRTC contracts that pair back into one instruction unless {@code --fmad=false} stops it.
 *
 * <h2>Why a probe rather than reading the flags</h2>
 *
 * <p>Both controls are JVM-wide system properties read into {@code static final} fields as
 * TornadoVM loads, so whether they took effect depends on ordering this code does not control. A
 * flag that was set too late reads back exactly like one that worked. The per-plan alternative,
 * {@code withCompilerFlags(CUDA, ...)}, does not work in 6.1.1: it writes to the task graph's meta
 * while {@code CUDACodeCache} reads the task's, and the two do not share the map — verified in the
 * source and by measurement, the flag changed nothing.
 *
 * <p>So this asks the device. The probe is a cancellation: {@code c} is exactly minus the rounded
 * product, so separate rounding gives {@code 0.0} exactly and fusion gives the rounding error
 * itself. There is no tolerance to choose and no third answer.
 *
 * <p>On failure the provider declines rather than returning a fast wrong answer. That is the whole
 * contract: <b>the device is used only where it has been shown to agree.</b>
 */
public final class StrictArithmetic {

    private static final Logger LOG = LoggerFactory.getLogger(StrictArithmetic.class);

    /** Named so an operator can repeat the deployment fix without reading this file. */
    public static final String REQUIRED_FLAGS =
            "-Dtornado.enable.fma=false -Dtornado.cuda.compile.profile=repro";

    /** An escape hatch for measuring what strictness costs. Never set by Flink, never by SQL. */
    private static final String ALLOW_FUSED_PROPERTY =
            "tornadovm.accelerator.allow-fused-arithmetic";

    /** Computed once per JVM: the check compiles a kernel, which is not free. */
    /**
     * The properties that decide how a kernel is compiled, and therefore what a verdict is evidence
     * about.
     *
     * <p>A verdict obtained under one compilation policy says nothing about another, so the cached
     * answer is keyed by the policy rather than simply memoised. These are read into {@code static
     * final} fields as TornadoVM loads and cannot change within a JVM, so one entry suffices in
     * practice; keying it anyway means a future per-plan control cannot silently reuse a verdict it
     * did not earn.
     */
    private static final String[] POLICY_PROPERTIES = {
        "tornado.enable.fma",
        "tornado.cuda.compile.profile",
        "tornado.cuda.compiler.flags",
        "tornado.device",
        ALLOW_FUSED_PROPERTY,
    };

    /** Keyed by compilation policy; the value is the refusal, or {@link #AGREES} for none. */
    private static final Map<String, String> VERDICTS = new ConcurrentHashMap<>();

    /** Sentinel, because a map cannot hold null and "no refusal" is the common answer. */
    private static final String AGREES = "";

    /** The device the probe actually ran on, recorded so the claim names its subject. */
    private static volatile @Nullable String qualifiedDevice = null;

    private StrictArithmetic() {}

    /**
     * The reason this device must not be used for computed values, or {@code null} if it agrees.
     *
     * <p>A probe that cannot run at all is also a refusal. If the device is unreachable the caller
     * has nothing to offload to anyway, and guessing "probably fine" is the one answer that can be
     * silently wrong.
     */
    public static @Nullable String refuse() {
        final String key = policyKey();
        final String cached = VERDICTS.get(key);
        if (cached != null) {
            return AGREES.equals(cached) ? null : cached;
        }
        synchronized (StrictArithmetic.class) {
            final String again = VERDICTS.get(key);
            if (again != null) {
                return AGREES.equals(again) ? null : again;
            }
            final String result = probe();
            VERDICTS.put(key, result == null ? AGREES : result);
            if (result == null) {
                LOG.info(
                        "device arithmetic agrees with the CPU on {}: DOUBLE multiply and add round"
                                + " separately, under compilation policy [{}]. That qualifies this"
                                + " policy on this device for DOUBLE multiply-add. It is a smoke"
                                + " test for the compilation contract, not a proof about other"
                                + " types, operators or special values.",
                        qualifiedDevice,
                        key);
            } else {
                LOG.warn("declining every computed expression on this device: {}", result);
            }
            return result;
        }
    }

    /** The compilation policy a verdict is about. Different policy, different question. */
    private static String policyKey() {
        final StringBuilder key = new StringBuilder();
        for (String property : POLICY_PROPERTIES) {
            key.append(property).append('=').append(System.getProperty(property, "")).append(' ');
        }
        return key.toString().trim();
    }

    /** The device the probe qualified, or {@code null} if it has not run. */
    public static @Nullable String qualifiedDevice() {
        return qualifiedDevice;
    }

    /** Visible for testing: forget the cached verdict so a test can probe again. */
    static synchronized void forget() {
        VERDICTS.clear();
        qualifiedDevice = null;
    }

    private static @Nullable String probe() {
        if (Boolean.parseBoolean(System.getProperty(ALLOW_FUSED_PROPERTY))) {
            LOG.warn(
                    "{} is set: fused device arithmetic is permitted and results may differ from"
                            + " the same SQL on the CPU. This is not a supported configuration.",
                    ALLOW_FUSED_PROPERTY);
            return null;
        }
        final int n = CASES.length;
        final DoubleArray a = new DoubleArray(n);
        final DoubleArray b = new DoubleArray(n);
        final DoubleArray c = new DoubleArray(n);
        final DoubleArray out = new DoubleArray(n);
        for (int i = 0; i < n; i++) {
            a.set(i, CASES[i][0]);
            b.set(i, CASES[i][1]);
            // Exactly minus the ROUNDED product, so the two cancel iff the rounding happened.
            c.set(i, -(CASES[i][0] * CASES[i][1]));
            out.set(i, Double.NaN);
        }
        try {
            final TaskGraph graph =
                    new TaskGraph("accelStrictCheck")
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b, c)
                            .task("mulAdd", StrictArithmetic::mulAdd, a, b, c, out)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            final ImmutableTaskGraph immutable = graph.snapshot();
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutable)) {
                // No withDevice() here, deliberately: GeneratedKernelEngine does not call it
                // either, so both take TornadoVM's default device and the probe qualifies the
                // device that will actually run the query. If the engine ever selects a device
                // explicitly, this must select the same one or it stops being evidence.
                plan.execute();
                qualifiedDevice = plan.getDevice(0).getPhysicalDevice().getDeviceName();
            }
        } catch (Throwable t) {
            return "the arithmetic conformance probe could not run on this device ("
                    + t
                    + "), so it is not known whether a multiply-add rounds once or twice here";
        }
        for (int i = 0; i < n; i++) {
            final double expected = CASES[i][0] * CASES[i][1] + c.get(i);
            final double got = out.get(i);
            if (Double.compare(expected, got) != 0) {
                return "this device fuses multiply-add: for a*b+c with a="
                        + CASES[i][0]
                        + " b="
                        + CASES[i][1]
                        + " c="
                        + c.get(i)
                        + " the CPU computes "
                        + expected
                        + " and the device "
                        + got
                        + ". A filter over a computed value can therefore select different rows"
                        + " than the same SQL on the CPU. Start the TaskManager with "
                        + REQUIRED_FLAGS
                        + " to make the device round the way Flink does";
            }
        }
        return null;
    }

    private static void mulAdd(DoubleArray a, DoubleArray b, DoubleArray c, DoubleArray out) {
        for (@Parallel int i = 0; i < a.getSize(); i++) {
            out.set(i, a.get(i) * b.get(i) + c.get(i));
        }
    }

    /**
     * Products that are not exactly representable, so something is lost in the rounding.
     *
     * <p>Spread across magnitudes on purpose: a single pair could be the one case a device happens
     * to get right, and the last two are large enough that the lost bits are worth whole units
     * rather than ulps, which makes a failure obvious in a log line.
     */
    private static final double[][] CASES = {
        {1.0000000001, 1.0000000003},
        {3.0000000000000004, 7.000000000000001},
        {1.4142135623730951, 1.4142135623730951},
        {0.1, 0.3},
        {1e8 + 1, 1e8 + 3},
        {2.718281828459045, 3.141592653589793},
        {1.7976931348623157e16, 1.0000000000000002},
    };
}
