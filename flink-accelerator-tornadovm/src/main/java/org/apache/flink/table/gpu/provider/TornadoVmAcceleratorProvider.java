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

package org.apache.flink.table.gpu.provider;

import org.apache.flink.streaming.api.operators.SimpleOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.table.accelerator.AccelAggCall;
import org.apache.flink.table.accelerator.AccelAggFunction;
import org.apache.flink.table.accelerator.AccelAggregate;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelIrVersion;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.codegen.AccelKernelGenerator;
import org.apache.flink.table.gpu.codegen.GpuAggregateSpec;
import org.apache.flink.table.gpu.codegen.GpuCalcSpec;
import org.apache.flink.table.gpu.codegen.GpuKernelSource;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.operator.GpuCalcOperator;
import org.apache.flink.table.gpu.operator.GpuGroupedAggregateOperator;
import org.apache.flink.table.runtime.accelerator.AcceleratorContext;
import org.apache.flink.table.runtime.accelerator.AcceleratorCost;
import org.apache.flink.table.runtime.accelerator.AcceleratorPlan;
import org.apache.flink.table.runtime.accelerator.AcceleratorProvider;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;

import java.io.Serializable;
import java.util.Optional;

/**
 * Serves accelerator subtrees with TornadoVM, discovered through {@code META-INF/services}.
 *
 * <p>Flink holds no compile-time reference to TornadoVM, and after this class was written to the
 * {@link AcceleratorProvider} contract it holds no runtime one either: everything device-shaped --
 * kernel generation, buffer layout, staging widths, and every constant calibrated against a
 * particular card -- lives on this side of the interface. What crosses it is the IR, a count of
 * operations per row, and one claimed speedup.
 *
 * <h2>Declining is normal</h2>
 *
 * <p>Empty from {@link #accept} is a supported answer, not an error. Flink's policy decides whether
 * a claimed speedup is worth taking; this class decides whether the kernel generator can express
 * the subtree at all, and that is much the narrower question.
 */
public class TornadoVmAcceleratorProvider implements AcceleratorProvider {

    private static final Logger LOG = LoggerFactory.getLogger(TornadoVmAcceleratorProvider.class);

    /**
     * Whether to collect the gather / copy-in / kernel / copy-out breakdown.
     *
     * <p>On while the project is establishing where time goes. It costs one profiler query per
     * batch, not per record.
     */
    private static final boolean PROFILE = true;

    /**
     * Why TornadoVM cannot be used in this JVM, or null if it can.
     *
     * <p>Probed once, by loading a class compiled against TornadoVM's off-heap array types. Those
     * are built on {@code java.lang.foreign}, a preview API on JDK 21, so a JVM started without
     * {@code --enable-preview} cannot load them.
     *
     * <p>The probe exists because failing here is far better than failing later. Without it the
     * TaskManager accepts the subtree and then dies building the operator with {@code
     * UnsupportedClassVersionError: Preview features are not enabled} -- an error that says nothing
     * about accelerators and appears a long way from its cause. Declining instead leaves Flink
     * running the operator it generated anyway, and records the reason.
     */
    private static final String UNAVAILABLE = probeTornado();

    /**
     * Whether the cuDF binding can actually run here.
     *
     * <p>Two things have to be true and neither usually is: TornadoVM must ship the {@code
     * tornado-cudf} module, and the host must have {@code libtornado-cudf.so} built against RAPIDS.
     * Probed reflectively so that a deployment without the module gets a declined aggregate rather
     * than a {@code NoClassDefFoundError} while this class initialises.
     */
    private static final boolean CUDF_AVAILABLE = probeCudf();

    private static boolean probeCudf() {
        if (UNAVAILABLE != null) {
            return false;
        }
        try {
            Class<?> provider =
                    Class.forName(
                            "uk.ac.manchester.tornado.cudf.provider.CudfLibraryProvider",
                            true,
                            TornadoVmAcceleratorProvider.class.getClassLoader());
            return (Boolean) provider.getMethod("isAvailable").invoke(null);
        } catch (Throwable t) {
            LOG.debug("the cuDF binding is unavailable: {}", String.valueOf(t));
            return false;
        }
    }

    private static String probeTornado() {
        try {
            // Loading the class is the check; it drags in DoubleArray and IntArray.
            Class.forName(
                    "org.apache.flink.table.gpu.operator.GeneratedKernelEngine",
                    false,
                    TornadoVmAcceleratorProvider.class.getClassLoader());
            return null;
        } catch (Throwable t) {
            String detail = t.getMessage() == null ? t.getClass().getName() : t.getMessage();
            if (t instanceof UnsupportedClassVersionError) {
                return "TornadoVM needs --enable-preview on this JVM (" + detail + ")";
            }
            return "TornadoVM is not usable in this JVM: " + detail;
        }
    }

    @Override
    public String name() {
        return "tornadovm";
    }

    @Override
    public int supportedIrVersion() {
        return AccelIrVersion.CURRENT;
    }

    @Override
    public Optional<AcceleratorPlan> accept(AccelNode subtree, AccelWorkProfile work) {
        if (UNAVAILABLE != null) {
            return Optional.empty();
        }
        if (subtree instanceof AccelAggregate) {
            return acceptAggregate((AccelAggregate) subtree, work);
        }
        if (work.totalOpsPerRow() > MAX_OPS_PER_ROW) {
            LOG.debug(
                    "declining: {} operations a row exceeds the ceiling of {}",
                    work.totalOpsPerRow(),
                    MAX_OPS_PER_ROW);
            // A ceiling on expression size, and the reason for it has changed twice.
            //
            // It was first set at 64 from an OpenCL measurement on an RTX 4070 Laptop, where a
            // 98-operation expression died with CL_OUT_OF_RESOURCES -- asynchronously and
            // unchecked, so the batch reported complete having computed nothing and the run
            // claimed 81x for work that never happened. That is the failure worth being
            // conservative about.
            //
            // On CUDA it does not happen there. Re-measured 2026-09-17 on an RTX 5070 Ti with
            // common-subexpression elimination in the generator: see the sweep recorded in
            // VERIFY.md. What used to fail was the generated *source* rather than the device --
            // a tree that repeated itself faster than the operation count grew -- and naming
            // subexpressions fixed that rather than moving it.
            //
            // The default is therefore set from what this card sustains, and the property exists
            // because the next card will sustain something else.
            return Optional.empty();
        }
        Optional<GpuKernelSource> kernel =
                AccelKernelGenerator.generate(subtree, Integer.toHexString(subtree.hashCode()));
        if (!kernel.isPresent()) {
            // Worth a line: "the provider declined" is all Flink can say, and the reason lives
            // only here. It cost an afternoon once already.
            LOG.debug("declining: no kernel could be generated for {}", subtree);
            return Optional.empty();
        }
        GpuCalcSpec probe =
                new GpuCalcSpec(kernel.get(), kernel.get().outputLayout(), subtree.outputType(), 1);
        if (!probe.canStage()) {
            LOG.debug("declining: the row cannot be staged for {}", subtree.outputType());
            // The kernel is expressible but the row is not: a column of a type the staging buffers
            // have no primitive array for, or a computed column the kernel would write as a double
            // into a field that is not one.
            return Optional.empty();
        }
        return Optional.of(new TornadoPlan(kernel.get(), null, estimateCost(work)));
    }

    /**
     * A local {@code GROUP BY}, served by the projection kernel and cuDF's group-by in one task
     * graph.
     *
     * <p>Every refusal below is a shape {@code Cudf.groupSum} has no answer for, not a shape that
     * would merely be slow. One key and one {@code SUM} is what the binding exposes; the rest is
     * what the fused pair can be trusted with:
     *
     * <ul>
     *   <li><b>No filter.</b> The group-by reads the kernel's output buffers where they lie, and a
     *       filtered row is still in them — the kernel writes a selection mask rather than
     *       compacting. Summing it would count rows the {@code WHERE} excluded. Compacting on the
     *       device first is the fix, and it is a task of its own.
     *   <li><b>No nullable operand.</b> A null key becomes a group and a null summand poisons one.
     *       {@code NOT NULL} in the DDL is the sanctioned answer, as it is for a Calc.
     *   <li><b>An {@code INT} key and a {@code DOUBLE} sum</b>, because that is the one shim entry
     *       point; widening it adds entry points rather than changing anything here.
     * </ul>
     */
    private Optional<AcceleratorPlan> acceptAggregate(AccelAggregate agg, AccelWorkProfile work) {
        // Louder than the Calc path's reasons, which are DEBUG. A declined Calc is the ordinary
        // case and there is one per subtree; a declined aggregate means the planner built a fused
        // node for this query and then found nothing to run it, which happens once per operator and
        // is the question an operator will actually ask. Finding out took a cluster round trip once
        // already, because a DEBUG line is only useful on a cluster configured to print it.
        if (!CUDF_AVAILABLE) {
            LOG.info("declining the aggregate: the cuDF binding is not usable in this JVM");
            return Optional.empty();
        }
        if (agg.grouping().length != 1) {
            LOG.info("declining the aggregate: {} grouping columns, not 1", agg.grouping().length);
            return Optional.empty();
        }
        if (agg.calls().size() != 1) {
            LOG.info("declining the aggregate: {} aggregate calls, not 1", agg.calls().size());
            return Optional.empty();
        }
        AccelAggCall call = agg.calls().get(0);
        if (call.function() != AccelAggFunction.SUM
                || call.inputField() == AccelAggCall.NO_INPUT_FIELD) {
            LOG.info("declining the aggregate: {} is not a SUM over a column", call);
            return Optional.empty();
        }
        AccelNode projection = agg.inputs().get(0);
        Optional<GpuKernelSource> kernel =
                AccelKernelGenerator.generate(
                        projection, Integer.toHexString(projection.hashCode()));
        if (!kernel.isPresent()) {
            LOG.info("declining the aggregate: no kernel for the projection {}", projection);
            return Optional.empty();
        }
        if (kernel.get().hasFilter() || kernel.get().carriesValidity()) {
            LOG.info(
                    "declining the aggregate: the projection filters ({}) or carries nulls ({})",
                    kernel.get().hasFilter(),
                    kernel.get().carriesValidity());
            return Optional.empty();
        }
        GpuCalcSpec probe =
                new GpuCalcSpec(
                        kernel.get(), kernel.get().outputLayout(), projection.outputType(), 1);
        if (!probe.canStage()) {
            LOG.info("declining the aggregate: cannot stage {}", projection.outputType());
            return Optional.empty();
        }
        int keyField = agg.grouping()[0];
        int valueField = call.inputField();
        if (probe.fieldType(keyField) != GpuValueType.INT
                || probe.fieldType(valueField) != GpuValueType.DOUBLE) {
            LOG.info(
                    "declining the aggregate: cuDF groupSum takes an INT key and a DOUBLE value, "
                            + "not {} and {}",
                    probe.fieldType(keyField),
                    probe.fieldType(valueField));
            return Optional.empty();
        }
        GpuAggregateSpec aggregate =
                new GpuAggregateSpec(
                        keyField, valueField, projection.outputType(), agg.outputType());
        return Optional.of(new TornadoPlan(kernel.get(), aggregate, estimateCost(work)));
    }

    @Override
    public StreamOperatorFactory<RowData> createOperator(
            AcceleratorPlan plan, AcceleratorContext context) {
        GpuKernelSource kernel = ((TornadoPlan) plan).kernel;
        GpuAggregateSpec aggregate = ((TornadoPlan) plan).aggregate;
        if (aggregate != null) {
            // The spec's output row is the projection's, not the operator's: the kernel produces
            // the aggregate's *input*, and what leaves the operator is one (key, sum) row a group.
            GpuCalcSpec spec =
                    new GpuCalcSpec(
                            kernel,
                            kernel.outputLayout(),
                            aggregate.projectionType(),
                            context.maxBatchSize());
            return SimpleOperatorFactory.of(
                    new GpuGroupedAggregateOperator(
                            spec, aggregate, context.maxBatchSize(), PROFILE, null));
        }
        GpuCalcSpec spec =
                new GpuCalcSpec(
                        kernel,
                        kernel.outputLayout(),
                        context.outputType(),
                        context.maxBatchSize());
        return SimpleOperatorFactory.of(new GpuCalcOperator(spec, context.maxBatchSize(), PROFILE));
    }

    @Override
    public String toString() {
        return UNAVAILABLE != null ? UNAVAILABLE : "TornadoVM (profile=" + PROFILE + ")";
    }

    // ------------------------------------------------------------------------------------------
    // Costing
    //
    // Every constant below is a property of one card and one host, and none of them belong in
    // Flink. They were measured on an RTX 4070 Laptop against an i9-13900H, 4M rows through the
    // generated-kernel path, with gather, copy-in, kernel, copy-out and drain all attributed to the
    // device side (see FINDINGS.md). The expression swept was a sin/cos series over one DOUBLE
    // column:
    //
    //     operations per row     2      8     14      26      50
    //     measured speedup    0.11x  3.54x  7.01x  13.42x  21.62x
    //
    // The model below reproduces those five points within about 20%, which is well inside the
    // run-to-run spread, and -- unlike the single weighted floor it replaces -- it distinguishes a
    // transcendental from a multiply. That distinction is the reason the old floor could not hold:
    // a sqrt-heavy sweep put break-even near 70 weighted operations and this sin/cos one put it
    // near 17, on the same card. The difference is not the device, it is how much worse the CPU is
    // at the particular function, and only a per-function model can see it.
    // ------------------------------------------------------------------------------------------

    /** The property a deployment sets to move {@link #MAX_OPS_PER_ROW}. */
    public static final String MAX_OPS_PER_ROW_PROPERTY =
            "flink.accelerator.tornadovm.maxOpsPerRow";

    /**
     * Largest expression this provider will accept, in operations per row.
     *
     * <p>Not a performance threshold -- a correctness one. See {@link #accept}.
     *
     * <p>Configurable since 2026-09-17, because the number is a property of a card and a driver and
     * this is the only place that knows either. It was a compile-time constant, so a deployment
     * that had measured its own hardware had no way to say so.
     */
    private static final int MAX_OPS_PER_ROW = Integer.getInteger(MAX_OPS_PER_ROW_PROPERTY, 1024);

    /** Nanoseconds the host spends per byte staged, transferred and drained. */
    private static final double HOST_NANOS_PER_BYTE = 0.454;

    /** A scalar CPU double operation, including the interpreter and bounds-check overhead. */
    private static final double CPU_NANOS_CHEAP = 0.5;

    /** {@code divsd} and friends: a few times a multiply, but nothing like a library call. */
    private static final double CPU_NANOS_DIVIDE = 2.0;

    /** Hardware {@code sqrtsd}. */
    private static final double CPU_NANOS_SQRT = 4.0;

    /** A libm call: sin, cos, exp, log, pow. This is where the device wins. */
    private static final double CPU_NANOS_TRANSCENDENTAL = 12.0;

    /** The same operations on the device, amortised over its lanes. */
    private static final double GPU_NANOS_CHEAP = 0.005;

    private static final double GPU_NANOS_MEDIUM = 0.02;

    private static final double GPU_NANOS_TRANSCENDENTAL = 0.05;

    /**
     * Compiling one kernel and opening a device context, paid once per subtask.
     *
     * <p>Measured between 300 ms and 700 ms depending on expression size; the higher end is used
     * because being wrong in this direction costs a fallback to the CPU, and the other direction
     * costs a job that offloads work it never earns back.
     */
    private static final long FIXED_COST_NANOS = 700_000_000L;

    /**
     * Dispatch and synchronisation per batch, independent of what the batch contains.
     *
     * <p>From the batch-size sweep: 37.1 ms of dispatch overhead across 31 batches of 262,144 rows.
     */
    private static final long PER_BATCH_NANOS = 1_200_000L;

    private static AcceleratorCost estimateCost(AccelWorkProfile work) {
        double cpuNanos = 0.0;
        double gpuNanos = 0.0;
        for (AccelFunction function : AccelFunction.values()) {
            int count = work.count(function);
            if (count == 0) {
                continue;
            }
            cpuNanos += count * cpuNanosOf(function);
            gpuNanos += count * gpuNanosOf(function);
        }
        // Everything the row costs to move: staged in, and the result drained back out.
        gpuNanos += HOST_NANOS_PER_BYTE * (work.inputBytesPerRow() + work.outputBytesPerRow());

        // A row with no arithmetic at all would divide by zero on the CPU side and claim an
        // infinite slowdown; floor it at one cheap operation, which is also roughly what reading
        // the row costs.
        cpuNanos = Math.max(cpuNanos, CPU_NANOS_CHEAP);
        return new AcceleratorCost(cpuNanos / gpuNanos, FIXED_COST_NANOS, PER_BATCH_NANOS);
    }

    private static double cpuNanosOf(AccelFunction function) {
        switch (function) {
            case DIVIDE:
                return CPU_NANOS_DIVIDE;
            case SQRT:
                return CPU_NANOS_SQRT;
            case EXP:
            case LN:
            case LOG2:
            case POWER:
            case SIN:
            case COS:
            case TAN:
            case ASIN:
            case ACOS:
            case ATAN:
            case ATAN2:
            case TANH:
                return CPU_NANOS_TRANSCENDENTAL;
            default:
                return CPU_NANOS_CHEAP;
        }
    }

    private static double gpuNanosOf(AccelFunction function) {
        switch (function) {
            case DIVIDE:
            case SQRT:
                return GPU_NANOS_MEDIUM;
            case EXP:
            case LN:
            case LOG2:
            case POWER:
            case SIN:
            case COS:
            case TAN:
            case ASIN:
            case ACOS:
            case ATAN:
            case ATAN2:
            case TANH:
                return GPU_NANOS_TRANSCENDENTAL;
            default:
                return GPU_NANOS_CHEAP;
        }
    }

    /** What this provider hands back to itself. Opaque to Flink, as the contract intends. */
    private static final class TornadoPlan implements AcceleratorPlan {

        private static final long serialVersionUID = 1L;

        private final GpuKernelSource kernel;
        private final @Nullable GpuAggregateSpec aggregate;
        private final AcceleratorCost cost;

        private TornadoPlan(
                GpuKernelSource kernel,
                @Nullable GpuAggregateSpec aggregate,
                AcceleratorCost cost) {
            this.kernel = kernel;
            this.aggregate = aggregate;
            this.cost = cost;
        }

        @Override
        public AcceleratorCost cost() {
            return cost;
        }

        @Override
        public Serializable payload() {
            return kernel;
        }
    }
}
