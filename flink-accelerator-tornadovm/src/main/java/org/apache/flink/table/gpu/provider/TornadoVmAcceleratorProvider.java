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
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelIrVersion;
import org.apache.flink.table.accelerator.AccelJoin;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelOverAggregate;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelSort;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.codegen.AccelKernelGenerator;
import org.apache.flink.table.gpu.codegen.GpuAggregateSpec;
import org.apache.flink.table.gpu.codegen.GpuCalcSpec;
import org.apache.flink.table.gpu.codegen.GpuGramSpec;
import org.apache.flink.table.gpu.codegen.GpuJoinSpec;
import org.apache.flink.table.gpu.codegen.GpuKernelSource;
import org.apache.flink.table.gpu.codegen.GpuOverAggregateSpec;
import org.apache.flink.table.gpu.codegen.GpuSortSpec;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.operator.GpuCalcOperator;
import org.apache.flink.table.gpu.operator.GpuGramOperator;
import org.apache.flink.table.gpu.operator.GpuGroupedAggregateOperator;
import org.apache.flink.table.gpu.operator.GpuJoinOperator;
import org.apache.flink.table.gpu.operator.GpuOverAggregateOperator;
import org.apache.flink.table.gpu.operator.GpuSortOperator;
import org.apache.flink.table.runtime.accelerator.AcceleratorContext;
import org.apache.flink.table.runtime.accelerator.AcceleratorCost;
import org.apache.flink.table.runtime.accelerator.AcceleratorPlan;
import org.apache.flink.table.runtime.accelerator.AcceleratorProvider;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

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
        if (subtree instanceof AccelSort) {
            return acceptSort((AccelSort) subtree, work);
        }
        if (subtree instanceof AccelJoin) {
            return acceptJoin((AccelJoin) subtree, work);
        }
        if (subtree instanceof AccelOverAggregate) {
            return acceptOverAggregate((AccelOverAggregate) subtree, work);
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
        if (agg.grouping().length == 0) {
            return acceptGram(agg, work);
        }
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

    /**
     * An {@code ORDER BY}, served by cuDF's {@code stable_sorted_order} and a host-side gather.
     *
     * <p>Every refusal here is the binding's shape rather than a device's. cuDF orders far more
     * than this; {@code Cudf.sortedOrder} is one entry point over an {@code INT32} column with no
     * null mask, and widening it is more shim rather than a different design. What is <em>not</em>
     * a refusal is the payload: no column but the key reaches the device, so a row carrying a
     * {@code BIGINT} — which the kernel generator refuses outright — is sorted here perfectly well.
     *
     * <ul>
     *   <li><b>Ascending only.</b> The shim asks for {@code cudf::order::ASCENDING} and takes no
     *       direction. Reversing the permutation on the host is not the fix it looks like: it
     *       reverses runs of equal keys too, and a stable sort that loses stability is a wrong
     *       answer for a query that sorts twice.
     *   <li><b>A {@code NOT NULL INT} key.</b> The shim builds a column view with no null mask, so
     *       a null key would order as whatever its bits happen to be — silently. {@code NOT NULL}
     *       in the DDL is the sanctioned answer here as it is for a Calc, and {@link
     *       AccelSort#nullsLast()} is consequently not consulted.
     *   <li><b>A row this can hold for the whole partition</b>, at fixed width, in at most {@link
     *       GpuSortSpec#MAX_FIELDS} columns.
     *   <li><b>A cardinality estimate, within a stated ceiling.</b> A sort undertakes to hold its
     *       whole input, so an unknown input is one this cannot undertake at all.
     * </ul>
     */
    private Optional<AcceleratorPlan> acceptSort(AccelSort sort, AccelWorkProfile work) {
        // INFO for the same reason the aggregate's reasons are: one per operator rather than one
        // per subtree, and it is the question an operator will actually ask.
        if (!CUDF_AVAILABLE) {
            LOG.info("declining the sort: the cuDF binding is not usable in this JVM");
            return Optional.empty();
        }
        String refusal = GpuSortSpec.refuse(sort, work.estimatedRows(), MAX_SORT_BYTES);
        if (refusal != null) {
            LOG.info("declining the sort: {}", refusal);
            return Optional.empty();
        }
        GpuSortSpec spec =
                new GpuSortSpec(sort.sortField(), sort.outputType(), work.estimatedRows());
        return Optional.of(
                new TornadoPlan(
                        null, null, spec, sortCost(work, sort.outputType().getFieldCount())));
    }

    /**
     * What ordering a row costs here, against what it costs Flink's sorter.
     *
     * <p>Deliberately not the model above. That one prices arithmetic, and a sort has none — the
     * planner charges its comparisons as {@code LESS_THAN} so the subtree is not read as having no
     * work at all, but pricing {@code log2(n)} imaginary multiplies against a device would be a
     * number about nothing.
     *
     * <p>{@link #CPU_NANOS_COMPARISON} is measured rather than modelled, and stating it is the
     * point: without it Flink prices a comparison at a nanosecond and refuses a sort whose CPU
     * operator it has just been told takes two seconds.
     */
    private static AcceleratorCost sortCost(AccelWorkProfile work, int fields) {
        double comparisons = Math.max(1, work.totalOpsPerRow());
        double cpuNanosPerRow = comparisons * CPU_NANOS_COMPARISON;
        // Per field rather than per byte, which is the thing this measurement changed. Everything
        // above prices host work as bytes moved, because for a Calc it is: a columnar gather is a
        // bulk copy. A sort's host work is not -- it writes each field into staging on arrival and
        // reads it back out into a binary row on the way out, and both are per field whatever the
        // field is worth. Priced per byte this claimed 23x and measured 2.2x.
        //
        // The key out and the permutation back are the only part that really is bytes, and they
        // are eight of them.
        double gpuNanosPerRow = SORT_NANOS_PER_FIELD * fields + HOST_NANOS_PER_BYTE * 8;
        return new AcceleratorCost(
                cpuNanosPerRow / gpuNanosPerRow,
                SORT_FIXED_COST_NANOS,
                PER_BATCH_NANOS,
                cpuNanosPerRow);
    }

    /**
     * An inner equi-join, served by cuDF's ordering of the build side and a generated probe.
     *
     * <p>The split is §T5's and it measured both halves: a sort is not expressible as a per-row map
     * so the ordering is bound, and a probe is one, so the probe is generated. Binding both would
     * be slower and larger.
     */
    private Optional<AcceleratorPlan> acceptJoin(AccelJoin join, AccelWorkProfile work) {
        if (!CUDF_AVAILABLE) {
            LOG.info("declining the join: the cuDF binding is not usable in this JVM");
            return Optional.empty();
        }
        String refusal = GpuJoinSpec.refuse(join, MAX_JOIN_BYTES);
        if (refusal != null) {
            LOG.info("declining the join: {}", refusal);
            return Optional.empty();
        }
        GpuJoinSpec spec =
                new GpuJoinSpec(
                        join.buildKeyField(),
                        join.probeKeyField(),
                        join.buildIsLeft(),
                        join.estimatedBuildRows(),
                        join.inputs().get(0).outputType(),
                        join.inputs().get(1).outputType(),
                        join.outputType());
        return Optional.of(new TornadoPlan(null, null, null, spec, joinCost(work, spec)));
    }

    /**
     * What probing a row costs here, against what it costs Flink's hash join.
     *
     * <p>{@link #CPU_NANOS_COMPARISON} is reused deliberately: what it measured is what a
     * comparison costs inside one of Flink's binary-row operators, and a hash join's per-row work
     * is the same kind of thing as a sort's. The device side is priced per field like the sort's,
     * for the same reason — this operator's cost is staging a row and gathering it back, not the
     * search between them.
     */
    private static AcceleratorCost joinCost(AccelWorkProfile work, GpuJoinSpec spec) {
        double comparisons = Math.max(1, work.totalOpsPerRow());
        double cpuNanosPerRow = comparisons * CPU_NANOS_COMPARISON;
        double gpuNanosPerRow =
                SORT_NANOS_PER_FIELD
                                * (spec.probeType().getFieldCount()
                                        + spec.buildType().getFieldCount())
                        + HOST_NANOS_PER_BYTE * 12;
        return new AcceleratorCost(
                cpuNanosPerRow / gpuNanosPerRow,
                SORT_FIXED_COST_NANOS,
                PER_BATCH_NANOS,
                cpuNanosPerRow);
    }

    /** The property a deployment sets to move {@link #MAX_JOIN_BYTES}. */
    public static final String MAX_JOIN_BYTES_PROPERTY = "flink.accelerator.tornadovm.maxJoinBytes";

    /**
     * The largest build side this provider will undertake to hold, in bytes of staging.
     *
     * <p>Stated here for the reason {@link #MAX_SORT_BYTES} is: {@code accept} runs before the
     * slot's staging is reserved, and a join that finds out it does not fit has already consumed
     * rows it cannot give back. {@code AcceleratorContext.stagingBytes()} narrows it afterwards.
     */
    private static final long MAX_JOIN_BYTES = Long.getLong(MAX_JOIN_BYTES_PROPERTY, 1L << 30);

    /** Rows the build side may hold, from what this machine actually reserved. */
    private static int joinBuildCapacity(GpuJoinSpec spec, AcceleratorContext context) {
        int bytesPerRow = GpuJoinSpec.bytesPerRow(spec.buildType());
        long wanted = spec.estimatedBuildRows() + spec.estimatedBuildRows() / 8 + 1;
        long budget = context.stagingBytes();
        // Half the budget, because the probe batch is staged out of the same arena and a build
        // side that claimed all of it would leave the probe nowhere to go. A page off the top for
        // the array headers, which are per buffer rather than per row.
        long fits = budget > 0 ? (budget / 2 - 4096) / bytesPerRow : wanted;
        long rows = Math.min(wanted, Math.max(1, fits));
        return (int) Math.max(1, Math.min(Integer.MAX_VALUE - 8, rows));
    }

    /**
     * A running total, served by {@code Cudf.runningSum} a batch at a time.
     *
     * <p>The shortest of these methods and the shortest list of refusals, because the planner has
     * already turned away every frame, partition and function this cannot express. What is left is
     * whether the columns can be staged.
     *
     * <p>No size refusal, alone among the cross-row nodes here: a prefix sum composes across
     * batches, so this holds one {@code double} rather than a partition.
     */
    private Optional<AcceleratorPlan> acceptOverAggregate(
            AccelOverAggregate over, AccelWorkProfile work) {
        if (!CUDF_AVAILABLE) {
            LOG.info("declining the window: the cuDF binding is not usable in this JVM");
            return Optional.empty();
        }
        String refusal = GpuOverAggregateSpec.refuse(over);
        if (refusal != null) {
            LOG.info("declining the window: {}", refusal);
            return Optional.empty();
        }
        GpuOverAggregateSpec spec =
                new GpuOverAggregateSpec(
                        over.valueField(), over.inputs().get(0).outputType(), over.outputType());
        return Optional.of(new TornadoPlan(null, null, null, null, spec, overCost(work, spec)));
    }

    /**
     * What a running total costs here, against what it costs {@code NonBufferOverWindowOperator}.
     *
     * <p>The one place a claim here is made against Flink's <em>cheap</em> path rather than its
     * expensive one, and the constant says so. {@link #CPU_NANOS_COMPARISON} would be wrong: there
     * is no comparator, no normalized key and no managed memory in what is being replaced, only one
     * add per row through a generated aggs handler. {@link #CPU_NANOS_ACCUMULATION} is that, and it
     * is a fifteenth of a comparison.
     *
     * <p>The device side is priced per field like the sort's and the join's, because it is the same
     * work: stage a row, read it back into a binary row. On any row of more than about two fields
     * that dominates, which is the model saying out loud that this offload is unlikely to pay.
     */
    private static AcceleratorCost overCost(AccelWorkProfile work, GpuOverAggregateSpec spec) {
        // One accumulation a row, and deliberately not work.totalOpsPerRow(). The profile is a
        // property of the whole subtree, and this node sits above the Sort its own ORDER BY
        // created -- so the total includes that sort's log2(n) comparisons. Charging those at an
        // accumulation's rate is how this claimed 45x against a measured 1.01x.
        double cpuNanosPerRow = CPU_NANOS_ACCUMULATION;
        double gpuNanosPerRow =
                SORT_NANOS_PER_FIELD * spec.inputType().getFieldCount() + HOST_NANOS_PER_BYTE * 16;
        return new AcceleratorCost(
                cpuNanosPerRow / gpuNanosPerRow,
                SORT_FIXED_COST_NANOS,
                PER_BATCH_NANOS,
                cpuNanosPerRow);
    }

    /**
     * A Gram matrix, served by a generated feature kernel and {@code cublasDgemm} in one task
     * graph.
     *
     * <p>The one shape in this project whose advantage grows with the query rather than being
     * fixed. Every other operator does the same work per row, which is why §T9, §T10, §T12, §T13
     * and §T14 all found the device is not the constraint; a Gram matrix over {@code d} features is
     * {@code O(n·d²)} of arithmetic on {@code O(n·d)} of data. §T15 measured 1.05x at eight
     * features and 12.09x at sixty-four.
     */
    private Optional<AcceleratorPlan> acceptGram(AccelAggregate agg, AccelWorkProfile work) {
        if (!CUBLAS_AVAILABLE) {
            LOG.info("declining the Gram matrix: the cuBLAS binding is not usable in this JVM");
            return Optional.empty();
        }
        GpuGramSpec.Recognition recognised = GpuGramSpec.recognise(agg);
        if (!recognised.recognised()) {
            LOG.info("declining the ungrouped aggregate: {}", recognised.reason());
            return Optional.empty();
        }
        GpuGramSpec spec = recognised.spec();
        // Generated once here purely to find out whether the feature map is expressible at all;
        // the stride the real kernel is packed at is the batch size, which only the TaskManager's
        // context knows, so the source that actually runs is generated again in createOperator.
        if (!AccelKernelGenerator.generate(featureProjection(spec), "probe", 1).isPresent()) {
            LOG.info("declining the Gram matrix: no kernel for its feature map");
            return Optional.empty();
        }
        return Optional.of(
                new TornadoPlan(null, null, null, null, null, spec, gramCost(work, spec)));
    }

    /** The {@code d} features as a projection, which is what the kernel generator takes. */
    private static AccelProject featureProjection(GpuGramSpec spec) {
        LogicalType[] fields = new LogicalType[spec.featureCount()];
        for (int i = 0; i < fields.length; i++) {
            fields[i] = spec.features().get(i).outputType();
        }
        return new AccelProject(
                spec.features(), new AccelInput(spec.inputType()), RowType.of(fields));
    }

    /**
     * What a Gram matrix costs here, against materialising its products a row at a time.
     *
     * <p>The only cost model in this provider whose ratio depends on the query's shape rather than
     * only on its row width, and that is the point. The CPU plan computes and sums {@code d(d+1)/2}
     * products per row; this computes {@code d} features per row and contracts them on the device.
     * So the work saved grows as {@code d²} while what moves grows as {@code d} — which is the
     * claim §T15 tested and the reason this node is worth having when nothing else here is.
     */
    private static AcceleratorCost gramCost(AccelWorkProfile work, GpuGramSpec spec) {
        int d = spec.featureCount();
        int products = d * (d + 1) / 2;
        double cpuNanosPerRow = products * CPU_NANOS_PRODUCT_SUM;
        double gpuNanosPerRow = d * GRAM_NANOS_PER_FEATURE;
        return new AcceleratorCost(
                cpuNanosPerRow / gpuNanosPerRow,
                SORT_FIXED_COST_NANOS,
                PER_BATCH_NANOS,
                cpuNanosPerRow);
    }

    /**
     * Whether the cuBLAS binding can run here.
     *
     * <p>Probed like {@link #CUDF_AVAILABLE} and for the same reason: cuBLAS ships with TornadoVM
     * but its native library does not always, and a deployment without it should get a declined
     * aggregate rather than a {@code NoClassDefFoundError} while this class initialises.
     */
    private static final boolean CUBLAS_AVAILABLE = probeCublas();

    private static boolean probeCublas() {
        if (UNAVAILABLE != null) {
            return false;
        }
        try {
            Class<?> provider =
                    Class.forName(
                            "uk.ac.manchester.tornado.cublas.provider.CuBlasLibraryProvider",
                            true,
                            TornadoVmAcceleratorProvider.class.getClassLoader());
            return (Boolean) provider.getMethod("isAvailable").invoke(null);
        } catch (Throwable t) {
            LOG.debug("the cuBLAS binding is unavailable: {}", String.valueOf(t));
            return false;
        }
    }

    @Override
    public StreamOperatorFactory<RowData> createOperator(
            AcceleratorPlan plan, AcceleratorContext context) {
        GpuKernelSource kernel = ((TornadoPlan) plan).kernel;
        GpuAggregateSpec aggregate = ((TornadoPlan) plan).aggregate;
        GpuGramSpec gram = ((TornadoPlan) plan).gram;
        if (gram != null) {
            // Generated here rather than in accept: the stride the features are packed at is the
            // batch size, and only this machine knows it.
            GpuKernelSource features =
                    AccelKernelGenerator.generate(
                                    featureProjection(gram),
                                    Integer.toHexString(gram.hashCode()),
                                    context.maxBatchSize())
                            .orElseThrow(
                                    () ->
                                            new IllegalStateException(
                                                    "accept admitted a feature map the generator"
                                                            + " cannot express"));
            return SimpleOperatorFactory.of(
                    new GpuGramOperator(gram, features, context.maxBatchSize()));
        }
        GpuOverAggregateSpec over = ((TornadoPlan) plan).over;
        if (over != null) {
            return SimpleOperatorFactory.of(
                    new GpuOverAggregateOperator(
                            over,
                            context.maxBatchSize(),
                            context.providesOffHeap() ? context::allocateOffHeap : null));
        }
        GpuJoinSpec join = ((TornadoPlan) plan).join;
        if (join != null) {
            return SimpleOperatorFactory.of(
                    new GpuJoinOperator(
                            join,
                            joinBuildCapacity(join, context),
                            context.maxBatchSize(),
                            context.providesOffHeap() ? context::allocateOffHeap : null));
        }
        GpuSortSpec sort = ((TornadoPlan) plan).sort;
        if (sort != null) {
            return SimpleOperatorFactory.of(
                    new GpuSortOperator(
                            sort,
                            sortCapacity(sort, context),
                            context.providesOffHeap() ? context::allocateOffHeap : null));
        }
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

    /** The property a deployment sets to move {@link #MAX_SORT_BYTES}. */
    public static final String MAX_SORT_BYTES_PROPERTY = "flink.accelerator.tornadovm.maxSortBytes";

    /**
     * The largest partition this provider will undertake to hold for a sort, in bytes of staging.
     *
     * <p>A ceiling stated here rather than derived from the slot, because {@link #accept} runs
     * before the slot's staging has been reserved and the answer is needed then: a sort that finds
     * out it does not fit has already consumed rows it cannot give back. {@link
     * AcceleratorContext#stagingBytes()} narrows it afterwards, where the reservation is known.
     *
     * <p>One gibibyte by default, which on the rows measured so far is a few tens of millions. The
     * figure is a property of a host rather than of a query, so a deployment that has measured its
     * own can say so.
     */
    private static final long MAX_SORT_BYTES = Long.getLong(MAX_SORT_BYTES_PROPERTY, 1L << 30);

    /**
     * Rows the sort operator may hold, from what this machine actually reserved.
     *
     * <p>{@link AcceleratorContext#maxBatchSize()} is the wrong number here and it is worth saying
     * why: it is capped by the configured batch size, which bounds how much may be in flight and
     * has nothing to say about how large an input may be held. A sort that sized itself from it
     * would decline every partition larger than one batch, which is every partition worth
     * offloading.
     *
     * <p>The budget is a ceiling and the estimate is the need, so this is the smaller of the two --
     * the same shape {@code batchSizeFor} has for a Calc. Sizing from the budget alone is not
     * merely wasteful: the arena is the slot's whole operator share, so a 4M-row partition claimed
     * six gigabytes, spent two and a half seconds having it zeroed, and then failed to fit because
     * the native arrays carry a header the row width does not count. It fell back and the offload
     * measured as a slowdown. An eighth over the estimate is the allowance for an estimate that is
     * a little low; past that the operator degrades to the host rather than failing.
     */
    private static int sortCapacity(GpuSortSpec sort, AcceleratorContext context) {
        int bytesPerRow = GpuSortSpec.bytesPerRow(sort.rowType());
        long wanted = sort.estimatedRows() + sort.estimatedRows() / 8 + 1;
        long budget = context.stagingBytes();
        // A page off the budget for the array headers, which are per buffer rather than per row
        // and which the width above therefore cannot express.
        long fits = budget > 0 ? (budget - 4096) / bytesPerRow : wanted;
        long rows = Math.min(wanted, fits);
        return (int) Math.max(1, Math.min(Integer.MAX_VALUE - 8, rows));
    }

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
     * What one comparison costs Flink's own sorter, per row sorted.
     *
     * <p>Measured 2026-09-18 on this host, not modelled: an RTX 4070 Laptop machine sorting 4M CSV
     * rows by one {@code INT}, chaining disabled so the {@code Sort} vertex stands alone. 1948 ms
     * over 4,000,000 rows at {@code log2(4M) = 22} comparisons a row is 22.1 ns per comparison-row
     * — against the one nanosecond {@code AcceleratorChoice} assumes where nobody tells it
     * otherwise.
     *
     * <p>Not {@link #CPU_NANOS_CHEAP}, and the gap is the whole reason this constant exists. A
     * comparison in a sorter is not a scalar compare: it is a normalized-key fetch, a binary-row
     * dereference on a tie, and a cache miss, inside an operator that is also serialising every row
     * into managed memory. Pricing it as arithmetic prices the wrong thing.
     */
    private static final double CPU_NANOS_COMPARISON = 22.1;

    /**
     * What one field of one row costs this operator on the host, staged in and written back out.
     *
     * <p>Measured 2026-09-18 alongside {@link #CPU_NANOS_COMPARISON}, on the same 4M-row sort of a
     * three-field row: the {@code Sort} vertex took 938 ms, which is 234 ns a row over three
     * fields. The operator's own breakdown puts 616 ms of that in the drain -- building four
     * million {@code BinaryRowData} -- and the cuDF ordering itself at 15.7 ms, so this constant is
     * almost entirely the cost of touching a row twice and almost not at all the device.
     *
     * <p>Which is the finding, and it is the same one §T9 and §T10 reached from the other side: the
     * device is not the constraint. A sort that wins 2.2x wins it on 15 ms of ordering against 2081
     * ms of Flink's sorter, and gives most of it back moving rows around.
     */
    private static final double SORT_NANOS_PER_FIELD = 78.0;

    /**
     * What one product-and-accumulate costs Flink's own plan, per row.
     *
     * <p>Derived from §T15's sweep rather than assumed, and it is the fourth constant in this file
     * to come out an order of magnitude above what the scalar model would have guessed. A product
     * in the CPU plan is not a multiply: it is a generated expression evaluated over a binary row
     * and folded into an accumulator through an {@code AggsHandleFunction}.
     *
     * <p>Calibrated conservatively. The measured ratios imply a rate that <em>grows</em> with the
     * feature count — the CPU arm degrades faster than its own product count, 528 to 2080 products
     * costing 15.6 s to 135.8 s — and a model of this shape cannot express that. Taking the
     * smallest rate the sweep supports makes the claim an under-statement at every width, which is
     * the direction that errs toward the CPU.
     */
    private static final double CPU_NANOS_PRODUCT_SUM = 27.0;

    /**
     * What one feature costs this operator per row, staged and moved.
     *
     * <p>The contraction is not in here and does not need to be: a GEMM over a resident matrix
     * never touches the host, and §T15's operator spends its time on the {@code d} columns going in
     * rather than the {@code d x d} coming back.
     *
     * <p>Together with the constant above this claims 1.15x at sixteen features, 2.23x at
     * thirty-two and 4.39x at sixty-four, against 1.15x, 2.80x and 12.09x measured. So the offload
     * is taken exactly where §T15 says it decisively wins and declined where the two arms are
     * inside each other's noise — sixteen features claims 1.15x, which is below the 1.3x floor, and
     * that is the right answer to a measured 1.15x.
     */
    private static final double GRAM_NANOS_PER_FEATURE = 200.0;

    /**
     * What one accumulation costs Flink's own over-aggregate, per row.
     *
     * <p>Measured 2026-09-18 on this host, like {@link #CPU_NANOS_COMPARISON}: the {@code
     * OverAggregate} vertex takes 527 ms over 4,000,000 rows with chaining off, which is 132 ns a
     * row for one add through a generated aggs handler.
     *
     * <p>A sixth of {@link #CPU_NANOS_COMPARISON}, and that is the point rather than an oversight.
     * {@code NonBufferOverWindowOperator} buffers nothing and reserves no managed memory, so there
     * is none of the cost a sorter pays and none of the advantage a device takes from it. Against
     * the per-field staging below, this model now claims well under 1x on any row worth offloading
     * — which is what the measurement found, and the reason the node is left eligible but
     * unattractive rather than removed.
     */
    private static final double CPU_NANOS_ACCUMULATION = 132.0;

    /**
     * What a sort pays once, which is not what a Calc pays once.
     *
     * <p>{@link #FIXED_COST_NANOS} is dominated by compiling a kernel, and a sort compiles none:
     * the whole device half is one cuDF library task. What is left is opening a device context and
     * loading the library, and the 300 ms to 700 ms bracket that figure came from had compilation
     * inside it at both ends — so the low end is an upper bound on a sort rather than an estimate
     * of one.
     *
     * <p>It is stated as the upper bound deliberately, because being wrong this way costs a sort
     * that stayed on the CPU and the other way costs a job that set a device up and never earned it
     * back. Replacing it with a measurement is part of what M5.4 still owes.
     */
    private static final long SORT_FIXED_COST_NANOS = 300_000_000L;

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

        private final @Nullable GpuKernelSource kernel;
        private final @Nullable GpuAggregateSpec aggregate;
        private final @Nullable GpuSortSpec sort;
        private final @Nullable GpuJoinSpec join;
        private final @Nullable GpuOverAggregateSpec over;
        private final @Nullable GpuGramSpec gram;
        private final AcceleratorCost cost;

        private TornadoPlan(
                @Nullable GpuKernelSource kernel,
                @Nullable GpuAggregateSpec aggregate,
                AcceleratorCost cost) {
            this(kernel, aggregate, null, null, cost);
        }

        private TornadoPlan(
                @Nullable GpuKernelSource kernel,
                @Nullable GpuAggregateSpec aggregate,
                @Nullable GpuSortSpec sort,
                AcceleratorCost cost) {
            this(kernel, aggregate, sort, null, cost);
        }

        private TornadoPlan(
                @Nullable GpuKernelSource kernel,
                @Nullable GpuAggregateSpec aggregate,
                @Nullable GpuSortSpec sort,
                @Nullable GpuJoinSpec join,
                AcceleratorCost cost) {
            this(kernel, aggregate, sort, join, null, cost);
        }

        private TornadoPlan(
                @Nullable GpuKernelSource kernel,
                @Nullable GpuAggregateSpec aggregate,
                @Nullable GpuSortSpec sort,
                @Nullable GpuJoinSpec join,
                @Nullable GpuOverAggregateSpec over,
                AcceleratorCost cost) {
            this(kernel, aggregate, sort, join, over, null, cost);
        }

        private TornadoPlan(
                @Nullable GpuKernelSource kernel,
                @Nullable GpuAggregateSpec aggregate,
                @Nullable GpuSortSpec sort,
                @Nullable GpuJoinSpec join,
                @Nullable GpuOverAggregateSpec over,
                @Nullable GpuGramSpec gram,
                AcceleratorCost cost) {
            this.kernel = kernel;
            this.aggregate = aggregate;
            this.sort = sort;
            this.join = join;
            this.over = over;
            this.gram = gram;
            this.cost = cost;
        }

        @Override
        public AcceleratorCost cost() {
            return cost;
        }

        @Override
        public Serializable payload() {
            if (kernel != null) {
                return kernel;
            }
            if (sort != null) {
                return sort;
            }
            if (join != null) {
                return join;
            }
            return over != null ? over : gram;
        }
    }
}
