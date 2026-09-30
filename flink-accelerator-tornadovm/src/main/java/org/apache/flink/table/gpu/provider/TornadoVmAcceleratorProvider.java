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
import org.apache.flink.table.accelerator.AccelExpression;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelIrVersion;
import org.apache.flink.table.accelerator.AccelJoin;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelOverAggregate;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelSort;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.codegen.AccelKernelGenerator;
import org.apache.flink.table.gpu.codegen.DecimalCoercion;
import org.apache.flink.table.gpu.codegen.DriftSensitivePredicate;
import org.apache.flink.table.gpu.codegen.ExactPowers;
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
import org.apache.flink.table.gpu.operator.GpuProjectedSortOperator;
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

    /**
     * Turns the cuDF binding off without uninstalling it.
     *
     * <p>A deployment switch rather than a query one, and it exists to make the difference the
     * library makes visible: the same SQL, the same jar, the same device, with the relational
     * operators either served by cuDF or left to Flink. Without it the only way to run that
     * comparison is to break {@code LD_LIBRARY_PATH}, which is indistinguishable from a
     * misconfigured cluster and produces the same silent CPU fallback.
     */
    private static final String DISABLE_CUDF = "flink.accelerator.tornadovm.disableCudf";

    private static boolean probeCudf() {
        if (UNAVAILABLE != null) {
            return false;
        }
        if (Boolean.getBoolean(DISABLE_CUDF)) {
            LOG.info("cuDF is installed but {} is set, so the binding is off", DISABLE_CUDF);
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
    public Optional<AcceleratorPlan> accept(AccelNode originalSubtree, AccelWorkProfile work) {
        Optional<AcceleratorPlan> plan = acceptSubtree(originalSubtree, work);
        if (plan.isPresent() && work.requiresStrictArithmetic()) {
            // The planner found that a value this subtree computes can reach something that
            // decides which rows come back. A device is free to fuse a multiply and an add into
            // one rounding, which is a different number from the CPU's two roundings, so the
            // kernel has to be compiled without that freedom or the same SQL answers differently
            // depending on where it ran.
            ((TornadoPlan) plan.get()).strictArithmetic = true;
        }
        return plan;
    }

    private Optional<AcceleratorPlan> acceptSubtree(
            AccelNode originalSubtree, AccelWorkProfile work) {
        if (UNAVAILABLE != null) {
            return Optional.empty();
        }
        // Before anything looks at the expressions, replace the two exponents a device can compute
        // exactly: POWER(x, 2) becomes a multiply and POWER(x, 0.5) a square root. Both are faster
        // and both agree with java.lang.Math bit for bit where a device pow does not, so this runs
        // ahead of the drift analysis rather than after it -- the point is that the rewritten form
        // is no longer drift-prone. See ExactPowers.
        AccelNode subtree = ExactPowers.rewrite(originalSubtree);
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
        // A filter whose outcome depends on a device transcendental can select a different set
        // of rows than the same SQL on the CPU -- demonstrated, not hypothetical. Refusing is the
        // conservative answer and it is the default; see DriftSensitivePredicate.
        String unsafePredicate = DriftSensitivePredicate.refuse(subtree);
        if (unsafePredicate != null) {
            LOG.debug("declining: {}", unsafePredicate);
            return Optional.empty();
        }
        // No multiply-add conformance probe here. A device that fuses multiply-add computes a
        // different value from the CPU, which can move a projected column and move a row past a
        // filter -- so the TaskManager must be started with
        // -Dtornado.enable.fma=false -Dtornado.cuda.compile.profile=repro, and that is now a
        // deployment obligation rather than something this provider verifies. See DEPLOYMENT.md.
        // A third way to the same failure, and the one the device is not at fault for: Flink does
        // not evaluate arithmetic with a DECIMAL operand as double arithmetic, and the kernel
        // does. Same 24-versus-23 row disagreement, same query, no transcendental involved.
        String decimal = DecimalCoercion.refuse(subtree);
        if (decimal != null) {
            LOG.debug("declining: {}", decimal);
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
    /**
     * The input field a grouping key is passed through from, as the one-element list the generator
     * wants; empty when the key is computed and therefore already a device column.
     */
    private static int[] keyInputFields(AccelNode projection, int keyOutputField) {
        if (!(projection instanceof AccelProject)) {
            return new int[0];
        }
        final AccelProject project = (AccelProject) projection;
        if (keyOutputField >= project.projections().size()) {
            return new int[0];
        }
        final AccelExpression key = project.projections().get(keyOutputField);
        return key instanceof AccelInputRef
                ? new int[] {((AccelInputRef) key).index()}
                : new int[0];
    }

    /** Whether the kernel stages the input column this output field is copied from. */
    private static boolean stages(GpuKernelSource kernel, GpuCalcSpec spec, int field) {
        final int source = spec.outputLayout()[field];
        for (int staged : kernel.inputFieldIndexes()) {
            if (staged == source) {
                return true;
            }
        }
        return false;
    }

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
        // cudf::groupby reads the key column from device memory, so the key has to be staged even
        // when the projection only passes it through -- which is the ordinary shape of the query,
        // SELECT region, SUM(f(x)) ... GROUP BY region. Naming it here is what makes a plain
        // column groupable; without it only a computed key worked, and a pass-through key threw
        // out of stagedColumn and took the whole query to the CPU.
        final int keyOutputField = agg.grouping()[0];
        Optional<GpuKernelSource> kernel =
                AccelKernelGenerator.generate(
                        projection,
                        Integer.toHexString(projection.hashCode()),
                        0,
                        0,
                        keyInputFields(projection, keyOutputField));
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
        int keyField = keyOutputField;
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
        // The engine reads the key column through stagedColumn, which throws when the field is
        // neither computed nor staged. That is a programming error rather than a plan this cannot
        // serve, so it is checked here and declined -- a provider that throws out of accept()
        // violates the SPI and is caught only by a safety net meant for third-party code.
        if (probe.computedSlot(keyField) < 0 && !stages(kernel.get(), probe, keyField)) {
            LOG.info(
                    "declining the aggregate: the grouping key (field {}) is neither computed nor"
                            + " staged, so the device has no column to group by",
                    keyField);
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
                new GpuSortSpec(
                        sort.sortField(), sort.outputType(), work.estimatedRows(), sort.limit());

        // A sort whose input is a projection rather than a leaf is a fused pair, and the whole
        // point of fusing is that the projection's output never leaves the device: the kernel
        // writes the columns, cuDF orders the key column where it lies, and one drain takes the
        // ordered rows back. Unfused, the same two nodes are two operators, two task graphs and
        // two round trips -- and the drain is 95% of what an offloaded sort costs (VERIFY T12),
        // so paying it twice is most of what there is to save.
        AccelNode input = sort.inputs().get(0);
        if (input instanceof AccelInput) {
            return Optional.of(
                    new TornadoPlan(
                            null,
                            null,
                            spec,
                            sortCost(
                                    work,
                                    sort.outputType().getFieldCount(),
                                    emittedFraction(sort, work),
                                    keyedRows(sort, work))));
        }
        Optional<GpuKernelSource> kernel =
                AccelKernelGenerator.generate(input, Integer.toHexString(input.hashCode()));
        if (!kernel.isPresent()) {
            LOG.info("declining the sort: no kernel for the fused projection {}", input);
            return Optional.empty();
        }
        if (kernel.get().hasFilter() || kernel.get().carriesValidity()) {
            // The same refusal the fused aggregate makes, for the same reason: the kernel writes a
            // selection mask rather than compacting, so the rows a filter excluded are still in
            // the buffers the sort would order.
            LOG.info("declining the sort: the fused projection filters or carries nulls");
            return Optional.empty();
        }
        GpuCalcSpec probe =
                new GpuCalcSpec(kernel.get(), kernel.get().outputLayout(), input.outputType(), 1);
        if (!probe.canStage()) {
            LOG.info(
                    "declining the sort: cannot stage the fused projection {}", input.outputType());
            return Optional.empty();
        }
        if (probe.fieldType(sort.sortField()) != GpuValueType.INT) {
            LOG.info(
                    "declining the sort: cuDF sortedOrder takes an INT key, not {}",
                    probe.fieldType(sort.sortField()));
            return Optional.empty();
        }
        return Optional.of(
                new TornadoPlan(
                        kernel.get(),
                        null,
                        spec,
                        sortCost(
                                work,
                                sort.outputType().getFieldCount(),
                                emittedFraction(sort, work),
                                keyedRows(sort, work))));
    }

    /**
     * What fraction of the rows read are emitted: 1.0 unbounded, {@code limit/rows} for a top-N.
     *
     * <p>Falls back to 1.0 when the cardinality is unknown, which is the conservative direction --
     * an unknown input may be no larger than the limit, in which case nothing is saved.
     */
    /**
     * How many rows Flink's sorter keeps ordered, which sets the depth of its comparisons.
     *
     * <p>All of them for a plain sort; the limit for a top-N, because {@code SortLimitOperator}
     * holds a heap of that size rather than ordering the input.
     */
    private static double keyedRows(AccelSort sort, AccelWorkProfile work) {
        long rows = work.estimatedRows();
        double n = rows > 0 ? rows : 2.0;
        return sort.isBounded() ? Math.min(n, (double) sort.limit()) : n;
    }

    private static double emittedFraction(AccelSort sort, AccelWorkProfile work) {
        if (!sort.isBounded()) {
            return 1.0;
        }
        long rows = work.estimatedRows();
        if (rows <= 0) {
            return 1.0;
        }
        return Math.min(1.0, (double) sort.limit() / (double) rows);
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
    private static AcceleratorCost sortCost(
            AccelWorkProfile work, int fields, double emitted, double keyedRows) {
        // M8.2. The planner charges an ordering exactly one LESS_THAN, so reading the CPU side
        // straight out of the work profile said Flink's sorter does one comparison a row. A
        // comparison sort does log2(n) -- about 23 at eight million -- and a top-N compares
        // against a heap of the limit, so log2(min(limit, n)), which is smaller and correctly
        // less flattering to us. Any operations a fused projection brought with it still cost
        // what they cost; only the ordering's own contribution changes.
        double charged = Math.max(1, work.totalOpsPerRow());
        double ordering = Math.log(Math.max(2.0, keyedRows)) / Math.log(2.0);
        double comparisons = (charged - 1.0) + ordering;
        double cpuNanosPerRow = comparisons * CPU_NANOS_COMPARISON + CPU_NANOS_SORT_FIELD * fields;
        // Per field rather than per byte, which is the thing this measurement changed. Everything
        // above prices host work as bytes moved, because for a Calc it is: a columnar gather is a
        // bulk copy. A sort's host work is not -- it writes each field into staging on arrival and
        // reads it back out into a binary row on the way out, and both are per field whatever the
        // field is worth. Priced per byte this claimed 23x and measured 2.2x.
        //
        // The key out and the permutation back are the only part that really is bytes, and they
        // are eight of them.
        //
        // The two halves are not both paid on every row once a limit is in play. Staging-in is per
        // input row and unavoidable -- cuDF orders the whole key column either way. Draining-out is
        // per *emitted* row, and a top-N emits N however many it read. `emitted` is that fraction,
        // 1.0 for an unbounded sort, and it is the only reason a bounded sort is worth offering
        // when an unbounded one is not: the drain is the half that dominates.
        double drainShare = 0.5 * Math.max(0.0, Math.min(1.0, emitted));
        double gpuNanosPerRow =
                SORT_NANOS_PER_FIELD * fields * (0.5 + drainShare) + HOST_NANOS_PER_BYTE * 8;
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
        if (!AccelKernelGenerator.generate(featureProjection(spec), "probe", 1, 1).isPresent()) {
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
                                    context.maxBatchSize(),
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
            if (kernel != null) {
                // A fused pair: the kernel writes its columns and cuDF orders them where they lie,
                // in one graph, so the projection's output never crosses the bus. See
                // GpuProjectedSortOperator for what that is worth.
                GpuCalcSpec projection =
                        new GpuCalcSpec(
                                kernel,
                                kernel.outputLayout(),
                                sort.rowType(),
                                sortCapacity(sort, context),
                                ((TornadoPlan) plan).strictArithmetic);
                return SimpleOperatorFactory.of(
                        new GpuProjectedSortOperator(
                                projection,
                                sort,
                                sortCapacity(sort, context),
                                PROFILE,
                                context.providesOffHeap() ? context::allocateOffHeap : null));
            }
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
                            context.maxBatchSize(),
                            ((TornadoPlan) plan).strictArithmetic);
            return SimpleOperatorFactory.of(
                    new GpuGroupedAggregateOperator(
                            spec, aggregate, context.maxBatchSize(), PROFILE, null));
        }
        GpuCalcSpec spec =
                new GpuCalcSpec(
                        kernel,
                        kernel.outputLayout(),
                        context.outputType(),
                        context.maxBatchSize(),
                        ((TornadoPlan) plan).strictArithmetic);
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
    private static final double CPU_NANOS_COMPARISON = 15.4;

    /**
     * What Flink's sorter pays per field of the row, per row, independent of the comparisons.
     *
     * <p>M8.2, measured 2026-09-30. The constant above was fitted on one point — 4M CSV rows of
     * three columns — and read as if a sort's cost were comparisons alone. It is not: {@code
     * SortLimitOperator} serialises every row into managed memory and copies it again on the way
     * out, so the cost carries a width term exactly as the device side does, and a model without
     * one is calibrated for the row it was measured on and wrong for every other.
     *
     * <p>The two points, both on this host:
     *
     * <ul>
     *   <li>4M CSV rows, 3 fields, one INT key, {@code log2(4M) = 22} -- 1948 ms, 487 ns a row.
     *   <li>8M Parquet rows, 32 fields, top-N to 1M, {@code log2(1M) = 19.93} -- 15,235 ms, 1900 ns
     *       a row. Isolated by difference: {@code COUNT(*)} over the same scan is 981 ms and the
     *       same query with {@code ORDER BY prio LIMIT} is 16,216 ms.
     * </ul>
     *
     * <p>Solving the pair gives 15.4 ns a comparison-row and 49.8 ns a field-row, and reproduces
     * both to within a nanosecond. <b>It is still a two-point fit</b> and should be treated as
     * such: it spans 3 to 32 fields and says nothing about a wider row, a multi-key sort, or a
     * machine that is not this one.
     */
    private static final double CPU_NANOS_SORT_FIELD = 49.8;

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

        /**
         * Whether the planner requires this subtree to round the way Flink's CPU operator does.
         *
         * <p>Set after construction because every {@code accept} path builds a plan and only one of
         * them has the work profile in hand; making it a constructor parameter would mean threading
         * it through six signatures to reach one assignment.
         */
        private boolean strictArithmetic;

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

    /**
     * A source that reads the scan itself, for a top-N directly over a Parquet file.
     *
     * <p>Narrow by design. The shape served is an {@link AccelSort} with a limit, one INT32 key,
     * over an {@link AccelScan} of {@code parquet} whose other columns are FP64 -- the operand
     * contract the cuDF binding states. Everything else declines, and declining costs nothing:
     * Flink plans the source it would have planned anyway.
     *
     * <p>Answered without touching a device. The claim is that this deployment has libcudf and the
     * shim, which is a property of the installation rather than of the moment, and the alternative
     * -- initialising CUDA on the JobManager to find out -- would cost every query a context on a
     * machine that may have no GPU at all.
     */
    @Override
    public Optional<org.apache.flink.api.connector.source.Source<RowData, ?, ?>> createScanSource(
            AccelNode subtree, RowType outputType) {
        if (!uk.ac.manchester.tornado.cudf.Cudf.isParquetAvailable()) {
            return Optional.empty();
        }
        // A local aggregate directly over a scan: read and group in one plan, no kernel involved.
        if (subtree instanceof org.apache.flink.table.accelerator.AccelAggregate) {
            return scanAggregateSource(
                    (org.apache.flink.table.accelerator.AccelAggregate) subtree, outputType);
        }
        if (!(subtree instanceof AccelSort)) {
            return Optional.empty();
        }
        final AccelSort sort = (AccelSort) subtree;
        if (!sort.isBounded() || sort.inputs().size() != 1) {
            return Optional.empty();
        }
        if (!(sort.inputs().get(0) instanceof org.apache.flink.table.accelerator.AccelScan)) {
            return Optional.empty();
        }
        final org.apache.flink.table.accelerator.AccelScan scan =
                (org.apache.flink.table.accelerator.AccelScan) sort.inputs().get(0);
        if (!"parquet".equals(scan.format())) {
            return Optional.empty();
        }
        final RowType row = scan.outputType();
        if (!(row.getTypeAt(sort.sortField())
                instanceof org.apache.flink.table.types.logical.IntType)) {
            return Optional.empty();
        }
        for (int i = 0; i < row.getFieldCount(); i++) {
            if (i == sort.sortField()) {
                continue;
            }
            if (!(row.getTypeAt(i) instanceof org.apache.flink.table.types.logical.DoubleType)) {
                return Optional.empty();
            }
        }
        for (int i = 0; i < row.getFieldCount(); i++) {
            if (row.getTypeAt(i).isNullable()) {
                // The read refuses a null rather than reading it as dense; a nullable column is
                // declined here so the refusal is a planning decision and not a runtime failure.
                return Optional.empty();
            }
        }
        if (!sort.ascending()) {
            return Optional.empty();
        }
        return Optional.of(
                new org.apache.flink.table.gpu.source.DeviceParquetTopNSource(
                        scan.paths(), sort.sortField(), sort.limit(), row));
    }

    /**
     * The two-operator region: cuDF reads the file and cuDF groups it, in one execution plan.
     *
     * <p>Served without generating anything, which is why this shape is supported and a projection
     * between the two is not: the grouping key and the summed column are read straight out of the
     * file, so the region is two library calls and a drain of the group partials.
     */
    private Optional<org.apache.flink.api.connector.source.Source<RowData, ?, ?>>
            scanAggregateSource(
                    org.apache.flink.table.accelerator.AccelAggregate aggregate,
                    RowType outputType) {
        if (!uk.ac.manchester.tornado.cudf.Cudf.isParquetAvailable()) {
            return Optional.empty();
        }
        if (aggregate.inputs().size() != 1
                || !(aggregate.inputs().get(0)
                        instanceof org.apache.flink.table.accelerator.AccelScan)) {
            return Optional.empty();
        }
        final org.apache.flink.table.accelerator.AccelScan scan =
                (org.apache.flink.table.accelerator.AccelScan) aggregate.inputs().get(0);
        if (!"parquet".equals(scan.format())) {
            return Optional.empty();
        }
        final int[] grouping = aggregate.grouping();
        if (grouping.length != 1 || aggregate.calls().size() != 1) {
            return Optional.empty();
        }
        final org.apache.flink.table.accelerator.AccelAggCall call = aggregate.calls().get(0);
        if (call.function() != org.apache.flink.table.accelerator.AccelAggFunction.SUM) {
            return Optional.empty();
        }
        final RowType row = scan.outputType();
        final int keyField = grouping[0];
        final int valueField = call.inputField();
        if (!(row.getTypeAt(keyField) instanceof org.apache.flink.table.types.logical.IntType)
                || !(row.getTypeAt(valueField)
                        instanceof org.apache.flink.table.types.logical.DoubleType)) {
            return Optional.empty();
        }
        if (row.getTypeAt(keyField).isNullable() || row.getTypeAt(valueField).isNullable()) {
            return Optional.empty();
        }
        return Optional.of(
                new org.apache.flink.table.gpu.source.DeviceParquetGroupSumSource(
                        scan.paths(), keyField, valueField, outputType));
    }
}
