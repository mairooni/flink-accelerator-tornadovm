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

import org.apache.flink.table.gpu.codegen.GpuAggregateSpec;
import org.apache.flink.table.gpu.codegen.GpuCalcSpec;
import org.apache.flink.table.gpu.codegen.GpuKernelSource;
import org.apache.flink.table.gpu.codegen.GpuSortSpec;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.gather.RowBlockGather;
import org.apache.flink.table.gpu.gather.RowGather;
import org.apache.flink.table.gpu.metrics.OffloadMetrics;

import uk.ac.manchester.tornado.api.DataRange;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;
import uk.ac.manchester.tornado.cudf.Cudf;

import javax.annotation.Nullable;

import java.io.IOException;
import java.lang.reflect.Method;
import java.net.URLClassLoader;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Compiles a generated kernel and runs it over batches of staged columns.
 *
 * <p>The kernel is not one of a few hand-written methods but Java source produced from the query's
 * own expressions, compiled once per operator instance. It replaced an earlier fixed catalogue that
 * could express 2 of the 13 shapes tested; generation covers 9.
 *
 * <h2>Why javac and a directory</h2>
 *
 * <p>TornadoVM reads a kernel's bytecode through {@code loader.getResourceAsStream(name +
 * ".class")}. A class defined only in memory is therefore unusable however it was produced, so the
 * compiled class is written to a temporary directory and loaded through a {@link URLClassLoader},
 * which serves it. {@code javax.tools} needs a JDK rather than a JRE, which TornadoVM already
 * requires.
 *
 * <p>Compilation happens once in {@link #open()}, alongside TornadoVM's own kernel compilation, not
 * per batch.
 */
public final class GeneratedKernelEngine implements AutoCloseable {

    private final GpuCalcSpec spec;
    private final boolean profile;

    /**
     * Staging buffers, one per column, each a {@code DoubleArray}, {@code FloatArray} or {@code
     * IntArray} according to the column's declared type. Held as {@code Object} because TornadoVM's
     * array classes share no interface that exposes {@code set}; every access goes through the
     * per-column accessors below, which know the concrete type.
     */
    private Object[] inputs;

    private Object[] outputs;
    private IntArray mask;

    /** The live row count for the batch about to run, in a one-element buffer. */
    private IntArray rows;

    /** The launch size, narrowed to the live row count before each execution. */
    private WorkerGrid1D grid;

    /** Which staged inputs are absent, a bit a column a row. Null when nothing can be. */
    private IntArray inNulls;

    /** Which computed outputs are absent, a bit a column a row. */
    private IntArray outNulls;

    private GeneratedKernel generated;
    private TornadoExecutionPlan plan;

    /**
     * Whether the filter compacts on the device instead of on the host.
     *
     * <p>Decided once in {@link #open()} from what the kernel turned out to need. See {@link
     * #compactsOnDevice()} for the conditions and why each one is there.
     */
    private boolean compacting;

    /** The mask as cuDF wants it: one byte a row, and zero past the live row count. */
    private ByteArray maskBytes;

    /** The surviving positions, in order, as {@code Cudf.selectedIndices} writes them. */
    private IntArray selectedIndices;

    /** How many survived, written by cuDF and read by the gather kernels and by the host. */
    private IntArray survivorCount;

    /** The compacted counterpart of each output column: survivors packed at the front. */
    private Object[] compacted;

    /**
     * The same kernel without the cuDF stage, built on the first short batch and only then.
     *
     * <p>{@code Cudf.groupSum} takes its row count as a value captured when the task graph is
     * built, unlike the kernel, which re-reads {@link #rows} on every execution. A partition's last
     * batch is short, and running the group-by over a full batch's worth of buffer would fold in
     * whatever the previous batch left in the tail -- a wrong answer, not a slow one. So the tail
     * runs the projection alone and is grouped on the host: one batch out of however many the
     * partition had, at a point where the stream is ending anyway.
     */
    private TornadoExecutionPlan tailPlan;

    private WorkerGrid1D tailGrid;

    /** Kept from {@link #open()} so the tail plan can be built from the same compiled kernel. */
    private Method entry;

    private Object[] kernelArgs;

    /** Where the cuDF stage writes: one row per distinct key, and the count of them. */
    private IntArray groupKeys;

    private DoubleArray groupSums;
    private IntArray groupCount;

    /** How many groups the last execution produced. */
    private int groups;

    private final OffloadMetrics metrics = new OffloadMetrics();

    private final @Nullable Staging staging;

    /** The grouped aggregate to run over the kernel's output, or null to stop at the kernel. */
    private final @Nullable GpuAggregateSpec aggregate;

    /**
     * The row block the device transposes, or null when staging stays column by column.
     *
     * <p>Allocated only once {@link #bindRowBlock} has been told the source row's arity, which is
     * not knowable until a row has arrived: {@code BinaryRowData}'s null-bit header is sized from
     * it, and the header decides where every field sits.
     */
    private @Nullable DoubleArray rowBlock;

    /** Slots in one staged row, header included. Zero while the block path is not in use. */
    private int blockSlots;

    /** The slot each staged column occupies within a row, parallel to {@code inputs}. */
    @Nullable private int[] columnSlots;

    /**
     * Built on first use rather than in {@link #open()}, because the block path needs the arity.
     */
    private boolean planBuilt;

    /** Present when this engine's projection feeds a device sort in the same graph. */
    private final @Nullable GpuSortSpec sort;

    /** The permutation cuDF writes back, allocated only for the fused sort. */
    private transient IntArray order;

    public GeneratedKernelEngine(GpuCalcSpec spec, boolean profile) {
        this(spec, profile, null);
    }

    /**
     * @param staging where staging buffers come from, or null to allocate privately. Flink offers
     *     one when the transformation declared managed memory; taking it is what makes the staging
     *     visible to the slot's budget instead of being memory this process took without telling
     *     anyone. Null is the ordinary answer for a benchmark driving the engine directly, and for
     *     a plan compiled before Flink declared anything.
     */
    public GeneratedKernelEngine(GpuCalcSpec spec, boolean profile, @Nullable Staging staging) {
        this(spec, null, profile, staging);
    }

    /**
     * @param aggregate a {@code SUM} grouped by one of the kernel's own output columns, run by cuDF
     *     in the same task graph. The intermediate the kernel writes is read by the group-by where
     *     it lies, so the only thing that crosses the interconnect is one row per group -- which is
     *     the entire reason for composing them rather than running two graphs.
     */
    public GeneratedKernelEngine(
            GpuCalcSpec spec,
            @Nullable GpuAggregateSpec aggregate,
            boolean profile,
            @Nullable Staging staging) {
        this(spec, aggregate, null, profile, staging);
    }

    public GeneratedKernelEngine(
            GpuCalcSpec spec,
            @Nullable GpuAggregateSpec aggregate,
            @Nullable GpuSortSpec sort,
            boolean profile,
            @Nullable Staging staging) {
        this.spec = spec;
        this.aggregate = aggregate;
        this.sort = sort;
        this.profile = profile;
        this.staging = staging;
    }

    /**
     * Orders the projection's own output, on the device, without it ever coming back.
     *
     * <p>The graph is built here rather than in {@code open()} because {@code Cudf.sortedOrder}
     * captures its row count when the graph is built, the way {@code Cudf.runningSum} does, and a
     * sort's row count is not known until the last row has arrived. A sort flushes once per
     * partition, so there is nothing to amortise a prebuilt graph over.
     *
     * @return the permutation, as row positions into the projection's output columns
     */
    public int[] projectAndOrder(int count) throws Exception {
        buildPlan();
        if (sort == null) {
            throw new IllegalStateException("this engine was not built with a sort stage");
        }
        rows.set(0, count);
        TaskGraph graph = new TaskGraph("calc-sort");
        graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, inputs);
        graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, rows);
        graph = graph.task("kernel", entry, kernelArgs);
        // The key the ordering reads is a column the task above just wrote, still on the device.
        graph =
                graph.libraryTask(
                        "order",
                        Cudf::sortedOrder,
                        count,
                        (IntArray) bufferFor(sort.sortField()),
                        order);
        graph = graph.transferToHost(DataTransferMode.EVERY_EXECUTION, concat(outputs, order));

        WorkerGrid1D sortGrid = new WorkerGrid1D(count);
        GridScheduler sortScheduler = new GridScheduler();
        sortScheduler.addWorkerGrid("calc-sort.kernel", sortGrid);
        TornadoExecutionPlan sortPlan =
                new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(sortScheduler);
        if (spec.requiresStrictArithmetic()) {
            sortPlan = sortPlan.withStrictFloatingPoint();
        }
        try (TornadoExecutionPlan closeable = sortPlan) {
            closeable.execute();
        }
        int[] permutation = new int[count];
        for (int i = 0; i < count; i++) {
            permutation[i] = order.get(i);
        }
        return permutation;
    }

    /** The staged column for a field of the projection's output, for the fused drain. */
    public Object outputBuffer(int field) {
        return bufferFor(field);
    }

    /** Where a staging buffer comes from. Exactly {@code AcceleratorContext::allocateOffHeap}. */
    @FunctionalInterface
    public interface Staging {
        ByteBuffer allocate(int bytes);
    }

    public void open() throws Exception {
        final GpuKernelSource kernel = spec.kernel();
        final int batchSize = spec.batchSize();

        inputs = new Object[kernel.inputFieldIndexes().length];
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = allocate(kernel.inputTypes()[i], batchSize);
        }
        outputs = new Object[kernel.outputCount()];
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = allocate(kernel.outputTypes()[i], batchSize);
        }
        if (sort != null) {
            order = new IntArray(batchSize);
        }
        if (kernel.hasFilter()) {
            mask = new IntArray(batchSize);
            mask.init(0);
        }
        compacting = compactsOnDevice();
        if (compacting) {
            maskBytes = new ByteArray(batchSize);
            maskBytes.init((byte) 0);
            selectedIndices = new IntArray(batchSize);
            selectedIndices.init(0);
            survivorCount = new IntArray(1);
            survivorCount.set(0, 0);
            compacted = new Object[outputs.length];
            for (int i = 0; i < compacted.length; i++) {
                compacted[i] = allocate(kernel.outputTypes()[i], batchSize);
            }
        }
        // How many rows the kernel should actually process, re-read by the device on every
        // execution. A one-element array rather than a scalar argument, because a scalar is
        // captured when the graph is built and the last batch of a partition is short.
        rows = new IntArray(1);
        rows.set(0, batchSize);
        if (kernel.carriesValidity()) {
            // One int a row covers every staged column: bit k says column k is absent in this row.
            // Four bytes a row whatever the column count, against four bytes per column the naive
            // way, on a path where every measurement so far is bound by moving the input.
            inNulls = new IntArray(batchSize);
            inNulls.init(0);
            outNulls = new IntArray(batchSize);
            outNulls.init(0);
        }

        entry = compile(kernel);
        if (aggregate != null) {
            // Sized to the batch rather than to the group count, which nobody knows in advance:
            // every row its own group is the worst case and it is the only safe one.
            groupKeys = new IntArray(batchSize);
            groupSums = new DoubleArray(batchSize);
            groupCount = new IntArray(1);
            groupCount.set(0, 0);
        }

        Object[] args =
                new Object
                        [inputs.length
                                + outputs.length
                                + (mask == null ? 0 : 1)
                                + (kernel.carriesValidity() ? 2 : 0)
                                + 1];
        int at = 0;
        for (Object in : inputs) {
            args[at++] = in;
        }
        for (Object out : outputs) {
            args[at++] = out;
        }
        if (mask != null) {
            args[at++] = mask;
        }
        if (inNulls != null) {
            args[at++] = inNulls;
            args[at++] = outNulls;
        }
        args[at] = rows;
        kernelArgs = args;
    }

    /**
     * Whether this engine will take the row block, and with what layout.
     *
     * <p>Called by the operator once it has seen a row, because the layout depends on the row: the
     * null-bit header is sized from the source arity and everything after it moves with it. Returns
     * false when the block path does not apply, and the operator then stages column by column as
     * before.
     *
     * <p>Refused for anything but all-{@code DOUBLE} inputs. The block is a {@code DoubleArray} and
     * the device reads slots out of it as doubles; a four-byte {@code INT} sits in the low half of
     * its slot and reading the slot as a double gives a number, not a failure. Recovering it needs
     * a bit reinterpretation and an {@code IntArray} output, which is a second family of kernels
     * for a case no measured query here has.
     */
    public boolean bindRowBlock(int sourceArity) {
        if (planBuilt || !BLOCK_STAGING) {
            return rowBlock != null;
        }
        final GpuKernelSource kernel = spec.kernel();
        final int[] fields = kernel.inputFieldIndexes();
        if (fields.length == 0 || fields.length > DeviceDeinterleave.MAX_COLUMNS) {
            return false;
        }
        for (GpuValueType type : kernel.inputTypes()) {
            if (type != GpuValueType.DOUBLE) {
                return false;
            }
        }
        if (kernel.carriesValidity()) {
            // The kernel reads a validity mask the host builds per column; the block path writes
            // no such mask, and a projection over nullable inputs is refused elsewhere anyway.
            return false;
        }
        blockSlots = RowBlockGather.slotsFor(sourceArity);
        columnSlots = new int[fields.length];
        for (int c = 0; c < fields.length; c++) {
            columnSlots[c] = RowBlockGather.slotOf(sourceArity, fields[c]);
        }
        rowBlock = (DoubleArray) allocate(GpuValueType.DOUBLE, spec.batchSize() * blockSlots);
        return true;
    }

    /** The block's buffer and TornadoVM's header size, for the gather that fills it. */
    public @Nullable RowBlockGather blockGather() {
        if (rowBlock == null) {
            return null;
        }
        return new RowBlockGather(GeneratedKernel.segmentOf(rowBlock).asByteBuffer(), blockSlots);
    }

    /** Builds the task graph, on first use, once the block decision is settled. */
    private void buildPlan() throws Exception {
        if (planBuilt) {
            return;
        }
        planBuilt = true;
        final GpuKernelSource kernel = spec.kernel();
        final int batchSize = spec.batchSize();

        TaskGraph graph = new TaskGraph("calc");
        if (rowBlock != null) {
            // C. One transfer of the row block, and the columns the kernel reads are produced on
            // the device by the task below rather than sent. Nothing about `inputs` changes --
            // they are the same buffers, filled in a different place -- so the generated kernel is
            // untouched by this path.
            graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, rowBlock);
        } else {
            graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, inputs);
        }
        graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, rows);
        if (inNulls != null) {
            graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, inNulls);
        }
        if (rowBlock != null) {
            graph = graph.task("deinterleave", deinterleaveEntry(), deinterleaveArgs());
        }
        // Naming the kernel by Method rather than by a method reference is what makes a generated
        // kernel possible at all: a method reference would have to exist in source.
        graph = graph.task("kernel", entry, kernelArgs);
        Object[] results;
        if (aggregate != null) {
            // The two columns the group-by reads are the ones the task above just wrote, still on
            // the device. Naming them by field lets a grouping key be either a computed column or
            // a staged one that the projection only passed through.
            graph =
                    graph.libraryTask(
                            "agg",
                            Cudf::groupSum,
                            batchSize,
                            (IntArray) bufferFor(aggregate.keyField()),
                            (DoubleArray) bufferFor(aggregate.valueField()),
                            groupKeys,
                            groupSums,
                            groupCount);
            // One row a group comes back, not one a row. On the query this was first measured on
            // that is four orders of magnitude less to copy out.
            results = new Object[] {groupKeys, groupSums, groupCount};
        } else {
            List<Object> back = new ArrayList<>(Arrays.asList(outputs));
            if (mask != null) {
                back.add(mask);
            }
            if (outNulls != null) {
                back.add(outNulls);
            }
            results = back.toArray();
        }
        if (compacting) {
            // Nothing the kernel wrote goes to the host from here: the projections and the mask
            // stay where they are and the next graph reads them in place. That alone saves no
            // bytes -- the next graph would copy out buffers of the same size -- and the saving
            // arrives with the prefix fetch in fetchSurvivors.
            graph = graph.persistOnDevice(concat(outputs, mask, rows));
        } else {
            graph = graph.transferToHost(DataTransferMode.EVERY_EXECUTION, results);
        }

        // An explicit iteration space, and it is not optional.
        //
        // TornadoVM infers one from the @Parallel loop's bound when it can. It cannot here: since
        // M2.5 the bound is read from a buffer, so it is not known when the kernel is compiled,
        // and the inference quietly gives up and emits a sequential loop. Quietly is the word --
        // results stay correct and the kernel got about 1,500x slower, 4ms to 6s over 2M rows,
        // which no correctness test can see. This project has now been caught by a silently
        // sequential kernel twice.
        //
        // Stating the grid removes the inference from the path entirely: the launch is this many
        // threads because it was asked for, and setGlobalWork below narrows it per batch so a
        // short batch launches short.
        grid = new WorkerGrid1D(batchSize);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("calc.kernel", grid);
        if (rowBlock != null) {
            // Same iteration space and the same narrowing, for the same reason the kernel needs
            // its grid stated: the loop bound is read from a buffer, so TornadoVM cannot infer one
            // and silently emits a sequential loop if it is not told.
            scheduler.addWorkerGrid("calc.deinterleave", grid);
        }
        if (compacting) {
            TaskGraph compact = compactionGraph(batchSize, scheduler);
            plan =
                    new TornadoExecutionPlan(graph.snapshot(), compact.snapshot())
                            .withGridScheduler(scheduler);
        } else {
            plan = new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(scheduler);
        }
        if (spec.requiresStrictArithmetic()) {
            // The planner found this kernel's value can reach something that decides which rows
            // come back, so the device must round a multiply and an add separately, as Flink's
            // own operator does. Scoped to this plan rather than the JVM, and it can only make
            // the arithmetic stricter than the deployment already asked for.
            plan = plan.withStrictFloatingPoint();
        }
        if (profile) {
            plan = plan.withProfiler(ProfilerMode.SILENT);
        }
    }

    /**
     * Off by default.
     *
     * <p>The kernel is 1.96x over the host gather on the transpose itself, but the transpose is a
     * small share of any pipeline measured here -- a Parquet source never reaches this path at all,
     * and on the CSV source the parse outweighs the whole operator. Default-on would put a second
     * staging mode in front of every row-major batch to move an end-to-end number by about a
     * percent, so it is opt-in until a pipeline is found where it is not.
     */
    static final boolean BLOCK_STAGING =
            Boolean.parseBoolean(
                    System.getProperty("flink.accelerator.tornadovm.blockStaging", "false"));

    private Method deinterleaveEntry() throws NoSuchMethodException {
        return DeviceDeinterleave.entryFor(inputs.length);
    }

    /** {@code (block, c0..cN, slots, off0..offN, rows)}, matching the arity's signature. */
    private Object[] deinterleaveArgs() {
        final Object[] args = new Object[1 + inputs.length + 1 + inputs.length + 1];
        int at = 0;
        args[at++] = rowBlock;
        for (Object in : inputs) {
            args[at++] = in;
        }
        args[at++] = blockSlots;
        for (int slot : columnSlots) {
            args[at++] = slot;
        }
        args[at] = rows;
        return args;
    }

    /**
     * Whether this kernel's filter can compact on the device rather than on the host.
     *
     * <p>Four conditions, and each one is a thing the compaction would otherwise get wrong rather
     * than merely a thing it has not been tried against:
     *
     * <ul>
     *   <li><b>There is a filter.</b> Without a mask there is nothing to compact, and the gather
     *       would be an identity copy costing a kernel and a buffer.
     *   <li><b>There is no grouped aggregate.</b> That path already ends on the device and returns
     *       one row a group; compacting before it would be work in front of a smaller answer.
     *   <li><b>Nothing carries validity.</b> Compacting a column means compacting its null bits
     *       too, in the same permutation, and the validity here is a packed word per row covering
     *       every column at once. Gathering that word is expressible; getting it wrong is a silent
     *       wrong answer, so it waits for a test that can see it.
     *   <li><b>There is at least one computed column.</b> A projection of nothing but pass-through
     *       columns has no output buffer to gather, and its rows are read from staging on the host
     *       whatever the device does.
     *   <li><b>cuDF can run here.</b> The compaction is a library task, and a library task for a
     *       backend that cannot serve it fails from {@code execute()} — after the batch has been
     *       consumed, where nothing can recover it. See {@link DeviceCompaction#available()}.
     * </ul>
     *
     * <p>Failing any of them is not a decline: the operator runs exactly as it did before, copying
     * the batch back and skipping masked-out rows while it builds output rows.
     */
    private boolean compactsOnDevice() {
        return mask != null
                && aggregate == null
                && !spec.kernel().carriesValidity()
                && outputs.length > 0
                && DeviceCompaction.available();
    }

    /** The consumer graph: widen the mask, compact with cuDF, gather, and return the survivors. */
    private TaskGraph compactionGraph(int batchSize, GridScheduler scheduler) {
        TaskGraph compact =
                new TaskGraph("compact")
                        .consumeFromDevice("calc", concat(outputs, mask, rows))
                        // FIRST_EXECUTION, not EVERY: every one of these is written by this
                        // graph before it is read, so copying the host's copy in on each batch
                        // allocates the buffer and then moves bytes nobody will look at. Measured
                        // on a 2M-row screening job, EVERY_EXECUTION cost 16 MiB of copy-in per
                        // job for the packed column alone -- a transfer the compaction exists to
                        // avoid, spent on the wrong side of the bus.
                        .transferToDevice(
                                DataTransferMode.FIRST_EXECUTION,
                                maskBytes,
                                selectedIndices,
                                survivorCount)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, compacted)
                        // rows is read here as well as in the kernel: cuDF captures its row count
                        // when the graph is built, so a partition's short last batch would compact
                        // whatever the previous batch left in the tail of the mask. Zeroing the
                        // tail while widening is what makes the count safe to fix at batchSize.
                        .task("widen", DeviceCompaction::widenMask, mask, maskBytes, rows)
                        .libraryTask(
                                "select",
                                Cudf::selectedIndices,
                                batchSize,
                                batchSize,
                                maskBytes,
                                selectedIndices,
                                survivorCount);
        scheduler.addWorkerGrid("compact.widen", new WorkerGrid1D(batchSize));
        for (int i = 0; i < outputs.length; i++) {
            String task = "gather" + i;
            compact = gatherTask(compact, task, outputs[i], compacted[i]);
            scheduler.addWorkerGrid("compact." + task, new WorkerGrid1D(batchSize));
        }
        // Only the count comes back in the graph: four bytes, and it is what says how much of
        // everything else is worth moving. The packed columns and the index array stay on the
        // device and are fetched afterwards, as a prefix. See drainCompacted below.
        return compact.transferToHost(DataTransferMode.EVERY_EXECUTION, survivorCount)
                .persistOnDevice(concat(compacted, selectedIndices));
    }

    /**
     * Fetches the survivors, and only the survivors.
     *
     * <p>This is the step that makes the compaction worth doing, and without it the whole design
     * saves nothing. A {@code transferToHost} inside a task graph moves a <em>whole buffer</em>,
     * and the packed columns are sized to the batch because nothing knows in advance how many rows
     * will survive — so packing the survivors at the front and then copying the entire buffer moves
     * exactly as many bytes as never packing them at all. Measured before this existed: 0.75 MiB
     * out for 65,536 rows at 10% selectivity, with the compaction on and with it off, to three
     * decimal places the same number.
     *
     * <p>{@code TornadoExecutionResult.transferToHost(DataRange)} is the way out. It is a transfer
     * issued <em>after</em> execution, on demand, for a sub-range — so the count can be read first
     * and the prefix sized from it. The count is the only thing the graph itself brings back.
     */
    private void fetchSurvivors(TornadoExecutionResult result) {
        int n = survivorCount.get(0);
        if (n <= 0) {
            // Nothing survived, so nothing is worth a transfer -- and a DataRange of size 0 means
            // "to the end of the array" rather than "nothing", which would copy the whole buffer.
            return;
        }
        long bytes = 0;
        for (Object column : compacted) {
            TornadoNativeArray array = (TornadoNativeArray) column;
            result.transferToHost(new DataRange(array).withSize(n));
            bytes += (long) n * array.getElementSize();
        }
        result.transferToHost(new DataRange(selectedIndices).withSize(n));
        bytes += (long) n * selectedIndices.getElementSize();

        // Counted here rather than read back from the profiler, because the profiler does not see
        // these at all: a partial transfer is issued after execution and clears the per-execution
        // counters on its way through. The figure is exact -- it is the size this method asked for.
        metrics.addOnDemandBytesOut(bytes);
    }

    /** One gather per column, bound to the pair of buffers whose width it knows how to move. */
    private TaskGraph gatherTask(TaskGraph graph, String task, Object source, Object target) {
        if (source instanceof IntArray) {
            return graph.task(
                    task,
                    DeviceCompaction::gatherInts,
                    selectedIndices,
                    survivorCount,
                    (IntArray) source,
                    (IntArray) target);
        }
        if (source instanceof FloatArray) {
            return graph.task(
                    task,
                    DeviceCompaction::gatherFloats,
                    selectedIndices,
                    survivorCount,
                    (FloatArray) source,
                    (FloatArray) target);
        }
        return graph.task(
                task,
                DeviceCompaction::gatherDoubles,
                selectedIndices,
                survivorCount,
                (DoubleArray) source,
                (DoubleArray) target);
    }

    private static Object[] concat(Object[] head, Object... tail) {
        Object[] all = new Object[head.length + tail.length];
        System.arraycopy(head, 0, all, 0, head.length);
        System.arraycopy(tail, 0, all, head.length, tail.length);
        return all;
    }

    private Method compile(GpuKernelSource kernel) throws Exception {
        generated = new GeneratedKernel(kernel);
        return generated.compile();
    }

    /** Write side of one staged input column. */
    /**
     * A newly allocated staging buffer of the given width.
     *
     * <p>TornadoVM's native arrays hold garbage on allocation, unlike Java arrays. The tail of a
     * partial batch is never read, but leaving it undefined would make device results differ
     * between runs.
     */
    /**
     * One staging buffer, on the slot's managed memory where Flink offered some.
     *
     * <p>Falling back to a private allocation is not a failure path and must not become one: a
     * benchmark driving this engine directly has no slot behind it, and neither does a plan
     * compiled before Flink learned to declare the memory.
     */
    private Object allocate(GpuValueType type, int batchSize) {
        if (staging == null) {
            return GeneratedKernel.allocate(type, batchSize);
        }
        return GeneratedKernel.allocateOn(
                type, batchSize, staging.allocate(GeneratedKernel.sizeOf(type, batchSize)));
    }

    /**
     * The write side of one staged column, bound once to its concrete buffer.
     *
     * <p>Returned rather than exposed as a {@code setInput(column, ...)} method so the narrowing
     * happens against a buffer the lambda already holds, instead of switching on the column's type
     * once per value.
     */
    public RowGather.StagingColumn inputColumn(int column) {
        return GeneratedKernel.writerFor(inputs[column]);
    }

    /** Off-heap buffer for one staged input column, for gathers that can bulk-copy into it. */
    public java.lang.foreign.MemorySegment inputSegment(int column) {
        return GeneratedKernel.segmentOf(inputs[column]);
    }

    /**
     * One computed value, boxed as the row type declares it.
     *
     * <p>A {@code FLOAT} output has to arrive downstream as a {@link Float} and an {@code INTEGER}
     * as an {@link Integer}: the field is written into a {@link
     * org.apache.flink.table.data.GenericRowData}, whose serializer reads it back at the declared
     * type and does not convert. Getting this wrong is a {@code ClassCastException} deep in the
     * drain rather than anything the type system catches, which is how an integer grouping key
     * announced itself the first time one reached here.
     */
    public Object output(int column, int position) {
        Object buffer = outputs[column];
        if (buffer instanceof FloatArray floats) {
            return floats.get(position);
        }
        if (buffer instanceof IntArray ints) {
            return ints.get(position);
        }
        return ((DoubleArray) buffer).get(position);
    }

    /** Marks a staged input column absent for this row. */
    public void setInputNull(int column, int position) {
        inNulls.set(position, inNulls.get(position) | (1 << column));
    }

    /** Clears every input validity bit for this row, which is the staging default. */
    public void clearInputNulls(int position) {
        inNulls.set(position, 0);
    }

    /** Whether this kernel moves validity at all. */
    public boolean carriesValidity() {
        return inNulls != null;
    }

    /** Whether the computed column at this position came back absent. */
    public boolean outputIsNull(int column, int position) {
        return outNulls != null && ((outNulls.get(position) >>> column) & 1) != 0;
    }

    public boolean selected(int position) {
        return mask == null || mask.get(position) != 0;
    }

    /**
     * Whether the filter compacted on the device, so the caller must drain survivors by index.
     *
     * <p>The two drains are not interchangeable and the difference is not cosmetic. Without
     * compaction the mask is on the host and {@link #selected} answers for every staged position;
     * with it, neither the mask nor the uncompacted outputs ever came back, so {@link #selected}
     * would be reading whatever the host copy last held.
     */
    public boolean compactsOnDeviceNow() {
        return compacting;
    }

    /** How many rows of the last batch survived the filter. Only meaningful while compacting. */
    public int survivors() {
        return survivorCount.get(0);
    }

    /**
     * Where the <em>j</em>th survivor was staged.
     *
     * <p>Needed because a projection's pass-through fields are read from the staging buffers on the
     * host, which the device never compacted and had no reason to: they are already there.
     */
    public int survivorPosition(int j) {
        return selectedIndices.get(j);
    }

    /** One computed value of the <em>j</em>th survivor, boxed as the row type declares it. */
    public Object compactedOutput(int column, int j) {
        Object buffer = compacted[column];
        if (buffer instanceof FloatArray floats) {
            return floats.get(j);
        }
        if (buffer instanceof IntArray ints) {
            return ints.get(j);
        }
        return ((DoubleArray) buffer).get(j);
    }

    /**
     * How many kernels this JVM has actually compiled.
     *
     * <p>Exposed for tests, and only meaningful as a difference across an operation: the cache is
     * shared by everything in the process, so an absolute value says nothing.
     */
    public static int compilationCount() {
        return GeneratedKernel.COMPILATIONS.get();
    }

    public OffloadMetrics metrics() {
        return metrics;
    }

    /** One batch's device timings, pending the caller's gather and drain numbers. */
    public static final class Execution {
        private final long wallNanos;
        private final @Nullable DeviceProfile profile;

        Execution(long wallNanos, @Nullable DeviceProfile profile) {
            this.wallNanos = wallNanos;
            this.profile = profile;
        }
    }

    /**
     * The runtime's numbers for one batch, read eagerly.
     *
     * <p>A {@code TornadoProfilerResult} is a live view of the executor, not a snapshot: its
     * getters answer with whatever the counters hold when they are called. The compaction's
     * on-demand prefix transfer resets those counters, so holding the result and reading it later
     * yields zeros for a batch that demonstrably moved data. Copying the six figures out at the
     * moment they are true is the whole of the fix.
     */
    public static final class DeviceProfile {
        private final long copyInNanos;
        private final long kernelNanos;
        private final long copyOutNanos;
        private final long compileNanos;
        private final long bytesIn;
        private final long bytesOut;

        public long copyInNanos() {
            return copyInNanos;
        }

        public long kernelNanos() {
            return kernelNanos;
        }

        public long copyOutNanos() {
            return copyOutNanos;
        }

        public long compileNanos() {
            return compileNanos;
        }

        public long bytesIn() {
            return bytesIn;
        }

        public long bytesOut() {
            return bytesOut;
        }

        DeviceProfile(TornadoProfilerResult result) {
            this.copyInNanos = result.getDeviceWriteTime();
            this.kernelNanos = result.getDeviceKernelTime();
            this.copyOutNanos = result.getDeviceReadTime();
            this.compileNanos = result.getCompileTime();
            this.bytesIn = result.getTotalBytesCopyIn();
            this.bytesOut = result.getTotalBytesCopyOut();
        }
    }

    /**
     * Runs the staged batch.
     *
     * @param stagedRows how many rows were staged. The kernel processes exactly this many; the tail
     *     of the buffers is left alone rather than evaluated and discarded.
     */
    public Execution execute(int stagedRows) {
        try {
            // Built here rather than in open(), because the row-block layout is not knowable
            // until a row has arrived. A no-op after the first batch.
            buildPlan();
        } catch (Exception cannotBuild) {
            throw new IllegalStateException("could not build the device plan", cannotBuild);
        }
        rows.set(0, stagedRows);
        boolean tail = aggregate != null && stagedRows != spec.batchSize();
        WorkerGrid1D live = tail ? tailGrid() : grid;
        live.setGlobalWork(stagedRows, 1, 1);
        TornadoExecutionPlan livePlan = tail ? tailPlan : plan;
        long t0 = System.nanoTime();
        TornadoExecutionResult result;
        DeviceProfile profiled = null;
        if (compacting) {
            // Two graphs, one plan. The second reads what the first left on the device; running
            // them as two plans would give the consumer its own, separate buffers.
            TornadoExecutionResult staged = withKernelLoader(() -> livePlan.withGraph(0).execute());
            // Snapshotted after the producer and before the consumer, which is where the staging
            // figures are true. The runtime's counters are per execution: reading them after the
            // second graph reports that graph's transfers and loses the batch's input staging
            // entirely, which showed up as a copy-in of exactly one column's worth where five
            // columns had demonstrably been staged.
            profiled = profile ? new DeviceProfile(staged.getProfilerResult()) : null;
            result = withKernelLoader(() -> livePlan.withGraph(1).execute());
            fetchSurvivors(result);
        } else {
            result = withKernelLoader(livePlan::execute);
        }
        if (tail) {
            groupOnHost(stagedRows);
        } else if (aggregate != null) {
            groups = groupCount.get(0);
        }
        long wall = System.nanoTime() - t0;
        if (!compacting) {
            profiled = profile ? new DeviceProfile(result.getProfilerResult()) : null;
        }
        return new Execution(wall, profiled);
    }

    /**
     * How many groups the last execution produced. Only meaningful on an engine with an aggregate.
     */
    public int groups() {
        return groups;
    }

    /** The grouping key of one group of the last execution. */
    public int groupKey(int group) {
        return groupKeys.get(group);
    }

    /** The partial sum of one group of the last execution. */
    public double groupSum(int group) {
        return groupSums.get(group);
    }

    /**
     * The buffer holding one field of the kernel's output row.
     *
     * <p>A field is either computed -- the kernel's own outputs fill those slots in order -- or
     * copied straight from a staged input column, which the kernel never writes but which is on the
     * device all the same. Both are addressable, so a {@code GROUP BY} on a raw column costs no
     * more than one on an expression.
     */
    private Object bufferFor(int field) {
        int computed = spec.computedSlot(field);
        return computed >= 0 ? outputs[computed] : inputs[spec.stagedColumn(field)];
    }

    /**
     * Builds the projection-only plan for a partition's last, short batch.
     *
     * <p>Lazily, because most operator instances never reach the branch on a full partition and the
     * build costs a device compilation of the same kernel.
     */
    private WorkerGrid1D tailGrid() {
        if (tailPlan != null) {
            return tailGrid;
        }
        TaskGraph graph = new TaskGraph("tail");
        graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, inputs);
        graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, rows);
        if (inNulls != null) {
            graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, inNulls);
        }
        graph = graph.task("kernel", entry, kernelArgs);
        graph =
                graph.transferToHost(
                        DataTransferMode.EVERY_EXECUTION,
                        bufferFor(aggregate.keyField()),
                        bufferFor(aggregate.valueField()));
        tailGrid = new WorkerGrid1D(spec.batchSize());
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("tail.kernel", tailGrid);
        tailPlan = new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(scheduler);
        if (profile) {
            tailPlan = tailPlan.withProfiler(ProfilerMode.SILENT);
        }
        return tailGrid;
    }

    /**
     * Groups the tail batch the ordinary way, writing into the same buffers cuDF would have.
     *
     * <p>The caller downstream cannot tell the difference, and does not need to: a partial
     * aggregate is allowed to emit several rows per key, so nothing about the tail has to match
     * what the device produced for the batches before it.
     */
    private void groupOnHost(int stagedRows) {
        IntArray keys = (IntArray) bufferFor(aggregate.keyField());
        DoubleArray values = (DoubleArray) bufferFor(aggregate.valueField());
        Map<Integer, Double> sums = new HashMap<>();
        for (int i = 0; i < stagedRows; i++) {
            sums.merge(keys.get(i), values.get(i), Double::sum);
        }
        groups = 0;
        for (Map.Entry<Integer, Double> group : sums.entrySet()) {
            groupKeys.set(groups, group.getKey());
            groupSums.set(groups, group.getValue());
            groups++;
        }
    }

    /**
     * Runs {@code action} with the generated kernel's loader as the context class loader.
     *
     * <p>TornadoVM finds a kernel's {@code @Parallel} annotations by reading its class file as a
     * resource. A class compiled at run time lives in a loader of our making, so unless that loader
     * is reachable the annotation scan comes up empty -- and the failure is silent: the kernel is
     * emitted as a sequential loop which every device thread runs in full, giving correct results
     * about a thousand times slower than it should.
     */
    private <T> T withKernelLoader(java.util.function.Supplier<T> action) {
        Thread current = Thread.currentThread();
        ClassLoader previous = current.getContextClassLoader();
        current.setContextClassLoader(generated.loader());
        try {
            return action.get();
        } finally {
            current.setContextClassLoader(previous);
        }
    }

    public void recordBatch(
            int count, long gatherNanos, Execution execution, int emitted, long drainNanos) {
        metrics.recordBatch(
                count, emitted, gatherNanos, execution.wallNanos, drainNanos, execution.profile);
    }

    @Override
    public void close() throws Exception {
        if (plan != null) {
            plan.close();
            plan = null;
        }
        if (tailPlan != null) {
            tailPlan.close();
            tailPlan = null;
        }
        if (generated != null) {
            generated.close();
            generated = null;
        }
    }

    private static void deleteRecursively(Path dir) throws IOException {
        if (!Files.exists(dir)) {
            return;
        }
        try (Stream<Path> paths = Files.walk(dir)) {
            for (Path path : (Iterable<Path>) paths.sorted(Comparator.reverseOrder())::iterator) {
                Files.deleteIfExists(path);
            }
        }
    }
}
