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
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.gather.RowGather;
import org.apache.flink.table.gpu.metrics.OffloadMetrics;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
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
        this.spec = spec;
        this.aggregate = aggregate;
        this.profile = profile;
        this.staging = staging;
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
        if (kernel.hasFilter()) {
            mask = new IntArray(batchSize);
            mask.init(0);
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

        TaskGraph graph = new TaskGraph("calc");
        graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, inputs);
        graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, rows);
        if (inNulls != null) {
            graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, inNulls);
        }
        // Naming the kernel by Method rather than by a method reference is what makes a generated
        // kernel possible at all: a method reference would have to exist in source.
        graph = graph.task("kernel", entry, args);
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
        graph = graph.transferToHost(DataTransferMode.EVERY_EXECUTION, results);

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
        plan = new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(scheduler);
        if (profile) {
            plan = plan.withProfiler(ProfilerMode.SILENT);
        }
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
        private final TornadoProfilerResult profilerResult;

        Execution(long wallNanos, TornadoProfilerResult profilerResult) {
            this.wallNanos = wallNanos;
            this.profilerResult = profilerResult;
        }
    }

    /**
     * Runs the staged batch.
     *
     * @param stagedRows how many rows were staged. The kernel processes exactly this many; the tail
     *     of the buffers is left alone rather than evaluated and discarded.
     */
    public Execution execute(int stagedRows) {
        rows.set(0, stagedRows);
        boolean tail = aggregate != null && stagedRows != spec.batchSize();
        WorkerGrid1D live = tail ? tailGrid() : grid;
        live.setGlobalWork(stagedRows, 1, 1);
        TornadoExecutionPlan livePlan = tail ? tailPlan : plan;
        long t0 = System.nanoTime();
        TornadoExecutionResult result = withKernelLoader(livePlan::execute);
        if (tail) {
            groupOnHost(stagedRows);
        } else if (aggregate != null) {
            groups = groupCount.get(0);
        }
        long wall = System.nanoTime() - t0;
        return new Execution(wall, profile ? result.getProfilerResult() : null);
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
                count,
                emitted,
                gatherNanos,
                execution.wallNanos,
                drainNanos,
                execution.profilerResult);
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
