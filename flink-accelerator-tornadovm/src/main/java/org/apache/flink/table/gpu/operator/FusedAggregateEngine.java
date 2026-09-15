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
import org.apache.flink.table.gpu.codegen.GpuKernelSource;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.gather.RowGather;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/**
 * A {@code Calc} and the {@code SUM}s above it, as one device pass.
 *
 * <p>The projection kernel writes its columns to device buffers and the contraction reads those
 * buffers in place. Nothing about the intermediate returns to the host: the only thing copied back
 * per batch is one partial sum per output column, a few kilobytes whatever the batch size, against
 * the megabytes the same pair costs when the Calc has to emit rows for a separate aggregate to
 * consume.
 *
 * <h2>Why the contraction is a library call, and why in FP64</h2>
 *
 * <p>{@code SUM(e1), ..., SUM(ek)} is {@code ones' * M}, which is one matrix-vector product, so it
 * is {@link CuBlas#cublasDgemv} in a single call. That contracts every output column at once -- the
 * fused multiple reduction {@code @Reduce} cannot express, and that k separate kernels cannot scale
 * to, since 136 of them is a graph TornadoVM refuses to build.
 *
 * <p>The {@code D} is deliberate. An earlier version of this class used a generated FP64
 * contraction precisely because the binding exposed only {@code cublasSgemv}, and accumulating
 * hundreds of thousands of {@code DOUBLE} values in FP32 would hand back an answer several digits
 * worse than the CPU plan's -- in a path whose entire premise is that the user did not ask for a
 * device and cannot tell one was used. A transparent optimisation may be slower than expected; it
 * may not return different numbers. Binding {@code cublasDgemv} removed the choice between the two.
 */
public final class FusedAggregateEngine implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(FusedAggregateEngine.class);

    /**
     * Threads per reduced column.
     *
     * <p>Enough to fill a device and small enough that the host-side combine is free. It also fixes
     * the summation order for a given batch size, so two runs of the same query agree exactly even
     * though neither agrees bit for bit with the CPU's left-to-right order.
     */
    private final GpuAggregateSpec spec;

    private final int batchSize;

    /** Staged input columns, one per column the kernel reads, at their declared widths. */
    private Object[] inputs;

    /** The Calc's outputs, which the contraction reads without them leaving the device. */
    /**
     * The Calc's outputs as one column-major matrix, read in place by the contraction.
     *
     * <p>One buffer rather than one per column, because a task's argument list is fixed and
     * <em>k</em> buffers force <em>k</em> reduction tasks. That stops working: sixteen feature
     * columns make the gram query 136 sums, and a 137-task graph gets {@code Tornado Graph resize
     * not implemented yet}. One matrix is one task at any width, and is the layout a GEMV wants.
     */
    private DoubleArray projected;

    /** 0/1 per row when the Calc has a condition; all ones when it does not. */
    private IntArray mask;

    /** One sum per contracted column, written by the contraction and the only buffer read back. */
    private DoubleArray partials;

    /**
     * The vector the contraction multiplies by: 1.0 for a row that counts, 0.0 otherwise.
     *
     * <p>This is where the mask goes. {@code SUM} over the rows a filter kept is {@code weights' M}
     * with the filter in {@code weights}, so a masked row contributes zero through the arithmetic
     * rather than a branch, and the library call needs no notion of masking at all.
     */
    private DoubleArray weights;

    /** Which Calc output column each contracted slot came from, fixed at open. */
    private int[] contractedColumns;

    private final double[] totals;
    private long contributingRows;
    private long stagedRows;
    private int buffered;

    private GeneratedKernel kernel;
    private TornadoExecutionPlan plan;

    public FusedAggregateEngine(GpuAggregateSpec spec) {
        this.spec = spec;
        this.batchSize = spec.batchSize();
        this.totals = new double[spec.sumSources().length];
    }

    /** Which Calc output columns are contracted, in output-field order, ignoring COUNT(*). */
    private int[] contractedColumns() {
        List<Integer> columns = new ArrayList<>();
        for (int source : spec.sumSources()) {
            if (source != GpuAggregateSpec.COUNT_STAR) {
                columns.add(source);
            }
        }
        int[] result = new int[columns.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = columns.get(i);
        }
        return result;
    }

    public void open() throws Exception {
        final GpuKernelSource source = spec.kernel();
        kernel = new GeneratedKernel(source);
        Method entry = kernel.compile();

        inputs = new Object[source.inputFieldIndexes().length];
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = GeneratedKernel.allocate(source.inputTypes()[i], batchSize);
        }
        projected = new DoubleArray(batchSize * source.outputCount());
        projected.init(0.0);
        mask = new IntArray(batchSize);
        mask.init(1);
        weights = new DoubleArray(batchSize);
        weights.init(1.0);

        int[] contracted = contractedColumns();
        contractedColumns = contracted;
        partials = new DoubleArray(contracted.length);
        partials.init(0.0);

        Object[] kernelArgs = new Object[inputs.length + 1 + (source.hasFilter() ? 1 : 0)];
        int at = 0;
        for (Object in : inputs) {
            kernelArgs[at++] = in;
        }
        kernelArgs[at++] = projected;
        if (source.hasFilter()) {
            kernelArgs[at] = mask;
        }

        TaskGraph graph =
                new TaskGraph("fused-aggregate")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, inputs)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, mask, weights)
                        // `projected` is deliberately in no transfer list. It is written by the
                        // projection and read by the contraction, both on the device, and nothing
                        // on the host ever looks at it. It was once declared in both directions on
                        // the theory that TornadoVM will not keep an intermediate it has not been
                        // told about; that theory was wrong -- ReferenceJoinEngine's scores matrix
                        // appears in no list and is correct -- and the wrong sum it was meant to
                        // explain turned out to be a staging bug in the operator instead. Copying
                        // it cost `batchSize * outputCount` doubles in each direction per batch,
                        // which for a 32-column gram matrix is over a gigabyte each way.
                        .task("project", entry, kernelArgs);
        if (source.hasFilter()) {
            // Fold the device-computed mask into the weights, so the contraction stays a single
            // library call that knows nothing about filtering.
            graph = graph.task("mask", FusedAggregateEngine::applyMask, mask, weights);
        }
        // y = A' x: A is the rows-by-columns matrix the projection wrote, column-major, leading
        // dimension the batch size; x is the weights. One call contracts every column at once,
        // which is the fused multiple reduction @Reduce cannot express and k separate kernels
        // could not scale to -- 136 of them is a graph TornadoVM refuses to build.
        graph =
                graph.libraryTask(
                        "sum",
                        CuBlas::cublasDgemv,
                        CuBlasOperation.CUBLAS_OP_T.operation(),
                        batchSize,
                        contracted.length,
                        1.0,
                        projected,
                        batchSize,
                        weights,
                        1,
                        0.0,
                        partials,
                        1);
        graph = graph.transferToHost(DataTransferMode.EVERY_EXECUTION, partials);

        // Every task's iteration space, stated rather than inferred.
        GridScheduler grid = new GridScheduler();
        grid.addWorkerGrid("fused-aggregate.project", new WorkerGrid1D(batchSize));
        if (source.hasFilter()) {
            grid.addWorkerGrid("fused-aggregate.mask", new WorkerGrid1D(batchSize));
        }
        plan = new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(grid);
    }

    /** Folds a 0/1 mask into the weight vector the contraction multiplies by. */
    public static void applyMask(IntArray mask, DoubleArray weights) {
        for (@Parallel int i = 0; i < weights.getSize(); i++) {
            weights.set(i, weights.get(i) * mask.get(i));
        }
    }

    /** The write side of one staged column, bound to its buffer. */
    public RowGather.StagingColumn inputColumn(int column) {
        return GeneratedKernel.writerFor(inputs[column]);
    }

    /** Off-heap buffer for a staged column, for a gather that can bulk-copy into it. */
    public java.lang.foreign.MemorySegment inputSegment(int column) {
        return GeneratedKernel.segmentOf(inputs[column]);
    }

    /**
     * Where the next row's values belong in the staging buffers.
     *
     * <p>Exposed because the gather writes by position and only the engine knows how many rows are
     * buffered. The operator wrote every row to position 0 for five cluster runs, which produced a
     * wrong sum that survived four unrelated fixes to the kernel.
     */
    public int position() {
        return buffered;
    }

    /** Ends a row. Returns true if the batch filled and ran. */
    public boolean rowComplete() throws Exception {
        stagedRows++;
        if (++buffered == batchSize) {
            execute();
            return true;
        }
        return false;
    }

    /** Runs whatever is staged. Safe on an empty batch. */
    public void flush() throws Exception {
        if (buffered > 0) {
            execute();
        }
    }

    /**
     * Runs one batch and folds its partials into the running totals.
     *
     * <p>A short final batch is handled by zeroing the mask beyond what was staged rather than by
     * resizing: a {@link TornadoExecutionPlan} fixes buffer sizes, and a masked row contributes
     * exactly zero to a sum, so the padding is exact rather than approximate.
     */
    /** Log the shape of the first batch's partials once, so a wrong sum explains itself. */
    private static final boolean DIAGNOSE_FIRST_BATCH = true;

    private boolean diagnosed;

    private void execute() throws Exception {
        for (int i = buffered; i < batchSize; i++) {
            mask.set(i, 0);
            weights.set(i, 0.0);
        }
        withKernelLoader(
                () -> {
                    try {
                        return plan.execute();
                    } catch (Exception e) {
                        throw new IllegalStateException("fused aggregate execution failed", e);
                    }
                });
        for (int column = 0; column < contractedColumns.length; column++) {
            totals[column] += partials.get(column);
        }
        for (int i = 0; i < buffered; i++) {
            if (mask.get(i) != 0) {
                contributingRows++;
            }
        }
        if (buffered < batchSize) {
            mask.init(1);
            weights.init(1.0);
        }
        buffered = 0;
    }

    /**
     * Runs {@code action} with the generated kernel's loader as the context class loader.
     *
     * <p>The same reason the Calc engine does it: TornadoVM reads a kernel's bytecode as a resource
     * through the context loader, and a class compiled at run time lives in a loader of our making.
     * Without this the {@code @Parallel} annotation is not found and the loop is emitted as a
     * sequential one that every device thread runs in full.
     */
    private <T> T withKernelLoader(java.util.function.Supplier<T> action) {
        Thread current = Thread.currentThread();
        ClassLoader previous = current.getContextClassLoader();
        current.setContextClassLoader(kernel.loader());
        try {
            return action.get();
        } finally {
            current.setContextClassLoader(previous);
        }
    }

    /** Accumulated sum for the {@code i}-th contracted column. */
    public double total(int contractedIndex) {
        return totals[contractedIndex];
    }

    /** Rows that survived the Calc's condition, which is what {@code COUNT(*)} reports. */
    public long contributingRows() {
        return contributingRows;
    }

    /** Rows staged, whether or not they survived a filter. */
    public long stagedRows() {
        return stagedRows;
    }

    @Override
    public void close() throws Exception {
        if (plan != null) {
            plan.close();
            plan = null;
        }
        if (kernel != null) {
            kernel.close();
            kernel = null;
        }
    }

    /**
     * Whether the fused device path agrees with the CPU on hardware.
     *
     * <p>It did not, for five cluster runs, and the fault was never where it was looked for. The
     * device returned <b>31,919.56</b> against the CPU's <b>7.0779784665051e9</b>, identically
     * every time -- and that invariance was the signal: four changes to the contraction, the
     * transfer declarations, the class loader and the grid altered nothing because none of them was
     * on the path.
     *
     * <p>The operator wrote every row to staging position 0. So the input buffer held one real row
     * and 262,143 zeros, and the query made that almost invisible: the benchmark's twenty reference
     * points include (0, 0) exactly, so {@code LEAST(...)} over a zeroed row is 0. One non-zero
     * projected element, summed across eight batches, is 31,919.
     *
     * <p>What found it was running this engine standalone on a local GPU, where 1,000,000 rows
     * summed exactly. That put the engine beyond suspicion and left only the operator feeding it.
     * The lesson is the cheap one: the reproducer should have come after the second identical
     * result, not the fifth, and it should have driven the real code rather than a copy of its
     * shape -- a hand-built approximation of the same graph passed every variant.
     */
    private static final boolean VERIFIED_ON_HARDWARE = true;

    /** Whether a spec's contracted columns are all types this engine reduces. */
    public static boolean canContract(GpuAggregateSpec spec) {
        if (!VERIFIED_ON_HARDWARE) {
            return false;
        }
        GpuValueType[] outputs = spec.kernel().outputTypes();
        for (int source : spec.sumSources()) {
            if (source == GpuAggregateSpec.COUNT_STAR) {
                continue;
            }
            if (source < 0 || source >= outputs.length || outputs[source] != GpuValueType.DOUBLE) {
                // A FLOAT column would need its own reduction kernel, and is the case where the
                // library becomes the right answer rather than a demotion. Neither exists yet.
                return false;
            }
        }
        return true;
    }
}
