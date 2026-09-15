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

import org.apache.flink.table.gpu.codegen.GpuCalcSpec;
import org.apache.flink.table.gpu.codegen.GpuKernelSource;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.gather.RowGather;
import org.apache.flink.table.gpu.metrics.OffloadMetrics;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.io.IOException;
import java.lang.reflect.Method;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
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
    private GeneratedKernel generated;
    private TornadoExecutionPlan plan;

    private final OffloadMetrics metrics = new OffloadMetrics();

    public GeneratedKernelEngine(GpuCalcSpec spec, boolean profile) {
        this.spec = spec;
        this.profile = profile;
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

        Method entry = compile(kernel);

        Object[] args = new Object[inputs.length + outputs.length + (mask == null ? 0 : 1)];
        int at = 0;
        for (Object in : inputs) {
            args[at++] = in;
        }
        for (Object out : outputs) {
            args[at++] = out;
        }
        if (mask != null) {
            args[at] = mask;
        }

        TaskGraph graph = new TaskGraph("calc");
        graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, inputs);
        // Naming the kernel by Method rather than by a method reference is what makes a generated
        // kernel possible at all: a method reference would have to exist in source.
        graph = graph.task("kernel", entry, args);
        Object[] results =
                mask == null
                        ? outputs
                        : Stream.concat(Arrays.stream(outputs), Stream.<Object>of(mask)).toArray();
        graph = graph.transferToHost(DataTransferMode.EVERY_EXECUTION, results);

        plan = new TornadoExecutionPlan(graph.snapshot());
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
    private static Object allocate(GpuValueType type, int batchSize) {
        return GeneratedKernel.allocate(type, batchSize);
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
     * <p>A {@code FLOAT} output has to arrive downstream as a {@link Float}: the field is written
     * into a {@link org.apache.flink.table.data.GenericRowData}, whose serializer reads it back at
     * the declared type and does not convert.
     */
    public Object output(int column, int position) {
        Object buffer = outputs[column];
        return buffer instanceof FloatArray floats
                ? (Object) floats.get(position)
                : (Object) ((DoubleArray) buffer).get(position);
    }

    public boolean selected(int position) {
        return mask == null || mask.get(position) != 0;
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

    public Execution execute() {
        long t0 = System.nanoTime();
        TornadoExecutionResult result = withKernelLoader(plan::execute);
        long wall = System.nanoTime() - t0;
        return new Execution(wall, profile ? result.getProfilerResult() : null);
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
