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

import org.apache.flink.table.gpu.codegen.GpuKernelSource;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.gather.RowGather;

import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Method;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * Compiles a generated kernel and allocates the buffers it reads and writes.
 *
 * <p>Extracted so the {@code Calc} operator and the fused Calc-and-aggregate operator share this
 * rather than each carrying a copy. Little of it survives being reimplemented from memory: the
 * {@code -g} flag without which {@code @Parallel} is silently ignored and the kernel becomes a
 * sequential loop every thread runs in full; the classpath derived from where classes actually came
 * from rather than from {@code java.class.path}, which does not mention a module path; and the
 * width each column is staged at.
 */
final class GeneratedKernel implements AutoCloseable {

    private final GpuKernelSource kernel;
    private Path workDir;
    private URLClassLoader loader;

    GeneratedKernel(GpuKernelSource kernel) {
        this.kernel = kernel;
    }

    Method compile() throws Exception {
        final GpuKernelSource kernel = this.kernel;
        JavaCompiler javac = ToolProvider.getSystemJavaCompiler();
        if (javac == null) {
            throw new IllegalStateException(
                    "No Java compiler available: GPU offload generates kernels at run time and so "
                            + "needs a JDK, not a JRE. TornadoVM requires one in any case.");
        }
        workDir = Files.createTempDirectory("flink-gpu-kernel");
        // The generated class name contains '$' to keep it distinct from anything hand-written;
        // the file must be named after the class for javac to accept it.
        String fileName = kernel.className() + ".java";
        Path source = workDir.resolve(fileName);
        Files.writeString(source, kernel.source());

        List<String> options =
                new ArrayList<>(
                        Arrays.asList(
                                "-classpath",
                                classpathFor(),
                                "-d",
                                workDir.toString(),
                                "--enable-preview",
                                "-source",
                                "21",
                                "-target",
                                "21",
                                // Debug info is not optional. @Parallel is a local-variable
                                // annotation, and TornadoVM associates it with the loop induction
                                // variable through the local variable table. Without -g javac emits
                                // no such table, the annotation is silently ignored, and the kernel
                                // is generated as a sequential loop that every GPU thread runs in
                                // full -- correct results, catastrophically slow, and no warning.
                                "-g",
                                "-nowarn"));
        options.add(source.toString());

        int rc = javac.run(null, null, System.err, options.toArray(new String[0]));
        if (rc != 0) {
            throw new IllegalStateException(
                    "Generated kernel did not compile. This is a bug in the generator; the source "
                            + "was:\n"
                            + kernel.source());
        }

        loader =
                new URLClassLoader(
                        new URL[] {workDir.toUri().toURL()},
                        GeneratedKernelEngine.class.getClassLoader());
        Class<?> generated = loader.loadClass(kernel.className());
        for (Method m : generated.getDeclaredMethods()) {
            if (m.getName().equals(kernel.methodName())) {
                return m;
            }
        }
        throw new IllegalStateException("generated class has no method " + kernel.methodName());
    }

    private static String classpathFor() {
        LinkedHashSet<String> entries = new LinkedHashSet<>();
        String systemPath = System.getProperty("java.class.path", "");
        if (!systemPath.isEmpty()) {
            entries.addAll(Arrays.asList(systemPath.split(java.io.File.pathSeparator)));
        }
        for (Class<?> referenced :
                new Class<?>[] {
                    uk.ac.manchester.tornado.api.annotations.Parallel.class,
                    uk.ac.manchester.tornado.api.math.TornadoMath.class,
                    DoubleArray.class,
                    FloatArray.class,
                    IntArray.class
                }) {
            String location = codeSourceOf(referenced);
            if (location != null) {
                entries.add(location);
            }
        }
        return String.join(java.io.File.pathSeparator, entries);
    }

    private static String codeSourceOf(Class<?> type) {
        try {
            ProtectionDomain domain = type.getProtectionDomain();
            if (domain == null || domain.getCodeSource() == null) {
                return null;
            }
            URL location = domain.getCodeSource().getLocation();
            return location == null ? null : Paths.get(location.toURI()).toString();
        } catch (URISyntaxException | RuntimeException e) {
            return null;
        }
    }

    static Object allocate(GpuValueType type, int batchSize) {
        switch (type) {
            case INT:
                IntArray ints = new IntArray(batchSize);
                ints.init(0);
                return ints;
            case FLOAT:
                FloatArray floats = new FloatArray(batchSize);
                floats.init(0.0f);
                return floats;
            default:
                DoubleArray doubles = new DoubleArray(batchSize);
                doubles.init(0.0);
                return doubles;
        }
    }

    /**
     * The same buffer, but on memory Flink reserved rather than memory we took.
     *
     * <p>{@code fromSegmentShallow} wraps without copying, so the array's elements <em>are</em> the
     * slot's managed memory and the device writes straight into it. What that buys is accounting
     * rather than speed: the slot's budget knows the staging exists, two accelerated operators
     * sharing a slot cannot each assume the whole machine, and a deployment no longer has to be
     * told by hand how much off-heap to set aside.
     *
     * <p>The layout is TornadoVM's. Its native arrays reserve {@link
     * TornadoNativeArray#ARRAY_HEADER} bytes at the front for a header and store elements after it,
     * so the buffer has to be that much larger than the data and the element count falls out of the
     * remaining size. Getting this wrong does not fail loudly — it silently shifts every element —
     * so {@code sizeOf} below is the only place that arithmetic is written down.
     *
     * @param buffer off-heap memory from {@code AcceleratorContext.allocateOffHeap}, of exactly
     *     {@link #sizeOf} bytes
     */
    static Object allocateOn(GpuValueType type, int batchSize, ByteBuffer buffer) {
        MemorySegment segment = MemorySegment.ofBuffer(buffer);
        long expected = sizeOf(type, batchSize);
        if (segment.byteSize() != expected) {
            throw new IllegalArgumentException(
                    "staging buffer is "
                            + segment.byteSize()
                            + " bytes, expected "
                            + expected
                            + " for "
                            + batchSize
                            + " elements of "
                            + type);
        }
        switch (type) {
            case INT:
                return IntArray.fromSegmentShallow(segment);
            case FLOAT:
                return FloatArray.fromSegmentShallow(segment);
            default:
                return DoubleArray.fromSegmentShallow(segment);
        }
    }

    /** Bytes one staged column needs: TornadoVM's header, then the elements. */
    static int sizeOf(GpuValueType type, int batchSize) {
        return (int) TornadoNativeArray.ARRAY_HEADER + batchSize * type.widthInBytes();
    }

    static RowGather.StagingColumn writerFor(Object buffer) {
        if (buffer instanceof IntArray ints) {
            return (position, value) -> ints.set(position, (int) value);
        }
        if (buffer instanceof FloatArray floats) {
            return (position, value) -> floats.set(position, (float) value);
        }
        DoubleArray doubles = (DoubleArray) buffer;
        return doubles::set;
    }

    static java.lang.foreign.MemorySegment segmentOf(Object buffer) {
        return buffer instanceof DoubleArray doubles ? doubles.getSegment() : null;
    }

    /** The loader the generated class lives in; TornadoVM reads its bytecode as a resource. */
    ClassLoader loader() {
        return loader;
    }

    @Override
    public void close() throws IOException {
        if (loader != null) {
            loader.close();
            loader = null;
        }
        if (workDir != null && Files.exists(workDir)) {
            try (java.util.stream.Stream<Path> paths = Files.walk(workDir)) {
                paths.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
            workDir = null;
        }
    }
}
