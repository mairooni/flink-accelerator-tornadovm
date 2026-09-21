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

import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.FileObject;
import javax.tools.ForwardingJavaFileManager;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Method;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.file.Paths;
import java.security.ProtectionDomain;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Compiles a generated kernel and allocates the buffers it reads and writes.
 *
 * <p>Extracted so the {@code Calc} operator and the fused Calc-and-aggregate operator share this
 * rather than each carrying a copy. Little of it survives being reimplemented from memory: the
 * {@code -g} flag without which {@code @Parallel} is silently ignored and the kernel becomes a
 * sequential loop every thread runs in full; the classpath derived from where classes actually came
 * from rather than from {@code java.class.path}, which does not mention a module path; and the
 * width each column is staged at.
 *
 * <p>Nothing reaches the filesystem. The source is compiled from a {@link SimpleJavaFileObject} and
 * the bytes are captured by a {@link ForwardingJavaFileManager}, so a kernel costs no temporary
 * directory and no cleanup that a killed TaskManager could skip. The one constraint this has to
 * respect is {@link KernelClassLoader}'s: TornadoVM re-reads the class <em>file</em> to find
 * {@code @Parallel}, so the loader has to serve it as a resource.
 */
final class GeneratedKernel implements AutoCloseable {

    private final GpuKernelSource kernel;

    /** The shared compilation this handle is using, or null before {@code compile}. */
    private Compiled compiled;

    /**
     * Kernels compiled in this JVM, keyed on the source they were compiled from.
     *
     * <p>Compiling costs a few hundred milliseconds and every subtask of a job generates the same
     * source from the same plan, so a TaskManager running eight subtasks was paying for the same
     * kernel eight times, on the task-startup path.
     *
     * <p>Keyed on the source rather than on the spec because the source is what {@code javac} is
     * given: two specs that differ in a way the generator does not emit -- a batch size, an output
     * layout -- produce one kernel and should share it, and two that emit different text must not.
     *
     * <p>Reference counted rather than weakly held. A cached entry owns a class loader holding the
     * only copy of the kernel's bytes, and an operator closing its handle must not take it from
     * another subtask still running it; when the last handle goes, so does the entry. That also
     * makes the lifetime answer the question the task asks: nothing here outlives the last operator
     * using it, let alone the class loader it came from.
     */
    private static final Map<String, Compiled> CACHE = new HashMap<>();

    /** How many times {@code javac} has actually run. For tests, and for nothing else. */
    static final AtomicInteger COMPILATIONS = new AtomicInteger();

    /** One compiled kernel, shared by every handle that asked for the same source. */
    private static final class Compiled {
        final String source;
        final Method entry;
        final KernelClassLoader loader;
        int handles;

        Compiled(String source, Method entry, KernelClassLoader loader) {
            this.source = source;
            this.entry = entry;
            this.loader = loader;
        }
    }

    GeneratedKernel(GpuKernelSource kernel) {
        this.kernel = kernel;
    }

    Method compile() throws Exception {
        synchronized (CACHE) {
            Compiled hit = CACHE.get(kernel.source());
            if (hit != null) {
                hit.handles++;
                compiled = hit;
                return hit.entry;
            }
        }
        Compiled fresh = compileFresh();
        synchronized (CACHE) {
            // Another subtask may have compiled the same source while this one was doing it. Its
            // copy is as good as this one; drop this rather than leave two loaders for one kernel.
            // Dropping it is now just letting go of the reference -- there is no directory to
            // remove and no file handle to close.
            Compiled hit = CACHE.get(kernel.source());
            if (hit != null) {
                hit.handles++;
                compiled = hit;
                return hit.entry;
            }
            fresh.handles = 1;
            CACHE.put(fresh.source, fresh);
            compiled = fresh;
            return fresh.entry;
        }
    }

    private Compiled compileFresh() throws Exception {
        COMPILATIONS.incrementAndGet();
        final GpuKernelSource kernel = this.kernel;
        JavaCompiler javac = ToolProvider.getSystemJavaCompiler();
        if (javac == null) {
            throw new IllegalStateException(
                    "No Java compiler available: GPU offload generates kernels at run time and so "
                            + "needs a JDK, not a JRE. TornadoVM requires one in any case.");
        }

        List<String> options =
                Arrays.asList(
                        "-classpath",
                        classpathFor(),
                        "--enable-preview",
                        "-source",
                        "21",
                        "-target",
                        "21",
                        // Debug info is not optional. @Parallel is a local-variable annotation, and
                        // TornadoVM associates it with the loop induction variable through the
                        // local variable table. Without -g javac emits no such table, the
                        // annotation is silently ignored, and the kernel is generated as a
                        // sequential loop that every GPU thread runs in full -- correct results,
                        // catastrophically slow, and no warning.
                        "-g",
                        "-nowarn");

        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
        Map<String, byte[]> classes = new LinkedHashMap<>();
        boolean ok;
        try (StandardJavaFileManager standard =
                        javac.getStandardFileManager(diagnostics, null, null);
                CapturingFileManager manager = new CapturingFileManager(standard, classes)) {
            ok =
                    javac.getTask(
                                    null,
                                    manager,
                                    diagnostics,
                                    options,
                                    null,
                                    List.of(new SourceFile(kernel.className(), kernel.source())))
                            .call();
        }

        if (!ok || classes.isEmpty()) {
            // The diagnostics are the useful half. Before they were collected javac wrote them to
            // System.err, where a TaskManager's logging buries them a long way from the exception
            // that says a kernel failed to compile.
            String errors =
                    diagnostics.getDiagnostics().stream()
                            .filter(d -> d.getKind() == Diagnostic.Kind.ERROR)
                            .map(d -> "  line " + d.getLineNumber() + ": " + d.getMessage(null))
                            .collect(Collectors.joining("\n"));
            throw new IllegalStateException(
                    "Generated kernel did not compile. This is a bug in the generator.\n"
                            + errors
                            + "\nThe source was:\n"
                            + kernel.source());
        }

        KernelClassLoader loader =
                new KernelClassLoader(classes, GeneratedKernelEngine.class.getClassLoader());
        Class<?> generated = loader.loadClass(kernel.className());
        for (Method m : generated.getDeclaredMethods()) {
            if (m.getName().equals(kernel.methodName())) {
                return new Compiled(kernel.source(), m, loader);
            }
        }
        throw new IllegalStateException("generated class has no method " + kernel.methodName());
    }

    /** The generated source, handed to javac without ever being a file. */
    private static final class SourceFile extends SimpleJavaFileObject {
        private final String code;

        SourceFile(String className, String code) {
            // javac requires the URI's last path segment to be the class name; "string" as the
            // scheme keeps it clear in a diagnostic that nothing was read from disk.
            super(URI.create("string:///" + className + ".java"), Kind.SOURCE);
            this.code = code;
        }

        @Override
        public CharSequence getCharContent(boolean ignoreEncodingErrors) {
            return code;
        }
    }

    /** Keeps what javac emits in a map instead of writing it under {@code -d}. */
    private static final class CapturingFileManager
            extends ForwardingJavaFileManager<StandardJavaFileManager> {
        private final Map<String, byte[]> classes;

        CapturingFileManager(StandardJavaFileManager delegate, Map<String, byte[]> classes) {
            super(delegate);
            this.classes = classes;
        }

        @Override
        public JavaFileObject getJavaFileForOutput(
                Location location, String className, JavaFileObject.Kind kind, FileObject sibling) {
            return new SimpleJavaFileObject(
                    URI.create("mem:///" + className.replace('.', '/') + kind.extension), kind) {
                @Override
                public OutputStream openOutputStream() {
                    return new ByteArrayOutputStream() {
                        @Override
                        public void close() {
                            // One entry per class javac emits, which is the kernel and any nested
                            // class the generator produces.
                            classes.put(className, toByteArray());
                        }
                    };
                }
            };
        }
    }

    /**
     * Holds the kernel's bytes and hands them out twice over.
     *
     * <p>Defining the class is the obvious half. The half that is not obvious, and that rules out
     * {@code defineHiddenClass} and a bare {@code defineClass} on the parent, is that TornadoVM
     * does not find {@code @Parallel} by reflection: it turns the declaring class's name into a
     * resource path and asks a class loader to <em>re-read the class file</em>, then scans the
     * local-variable type annotations with ASM. A class with no loadable {@code .class} resource is
     * not an error there -- {@code getParallelAnnotations} returns an empty array, and the kernel
     * is emitted as a sequential loop every device thread runs in full.
     *
     * <p>So {@link #getResourceAsStream} is not a convenience. It is the contract, and it is the
     * one TornadoVM calls; {@code findResource} would need a {@code URL} over memory, which buys
     * nothing here because nothing asks for one.
     */
    private static final class KernelClassLoader extends ClassLoader {
        private final Map<String, byte[]> classes;

        static {
            registerAsParallelCapable();
        }

        KernelClassLoader(Map<String, byte[]> classes, ClassLoader parent) {
            super("flink-gpu-kernel", parent);
            this.classes = Collections.unmodifiableMap(new LinkedHashMap<>(classes));
        }

        @Override
        protected Class<?> findClass(String name) throws ClassNotFoundException {
            byte[] bytes = classes.get(name);
            if (bytes == null) {
                throw new ClassNotFoundException(name);
            }
            return defineClass(name, bytes, 0, bytes.length);
        }

        @Override
        public InputStream getResourceAsStream(String name) {
            if (name.endsWith(".class")) {
                byte[] bytes =
                        classes.get(
                                name.substring(0, name.length() - ".class".length())
                                        .replace('/', '.'));
                if (bytes != null) {
                    return new ByteArrayInputStream(bytes);
                }
            }
            return super.getResourceAsStream(name);
        }
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

    /**
     * The loader the generated class lives in, and the only place its bytecode exists.
     *
     * <p>Callers install this as the thread's context class loader around anything that makes
     * TornadoVM compile, because that is one of the loaders it asks for the class file.
     */
    ClassLoader loader() {
        return compiled == null ? null : compiled.loader;
    }

    @Override
    public void close() {
        Compiled mine = compiled;
        compiled = null;
        if (mine == null) {
            return;
        }
        synchronized (CACHE) {
            if (--mine.handles > 0) {
                // Another subtask is still running this kernel, so its loader has to stay. Dropping
                // it here is what a per-operator cache would have done.
                return;
            }
            CACHE.remove(mine.source);
        }
        // Nothing to close and nothing to delete: once the cache lets go, the loader and the only
        // copy of the kernel's bytes are garbage like anything else.
    }
}
