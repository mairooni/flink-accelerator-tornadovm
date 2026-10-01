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

package org.apache.flink.table.gpu.source;

import org.apache.flink.api.connector.source.Boundedness;
import org.apache.flink.api.connector.source.ReaderOutput;
import org.apache.flink.api.connector.source.Source;
import org.apache.flink.api.connector.source.SourceReader;
import org.apache.flink.api.connector.source.SourceReaderContext;
import org.apache.flink.api.connector.source.SourceSplit;
import org.apache.flink.api.connector.source.SplitEnumerator;
import org.apache.flink.api.connector.source.SplitEnumeratorContext;
import org.apache.flink.core.io.InputStatus;
import org.apache.flink.core.io.SimpleVersionedSerializer;
import org.apache.flink.table.accelerator.AccelScan;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.codegen.AccelKernelGenerator;
import org.apache.flink.table.gpu.codegen.GpuGramSpec;
import org.apache.flink.table.gpu.codegen.GpuKernelSource;
import org.apache.flink.table.gpu.operator.GeneratedKernelEngine;
import org.apache.flink.table.types.logical.RowType;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;
import uk.ac.manchester.tornado.cudf.Cudf;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * A four-stage device region: cuDF reads the file, a generated kernel computes {@code d} features,
 * and cuBLAS contracts them into a Gram matrix — in one execution plan, with nothing on the host.
 *
 * <p>The shape whose advantage grows with the query rather than being fixed. Over {@code n} rows of
 * {@code d} features a Gram matrix is {@code O(n·d²)} of arithmetic on {@code O(n·d)} of data, so
 * arithmetic per byte rises with {@code d} — which is the one axis this project has found that
 * moves a device result. Every other operator does fixed work per row.
 *
 * <p>The device never computes the {@code d(d+1)/2} products the SQL asks for. The kernel computes
 * {@code d} features and {@code A'A} forms the products, so the region is {@code O(n·d)} of kernel
 * work and one GEMM. That is also what keeps the kernel under TornadoVM's argument ceiling: packed
 * input and output are two buffers whatever {@code d} is.
 *
 * <p>FP32 or FP64 by the query's own declaration — Flink sums a {@code FLOAT} column in a
 * {@code FLOAT} accumulator, so an FP32 query is FP32 on both arms. FP64 remains what a
 * {@code DOUBLE} query gets, and for the reason {@code GpuGramEngine} records: a Gram matrix is the
 * left-hand side of a normal equation, which is where a narrowed contraction propagates.
 */
public final class DeviceParquetGramSource
        implements Source<
                        RowData,
                        DeviceParquetGramSource.FileSplit,
                        List<DeviceParquetGramSource.FileSplit>>,
                Serializable {

    private static final long serialVersionUID = 1L;

    private final List<String> paths;
    private final GpuGramSpec spec;
    private final AccelScan scan;
    private final RowType outputType;

    public DeviceParquetGramSource(
            List<String> paths, GpuGramSpec spec, AccelScan scan, RowType outputType) {
        this.paths = new ArrayList<>(paths);
        this.spec = spec;
        this.scan = scan;
        this.outputType = outputType;
    }

    /** One Parquet file. */
    public static final class FileSplit implements SourceSplit, Serializable {
        private static final long serialVersionUID = 1L;
        final String path;

        FileSplit(String path) {
            this.path = path;
        }

        @Override
        public String splitId() {
            return path;
        }
    }

    @Override
    public Boundedness getBoundedness() {
        return Boundedness.BOUNDED;
    }

    @Override
    public SourceReader<RowData, FileSplit> createReader(SourceReaderContext context) {
        return new GramReader(spec, scan, outputType, context);
    }

    @Override
    public SplitEnumerator<FileSplit, List<FileSplit>> createEnumerator(
            SplitEnumeratorContext<FileSplit> context) {
        return new OnRequest(context, expand(paths));
    }

    @Override
    public SplitEnumerator<FileSplit, List<FileSplit>> restoreEnumerator(
            SplitEnumeratorContext<FileSplit> context, List<FileSplit> state) {
        return new OnRequest(context, new ArrayList<>(state));
    }

    @Override
    public SimpleVersionedSerializer<FileSplit> getSplitSerializer() {
        return new JavaSerializer<>();
    }

    @Override
    public SimpleVersionedSerializer<List<FileSplit>> getEnumeratorCheckpointSerializer() {
        return new JavaSerializer<>();
    }

    private static List<FileSplit> expand(List<String> paths) {
        final List<FileSplit> splits = new ArrayList<>();
        for (String path : paths) {
            final File file =
                    new File(
                            java.net
                                    .URI
                                    .create(path.contains("://") ? path : "file://" + path)
                                    .getPath());
            if (file.isDirectory()) {
                final File[] parts =
                        file.listFiles(f -> f.isFile() && !f.getName().startsWith("."));
                if (parts != null) {
                    Arrays.sort(parts);
                    for (File part : parts) {
                        splits.add(new FileSplit(part.getAbsolutePath()));
                    }
                }
            } else {
                splits.add(new FileSplit(file.getAbsolutePath()));
            }
        }
        return splits;
    }

    private static final class JavaSerializer<T> implements SimpleVersionedSerializer<T> {
        @Override
        public int getVersion() {
            return 1;
        }

        @Override
        public byte[] serialize(T obj) throws IOException {
            final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
                out.writeObject(obj);
            }
            return bytes.toByteArray();
        }

        @SuppressWarnings("unchecked")
        @Override
        public T deserialize(int version, byte[] serialized) throws IOException {
            try (ObjectInputStream in =
                    new ObjectInputStream(new ByteArrayInputStream(serialized))) {
                return (T) in.readObject();
            } catch (ClassNotFoundException e) {
                throw new IOException(e);
            }
        }
    }

    /** Hands out one split per request; see {@code DeviceParquetProjectSumSource} for why. */
    private static final class OnRequest implements SplitEnumerator<FileSplit, List<FileSplit>> {
        private final SplitEnumeratorContext<FileSplit> context;
        private final ArrayDeque<FileSplit> remaining;

        OnRequest(SplitEnumeratorContext<FileSplit> context, List<FileSplit> splits) {
            this.context = context;
            this.remaining = new ArrayDeque<>(splits);
        }

        @Override
        public void start() {}

        @Override
        public void handleSplitRequest(int subtask, String hostname) {
            if (!remaining.isEmpty()) {
                context.assignSplit(remaining.poll(), subtask);
            } else {
                context.signalNoMoreSplits(subtask);
            }
        }

        @Override
        public void addReader(int subtask) {}

        @Override
        public void addSplitsBack(List<FileSplit> splits, int subtask) {
            remaining.addAll(splits);
        }

        @Override
        public List<FileSplit> snapshotState(long checkpointId) {
            return new ArrayList<>(remaining);
        }

        @Override
        public void close() {}
    }

    private static final class GramReader implements SourceReader<RowData, FileSplit> {
        private final GpuGramSpec spec;
        private final AccelScan scan;
        private final RowType outputType;
        private final SourceReaderContext context;
        private final ArrayDeque<FileSplit> pending = new ArrayDeque<>();
        private final int d;
        private final boolean fp32;
        private final double[] total;
        private boolean noMore;
        private boolean emitted;
        private boolean outstanding;

        private GeneratedKernelEngine.Compiled kernel;
        private TornadoExecutionPlan plan;
        private StringBuilder pathHolder;
        private IntArray unusedKeys;
        private IntArray rowCount;
        private Object staged;
        private Object packed;
        private Object gram;
        private int[] fileColumns;
        private int planRows = -1;

        GramReader(
                GpuGramSpec spec,
                AccelScan scan,
                RowType outputType,
                SourceReaderContext context) {
            this.spec = spec;
            this.scan = scan;
            this.outputType = outputType;
            this.context = context;
            this.d = spec.featureCount();
            this.fp32 = spec.isFloat();
            this.total = new double[d * d];
        }

        @Override
        public void start() {
            request();
        }

        private void request() {
            if (!outstanding && !noMore) {
                outstanding = true;
                context.sendSplitRequest();
            }
        }

        @Override
        public InputStatus pollNext(ReaderOutput<RowData> output) throws Exception {
            if (!pending.isEmpty()) {
                one(pending.poll().path);
                request();
                return InputStatus.NOTHING_AVAILABLE;
            }
            if (!noMore) {
                request();
                return InputStatus.NOTHING_AVAILABLE;
            }
            if (!emitted) {
                // The upper triangle in row-major order, which is the order the recogniser
                // insisted the aggregate's sums were in.
                final GenericRowData row = new GenericRowData(outputType.getFieldCount());
                int at = 0;
                for (int i = 0; i < d; i++) {
                    for (int j = i; j < d; j++) {
                        final double value = total[j * d + i];
                        row.setField(at++, fp32 ? (Object) (float) value : (Object) value);
                    }
                }
                output.collect(row);
                emitted = true;
            }
            return InputStatus.END_OF_INPUT;
        }

        /** Read, compute the features and contract them, on the device, in one plan. */
        private void one(String file) throws Exception {
            final int rows = (int) Cudf.parquetMetadata(file)[0];
            if (rows <= 0) {
                return;
            }
            if (plan == null || planRows != rows) {
                close();
                build(rows);
                planRows = rows;
            }
            pathHolder.setLength(0);
            pathHolder.append(file);
            rowCount.set(0, rows);
            plan.execute();
            // Accumulated at the width the query declared, not wider. Summing FP32 partials in a
            // double would make this arm more accurate than the CPU arm it is compared against,
            // which is a difference in the answer and not only in the clock.
            for (int i = 0; i < total.length; i++) {
                final double value =
                        fp32 ? ((FloatArray) gram).get(i) : ((DoubleArray) gram).get(i);
                total[i] = fp32 ? (float) (total[i] + value) : total[i] + value;
            }
        }

        private void build(int rows) throws Exception {
            final GpuKernelSource source =
                    AccelKernelGenerator.generate(
                                    spec.featureProjection(scan), "gram" + rows, rows, rows)
                            .orElseThrow(
                                    () ->
                                            new IllegalStateException(
                                                    "the feature map has no kernel, but planning"
                                                            + " accepted it"));
            kernel = GeneratedKernelEngine.compileStandalone(source);

            final int[] stagedFields = source.inputFieldIndexes();
            fileColumns = new int[stagedFields.length];
            final int[] scanColumns = scan.projectedFields();
            for (int k = 0; k < stagedFields.length; k++) {
                fileColumns[k] = scanColumns[stagedFields[k]];
            }

            pathHolder = new StringBuilder();
            unusedKeys = new IntArray(1);
            rowCount = new IntArray(1);
            if (fp32) {
                staged = new FloatArray(rows * stagedFields.length);
                packed = new FloatArray(rows * d);
                gram = new FloatArray(d * d);
            } else {
                staged = new DoubleArray(rows * stagedFields.length);
                packed = new DoubleArray(rows * d);
                gram = new DoubleArray(d * d);
            }

            final Object[] kernelArgs = new Object[] {staged, packed, rowCount};
            TaskGraph region =
                    new TaskGraph("gram")
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION,
                                    unusedKeys,
                                    rowCount,
                                    staged,
                                    packed,
                                    gram);
            region =
                    fp32
                            ? region.libraryTask(
                                    "read",
                                    Cudf::readParquet,
                                    pathHolder,
                                    0,
                                    0,
                                    -1,
                                    fileColumns,
                                    (long) rows,
                                    unusedKeys,
                                    (FloatArray) staged)
                            : region.libraryTask(
                                    "read",
                                    Cudf::readParquet,
                                    pathHolder,
                                    0,
                                    0,
                                    -1,
                                    fileColumns,
                                    (long) rows,
                                    unusedKeys,
                                    (DoubleArray) staged);
            region = region.task("features", kernel.entry(), kernelArgs);
            // C = A'A. A is rows x d held column-major, so its leading dimension is the row count
            // and the transpose costs nothing: cuBLAS reads the buffer the kernel just wrote.
            region =
                    fp32
                            ? region.libraryTask(
                                    "gemm",
                                    CuBlas::cublasSgemm,
                                    CuBlasOperation.CUBLAS_OP_T.operation(),
                                    CuBlasOperation.CUBLAS_OP_N.operation(),
                                    d,
                                    d,
                                    rows,
                                    1.0f,
                                    (FloatArray) packed,
                                    rows,
                                    (FloatArray) packed,
                                    rows,
                                    0.0f,
                                    (FloatArray) gram,
                                    d)
                            : region.libraryTask(
                                    "gemm",
                                    CuBlas::cublasDgemm,
                                    CuBlasOperation.CUBLAS_OP_T.operation(),
                                    CuBlasOperation.CUBLAS_OP_N.operation(),
                                    d,
                                    d,
                                    rows,
                                    1.0,
                                    (DoubleArray) packed,
                                    rows,
                                    (DoubleArray) packed,
                                    rows,
                                    0.0,
                                    (DoubleArray) gram,
                                    d);
            region = region.transferToHost(DataTransferMode.EVERY_EXECUTION, gram);

            // Stated, not inferred: the kernel's loop bound is read from a buffer, and left to
            // infer TornadoVM emits a sequential loop -- correct, and about a thousand times
            // slower, with nothing in the output to say so.
            final GridScheduler scheduler = new GridScheduler();
            scheduler.addWorkerGrid("gram.features", new WorkerGrid1D(rows));
            plan = new TornadoExecutionPlan(region.snapshot()).withGridScheduler(scheduler);
        }

        @Override
        public List<FileSplit> snapshotState(long checkpointId) {
            return new ArrayList<>(pending);
        }

        @Override
        public CompletableFuture<Void> isAvailable() {
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public void addSplits(List<FileSplit> splits) {
            outstanding = false;
            pending.addAll(splits);
        }

        @Override
        public void notifyNoMoreSplits() {
            outstanding = false;
            noMore = true;
        }

        @Override
        public void close() {
            if (plan != null) {
                try {
                    plan.close();
                } catch (Throwable ignored) {
                    // A plan that never executed throws on some drivers; the reader is done.
                }
                plan = null;
            }
            if (kernel != null) {
                kernel.close();
                kernel = null;
            }
            // Dropped so the collector can reclaim them. TornadoVM allocates its arrays from an
            // Arena.ofAuto(), so the off-heap behind them is freed only when the array itself is
            // collected -- there is no close() to call. Holding the fields past the reader's work
            // keeps a few hundred megabytes of direct memory alive against the TaskManager's
            // off-heap budget, which the next job in the same process then does not have.
            staged = null;
            packed = null;
            gram = null;
        }
    }
}
