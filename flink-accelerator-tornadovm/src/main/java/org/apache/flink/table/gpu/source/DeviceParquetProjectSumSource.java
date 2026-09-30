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
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.codegen.AccelKernelGenerator;
import org.apache.flink.table.gpu.codegen.GpuKernelSource;
import org.apache.flink.table.gpu.operator.GeneratedKernelEngine;
import org.apache.flink.table.types.logical.RowType;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
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
 * A three-operator device region: cuDF reads the file, a generated kernel evaluates the projection
 * over it, and cuDF reduces the result — in one execution plan, with nothing staged on the host.
 *
 * <p>What {@code SELECT SUM(f(a, b)) FROM t} becomes when the whole region runs on the device.
 * {@link DeviceParquetGroupSumSource} serves the case where the summed column is read straight out
 * of the file; this serves the case where it does not exist until something computes it, which is
 * the shape of every query whose arithmetic is worth moving to a device at all.
 *
 * <p>The kernel is compiled in the reader rather than on the planner, because the generator bakes
 * the packed-input stride — the row count — into the generated source, and that is not known until
 * a file is opened. It is compiled once and reused across splits; a file whose row count differs
 * from the last forces a recompile, so a dataset of equal-sized parts compiles exactly one class.
 *
 * <p>Emits one row per subtask, not one per query: Flink leaves its {@code Final} aggregate above
 * the substituted local one, and that is what combines the partials.
 */
public final class DeviceParquetProjectSumSource
        implements Source<
                        RowData,
                        DeviceParquetProjectSumSource.FileSplit,
                        List<DeviceParquetProjectSumSource.FileSplit>>,
                Serializable {

    private static final long serialVersionUID = 1L;

    private final List<String> paths;
    private final AccelProject project;
    private final int[] scanColumns;
    private final int valueField;
    private final RowType outputType;

    /**
     * @param paths the files or directories the scan names
     * @param project the projection, rooted at the scan it reads
     * @param scanColumns the file column index of each field of the scan's produced row
     * @param valueField which of the projection's outputs the sum is over
     * @param outputType the local aggregate's row, which is the one partial this emits
     */
    public DeviceParquetProjectSumSource(
            List<String> paths,
            AccelProject project,
            int[] scanColumns,
            int valueField,
            RowType outputType) {
        this.paths = new ArrayList<>(paths);
        this.project = project;
        this.scanColumns = scanColumns.clone();
        this.valueField = valueField;
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
        return new ProjectReader(project, scanColumns, valueField, context);
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

    /**
     * Hands out one split per request and says "no more" only when the queue is empty.
     *
     * <p>Splits are work to be claimed, not a partition to be computed. Partitioning them by
     * subtask against a parallelism that changes as readers register loses whole files, silently.
     */
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

    /**
     * One outstanding request at a time.
     *
     * <p>Asking for the next split after finishing each one lets a request outlive the reader: the
     * enumerator answers it after {@code notifyNoMoreSplits} has arrived and the split is never
     * read. Ask only when idle; finish only when nothing is pending and nothing is outstanding.
     */
    private static final class ProjectReader implements SourceReader<RowData, FileSplit> {
        private final AccelProject project;
        private final int[] scanColumns;
        private final int valueField;
        private final SourceReaderContext context;
        private final ArrayDeque<FileSplit> pending = new ArrayDeque<>();
        private boolean noMore;
        private boolean emitted;
        private boolean outstanding;
        private double total;

        // The compiled kernel and the plan around it, both built on the first split and kept.
        // Building a plan costs more than the read it performs, so rebuilding one per file was
        // most of the cost of the first version of this path.
        private GeneratedKernelEngine.Compiled kernel;
        private TornadoExecutionPlan plan;
        private StringBuilder pathHolder;
        private IntArray unusedKeys;
        private IntArray rowCount;
        private DoubleArray packedIn;
        private DoubleArray[] outputs;
        private DoubleArray sum;
        private int[] fileColumns;
        private int planRows = -1;

        ProjectReader(
                AccelProject project,
                int[] scanColumns,
                int valueField,
                SourceReaderContext context) {
            this.project = project;
            this.scanColumns = scanColumns;
            this.valueField = valueField;
            this.context = context;
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
                total += one(pending.poll().path);
                request();
                return InputStatus.NOTHING_AVAILABLE;
            }
            if (!noMore) {
                request();
                return InputStatus.NOTHING_AVAILABLE;
            }
            if (!emitted) {
                final GenericRowData row = new GenericRowData(1);
                row.setField(0, total);
                output.collect(row);
                emitted = true;
            }
            return InputStatus.END_OF_INPUT;
        }

        /** Read, compute and reduce one file, on the device, in one plan. */
        private double one(String file) throws Exception {
            final long[] meta = Cudf.parquetMetadata(file);
            final int rows = (int) meta[0];
            if (rows <= 0) {
                return 0.0;
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
            return sum.get(0);
        }

        /**
         * Compiles the kernel for this row count and builds the plan around it.
         *
         * <p>The packed input buffer is column-major with the row count as its stride, which is
         * both what {@code cudf::read_parquet} writes and what the generator indexes — so the read
         * and the kernel agree on the layout without anything copying between them.
         */
        private void build(int rows) throws Exception {
            final GpuKernelSource source =
                    AccelKernelGenerator.generate(project, "scan" + rows, 0, rows)
                            .orElseThrow(
                                    () ->
                                            new IllegalStateException(
                                                    "the projection has no kernel, but planning"
                                                            + " accepted it"));
            kernel = GeneratedKernelEngine.compileStandalone(source);

            // Buffer column k holds the scan field the generator staged k-th, which is the order
            // the expressions first referenced them and not necessarily ascending. cuDF writes the
            // columns in the order it is asked for them, so asking in exactly this order is what
            // makes the two layouts the same one.
            final int[] staged = source.inputFieldIndexes();
            fileColumns = new int[staged.length];
            for (int k = 0; k < staged.length; k++) {
                fileColumns[k] = scanColumns[staged[k]];
            }

            pathHolder = new StringBuilder();
            unusedKeys = new IntArray(1);
            rowCount = new IntArray(1);
            packedIn = new DoubleArray(rows * staged.length);
            outputs = new DoubleArray[source.outputCount()];
            for (int i = 0; i < outputs.length; i++) {
                outputs[i] = new DoubleArray(rows);
            }
            sum = new DoubleArray(1);

            final Object[] kernelArgs = new Object[outputs.length + 2];
            kernelArgs[0] = packedIn;
            System.arraycopy(outputs, 0, kernelArgs, 1, outputs.length);
            kernelArgs[kernelArgs.length - 1] = rowCount;

            TaskGraph region =
                    new TaskGraph("region")
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION,
                                    unusedKeys,
                                    rowCount,
                                    packedIn);
            for (DoubleArray out : outputs) {
                region = region.transferToDevice(DataTransferMode.EVERY_EXECUTION, out);
            }
            region =
                    region.libraryTask(
                                    "read",
                                    Cudf::readParquet,
                                    pathHolder,
                                    0,
                                    0,
                                    -1,
                                    fileColumns,
                                    (long) rows,
                                    unusedKeys,
                                    packedIn)
                            .task("kernel", kernel.entry(), kernelArgs)
                            .libraryTask("sum", Cudf::reduce, rows, 0, outputs[valueField], sum)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, sum);

            // State the iteration space rather than letting it be inferred. The generated kernel
            // bounds its loop by rows.get(0) -- a value that does not exist until the graph runs --
            // and TornadoVM cannot read a thread count out of that. Left to infer, it compiles the
            // loop sequentially: one GPU thread walking every row, correct and about three orders
            // of magnitude slower, with nothing in the output to say so.
            final GridScheduler scheduler = new GridScheduler();
            scheduler.addWorkerGrid("region.kernel", new WorkerGrid1D(rows));
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
                    // Closing a plan that never executed throws on some drivers; the reader is
                    // finished either way and there is nothing a failure here could be acted on.
                }
                plan = null;
            }
            if (kernel != null) {
                kernel.close();
                kernel = null;
            }
        }
    }
}
