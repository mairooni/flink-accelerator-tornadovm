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
import org.apache.flink.table.gpu.codegen.GpuGrokSpec;
import org.apache.flink.table.gpu.operator.DeviceGrok;
import org.apache.flink.table.types.logical.RowType;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
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
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * A grok-shaped region: cuDF reads a string column, k regexes match it, a generated kernel folds
 * the masks, and one partial count comes back — one task graph, one plan for the whole partition.
 *
 * <p>The strings are read once into TornadoVM-owned buffers and every pattern after that matches
 * the same device memory. That is the whole reason this shape pays: the read is a fixed cost and
 * the patterns are the marginal one, and the marginal one is where a library and a CPU diverge by
 * about 41x a pattern (§T56).
 *
 * <p>Buffers and the plan are built once for the widest file in the partition and reused. Doing
 * either per file is host work proportional to the data — §T38 measured per-plan allocation as the
 * dominant cost of a device read — and the first draft of this region spent half its time there.
 */
public final class DeviceParquetGrokSource
        implements Source<
                        RowData,
                        DeviceParquetGrokSource.FileSplit,
                        List<DeviceParquetGrokSource.FileSplit>>,
                Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Bytes of decoded characters this sizes the blob for, per byte of Parquet on disk.
     *
     * <p>A footer gives the row count and not the decoded size, so a caller has to guess and be
     * told. Eight covers the compression ratios log text reaches; the read refuses rather than
     * overruns when it does not, and the refusal names the size it needed.
     */
    private static final long CHARS_PER_FILE_BYTE = 8L;

    private final List<String> paths;
    private final GpuGrokSpec spec;
    private final int fileColumn;
    private final RowType outputType;

    public DeviceParquetGrokSource(
            List<String> paths, GpuGrokSpec spec, AccelScan scan, RowType outputType) {
        this.paths = new ArrayList<>(paths);
        this.spec = spec;
        this.fileColumn = scan.projectedFields()[spec.stringField()];
        this.outputType = outputType;
    }

    /** A table's path is a directory and a reader needs files. */
    public static List<String> expand(List<String> paths) {
        final List<String> files = new ArrayList<>();
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
                    java.util.Arrays.sort(parts);
                    for (File part : parts) {
                        files.add(part.getAbsolutePath());
                    }
                }
            } else {
                files.add(file.getAbsolutePath());
            }
        }
        return files;
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
        return new GrokReader(spec, fileColumn, outputType, context);
    }

    @Override
    public SplitEnumerator<FileSplit, List<FileSplit>> createEnumerator(
            SplitEnumeratorContext<FileSplit> context) {
        final List<FileSplit> splits = new ArrayList<>();
        for (String file : expand(paths)) {
            splits.add(new FileSplit(file));
        }
        return new FileEnumerator(context, splits);
    }

    @Override
    public SplitEnumerator<FileSplit, List<FileSplit>> restoreEnumerator(
            SplitEnumeratorContext<FileSplit> context, List<FileSplit> checkpoint) {
        return new FileEnumerator(context, new ArrayList<>(checkpoint));
    }

    @Override
    public SimpleVersionedSerializer<FileSplit> getSplitSerializer() {
        return new JavaSerializer<>();
    }

    @Override
    public SimpleVersionedSerializer<List<FileSplit>> getEnumeratorCheckpointSerializer() {
        return new JavaSerializer<>();
    }

    private static final class JavaSerializer<T> implements SimpleVersionedSerializer<T> {
        @Override
        public int getVersion() {
            return 1;
        }

        @Override
        public byte[] serialize(T object) throws IOException {
            final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
                out.writeObject(object);
            }
            return bytes.toByteArray();
        }

        @Override
        @SuppressWarnings("unchecked")
        public T deserialize(int version, byte[] serialized) throws IOException {
            try (ObjectInputStream in =
                    new ObjectInputStream(new ByteArrayInputStream(serialized))) {
                return (T) in.readObject();
            } catch (ClassNotFoundException e) {
                throw new IOException(e);
            }
        }
    }

    private static final class FileEnumerator
            implements SplitEnumerator<FileSplit, List<FileSplit>> {
        private final SplitEnumeratorContext<FileSplit> context;
        private final ArrayDeque<FileSplit> remaining;

        FileEnumerator(SplitEnumeratorContext<FileSplit> context, List<FileSplit> splits) {
            this.context = context;
            this.remaining = new ArrayDeque<>(splits);
        }

        @Override
        public void start() {}

        @Override
        public void handleSplitRequest(int subtaskId, String requesterHostname) {
            if (remaining.isEmpty()) {
                context.signalNoMoreSplits(subtaskId);
            } else {
                context.assignSplit(remaining.poll(), subtaskId);
            }
        }

        @Override
        public void addSplitsBack(List<FileSplit> splits, int subtaskId) {
            remaining.addAll(splits);
        }

        @Override
        public void addReader(int subtaskId) {}

        @Override
        public List<FileSplit> snapshotState(long checkpointId) {
            return new ArrayList<>(remaining);
        }

        @Override
        public void close() {}
    }

    private static final class GrokReader implements SourceReader<RowData, FileSplit> {
        private static final org.slf4j.Logger LOG =
                org.slf4j.LoggerFactory.getLogger(GrokReader.class);

        private final GpuGrokSpec spec;
        private final int fileColumn;
        private final RowType outputType;
        private final SourceReaderContext context;
        private final ArrayDeque<FileSplit> pending = new ArrayDeque<>();
        private final int patterns;
        private boolean noMore;
        private boolean emitted;
        private boolean outstanding;

        private TornadoExecutionPlan plan;
        private StringBuilder pathHolder;
        private IntArray offsets;
        private ByteArray chars;
        private IntArray dims;
        private IntArray hits;
        private ByteArray[] masks;
        private int sizedRows = -1;
        private long sizedChars = -1;

        private long count;
        private long rowsIn;
        private long deviceNanos;

        GrokReader(
                GpuGrokSpec spec, int fileColumn, RowType outputType, SourceReaderContext context) {
            this.spec = spec;
            this.fileColumn = fileColumn;
            this.outputType = outputType;
            this.context = context;
            this.patterns = spec.patterns().size();
        }

        @Override
        public void start() {
            // Refused above one subtask, loudly, because the failure above it is not always loud.
            // Measured 2026-10-01: at parallelism 4 this region returned 3,500,000 where every
            // other arm returned 4,000,000 -- one file's worth missing, with no error -- and other
            // runs of the same configuration died inside TornadoVM with
            // "unimplemented: field type [B". A crash is survivable and a silently short count is
            // not, so until the cause is found the shape that produced it cannot be allowed to
            // run. The region is correct and stable at parallelism 1 (twelve runs, §T56).
            final int parallelism = context.currentParallelism();
            if (parallelism > 1) {
                throw new IllegalStateException(
                        "the grok region is not correct above one subtask and this job has "
                                + parallelism
                                + "; it has been seen to return a short count. Run at parallelism 1,"
                                + " or set -Dflink.accelerator.device-scan=off to take the CPU path.");
            }
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
                // One partial per subtask. Flink's own Final aggregate above the substitution adds
                // them up, which is what makes substituting only the local half correct.
                final GenericRowData row = new GenericRowData(1);
                row.setField(0, count);
                output.collect(row);
                emitted = true;
                LOG.info(
                        "grok region: {} rows x {} patterns, {} matched, {} ms on the device",
                        rowsIn,
                        patterns,
                        count,
                        deviceNanos / 1_000_000);
            }
            return InputStatus.END_OF_INPUT;
        }

        private void one(String file) throws Exception {
            final int rows = (int) Cudf.parquetMetadata(file)[0];
            if (rows <= 0) {
                return;
            }
            final long capacity = new File(file).length() * CHARS_PER_FILE_BYTE;
            if (plan == null || rows > sizedRows || capacity > sizedChars) {
                close();
                build(rows, capacity);
            }
            pathHolder.setLength(0);
            pathHolder.append(file);
            dims.set(0, rows);

            final long started = System.nanoTime();
            plan.execute();
            deviceNanos += System.nanoTime() - started;

            rowsIn += rows;
            for (int i = 0; i < rows; i++) {
                count += hits.get(i);
            }
        }

        private void build(int rows, long capacity) {
            sizedRows = rows;
            sizedChars = capacity;
            pathHolder = new StringBuilder(256);
            offsets = new IntArray(rows + 1);
            chars = new ByteArray((int) capacity);
            dims = new IntArray(2);
            dims.set(0, rows);
            dims.set(1, patterns);
            hits = new IntArray(rows);
            masks = new ByteArray[DeviceGrok.MAX_PATTERNS];
            for (int i = 0; i < DeviceGrok.MAX_PATTERNS; i++) {
                masks[i] = new ByteArray(i < patterns ? rows : 1);
            }
            final List<String> raw = spec.patterns();
            final StringBuilder[] pats = new StringBuilder[DeviceGrok.MAX_PATTERNS];
            for (int i = 0; i < DeviceGrok.MAX_PATTERNS; i++) {
                pats[i] = new StringBuilder(i < patterns ? raw.get(i) : ".");
            }

            // Only `dims` moves per execution. The rest are written by the device and read back
            // once, so uploading them per file would move hundreds of megabytes of nothing.
            TaskGraph region =
                    new TaskGraph("grok")
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, dims)
                            .transferToDevice(
                                    DataTransferMode.FIRST_EXECUTION, offsets, chars, hits)
                            .libraryTask(
                                    "read",
                                    Cudf::readParquetStrings,
                                    pathHolder,
                                    0,
                                    0,
                                    fileColumn,
                                    (long) rows,
                                    offsets,
                                    chars,
                                    capacity);
            for (int i = 0; i < patterns; i++) {
                region =
                        region.libraryTask(
                                "re" + i,
                                Cudf::containsRe,
                                (long) rows,
                                offsets,
                                chars,
                                capacity,
                                pats[i],
                                masks[i]);
            }
            region =
                    region.task(
                                    "combine",
                                    DeviceGrok::combine,
                                    masks[0], masks[1], masks[2], masks[3],
                                    masks[4], masks[5], masks[6], masks[7],
                                    hits, dims)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, hits);

            final GridScheduler grid = new GridScheduler();
            grid.addWorkerGrid("grok.combine", new WorkerGrid1D(rows));
            plan = new TornadoExecutionPlan(region.snapshot()).withGridScheduler(grid);
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
            pending.addAll(splits);
            outstanding = false;
        }

        @Override
        public void notifyNoMoreSplits() {
            noMore = true;
            outstanding = false;
        }

        @Override
        public void close() {
            if (plan != null) {
                try {
                    plan.close();
                } catch (Exception e) {
                    LOG.warn("closing the grok plan", e);
                }
                plan = null;
            }
            offsets = null;
            chars = null;
            hits = null;
            masks = null;
        }
    }
}
