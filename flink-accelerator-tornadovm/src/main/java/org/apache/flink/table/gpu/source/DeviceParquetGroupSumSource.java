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
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.types.logical.RowType;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
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
 * A two-operator device region: cuDF reads the file and cuDF groups it, in one execution plan.
 *
 * <p>What a local {@code GROUP BY k SUM(v)} over a Parquet source becomes when both halves run on
 * the device. The key and value columns are read straight into device arrays and grouped where they
 * lie; only the group partials come back, and Flink's global aggregate above the exchange merges
 * them as it would any local aggregate's output.
 *
 * <p>Nothing is generated here. That is the reason this shape is served and a projection between
 * the scan and the aggregate is not: with a {@code Calc} in between the region would need the
 * projection's kernel compiled into the same plan, which is a larger change.
 */
public final class DeviceParquetGroupSumSource
        implements Source<
                        RowData,
                        DeviceParquetGroupSumSource.FileSplit,
                        List<DeviceParquetGroupSumSource.FileSplit>>,
                Serializable {

    private static final long serialVersionUID = 1L;

    private final List<String> paths;
    private final int keyField;
    private final int valueField;
    private final RowType outputType;

    public DeviceParquetGroupSumSource(
            List<String> paths, int keyField, int valueField, RowType outputType) {
        this.paths = new ArrayList<>(paths);
        this.keyField = keyField;
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
        return new GroupReader(keyField, valueField, context);
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
    private static final class GroupReader implements SourceReader<RowData, FileSplit> {
        private final int keyField;
        private final int valueField;
        private final SourceReaderContext context;
        private final ArrayDeque<FileSplit> pending = new ArrayDeque<>();
        private final java.util.HashMap<Integer, Double> totals = new java.util.HashMap<>();
        private boolean noMore;
        private boolean emitted;
        private boolean outstanding;

        GroupReader(int keyField, int valueField, SourceReaderContext context) {
            this.keyField = keyField;
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
                group(pending.poll().path);
                request();
                return InputStatus.NOTHING_AVAILABLE;
            }
            if (!noMore) {
                request();
                return InputStatus.NOTHING_AVAILABLE;
            }
            if (!emitted) {
                for (java.util.Map.Entry<Integer, Double> e : totals.entrySet()) {
                    final GenericRowData row = new GenericRowData(2);
                    row.setField(0, e.getKey());
                    row.setField(1, e.getValue());
                    output.collect(row);
                }
                emitted = true;
            }
            return InputStatus.END_OF_INPUT;
        }

        /** Read and group one file, on the device, in one plan. */
        private void group(String file) throws Exception {
            final long[] meta = Cudf.parquetMetadata(file);
            final int rows = (int) meta[0];
            if (rows <= 0) {
                return;
            }
            final IntArray keys = new IntArray(rows);
            final DoubleArray values = new DoubleArray(rows);
            final IntArray outKeys = new IntArray(rows);
            final DoubleArray outSums = new DoubleArray(rows);
            final IntArray groups = new IntArray(1);

            final TaskGraph region =
                    new TaskGraph("region")
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, keys, values)
                            .libraryTask(
                                    "read",
                                    Cudf::readParquet,
                                    file,
                                    0,
                                    0,
                                    keyField,
                                    new int[] {valueField},
                                    (long) rows,
                                    keys,
                                    values)
                            .libraryTask(
                                    "group",
                                    Cudf::groupSum,
                                    rows,
                                    keys,
                                    values,
                                    outKeys,
                                    outSums,
                                    groups)
                            .transferToHost(
                                    DataTransferMode.EVERY_EXECUTION, outKeys, outSums, groups);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(region.snapshot())) {
                plan.execute();
            }
            final int n = groups.get(0);
            for (int i = 0; i < n; i++) {
                totals.merge(outKeys.get(i), outSums.get(i), Double::sum);
            }
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
        public void close() {}
    }
}
