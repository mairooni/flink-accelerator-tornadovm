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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Reads a Parquet file onto the device and returns its top-N rows, without the rows ever reaching
 * the host on the way.
 *
 * <p>What makes a plain {@code SELECT ... ORDER BY k LIMIT n} take the device path. Flink's own
 * source would decode the file on the CPU (~17.3 ns a field) and the operator above it would stage
 * every field into a device array (~17.0 ns a field), against ~3.17 ns a field to read straight
 * onto the device. There is no way to recover that downstream, which is why this replaces the
 * source rather than decorating it.
 *
 * <p>One execution plan, two task graphs: the scan reads and persists, the sort consumes in place
 * and a gather kernel collects the surviving rows. Residency does not span two execution plans --
 * {@code consumeFromDevice} hands over a pointer reachable only through the producing graph -- so
 * the two have to be built together.
 */
public final class DeviceParquetTopNSource
        implements Source<
                        RowData,
                        DeviceParquetTopNSource.FileSplit,
                        List<DeviceParquetTopNSource.FileSplit>>,
                Serializable {

    private static final long serialVersionUID = 1L;

    private final List<String> paths;
    private final int sortField;
    private final long limit;
    private final RowType rowType;

    public DeviceParquetTopNSource(List<String> paths, int sortField, long limit, RowType rowType) {
        this.paths = new ArrayList<>(paths);
        this.sortField = sortField;
        this.limit = limit;
        this.rowType = rowType;
    }

    /** One file, or one directory of them, as a split. */
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
        return new DeviceReader(sortField, limit, rowType);
    }

    @Override
    public SplitEnumerator<FileSplit, List<FileSplit>> createEnumerator(
            SplitEnumeratorContext<FileSplit> context) {
        return new StaticEnumerator(context, expand(paths));
    }

    @Override
    public SplitEnumerator<FileSplit, List<FileSplit>> restoreEnumerator(
            SplitEnumeratorContext<FileSplit> context, List<FileSplit> state) {
        return new StaticEnumerator(context, new ArrayList<>(state));
    }

    @Override
    public SimpleVersionedSerializer<FileSplit> getSplitSerializer() {
        return new JavaSerializer<>();
    }

    @Override
    public SimpleVersionedSerializer<List<FileSplit>> getEnumeratorCheckpointSerializer() {
        return new JavaSerializer<>();
    }

    /** A path may name a directory of part files, which is how Flink writes a table. */
    private static List<FileSplit> expand(List<String> paths) {
        List<FileSplit> splits = new ArrayList<>();
        for (String path : paths) {
            java.io.File file = new java.io.File(java.net.URI.create(normalise(path)).getPath());
            if (file.isDirectory()) {
                java.io.File[] parts =
                        file.listFiles(f -> f.isFile() && !f.getName().startsWith("."));
                if (parts != null) {
                    java.util.Arrays.sort(parts);
                    for (java.io.File part : parts) {
                        splits.add(new FileSplit(part.getAbsolutePath()));
                    }
                }
            } else {
                splits.add(new FileSplit(file.getAbsolutePath()));
            }
        }
        return splits;
    }

    private static String normalise(String path) {
        return path.contains("://") ? path : "file://" + path;
    }

    private static final class JavaSerializer<T> implements SimpleVersionedSerializer<T> {
        @Override
        public int getVersion() {
            return 1;
        }

        @Override
        public byte[] serialize(T obj) throws IOException {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
                out.writeObject(obj);
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

    /**
     * Hands every split out as readers appear, then says there are no more.
     *
     * <p>Pushed rather than waited for: {@link DeviceReader} does not send split requests, and an
     * enumerator that only answered requests would leave the job running with nothing assigned.
     */
    private static final class StaticEnumerator
            implements SplitEnumerator<FileSplit, List<FileSplit>> {
        private final SplitEnumeratorContext<FileSplit> context;
        private final List<FileSplit> remaining;

        StaticEnumerator(SplitEnumeratorContext<FileSplit> context, List<FileSplit> splits) {
            this.context = context;
            this.remaining = splits;
        }

        @Override
        public void start() {}

        @Override
        public void handleSplitRequest(int subtask, String hostname) {
            assign(subtask);
        }

        @Override
        public void addReader(int subtask) {
            assign(subtask);
        }

        private void assign(int subtask) {
            if (!remaining.isEmpty()) {
                context.assignSplit(remaining.remove(0), subtask);
            }
            if (remaining.isEmpty()) {
                context.signalNoMoreSplits(subtask);
            }
        }

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

    /** The device work, and the only place a row becomes a {@link RowData}. */
    private static final class DeviceReader implements SourceReader<RowData, FileSplit> {
        private final int sortField;
        private final long limit;
        private final RowType rowType;
        private final List<FileSplit> pending = new ArrayList<>();
        private boolean noMoreSplits;

        DeviceReader(int sortField, long limit, RowType rowType) {
            this.sortField = sortField;
            this.limit = limit;
            this.rowType = rowType;
        }

        @Override
        public void start() {}

        @Override
        public InputStatus pollNext(ReaderOutput<RowData> output) throws Exception {
            if (!pending.isEmpty()) {
                DeviceTopNRead.emit(pending.remove(0).path, sortField, limit, rowType, output);
            }
            if (noMoreSplits && pending.isEmpty()) {
                return InputStatus.END_OF_INPUT;
            }
            return InputStatus.NOTHING_AVAILABLE;
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
        }

        @Override
        public void notifyNoMoreSplits() {
            noMoreSplits = true;
        }

        @Override
        public void close() {}
    }

    /** Kept apart so the Tornado types appear in one place. */
    static final class DeviceTopNRead {
        private DeviceTopNRead() {}

        static void emit(
                String file, int sortField, long limit, RowType rowType, ReaderOutput<RowData> out)
                throws Exception {
            final long[] meta = uk.ac.manchester.tornado.cudf.Cudf.parquetMetadata(file);
            final int rows = (int) meta[0];
            final int fields = rowType.getFieldCount();
            final int valueColumns = fields - 1;

            final int[] doubleColumns = new int[valueColumns];
            int next = 0;
            for (int i = 0; i < fields; i++) {
                if (i != sortField) {
                    doubleColumns[next++] = i;
                }
            }

            final uk.ac.manchester.tornado.api.types.arrays.IntArray keys =
                    new uk.ac.manchester.tornado.api.types.arrays.IntArray(rows);
            final uk.ac.manchester.tornado.api.types.arrays.DoubleArray values =
                    new uk.ac.manchester.tornado.api.types.arrays.DoubleArray(rows * valueColumns);
            final uk.ac.manchester.tornado.api.types.arrays.IntArray order =
                    new uk.ac.manchester.tornado.api.types.arrays.IntArray(rows);
            final int n = (int) Math.min(limit, rows);
            final uk.ac.manchester.tornado.api.types.arrays.IntArray topKeys =
                    new uk.ac.manchester.tornado.api.types.arrays.IntArray(n);
            final uk.ac.manchester.tornado.api.types.arrays.DoubleArray topValues =
                    new uk.ac.manchester.tornado.api.types.arrays.DoubleArray(n * valueColumns);

            // One task graph, not two. The read, the sort and the gather all operate on the
            // same device arrays, so the columns are resident between them by construction --
            // there is nothing to persist and nothing to consume.
            //
            // The two-graph shape works and was how this was first measured, but it aliases a
            // buffer across graphs: consumeFromDevice hands the consumer the producer's pointer,
            // and TornadoExecutionPlan.close() then frees every graph, so the second free reports
            // an invalid buffer. Tolerating that exception also swallowed the release and leaked
            // 4 MiB a query (VERIFY.md T31). One graph has no aliasing, frees each buffer once,
            // and is simpler.
            final uk.ac.manchester.tornado.api.TaskGraph graph =
                    new uk.ac.manchester.tornado.api.TaskGraph("scan")
                            .transferToDevice(
                                    uk.ac.manchester.tornado.api.enums.DataTransferMode
                                            .EVERY_EXECUTION,
                                    keys,
                                    values)
                            .libraryTask(
                                    "read",
                                    uk.ac.manchester.tornado.cudf.Cudf::readParquet,
                                    file,
                                    0,
                                    0,
                                    sortField,
                                    doubleColumns,
                                    (long) rows,
                                    keys,
                                    values)
                            .libraryTask(
                                    "order",
                                    uk.ac.manchester.tornado.cudf.Cudf::sortedOrder,
                                    rows,
                                    keys,
                                    order)
                            .task(
                                    "gather",
                                    DeviceTopNRead::gather,
                                    keys,
                                    values,
                                    order,
                                    topKeys,
                                    topValues,
                                    n,
                                    rows,
                                    valueColumns)
                            .transferToHost(
                                    uk.ac.manchester.tornado.api.enums.DataTransferMode
                                            .EVERY_EXECUTION,
                                    topKeys,
                                    topValues);

            try (uk.ac.manchester.tornado.api.TornadoExecutionPlan plan =
                    new uk.ac.manchester.tornado.api.TornadoExecutionPlan(graph.snapshot())) {
                plan.execute();
            }

            for (int i = 0; i < n; i++) {
                final GenericRowData row = new GenericRowData(fields);
                row.setField(sortField, topKeys.get(i));
                for (int c = 0; c < valueColumns; c++) {
                    row.setField(doubleColumns[c], topValues.get(c * n + i));
                }
                out.collect(row);
            }
        }

        /**
         * Collects the surviving rows, on the device.
         *
         * <p>A persisted array is not transferred back, so the rows have to be gathered where they
         * are; and gathering is what a limit means anyway -- only the survivors cross the
         * interconnect, which for a 1M-row file limited to 500,000 is half the bytes and for a
         * tighter limit is almost none of them.
         */
        static void gather(
                uk.ac.manchester.tornado.api.types.arrays.IntArray keys,
                uk.ac.manchester.tornado.api.types.arrays.DoubleArray values,
                uk.ac.manchester.tornado.api.types.arrays.IntArray order,
                uk.ac.manchester.tornado.api.types.arrays.IntArray outKeys,
                uk.ac.manchester.tornado.api.types.arrays.DoubleArray outValues,
                int n,
                int rows,
                int valueColumns) {
            for (@uk.ac.manchester.tornado.api.annotations.Parallel int i = 0; i < n; i++) {
                int source = order.get(i);
                outKeys.set(i, keys.get(source));
                for (int c = 0; c < valueColumns; c++) {
                    outValues.set(c * n + i, values.get(c * rows + source));
                }
            }
        }
    }
}
