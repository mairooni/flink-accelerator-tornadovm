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

package org.apache.flink.table.examples.java.gpu;

import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.connector.source.*;
import org.apache.flink.core.io.InputStatus;
import org.apache.flink.core.io.SimpleVersionedSerializer;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cudf.Cudf;

import java.io.*;
import java.util.*;
import java.util.ArrayDeque;
import java.util.concurrent.CompletableFuture;

/**
 * A three-operator device region, fused into one Flink operator (option (b)).
 *
 * <p>On the CPU this query is Scan -> Calc -> LocalHashAggregate. Here all three run inside one
 * TornadoExecutionPlan and the intermediate columns never leave the device:
 *
 * <p>cuDF read_parquet -> generated kernel -> cuDF group_sum
 *
 * <p>Only the group results come back -- 1000 rows a file instead of 4,000,000 -- which is the
 * point of a region: the row boundary is paid once at the edge, not between every pair of
 * operators.
 */
public final class DeviceRegionBenchmark {

    static final int BUCKETS = 1000;

    public static void main(String[] args) throws Exception {
        String dir = args[0];
        int valueColumns = Integer.parseInt(args[1]);
        int parallelism = args.length > 2 ? Integer.parseInt(args[2]) : 1;
        String resultFile = args.length > 3 ? args[3] : null;

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setRuntimeMode(RuntimeExecutionMode.BATCH);
        env.setParallelism(parallelism);

        env.fromSource(
                        new RegionSource(dir, valueColumns),
                        WatermarkStrategy.noWatermarks(),
                        "device-region")
                .keyBy(p -> p.key)
                .reduce((a, b) -> new Pair(a.key, a.sum + b.sum))
                .map(p -> p.sum * p.sum)
                .returns(Double.class)
                .keyBy(d -> 0)
                .reduce(Double::sum)
                .map(
                        d -> {
                            if (resultFile != null) {
                                try {
                                    java.nio.file.Files.write(
                                            java.nio.file.Paths.get(resultFile),
                                            String.valueOf(d).getBytes());
                                } catch (Exception ignored) {
                                }
                            }
                            return d;
                        })
                .returns(Double.class)
                .print();

        env.execute("device-region-agg");
    }

    /** One partial group sum. */
    public static final class Pair implements Serializable {
        private static final long serialVersionUID = 1L;
        public int key;
        public double sum;

        public Pair() {}

        public Pair(int key, double sum) {
            this.key = key;
            this.sum = sum;
        }
    }

    public static final class Split implements SourceSplit, Serializable {
        private static final long serialVersionUID = 1L;
        final String path;

        Split(String path) {
            this.path = path;
        }

        @Override
        public String splitId() {
            return path;
        }
    }

    public static final class RegionSource
            implements Source<Pair, Split, List<Split>>, Serializable {
        private static final long serialVersionUID = 1L;
        private final String dir;
        private final int valueColumns;

        public RegionSource(String dir, int valueColumns) {
            this.dir = dir;
            this.valueColumns = valueColumns;
        }

        @Override
        public Boundedness getBoundedness() {
            return Boundedness.BOUNDED;
        }

        @Override
        public SourceReader<Pair, Split> createReader(SourceReaderContext c) {
            return new RegionReader(valueColumns, c);
        }

        @Override
        public SplitEnumerator<Split, List<Split>> createEnumerator(
                SplitEnumeratorContext<Split> c) {
            return new Enum(c, expand(dir));
        }

        @Override
        public SplitEnumerator<Split, List<Split>> restoreEnumerator(
                SplitEnumeratorContext<Split> c, List<Split> st) {
            return new Enum(c, new ArrayList<>(st));
        }

        @Override
        public SimpleVersionedSerializer<Split> getSplitSerializer() {
            return new JS<>();
        }

        @Override
        public SimpleVersionedSerializer<List<Split>> getEnumeratorCheckpointSerializer() {
            return new JS<>();
        }
    }

    private static List<Split> expand(String dir) {
        List<Split> out = new ArrayList<>();
        File f = new File(dir);
        File[] parts =
                f.isDirectory()
                        ? f.listFiles(x -> x.isFile() && x.getName().startsWith("part-"))
                        : new File[] {f};
        if (parts != null) {
            Arrays.sort(parts);
            for (File p : parts) out.add(new Split(p.getAbsolutePath()));
        }
        return out;
    }

    private static final class JS<T> implements SimpleVersionedSerializer<T> {
        @Override
        public int getVersion() {
            return 1;
        }

        @Override
        public byte[] serialize(T o) throws IOException {
            ByteArrayOutputStream b = new ByteArrayOutputStream();
            try (ObjectOutputStream s = new ObjectOutputStream(b)) {
                s.writeObject(o);
            }
            return b.toByteArray();
        }

        @SuppressWarnings("unchecked")
        @Override
        public T deserialize(int v, byte[] d) throws IOException {
            try (ObjectInputStream s = new ObjectInputStream(new ByteArrayInputStream(d))) {
                return (T) s.readObject();
            } catch (ClassNotFoundException e) {
                throw new IOException(e);
            }
        }
    }

    /**
     * Hands out one split per request, and says "no more" only when the queue is actually empty.
     *
     * <p>The previous version partitioned by hash against ctx.currentParallelism(), which is the
     * number of readers *registered so far* and changes as they come up. At parallelism 2 that left
     * two of eight files assigned to nobody and the query returned about three quarters of the
     * right answer -- fast, plausible, and wrong. Splits are work, not a partitioning: whoever asks
     * next gets the next one.
     */
    private static final class Enum implements SplitEnumerator<Split, List<Split>> {
        private final SplitEnumeratorContext<Split> ctx;
        private final ArrayDeque<Split> left;

        Enum(SplitEnumeratorContext<Split> c, List<Split> s) {
            ctx = c;
            left = new ArrayDeque<>(s);
        }

        @Override
        public void start() {}

        @Override
        public void handleSplitRequest(int st, String h) {
            give(st);
        }

        // Nothing on addReader: the reader asks when it is ready. Assigning here as well produced
        // splits the reader had not asked for and could not account for.
        @Override
        public void addReader(int st) {}

        private void give(int st) {
            if (!left.isEmpty()) {
                ctx.assignSplit(left.poll(), st);
            } else {
                ctx.signalNoMoreSplits(st);
            }
        }

        @Override
        public void addSplitsBack(List<Split> s, int st) {
            left.addAll(s);
        }

        @Override
        public List<Split> snapshotState(long id) {
            return new ArrayList<>(left);
        }

        @Override
        public void close() {}
    }

    /**
     * One outstanding split request at a time.
     *
     * <p>The previous version asked for another split after finishing each one, including the last.
     * Requests then outlived the reader: the enumerator answered a request from a reader that had
     * already seen notifyNoMoreSplits and ended, and the split it handed over was never read. At
     * parallelism 4 that lost a quarter of the data and returned 2.397e14 for 3.131e14 -- a
     * different wrong answer each run, and one outright failure.
     *
     * <p>The discipline here: ask only when idle, never ask again until the answer arrives, and
     * only finish when there is nothing pending and nothing outstanding.
     */
    private static final class RegionReader implements SourceReader<Pair, Split> {
        private final int valueColumns;
        private final SourceReaderContext ctx;
        private final ArrayDeque<Split> pending = new ArrayDeque<>();
        private final double[] totals = new double[BUCKETS];
        private boolean noMore, emitted, outstanding;

        RegionReader(int valueColumns, SourceReaderContext ctx) {
            this.valueColumns = valueColumns;
            this.ctx = ctx;
        }

        @Override
        public void start() {
            request();
        }

        private void request() {
            if (!outstanding && !noMore) {
                outstanding = true;
                ctx.sendSplitRequest();
            }
        }

        @Override
        public InputStatus pollNext(ReaderOutput<Pair> out) throws Exception {
            if (!pending.isEmpty()) {
                region(pending.poll().path, valueColumns, totals);
                request();
                return InputStatus.NOTHING_AVAILABLE;
            }
            if (!noMore) {
                request();
                return InputStatus.NOTHING_AVAILABLE;
            }
            if (!emitted) {
                for (int k = 0; k < BUCKETS; k++) {
                    if (totals[k] != 0.0) out.collect(new Pair(k, totals[k]));
                }
                emitted = true;
            }
            return InputStatus.END_OF_INPUT;
        }

        @Override
        public List<Split> snapshotState(long id) {
            return new ArrayList<>(pending);
        }

        @Override
        public CompletableFuture<Void> isAvailable() {
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public void addSplits(List<Split> s) {
            outstanding = false;
            pending.addAll(s);
        }

        @Override
        public void notifyNoMoreSplits() {
            outstanding = false;
            noMore = true;
        }

        @Override
        public void close() {}
    }

    /** The region: read, compute, group — three stages, one plan, nothing crosses in between. */
    static void region(String file, int valueColumns, double[] totals) throws Exception {
        long[] meta = Cudf.parquetMetadata(file);
        int[] cols = new int[valueColumns];
        for (int i = 0; i < valueColumns; i++) cols[i] = i + 1;

        // A row group at a time, not a whole file. Device memory is then bounded by the largest
        // row group rather than by the file, which is what lets several subtasks share one card
        // and what a real reader would have to do anyway.
        {
            // One read for the whole file. Reading row group by row group cost 30% -- the same
            // fragmentation spark-rapids' COALESCING reader exists to avoid.
            int rows = (int) meta[0];

            IntArray prio = new IntArray(rows);
            DoubleArray values = new DoubleArray(rows * valueColumns);
            IntArray keys = new IntArray(rows);
            DoubleArray perRow = new DoubleArray(rows);
            IntArray outKeys = new IntArray(rows);
            DoubleArray outSums = new DoubleArray(rows);
            IntArray groups = new IntArray(1);

            TaskGraph region =
                    new TaskGraph("region")
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, prio, values)
                            .libraryTask(
                                    "read",
                                    Cudf::readParquet,
                                    file,
                                    0,
                                    0,
                                    0,
                                    cols,
                                    (long) rows,
                                    prio,
                                    values)
                            .task(
                                    "calc",
                                    DeviceRegionBenchmark::calc,
                                    prio,
                                    values,
                                    keys,
                                    perRow,
                                    rows,
                                    valueColumns)
                            .libraryTask(
                                    "group",
                                    Cudf::groupSum,
                                    rows,
                                    keys,
                                    perRow,
                                    outKeys,
                                    outSums,
                                    groups)
                            .transferToHost(
                                    DataTransferMode.EVERY_EXECUTION, outKeys, outSums, groups);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(region.snapshot())) {
                plan.execute();
            }
            int n = groups.get(0);
            for (int i = 0; i < n; i++) {
                int k = outKeys.get(i);
                if (k >= 0 && k < BUCKETS) totals[k] += outSums.get(i);
            }
        }
    }

    /** The Calc: bucket key and pairwise products, per row. */
    /**
     * Transcendental work per row, over few columns.
     *
     * <p>The pairwise-product version this replaces was 15 flops for 248 bytes read -- 0.06 flops a
     * byte, which is an I/O benchmark wearing a compute costume. This is the shape the whole
     * project's finding points at: a device wins in proportion to arithmetic per byte read.
     */
    public static void calc(
            IntArray prio, DoubleArray values, IntArray keys, DoubleArray out, int rows, int cols) {
        for (@Parallel int i = 0; i < rows; i++) {
            keys.set(i, prio.get(i) % BUCKETS);
            double acc = 0.0;
            for (int c = 0; c < cols; c++) {
                double v = values.get(c * rows + i);
                acc +=
                        Math.exp(v * 0.001) * Math.log(Math.abs(v) + 1.0)
                                + Math.sin(v) * Math.cos(v)
                                + Math.sqrt(Math.abs(v));
            }
            out.set(i, acc);
        }
    }

    private DeviceRegionBenchmark() {}
}
