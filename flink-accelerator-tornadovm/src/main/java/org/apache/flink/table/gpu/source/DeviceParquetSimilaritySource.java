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
import org.apache.flink.table.gpu.codegen.GpuSimilarityJoinSpec;
import org.apache.flink.table.gpu.operator.DeviceSimilarity;
import org.apache.flink.table.types.logical.RowType;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
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
 * A nearest-neighbour query as one device region: cuDF reads both tables, cuBLAS forms every pair's
 * score, and a generated kernel reduces them away — with the corpus resident across files and the
 * score matrix never leaving the device.
 *
 * <h2>Why this region and not the Gram one</h2>
 *
 * <p>{@link DeviceParquetGramSource} puts the same three stages in one plan and cannot show what
 * the library is worth, because its GEMM is a thousandth of its running time. The reason is
 * structural and is set out in {@link GpuSimilarityJoinSpec}: a Gram matrix is {@code d/2}
 * operations per byte read and its SQL spelling caps {@code d} near 32, while a device computes
 * floating point some three thousand times faster than a source delivers bytes. This query is
 * {@code nQ·nP / 2(nQ+nP)} operations a byte — four orders of magnitude more — so the GEMM is the
 * job and what tunes the GEMM is visible on the wall clock.
 *
 * <h2>The three arms</h2>
 *
 * <p>Selected by {@code -Dflink.accelerator.similarity.contraction}, which is a property of the
 * deployment and invisible to whoever wrote the SQL. All three compute the same answer from the
 * same buffers in the same plan; only the contraction differs, so the difference between them is
 * the contraction and nothing else.
 *
 * <ul>
 *   <li>{@code cublas} — {@code cublasSgemm} writes the score matrix, a kernel reduces it.
 *   <li>{@code tiled} — a {@link KernelContext} tiled GEMM writes it instead, same reduction.
 *   <li>{@code fused} — one kernel does both and never writes a score matrix at all.
 * </ul>
 *
 * <h2>Residency</h2>
 *
 * <p>Two task graphs in one execution plan. The first reads the corpus and {@code persistOnDevice}s
 * it; the second {@code consumeFromDevice}s it and runs once per probe file. So the corpus crosses
 * the interconnect once for the whole job rather than once per file, and the only things that move
 * per file are the probe rows in and {@code nQ} answers out. The {@code nQ × nB} scores — gigabytes
 * of them — are allocated once and never read by the host.
 *
 * <h2>What is restricted, and why</h2>
 *
 * <p>The corpus must be a single Parquet file. {@code Cudf::readParquet} fills a caller-owned
 * buffer per file, and concatenating several into one column-major matrix would need either a host
 * round trip or a device copy pass; neither is interesting and both would be measured. The probe
 * side has no such limit — it is read a file at a time, which is also how the region tiles.
 */
public final class DeviceParquetSimilaritySource
        implements Source<
                        RowData,
                        DeviceParquetSimilaritySource.FileSplit,
                        List<DeviceParquetSimilaritySource.FileSplit>>,
                Serializable {

    private static final long serialVersionUID = 1L;

    /** How the score matrix is formed. A deployment property, not a query option. */
    public enum Contraction {
        CUBLAS,
        TILED,
        FUSED;

        public static Contraction fromProperty() {
            final String value =
                    System.getProperty("flink.accelerator.similarity.contraction", "cublas");
            switch (value.toLowerCase()) {
                case "tiled":
                    return TILED;
                case "fused":
                    return FUSED;
                default:
                    return CUBLAS;
            }
        }
    }

    /** Tile side of {@link DeviceSimilarity#scoreTiled}; the shapes must be multiples of it. */
    private static final int TILE = 16;

    private final List<String> probePaths;
    private final String corpusPath;
    private final GpuSimilarityJoinSpec spec;
    private final int[] probeFileColumns;
    private final int probeKeyFileColumn;
    private final int[] corpusFileColumns;
    private final RowType outputType;
    private final Contraction contraction;

    public DeviceParquetSimilaritySource(
            List<String> probePaths,
            String corpusPath,
            GpuSimilarityJoinSpec spec,
            AccelScan probeScan,
            AccelScan corpusScan,
            RowType outputType,
            Contraction contraction) {
        this.probePaths = new ArrayList<>(probePaths);
        this.corpusPath = corpusPath;
        this.spec = spec;
        this.outputType = outputType;
        this.contraction = contraction;
        // The spec indexes the columns the scan produced; the reader needs the file's own indexes.
        final int[] probeScanColumns = probeScan.projectedFields();
        final int[] probeColumns = spec.probeColumns();
        this.probeFileColumns = new int[probeColumns.length];
        for (int i = 0; i < probeColumns.length; i++) {
            this.probeFileColumns[i] = probeScanColumns[probeColumns[i]];
        }
        this.probeKeyFileColumn = probeScanColumns[spec.groupKeyField()];
        final int[] corpusScanColumns = corpusScan.projectedFields();
        final int[] corpusColumns = spec.buildColumns();
        this.corpusFileColumns = new int[corpusColumns.length];
        for (int i = 0; i < corpusColumns.length; i++) {
            this.corpusFileColumns[i] = corpusScanColumns[corpusColumns[i]];
        }
    }

    /**
     * A table's path is a directory, and a reader needs files.
     *
     * <p>{@code AccelScan} carries what the DDL wrote, which for a filesystem table is the
     * directory the parts live in. Listing it here rather than in the planner keeps the IR a
     * statement about the table and not about one reader's idea of a split.
     */
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
                    Arrays.sort(parts);
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

    private static List<FileSplit> expand(List<String> paths, boolean asSplits) {
        final List<FileSplit> splits = new ArrayList<>();
        for (String file : expand(paths)) {
            splits.add(new FileSplit(file));
        }
        return splits;
    }

    /** One probe file. */
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
        return new SimilarityReader(this, context);
    }

    @Override
    public SplitEnumerator<FileSplit, List<FileSplit>> createEnumerator(
            SplitEnumeratorContext<FileSplit> context) {
        return new FileEnumerator(context, expand(probePaths, true));
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

    /** Java serialization, which is enough for a path and a list of them. */
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

    private static final class SimilarityReader implements SourceReader<RowData, FileSplit> {
        private static final org.slf4j.Logger LOG =
                org.slf4j.LoggerFactory.getLogger(SimilarityReader.class);

        private final DeviceParquetSimilaritySource source;
        private final SourceReaderContext context;
        private final ArrayDeque<FileSplit> pending = new ArrayDeque<>();
        private final ArrayDeque<RowData> ready = new ArrayDeque<>();
        private final int d;
        private boolean noMore;
        private boolean outstanding;

        /** The corpus, read once and kept on the device for the life of the plan. */
        private FloatArray corpus;
        private int corpusRows = -1;
        private StringBuilder corpusPathHolder;
        private IntArray corpusKeys;

        private TornadoExecutionPlan plan;
        private GridScheduler grid;
        private StringBuilder probePathHolder;
        private FloatArray probe;
        private IntArray probeKeys;
        private FloatArray scores;
        private FloatArray best;
        private IntArray dims;
        private int planRows = -1;

        private long rowsIn;
        private long pairs;
        private long deviceNanos;

        SimilarityReader(DeviceParquetSimilaritySource source, SourceReaderContext context) {
            this.source = source;
            this.context = context;
            this.d = source.spec.width();
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
            if (!ready.isEmpty()) {
                output.collect(ready.poll());
                return ready.isEmpty() && noMore && pending.isEmpty()
                        ? InputStatus.END_OF_INPUT
                        : InputStatus.MORE_AVAILABLE;
            }
            if (!pending.isEmpty()) {
                one(pending.poll().path);
                request();
                return ready.isEmpty() ? InputStatus.NOTHING_AVAILABLE : InputStatus.MORE_AVAILABLE;
            }
            if (!noMore) {
                request();
                return InputStatus.NOTHING_AVAILABLE;
            }
            LOG.info(
                    "similarity region: {} probe rows x {} corpus rows x {} columns = {} pairs,"
                            + " {} ms on the device, contraction by {}",
                    rowsIn,
                    corpusRows,
                    d,
                    pairs,
                    deviceNanos / 1_000_000,
                    source.contraction);
            return InputStatus.END_OF_INPUT;
        }

        /** One probe file: read it, score it against the resident corpus, reduce, drain. */
        private void one(String file) throws Exception {
            final int rows = (int) Cudf.parquetMetadata(file)[0];
            if (rows <= 0) {
                return;
            }
            if (corpus == null) {
                buildCorpus();
            }
            if (plan == null || planRows != rows) {
                closePlan();
                build(rows);
                planRows = rows;
            }
            probePathHolder.setLength(0);
            probePathHolder.append(file);
            dims.set(0, rows);
            best.init(DeviceSimilarity.NO_MATCH);

            final long started = System.nanoTime();
            if (corpusRows > 0) {
                plan.withGraph(1).execute();
            }
            deviceNanos += System.nanoTime() - started;

            rowsIn += rows;
            pairs += (long) rows * corpusRows;
            for (int q = 0; q < rows; q++) {
                final float score = best.get(q);
                if (score == DeviceSimilarity.NO_MATCH) {
                    // An inner join emits nothing for a probe row with no partner, which here
                    // means an empty corpus and nothing else.
                    continue;
                }
                final GenericRowData row = new GenericRowData(2);
                row.setField(0, probeKeys.get(q));
                row.setField(1, score);
                ready.add(row);
            }
        }

        /**
         * Size the corpus from its footer. The buffers are allocated here and filled by graph 0 of
         * the plan, which runs once; everything after that consumes them where they already are.
         */
        private void buildCorpus() {
            corpusRows = (int) Cudf.parquetMetadata(source.corpusPath)[0];
            corpus = new FloatArray(Math.max(1, corpusRows) * d);
            corpusKeys = new IntArray(1);
            corpusPathHolder = new StringBuilder(source.corpusPath);
        }

        private void build(int rows) {
            probePathHolder = new StringBuilder();
            probe = new FloatArray(rows * d);
            probeKeys = new IntArray(rows);
            best = new FloatArray(rows);
            dims = new IntArray(3);
            dims.set(0, rows);
            dims.set(1, corpusRows);
            dims.set(2, d);

            final boolean materialise = source.contraction != Contraction.FUSED;
            scores = new FloatArray(materialise ? (long) rows * corpusRows > Integer.MAX_VALUE
                            ? 1
                            : rows * corpusRows : 1);

            // Graph 0: the corpus, read once and left where it lands.
            final TaskGraph corpusGraph =
                    new TaskGraph("corpus")
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION, corpusKeys, corpus)
                            .libraryTask(
                                    "read",
                                    Cudf::readParquet,
                                    corpusPathHolder,
                                    0,
                                    0,
                                    -1,
                                    source.corpusFileColumns,
                                    (long) corpusRows,
                                    corpusKeys,
                                    corpus)
                            .persistOnDevice(corpus);

            TaskGraph region =
                    new TaskGraph("region")
                            .consumeFromDevice("corpus", corpus)
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION,
                                    probeKeys,
                                    probe,
                                    best,
                                    dims)
                            .libraryTask(
                                    "read",
                                    Cudf::readParquet,
                                    probePathHolder,
                                    0,
                                    0,
                                    source.probeKeyFileColumn,
                                    source.probeFileColumns,
                                    (long) rows,
                                    probeKeys,
                                    probe);

            final GridScheduler scheduler = new GridScheduler();
            switch (source.contraction) {
                case CUBLAS:
                    // Column-major throughout: the probe is nQ x d with leading dimension nQ, the
                    // corpus is nB x d with leading dimension nB -- which is the transpose cuBLAS
                    // wants for the second operand, so OP_T costs nothing and no pass transposes
                    // anything. The scores come out nQ x nB with leading dimension nQ, which is
                    // what makes the reduction below coalesced.
                    region =
                            region.libraryTask(
                                    "gemm",
                                    CuBlas::cublasSgemm,
                                    CuBlasOperation.CUBLAS_OP_N.operation(),
                                    CuBlasOperation.CUBLAS_OP_T.operation(),
                                    rows,
                                    corpusRows,
                                    d,
                                    1.0f,
                                    probe,
                                    rows,
                                    corpus,
                                    corpusRows,
                                    0.0f,
                                    scores,
                                    rows);
                    region = region.task("reduce", DeviceSimilarity::rowMax, scores, best, dims);
                    scheduler.addWorkerGrid("region.reduce", new WorkerGrid1D(rows));
                    break;
                case TILED:
                    region =
                            region.task(
                                    "gemm",
                                    DeviceSimilarity::scoreTiled,
                                    new KernelContext(),
                                    probe,
                                    corpus,
                                    scores,
                                    rows,
                                    corpusRows,
                                    d);
                    region = region.task("reduce", DeviceSimilarity::rowMax, scores, best, dims);
                    final WorkerGrid2D tiles = new WorkerGrid2D(rows, corpusRows);
                    tiles.setLocalWork(TILE, TILE, 1);
                    scheduler.addWorkerGrid("region.gemm", tiles);
                    scheduler.addWorkerGrid("region.reduce", new WorkerGrid1D(rows));
                    break;
                case FUSED:
                default:
                    region =
                            region.task(
                                    "gemm",
                                    DeviceSimilarity::scoreFused,
                                    probe,
                                    corpus,
                                    best,
                                    dims);
                    scheduler.addWorkerGrid("region.gemm", new WorkerGrid1D(rows));
                    break;
            }
            region =
                    region.transferToHost(DataTransferMode.EVERY_EXECUTION, best, probeKeys);

            grid = scheduler;
            plan =
                    new TornadoExecutionPlan(corpusGraph.snapshot(), region.snapshot())
                            .withGridScheduler(scheduler);
            // Once, before anything consumes it. Everything after this reads the corpus where it
            // already is.
            plan.withGraph(0).execute();
        }

        private void closePlan() {
            if (plan != null) {
                try {
                    plan.close();
                } catch (Exception e) {
                    LOG.warn("closing the similarity plan", e);
                }
                plan = null;
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
            closePlan();
            // TornadoVM allocates from an Arena.ofAuto(), so these are freed by the collector and
            // there is no deterministic release to call. Dropping the references is what lets it.
            corpus = null;
            probe = null;
            scores = null;
            best = null;
        }
    }
}
