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
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.gpu.operator.ReferenceJoinEngine;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.StringJoiner;
import java.util.stream.Stream;

/**
 * A benchmark for device residency, on a query that asks for it.
 *
 * <pre>{@code
 * SELECT q.id, MAX(q.d0*r.d0 + q.d1*r.d1 + ... ) AS best
 * FROM Queries q, Refs r
 * GROUP BY q.id
 * }</pre>
 *
 * <p>Nearest reference by inner product: the shape underneath vector search, recommendation
 * retrieval, entity resolution against a canonical set, and any scoring of a stream against a fixed
 * catalogue.
 *
 * <h2>Why this query exists</h2>
 *
 * <p>{@link LogisticRegressionBenchmark} measures residency at 48x and cannot defend the number.
 * The data stays on the device across fifty passes, but the fifty is a {@code for} loop in the
 * driver -- SQL has no iteration construct, so nothing in the query asked for it -- and the CPU arm
 * it is measured against re-reads the input file every pass because Flink SQL has no way to cache a
 * table. Most of that 48x is the baseline's inability to hold data, not the device's ability to.
 *
 * <p>This query has the same property without either weakness. {@code Refs} is invariant: the join
 * says so, in SQL, with no loop anywhere. And the reuse count is {@code queryRows / batch} -- the
 * cardinality of the probe side divided by the batch the operator chose. Both are properties of the
 * job.
 *
 * <h2>Four arms, because the end-to-end number is a product of three separate wins</h2>
 *
 * <p>Running this query with the integration beats vanilla Flink for three reasons at once, and
 * they are worth different amounts and belong to different parts of the system. Quoting only the
 * end-to-end ratio hides that, and invites the reader to attribute all of it to the GPU:
 *
 * <ul>
 *   <li>{@code sql} -- the query on vanilla Flink. A cross join: it materialises {@code queries x
 *       refs} tuples through the runtime.
 *   <li>{@code cpu} -- the same algorithm as the device arms, on the host. Blocked dot products
 *       against a reference array held in memory, no join intermediate, no device. Against {@code
 *       sql} this measures what not materialising the join is worth, which is a plan win rather
 *       than a device one.
 *   <li>{@code reupload} -- the device, with {@code Refs} under {@code EVERY_EXECUTION}. Against
 *       {@code cpu} this measures the device.
 *   <li>{@code resident} -- the device, with {@code Refs} under {@code FIRST_EXECUTION}. Against
 *       {@code reupload} this measures residency, and nothing else: same kernels, same arithmetic,
 *       same answers, one argument different.
 * </ul>
 *
 * <p>The three operator arms share one harness -- same gather, same batching, same emission, same
 * reference load -- so each neighbouring pair differs in exactly one thing. {@code --decompose}
 * runs all four and reports the chain.
 *
 * <h2>What is fair here and what is not</h2>
 *
 * <p>Every arm reads both tables and pays for both, and the reported wall time includes loading the
 * reference side. The {@code cpu} arm computes in double while the device arms are FP32, which
 * favours the device by up to the width of one SIMD register; it is written that way because the
 * SQL arm computes in double and the three are meant to answer the same question.
 *
 * <p>The comparison is against Flink as it actually runs, which is the same basis as {@link
 * HaversineBenchmark}. It is not a claim about CPUs in general: a host BLAS would close much of the
 * {@code cpu} to {@code resident} gap, and nothing here has measured that.
 *
 * <h2>Usage</h2>
 *
 * <pre>
 *   # once: write both tables
 *   flink run examples/table/ReferenceJoinBenchmark.jar --generate --data /tmp/refjoin
 *
 *   # all four arms, and the decomposition
 *   flink run examples/table/ReferenceJoinBenchmark.jar --data /tmp/refjoin --decompose
 *
 *   # residency alone, swept against batch size
 *   flink run examples/table/ReferenceJoinBenchmark.jar --data /tmp/refjoin --sweep
 *
 *   # correctness, at a size the SQL arm can finish
 *   flink run examples/table/ReferenceJoinBenchmark.jar --generate --verify \
 *       --data /tmp/refjoin-small --ref-count 512 --query-rows 2048 --dims 16
 * </pre>
 */
public final class ReferenceJoinBenchmark {

    /** Batch sizes the residency sweep visits, smallest first, so the ratio moves one way. */
    private static final int[] SWEEP = {128, 256, 512, 1024, 2048, 4096};

    /** Which compute the shared operator harness runs. */
    private enum Arm {
        SQL,
        CPU,
        GENERATED,
        REUPLOAD,
        RESIDENT;

        static Arm of(String name) {
            switch (name) {
                case "sql":
                    return SQL;
                case "cpu":
                    return CPU;
                case "generated":
                    return GENERATED;
                case "reupload":
                    return REUPLOAD;
                case "resident":
                    return RESIDENT;
                default:
                    throw new IllegalArgumentException(
                            "unknown arm " + name + " -- expected sql, cpu, reupload or resident");
            }
        }

        String label() {
            return name().toLowerCase();
        }
    }

    public static void main(String[] args) throws Exception {
        Args parsed = Args.parse(args);

        if (parsed.generate) {
            generate(parsed);
        }
        if (!Files.isDirectory(Paths.get(parsed.queryData))) {
            throw new IllegalStateException(
                    parsed.queryData + " does not exist -- run with --generate first");
        }

        System.out.printf(
                "data=%s  refs=%,d x %d (%.1f MB)  queries=%,d  parallelism=%d  ops/byte=%d%n",
                parsed.data,
                parsed.refCount,
                parsed.dims,
                parsed.refCount * (double) parsed.dims * Float.BYTES / (1 << 20),
                parsed.queryRows,
                parsed.parallelism,
                parsed.refCount / 4);

        if (parsed.verify) {
            verify(parsed);
            return;
        }
        if (parsed.decompose) {
            decompose(parsed);
            return;
        }
        if (parsed.sweep) {
            sweep(parsed);
            return;
        }

        Arm arm = Arm.of(parsed.arm);
        for (int run = 1; run <= parsed.runs; run++) {
            Measurement m = measure(parsed, parsed.batch, arm);
            System.out.printf(
                    "run %2d  %-8s  %10.0f ms wall  %8.1f ms compute  batches=%,d  checksum=%.6f%n",
                    run, arm.label(), m.wallMillis, m.computeMillis, m.executions, m.checksum);
        }
    }

    /**
     * All four arms at one batch size, and the chain of ratios between them.
     *
     * <p>Each line of the decomposition is a pair that differs in one thing. The end-to-end figure
     * is the product of the three, and is the number a user of the integration would see.
     */
    private static void decompose(Args args) throws Exception {
        System.out.printf("%nbatch=%,d  runs=%d%n%n", args.batch, args.runs);

        Measurement sql = args.skipSql ? null : best(args, args.batch, Arm.SQL);
        Measurement cpu = best(args, args.batch, Arm.CPU);
        Measurement generated = best(args, args.batch, Arm.GENERATED);
        Measurement reupload = best(args, args.batch, Arm.REUPLOAD);
        Measurement resident = best(args, args.batch, Arm.RESIDENT);

        System.out.printf(
                "%-22s %12s %12s %10s %14s%n", "arm", "wall", "compute", "batches", "checksum");
        if (sql != null) {
            System.out.printf(
                    "%-22s %9.0f ms %12s %10s %14.4f%n",
                    "vanilla SQL cross join", sql.wallMillis, "-", "-", sql.checksum);
        }
        for (Measurement m : new Measurement[] {cpu, generated, reupload, resident}) {
            System.out.printf(
                    "%-22s %9.0f ms %9.1f ms %,10d %14.4f%n",
                    m.label, m.wallMillis, m.computeMillis, m.executions, m.checksum);
        }

        // The arms must agree. FP32 against the CPU's double moves the last digits of a sum over
        // queryRows scores, so this is a precision tolerance, not a fudge factor.
        agree(cpu, generated);
        agree(cpu, reupload);
        agree(cpu, resident);
        if (sql != null) {
            agree(sql, resident);
        }

        System.out.printf("%n%-46s %9s%n", "decomposition", "speedup");
        if (sql != null) {
            System.out.printf(
                    "%-46s %8.2fx%n",
                    "SQL -> CPU operator   (no join materialised)",
                    sql.wallMillis / cpu.wallMillis);
        }
        System.out.printf(
                "%-46s %8.2fx%n",
                "CPU operator -> device  (the device, wall)", cpu.wallMillis / reupload.wallMillis);
        // The same pair on compute alone. The wall figure above is Amdahl-diluted -- both arms
        // read the same source and load the same reference table, and past a point that fixed
        // cost is the whole difference. This is the ratio the device is actually responsible for,
        // and it is the analogue of HaversineBenchmark's "expression only" column.
        System.out.printf(
                "%-46s %8.2fx%n",
                "  of which, compute only", cpu.computeMillis / reupload.computeMillis);
        // Split that compute figure into the two things it contains. The generated contraction is
        // what the automatic path could reach if it could emit this shape at all; the remainder is
        // what calling cuBLAS instead is worth. Compute rather than wall, because at these sizes
        // the wall times of the two device arms differ by less than the run-to-run noise.
        System.out.printf(
                "%-46s %8.2fx%n",
                "    device, generated contraction", cpu.computeMillis / generated.computeMillis);
        System.out.printf(
                "%-46s %8.2fx%n",
                "    what cuBLAS adds over generated",
                generated.computeMillis / reupload.computeMillis);
        // Assumes the device arm pays the same non-compute cost as the host arm, which it does by
        // construction -- same source, same reference load. Run to run that cost varies by a few
        // per cent, so a measured wall ratio can sit just above this line without meaning anything.
        double ceiling = cpu.wallMillis / (cpu.wallMillis - cpu.computeMillis);
        System.out.printf("%-46s %8.2fx%n", "  ceiling for any device (Amdahl)", ceiling);
        System.out.printf(
                "%-46s %8.2fx%n",
                "reupload -> resident    (residency)",
                reupload.computeMillis / resident.computeMillis);
        if (sql != null) {
            System.out.printf(
                    "%n%-46s %8.2fx%n",
                    "SQL -> resident device  (END TO END)", sql.wallMillis / resident.wallMillis);
        }
    }

    /**
     * The residency A/B, at every batch size.
     *
     * <p>Compute time rather than wall: both arms read the same input through the same source and
     * load the same reference table, and those costs are equal by construction, so including them
     * would dilute a ratio that is about one thing.
     */
    private static void sweep(Args args) throws Exception {
        System.out.printf(
                "%n%-8s %12s %12s %9s %12s %10s%n",
                "batch", "resident", "reupload", "speedup", "wasted MB", "GFLOP/exec");
        for (int batch : SWEEP) {
            if (batch > args.queryRows) {
                continue;
            }
            // scores is batch x refCount floats and is the largest buffer in the graph; skip a
            // point that would not fit rather than failing the whole sweep on the last row.
            long scoreBytes = (long) batch * args.refCount * Float.BYTES;
            if (scoreBytes > args.scoreLimitBytes) {
                System.out.printf(
                        "%-8d skipped -- scores would be %.1f MB, over the %.0f MB limit%n",
                        batch,
                        scoreBytes / (double) (1 << 20),
                        args.scoreLimitBytes / (double) (1 << 20));
                continue;
            }

            Measurement resident = best(args, batch, Arm.RESIDENT);
            Measurement reupload = best(args, batch, Arm.REUPLOAD);
            double wastedMb =
                    reupload.wastedUploads()
                            * args.refCount
                            * (double) args.dims
                            * Float.BYTES
                            / (1 << 20);
            System.out.printf(
                    "%-8d %9.1f ms %9.1f ms %8.2fx %12.1f %10.2f%n",
                    batch,
                    resident.computeMillis,
                    reupload.computeMillis,
                    reupload.computeMillis / resident.computeMillis,
                    wastedMb,
                    (double) batch * args.refCount * args.dims * 2 / 1e9);
            agree(resident, reupload);
        }
    }

    /** Fastest of {@code runs} attempts, to keep JIT and kernel warm-up out of the comparison. */
    private static Measurement best(Args args, int batch, Arm arm) throws Exception {
        Measurement best = null;
        for (int run = 0; run < args.runs; run++) {
            Measurement m = measure(args, batch, arm);
            if (best == null || m.wallMillis < best.wallMillis) {
                best = m;
            }
        }
        return best;
    }

    private static Measurement measure(Args args, int batch, Arm arm) throws Exception {
        return arm == Arm.SQL ? sqlJoin(args) : operatorJoin(args, batch, arm);
    }

    /**
     * The query on vanilla Flink: a cross join, a dot product per pair, a maximum per query row.
     *
     * <p>Written the way a user would, which is also the fastest form SQL offers -- one wide
     * expression rather than a join on a dimension index, which would materialise {@code queries x
     * refs x dims} tuples instead of {@code queries x refs}.
     */
    private static Measurement sqlJoin(Args args) throws Exception {
        StreamTableEnvironment env = environment(args);
        env.executeSql(refTable(args));
        env.executeSql(queryTable(args));

        StringJoiner dot = new StringJoiner(" + ");
        for (int d = 0; d < args.dims; d++) {
            dot.add(String.format("q.d%d * r.d%d", d, d));
        }
        String sql =
                String.format(
                        "SELECT q.id, MAX(%s) AS best FROM Queries q, Refs r GROUP BY q.id",
                        dot.toString());

        long start = System.nanoTime();
        double checksum = 0.0;
        long rows = 0;
        try (CloseableIterator<Row> it = env.executeSql(sql).collect()) {
            while (it.hasNext()) {
                checksum += ((Number) it.next().getField(1)).doubleValue();
                rows++;
            }
        }
        double wall = (System.nanoTime() - start) / 1e6;
        if (rows != args.queryRows) {
            throw new IllegalStateException(
                    "sql arm produced " + rows + " rows, expected " + args.queryRows);
        }
        return new Measurement("vanilla SQL", wall, 0.0, 0, 1, checksum);
    }

    /**
     * The same query through the shared operator harness, on the host or on the device.
     *
     * <p>The reference table is loaded by each subtask in {@code open()}, which is what a broadcast
     * join does with its build side. That load is inside the job, so the reported wall time
     * includes it for every operator arm.
     */
    private static Measurement operatorJoin(Args args, int batch, Arm arm) throws Exception {
        StreamTableEnvironment env = environment(args);
        env.executeSql(queryTable(args));

        DataStream<Row> results =
                env.toDataStream(env.from("Queries"))
                        .transform(
                                "reference-join-" + arm.label(),
                                receiptType(),
                                new ReferenceJoinOperator(
                                        args.refData, args.refCount, args.dims, batch, arm));

        long start = System.nanoTime();
        double checksum = 0.0;
        double computeMillis = 0.0;
        long executions = 0;
        int subtasks = 0;
        try (CloseableIterator<Row> it = results.executeAndCollect()) {
            while (it.hasNext()) {
                Row row = it.next();
                if (row.getField(0) == null) {
                    // The per-subtask receipt, emitted once at endInput. Compute time is the
                    // slowest subtask rather than the sum: they run concurrently, so the sum would
                    // report a duration nothing ever waited for.
                    computeMillis = Math.max(computeMillis, (Double) row.getField(3));
                    executions += (Long) row.getField(4);
                    subtasks++;
                } else {
                    checksum += (Double) row.getField(1);
                }
            }
        }
        if (subtasks == 0) {
            throw new IllegalStateException("no subtask receipt arrived -- the operator never ran");
        }
        double wall = (System.nanoTime() - start) / 1e6;
        return new Measurement(
                arm == Arm.CPU ? "CPU operator" : "device " + arm.label(),
                wall,
                computeMillis,
                executions,
                subtasks,
                checksum);
    }

    private static TypeInformation<Row> receiptType() {
        return new RowTypeInfo(
                TypeInformation.of(Long.class),
                TypeInformation.of(Double.class),
                TypeInformation.of(Integer.class),
                TypeInformation.of(Double.class),
                TypeInformation.of(Long.class));
    }

    /** Runs the device arm and the SQL arm and compares them row by row. */
    private static void verify(Args args) throws Exception {
        StreamTableEnvironment env = environment(args);
        env.executeSql(refTable(args));
        env.executeSql(queryTable(args));
        StringJoiner dot = new StringJoiner(" + ");
        for (int d = 0; d < args.dims; d++) {
            dot.add(String.format("q.d%d * r.d%d", d, d));
        }
        List<Row> expected = new ArrayList<>();
        try (CloseableIterator<Row> it =
                env.executeSql(
                                String.format(
                                        "SELECT q.id, MAX(%s) AS best FROM Queries q, Refs r"
                                                + " GROUP BY q.id",
                                        dot))
                        .collect()) {
            while (it.hasNext()) {
                expected.add(it.next());
            }
        }
        expected.sort(Comparator.comparingLong(r -> ((Number) r.getField(0)).longValue()));
        System.out.printf("sql arm produced %,d rows%n", expected.size());

        for (Arm arm : new Arm[] {Arm.CPU, Arm.RESIDENT}) {
            StreamTableEnvironment devEnv = environment(args);
            devEnv.executeSql(queryTable(args));
            DataStream<Row> results =
                    devEnv.toDataStream(devEnv.from("Queries"))
                            .transform(
                                    "reference-join-" + arm.label(),
                                    receiptType(),
                                    new ReferenceJoinOperator(
                                            args.refData,
                                            args.refCount,
                                            args.dims,
                                            args.batch,
                                            arm));
            List<Row> actual = new ArrayList<>();
            try (CloseableIterator<Row> it = results.executeAndCollect()) {
                while (it.hasNext()) {
                    Row row = it.next();
                    if (row.getField(0) != null) {
                        actual.add(row);
                    }
                }
            }
            actual.sort(Comparator.comparingLong(r -> ((Number) r.getField(0)).longValue()));
            if (actual.size() != expected.size()) {
                throw new IllegalStateException(
                        "row counts differ: sql "
                                + expected.size()
                                + ", "
                                + arm.label()
                                + " "
                                + actual.size());
            }
            double worst = 0.0;
            for (int i = 0; i < actual.size(); i++) {
                double a = ((Number) expected.get(i).getField(1)).doubleValue();
                double b = ((Number) actual.get(i).getField(1)).doubleValue();
                worst = Math.max(worst, Math.abs(a - b) / Math.max(1.0, Math.abs(a)));
            }
            System.out.printf(
                    "%-8s worst relative difference %.3e over %,d rows%n",
                    arm.label(), worst, actual.size());
            if (worst > 1e-5) {
                throw new IllegalStateException(
                        arm.label() + " answers differ from SQL by " + worst);
            }
        }
        System.out.println("verified");
    }

    /** What one arm computes, so the harness can hold either without knowing which. */
    private interface Join extends AutoCloseable {
        void setReference(int ref, int dim, double value);

        void open();

        void set(int dim, double value);

        int rowComplete();

        int flush();

        double bestScore(int index);

        int bestRef(int index);

        long executionCount();

        double computeMillis();
    }

    /**
     * The host arm: blocked dot products against a reference array held in memory.
     *
     * <p>Row-major on both sides, so the inner loop walks both operands contiguously. This is the
     * straightforward implementation, and it is deliberately not a strawman -- it does no join
     * materialisation, holds the references exactly as the device arm does, and skips the tail
     * padding the device pays for.
     */
    private static final class HostJoin implements Join {
        private final int refCount;
        private final int dims;
        private final int batchSize;
        private final double[] refs;
        private final double[] queries;
        private final double[] bestScore;
        private final int[] bestRef;
        private int staged;
        private long executions;
        private double computeMillis;

        private HostJoin(int refCount, int dims, int batchSize) {
            this.refCount = refCount;
            this.dims = dims;
            this.batchSize = batchSize;
            this.refs = new double[refCount * dims];
            this.queries = new double[batchSize * dims];
            this.bestScore = new double[batchSize];
            this.bestRef = new int[batchSize];
        }

        @Override
        public void setReference(int ref, int dim, double value) {
            refs[ref * dims + dim] = value;
        }

        @Override
        public void open() {}

        @Override
        public void set(int dim, double value) {
            queries[staged * dims + dim] = value;
        }

        @Override
        public int rowComplete() {
            staged++;
            if (staged == batchSize) {
                return execute();
            }
            return 0;
        }

        @Override
        public int flush() {
            return staged > 0 ? execute() : 0;
        }

        private int execute() {
            int rows = staged;
            long start = System.nanoTime();
            for (int q = 0; q < rows; q++) {
                int qb = q * dims;
                double top = Double.NEGATIVE_INFINITY;
                int at = 0;
                for (int r = 0; r < refCount; r++) {
                    int rb = r * dims;
                    double sum = 0.0;
                    for (int d = 0; d < dims; d++) {
                        sum += queries[qb + d] * refs[rb + d];
                    }
                    if (sum > top) {
                        top = sum;
                        at = r;
                    }
                }
                bestScore[q] = top;
                bestRef[q] = at;
            }
            computeMillis += (System.nanoTime() - start) / 1e6;
            executions++;
            staged = 0;
            return rows;
        }

        @Override
        public double bestScore(int index) {
            return bestScore[index];
        }

        @Override
        public int bestRef(int index) {
            return bestRef[index];
        }

        @Override
        public long executionCount() {
            return executions;
        }

        @Override
        public double computeMillis() {
            return computeMillis;
        }

        @Override
        public void close() {}
    }

    /** The device arm, delegating to the engine. */
    private static final class DeviceJoin implements Join {
        private final ReferenceJoinEngine engine;

        private DeviceJoin(
                int refCount, int dims, int batchSize, boolean resident, String contraction) {
            this.engine = new ReferenceJoinEngine(refCount, dims, batchSize, resident, contraction);
        }

        @Override
        public void setReference(int ref, int dim, double value) {
            engine.setReference(ref, dim, value);
        }

        @Override
        public void open() {
            engine.open();
        }

        @Override
        public void set(int dim, double value) {
            engine.set(dim, value);
        }

        @Override
        public int rowComplete() {
            return engine.rowComplete();
        }

        @Override
        public int flush() {
            return engine.flush();
        }

        @Override
        public double bestScore(int index) {
            return engine.bestScore(index);
        }

        @Override
        public int bestRef(int index) {
            return engine.bestRef(index);
        }

        @Override
        public long executionCount() {
            return engine.executionCount();
        }

        @Override
        public double computeMillis() {
            return engine.executeMillis();
        }

        @Override
        public void close() throws Exception {
            engine.close();
        }
    }

    /**
     * Buffers query rows, runs one batch at a time, emits one result row per query.
     *
     * <p>One harness for all three operator arms, so a pair of them differs only in the {@link
     * Join} it holds. Also emits a receipt row per subtask at {@code endInput} carrying compute
     * time and execution count, measured where they happen rather than inferred from wall time.
     */
    private static final class ReferenceJoinOperator extends AbstractStreamOperator<Row>
            implements OneInputStreamOperator<Row, Row>, BoundedOneInput {

        private static final long serialVersionUID = 1L;

        private final String refData;
        private final int refCount;
        private final int dims;
        private final int batchSize;
        private final Arm arm;

        private transient Join join;
        private transient long[] pending;
        private transient int pendingCount;

        private ReferenceJoinOperator(
                String refData, int refCount, int dims, int batchSize, Arm arm) {
            this.refData = refData;
            this.refCount = refCount;
            this.dims = dims;
            this.batchSize = batchSize;
            this.arm = arm;
        }

        @Override
        public void open() throws Exception {
            super.open();
            join =
                    arm == Arm.CPU
                            ? new HostJoin(refCount, dims, batchSize)
                            : new DeviceJoin(
                                    refCount,
                                    dims,
                                    batchSize,
                                    arm != Arm.REUPLOAD,
                                    arm == Arm.GENERATED ? "generated" : "library");
            long start = System.nanoTime();
            loadReferences();
            double loadMillis = (System.nanoTime() - start) / 1e6;
            join.open();
            pending = new long[batchSize];
            pendingCount = 0;
            LOG.info(
                    "reference-join {}: {} references x {} dims loaded in {} ms, batch {}",
                    arm.label(),
                    refCount,
                    dims,
                    loadMillis,
                    batchSize);
        }

        /** Reads the reference table straight off disk -- the build side, already materialised. */
        private void loadReferences() throws IOException {
            int loaded = 0;
            List<Path> parts = new ArrayList<>();
            try (Stream<Path> files = Files.walk(Paths.get(refData))) {
                // The filesystem sink leaves hidden in-progress files behind; they duplicate rows
                // the finalised part files already carry.
                files.filter(Files::isRegularFile)
                        .filter(f -> !f.getFileName().toString().startsWith("."))
                        .filter(f -> !f.getFileName().toString().startsWith("_"))
                        .sorted()
                        .forEach(parts::add);
            }
            for (Path part : parts) {
                for (String line : Files.readAllLines(part, StandardCharsets.UTF_8)) {
                    if (line.isEmpty()) {
                        continue;
                    }
                    String[] fields = line.split(",");
                    // Column 0 is the id; the dimensions follow it.
                    for (int d = 0; d < dims; d++) {
                        join.setReference(loaded, d, Double.parseDouble(fields[d + 1]));
                    }
                    loaded++;
                }
            }
            if (loaded != refCount) {
                throw new IllegalStateException(
                        "expected " + refCount + " references in " + refData + ", read " + loaded);
            }
        }

        @Override
        public void processElement(StreamRecord<Row> element) {
            Row row = element.getValue();
            pending[pendingCount++] = ((Number) row.getField(0)).longValue();
            for (int d = 0; d < dims; d++) {
                join.set(d, ((Number) row.getField(d + 1)).doubleValue());
            }
            emit(join.rowComplete());
        }

        @Override
        public void endInput() {
            emit(join.flush());
            // The receipt: a null id marks it, so the driver can tell it from a result.
            output.collect(
                    new StreamRecord<>(
                            Row.of(null, null, null, join.computeMillis(), join.executionCount())));
        }

        private void emit(int ready) {
            for (int i = 0; i < ready; i++) {
                output.collect(
                        new StreamRecord<>(
                                Row.of(
                                        pending[i],
                                        join.bestScore(i),
                                        join.bestRef(i),
                                        null,
                                        null)));
            }
            if (ready > 0) {
                pendingCount = 0;
            }
        }

        @Override
        public void close() throws Exception {
            if (join != null) {
                join.close();
            }
            super.close();
        }
    }

    /** Writes both tables once, so no arm pays for generating rows. */
    private static void generate(Args args) throws Exception {
        write(args, args.refData, "Refs", args.refCount);
        write(args, args.queryData, "Queries", args.queryRows);
    }

    private static void write(Args args, String path, String name, int rows) throws Exception {
        if (Files.isDirectory(Paths.get(path))) {
            System.out.printf("%s already exists, leaving it alone%n", path);
            return;
        }
        StreamTableEnvironment env = environment(args);

        StringJoiner schema = new StringJoiner(",\n  ");
        StringJoiner options = new StringJoiner(",\n  ");
        schema.add("id BIGINT");
        // Both the sequence bound and number-of-rows: the sequence gives each row a stable
        // identity to join and sort on, and the row count is what makes the source bounded, which
        // batch mode requires.
        options.add(String.format("'number-of-rows' = '%d'", rows));
        options.add("'fields.id.kind' = 'sequence'");
        options.add("'fields.id.start' = '0'");
        options.add(String.format("'fields.id.end' = '%d'", rows - 1));
        for (int d = 0; d < args.dims; d++) {
            schema.add(String.format("d%d DOUBLE", d));
            options.add(String.format("'fields.d%d.min' = '-1.0'", d));
            options.add(String.format("'fields.d%d.max' = '1.0'", d));
        }
        env.executeSql(
                String.format(
                        "CREATE TABLE Gen%s (%n  %s%n) WITH (%n  'connector' = 'datagen',%n  %s%n)",
                        name, schema, options));
        env.executeSql(table(args, name, path));

        System.out.printf("writing %,d rows x %d dims to %s%n", rows, args.dims, path);
        long start = System.nanoTime();
        env.executeSql(String.format("INSERT INTO %s SELECT * FROM Gen%s", name, name)).await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    private static String refTable(Args args) {
        return table(args, "Refs", args.refData);
    }

    private static String queryTable(Args args) {
        return table(args, "Queries", args.queryData);
    }

    private static String table(Args args, String name, String path) {
        StringJoiner schema = new StringJoiner(",\n  ");
        schema.add("id BIGINT");
        for (int d = 0; d < args.dims; d++) {
            schema.add(String.format("d%d DOUBLE", d));
        }
        return String.format(
                "CREATE TABLE %s (%n  %s%n) WITH (%n  'connector' = 'filesystem',%n"
                        + "  'path' = '%s',%n  'format' = 'csv'%n)",
                name, schema, path);
    }

    private static StreamTableEnvironment environment(Args args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setRuntimeMode(RuntimeExecutionMode.BATCH);
        env.setParallelism(args.parallelism);
        return StreamTableEnvironment.create(env);
    }

    /**
     * Checks two arms produced the same answers.
     *
     * <p>They must: every arm computes the same maxima over the same inputs. The tolerance is the
     * FP32 device arms against the double host ones, over a sum of {@code queryRows} scores.
     */
    private static void agree(Measurement first, Measurement second) {
        double drift =
                Math.abs(first.checksum - second.checksum)
                        / Math.max(1.0, Math.abs(first.checksum));
        if (drift > 1e-5) {
            throw new IllegalStateException(
                    "arms disagree: "
                            + first.label
                            + " "
                            + first.checksum
                            + " vs "
                            + second.label
                            + " "
                            + second.checksum);
        }
    }

    /** What one arm produced, at one batch size. */
    private static final class Measurement {
        private final String label;
        private final double wallMillis;
        private final double computeMillis;
        private final long executions;
        private final int subtasks;
        private final double checksum;

        private Measurement(
                String label,
                double wallMillis,
                double computeMillis,
                long executions,
                int subtasks,
                double checksum) {
            this.label = label;
            this.wallMillis = wallMillis;
            this.computeMillis = computeMillis;
            this.executions = executions;
            this.subtasks = subtasks;
            this.checksum = checksum;
        }

        /**
         * Uploads the reupload arm made that the resident arm did not.
         *
         * <p>Each subtask uploads once under {@code FIRST_EXECUTION} and once per batch under
         * {@code EVERY_EXECUTION}, so the difference is one upload per subtask fewer than the
         * total, not one overall.
         */
        private long wastedUploads() {
            return executions - subtasks;
        }
    }

    private static final class Args {
        private String data;
        private String refData;
        private String queryData;
        private int refCount = 32_768;
        private int dims = 128;
        private int queryRows = 32_768;
        private int batch = 1024;
        private int parallelism = 1;
        private int runs = 3;
        private String arm = "resident";
        private boolean generate;
        private boolean sweep;
        private boolean decompose;
        private boolean skipSql;
        private boolean verify;
        private long scoreLimitBytes = 512L << 20;

        static Args parse(String[] argv) {
            Args args = new Args();
            for (int i = 0; i < argv.length; i++) {
                String flag = argv[i];
                if ("--data".equals(flag)) {
                    args.data = argv[++i];
                } else if ("--ref-count".equals(flag)) {
                    args.refCount = Integer.parseInt(argv[++i]);
                } else if ("--query-rows".equals(flag)) {
                    args.queryRows = Integer.parseInt(argv[++i]);
                } else if ("--dims".equals(flag)) {
                    args.dims = Integer.parseInt(argv[++i]);
                } else if ("--batch".equals(flag)) {
                    args.batch = Integer.parseInt(argv[++i]);
                } else if ("--parallelism".equals(flag)) {
                    args.parallelism = Integer.parseInt(argv[++i]);
                } else if ("--runs".equals(flag)) {
                    args.runs = Integer.parseInt(argv[++i]);
                } else if ("--arm".equals(flag)) {
                    args.arm = argv[++i];
                } else if ("--score-limit-mb".equals(flag)) {
                    args.scoreLimitBytes = Long.parseLong(argv[++i]) << 20;
                } else if ("--generate".equals(flag)) {
                    args.generate = true;
                } else if ("--sweep".equals(flag)) {
                    args.sweep = true;
                } else if ("--decompose".equals(flag)) {
                    args.decompose = true;
                } else if ("--skip-sql".equals(flag)) {
                    args.skipSql = true;
                } else if ("--verify".equals(flag)) {
                    args.verify = true;
                } else {
                    throw new IllegalArgumentException("unknown argument " + flag);
                }
            }
            if (args.data == null) {
                args.data =
                        String.format(
                                "/tmp/flink-gpu-refjoin-%d-%d-%d",
                                args.refCount, args.dims, args.queryRows);
            }
            args.refData = args.data + "/refs";
            args.queryData = args.data + "/queries";
            return args;
        }
    }

    private ReferenceJoinBenchmark() {}
}
