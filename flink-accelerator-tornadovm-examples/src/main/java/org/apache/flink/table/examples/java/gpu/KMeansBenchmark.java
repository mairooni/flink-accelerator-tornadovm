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
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.gpu.operator.KMeansEngine;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.StringJoiner;

/**
 * One k-means iteration as ordinary batch SQL, with and without GPU offload.
 *
 * <h2>Why k-means</h2>
 *
 * <p>Every operator measured so far that moves rows rather than computing over them came out
 * between 1.0x and 2.2x, because the interconnect decided and not the device. The two that did
 * better -- the haversine {@code Calc} at 11.9x and the Gram matrix at 4.6x -- share a shape: a lot
 * of arithmetic per row, and an output far smaller than the input. K-means has that shape and is
 * something people actually run.
 *
 * <p>It also splits along the seam this integration exists for. Assigning each point to its nearest
 * centroid is {@code n * k * d} arithmetic and per-row, which is what TornadoVM compiles well.
 * Recomputing the centroids is a grouped aggregation, which needs cross-row cooperation and is what
 * cuDF is bound for. Both halves are in one query and the planner fuses them into one node.
 *
 * <h2>The two stages</h2>
 *
 * <ul>
 *   <li>{@code --stage assign} is the arithmetic alone: the squared distance to each of {@code k}
 *       centroids and the smallest of them, aggregated to one row so the drain measures nothing.
 *       This is the half that decides whether the whole thing pays.
 *   <li>{@code --stage iterate} is a whole iteration: assign each point to a cluster, then group by
 *       the cluster to get the member count and the coordinate sums that form the next centroids.
 *       The output is {@code k} rows.
 * </ul>
 *
 * <h2>Why the argmin has two spellings</h2>
 *
 * <p>An iteration needs the <em>index</em> of the nearest centroid, not the distance to it. The
 * natural SQL for that is a {@code CASE} chain, and the accelerator IR has no conditional, so the
 * {@code Calc} is refused and the whole query runs on the CPU. {@code --argmin sign} writes the
 * same thing arithmetically instead, using {@code SIGN(d - m)}, which is zero for the nearest
 * centroid and one for every other, so the index falls out of a weighted sum.
 *
 * <p>Both are offered because the difference between them is the measurement: it is what a missing
 * conditional costs, in a query nobody would otherwise write twice. {@code sign} is not the
 * spelling to put in front of a user -- a query author should be able to write {@code CASE} -- so
 * if the gap is worth closing this is the number that says so.
 *
 * <h2>Usage</h2>
 *
 * <pre>
 *   # once: write the input table
 *   flink run KMeansBenchmark.jar --generate --rows 2000000 --dims 16 --data /tmp/kmeans
 *
 *   # then: measure, one arm at a time
 *   flink run KMeansBenchmark.jar --data /tmp/kmeans --dims 16 --clusters 32 --offload false
 *   flink run KMeansBenchmark.jar --data /tmp/kmeans --dims 16 --clusters 32 --offload true
 * </pre>
 */
public final class KMeansBenchmark {

    private static final Logger LOG = LoggerFactory.getLogger(KMeansBenchmark.class);

    /** The range datagen fills, and therefore the range the centroids are drawn from. */
    private static final double SPREAD = 3.0;

    private KMeansBenchmark() {}

    public static void main(String[] args) throws Exception {
        Args parsed = Args.parse(args);

        if (parsed.generate) {
            generate(parsed);
            return;
        }
        if (!Files.isDirectory(Paths.get(parsed.data))) {
            System.out.printf("%s does not exist yet%n", parsed.data);
            generate(parsed);
        }

        double[][] centroids = centroids(parsed);
        System.out.printf(
                "data=%s  rows=%,d  dims=%d  clusters=%d  stage=%s  argmin=%s  parallelism=%d"
                        + "  offload=%s  fused=%s  runs=%d%n",
                parsed.data,
                parsed.rows,
                parsed.dims,
                parsed.clusters,
                parsed.stage,
                parsed.argmin,
                parsed.parallelism,
                parsed.offload,
                parsed.fuseAggregate,
                parsed.runs);
        System.out.printf(
                "expressions in the projection: about %d%n", 3 * parsed.clusters * parsed.dims);

        if ("train".equals(parsed.stage)) {
            double[][] firstCentroids = null;
            for (int run = 1; run <= parsed.runs; run++) {
                long start = System.nanoTime();
                double[][] result =
                        parsed.device
                                ? deviceTrain(parsed, centroids)
                                : sqlTrain(parsed, centroids);
                double millis = (System.nanoTime() - start) / 1e6;
                System.out.printf(
                        "run %2d  %10.0f ms  centroidNorm=%.6e%n", run, millis, norm(result));
                if (firstCentroids == null) {
                    firstCentroids = result;
                } else {
                    agreeCentroids(firstCentroids, result);
                }
            }
            return;
        }

        List<Row> first = null;
        for (int run = 1; run <= parsed.runs; run++) {
            long start = System.nanoTime();
            List<Row> rows = run(parsed, centroids);
            double millis = (System.nanoTime() - start) / 1e6;
            System.out.printf(
                    "run %2d  %10.0f ms  rows=%d  %s%n", run, millis, rows.size(), digest(rows));
            if (first == null) {
                first = rows;
            } else {
                agree(first, rows);
            }
        }
    }

    /**
     * The query, as a user would write it.
     *
     * <p>Squared distance rather than Euclidean: the nearest centroid is the same either way, and a
     * square root per centroid per row would be arithmetic that k-means does not need and that
     * would flatter the device.
     */
    private static List<Row> run(Args args, double[][] centroids) throws Exception {
        final StreamTableEnvironment env = environment(args);
        env.executeSql(sourceTable(args));

        String sql =
                "assign".equals(args.stage)
                        ? assignSql(args, centroids)
                        : iterateSql(args, centroids);
        if (args.printSql) {
            System.out.println(sql);
        }

        List<Row> rows = new ArrayList<>();
        try (CloseableIterator<Row> it = env.executeSql(sql).collect()) {
            while (it.hasNext()) {
                rows.add(it.next());
            }
        }
        return rows;
    }

    /**
     * Lloyd's algorithm as a user would write it without a device: one query per iteration.
     *
     * <p>The centroids go in as literals because there is nowhere else to put them -- a Flink SQL
     * plan is fixed, so new centroids mean a new query. That also means the points are read again
     * on every pass, which is the honest cost of doing this in SQL and is exactly what the device
     * arm does not pay.
     */
    private static double[][] sqlTrain(Args args, double[][] initial) throws Exception {
        final StreamTableEnvironment env = environment(args);
        env.executeSql(sourceTable(args));

        double[][] centroids = new double[args.clusters][args.dims];
        for (int j = 0; j < args.clusters; j++) {
            System.arraycopy(initial[j], 0, centroids[j], 0, args.dims);
        }

        for (int iteration = 0; iteration < args.iterations; iteration++) {
            try (CloseableIterator<Row> it =
                    env.executeSql(iterateSql(args, centroids)).collect()) {
                while (it.hasNext()) {
                    Row row = it.next();
                    int cid = ((Number) row.getField(0)).intValue();
                    double members = ((Number) row.getField(1)).doubleValue();
                    if (members > 0.0 && cid >= 0 && cid < args.clusters) {
                        for (int c = 0; c < args.dims; c++) {
                            centroids[cid][c] =
                                    ((Number) row.getField(2 + c)).doubleValue() / members;
                        }
                    }
                }
            }
        }
        return centroids;
    }

    /** The same algorithm with the points resident on the device for every pass. */
    private static double[][] deviceTrain(Args args, double[][] initial) throws Exception {
        final StreamTableEnvironment env = environment(args);
        env.executeSql(sourceTable(args));

        DataStream<double[]> trained =
                env.toDataStream(env.from("Points"))
                        .transform(
                                "kmeans",
                                PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO,
                                new KMeansOperator(
                                        args.rows,
                                        args.dims,
                                        args.clusters,
                                        args.iterations,
                                        flatten(initial)));

        try (CloseableIterator<double[]> it = trained.executeAndCollect()) {
            return unflatten(it.next(), args.clusters, args.dims);
        }
    }

    private static double[] flatten(double[][] centroids) {
        int dims = centroids[0].length;
        double[] flat = new double[centroids.length * dims];
        for (int j = 0; j < centroids.length; j++) {
            System.arraycopy(centroids[j], 0, flat, j * dims, dims);
        }
        return flat;
    }

    private static double[][] unflatten(double[] flat, int clusters, int dims) {
        double[][] centroids = new double[clusters][dims];
        for (int j = 0; j < clusters; j++) {
            System.arraycopy(flat, j * dims, centroids[j], 0, dims);
        }
        return centroids;
    }

    /** One number over every centroid, which moves if any of them does. */
    private static double norm(double[][] centroids) {
        double total = 0.0;
        for (double[] centroid : centroids) {
            for (double value : centroid) {
                total += value * value;
            }
        }
        return Math.sqrt(total);
    }

    /**
     * Checks two runs converged to the same centroids.
     *
     * <p>Looser than the row comparison below, because the device sums in a different order and a
     * centroid is a quotient of two sums. An assignment that differed would move a centroid far
     * more than reassociation can.
     */
    private static void agreeCentroids(double[][] first, double[][] second) {
        double worst = 0.0;
        for (int j = 0; j < first.length; j++) {
            for (int c = 0; c < first[j].length; c++) {
                double scale = Math.max(1.0, Math.abs(first[j][c]));
                worst = Math.max(worst, Math.abs(first[j][c] - second[j][c]) / scale);
            }
        }
        if (worst > 1e-6) {
            throw new IllegalStateException(
                    "runs disagree on the centroids by " + worst + ", beyond reassociation");
        }
    }

    /** Holds the engine, stages every row into it, and trains once the input ends. */
    private static final class KMeansOperator extends AbstractStreamOperator<double[]>
            implements OneInputStreamOperator<Row, double[]>, BoundedOneInput {

        private static final long serialVersionUID = 1L;

        private final int rows;
        private final int dims;
        private final int clusters;
        private final int iterations;
        private final double[] initial;
        private transient KMeansEngine engine;
        private transient int staged;

        private KMeansOperator(int rows, int dims, int clusters, int iterations, double[] initial) {
            this.rows = rows;
            this.dims = dims;
            this.clusters = clusters;
            this.iterations = iterations;
            this.initial = initial;
        }

        @Override
        public void open() throws Exception {
            super.open();
            engine = new KMeansEngine(rows, dims, clusters);
            for (int j = 0; j < clusters; j++) {
                for (int c = 0; c < dims; c++) {
                    engine.setCentroid(j, c, initial[j * dims + c]);
                }
            }
        }

        @Override
        public void processElement(StreamRecord<Row> element) {
            Row row = element.getValue();
            for (int c = 0; c < dims; c++) {
                engine.setPoint(staged, c, ((Number) row.getField(c)).doubleValue());
            }
            staged++;
        }

        @Override
        public void endInput() {
            engine.open();
            engine.train(iterations);
            LOG.info(
                    "kmeans: {} rows, {} dims, {} clusters, {} iterations, loop {} ms",
                    staged,
                    dims,
                    clusters,
                    iterations,
                    String.format("%.1f", engine.trainMillis()));
            double[] flat = new double[clusters * dims];
            for (int j = 0; j < clusters; j++) {
                for (int c = 0; c < dims; c++) {
                    flat[j * dims + c] = engine.centroid(j, c);
                }
            }
            output.collect(new StreamRecord<>(flat));
        }

        @Override
        public void close() throws Exception {
            if (engine != null) {
                engine.close();
            }
            super.close();
        }
    }

    /** The arithmetic alone, aggregated to one row so that nothing is measured but the compute. */
    private static String assignSql(Args args, double[][] centroids) {
        return String.format(
                "SELECT COUNT(*) AS points, SUM(nearest) AS total%n"
                        + "FROM (SELECT %s AS nearest FROM Points)",
                leastOfDistances(args, centroids));
    }

    /** A whole iteration: assign, then group by the assignment to form the next centroids. */
    private static String iterateSql(Args args, double[][] centroids) {
        StringJoiner sums = new StringJoiner(", ");
        StringJoiner carried = new StringJoiner(", ");
        for (int c = 0; c < args.dims; c++) {
            sums.add(String.format("SUM(c%d) AS s%d", c, c));
            carried.add(String.format("c%d", c));
        }
        return String.format(
                "SELECT cid, COUNT(*) AS members, %s%n"
                        + "FROM (SELECT %s AS cid, %s FROM Points)%n"
                        + "GROUP BY cid",
                sums, argmin(args, centroids), carried);
    }

    /** {@code LEAST(d0, d1, ...)} over the squared distance to every centroid. */
    private static String leastOfDistances(Args args, double[][] centroids) {
        if (args.clusters == 1) {
            return distance(centroids[0]);
        }
        StringJoiner least = new StringJoiner(",\n       ", "LEAST(", ")");
        for (double[] centroid : centroids) {
            least.add(distance(centroid));
        }
        return least.toString();
    }

    /**
     * {@code (c0 - v0) * (c0 - v0) + ...}, which is {@code d} multiplies and {@code d - 1} adds.
     */
    private static String distance(double[] centroid) {
        StringJoiner terms = new StringJoiner(" + ");
        for (int c = 0; c < centroid.length; c++) {
            terms.add(
                    String.format(
                            "(c%d - %s) * (c%d - %s)",
                            c, literal(centroid[c]), c, literal(centroid[c])));
        }
        return terms.toString();
    }

    /**
     * The index of the nearest centroid, in one of two spellings.
     *
     * <p>{@code case} is what a query author would write and what the IR cannot express. {@code
     * sign} says the same thing with arithmetic: {@code SIGN(dj - m)} is 0 for the nearest centroid
     * and 1 for every other, since no distance is below the minimum, so {@code sum(j * (1 - SIGN(dj
     * - m)))} is the index. Ties would count twice; with continuous coordinates they do not occur,
     * and k-means does not define which of two equidistant centroids wins anyway.
     */
    private static String argmin(Args args, double[][] centroids) {
        String least = leastOfDistances(args, centroids);
        if ("case".equals(args.argmin)) {
            StringBuilder sb = new StringBuilder("CASE");
            for (int j = 0; j < args.clusters - 1; j++) {
                sb.append(
                        String.format(
                                "%n       WHEN %s <= %s THEN %d",
                                distance(centroids[j]), least, j));
            }
            return sb.append(String.format("%n       ELSE %d END", args.clusters - 1)).toString();
        }
        StringJoiner terms = new StringJoiner(" + ");
        for (int j = 1; j < args.clusters; j++) {
            terms.add(
                    String.format("%d * (1 - SIGN(%s - (%s)))", j, distance(centroids[j]), least));
        }
        return args.clusters == 1 ? "0" : "CAST(" + terms + " AS INT)";
    }

    /** Full precision, so the two arms are given the same numbers and not the same rounding. */
    private static String literal(double value) {
        return String.format("CAST(%s AS DOUBLE)", Double.toString(value));
    }

    /** A deterministic spread, so every run and every arm compares against the same centroids. */
    private static double[][] centroids(Args args) {
        Random random = new Random(42);
        double[][] centroids = new double[args.clusters][args.dims];
        for (int j = 0; j < args.clusters; j++) {
            for (int c = 0; c < args.dims; c++) {
                centroids[j][c] = (random.nextDouble() * 2.0 - 1.0) * SPREAD;
            }
        }
        return centroids;
    }

    private static void generate(Args args) throws Exception {
        final StreamTableEnvironment env = environment(args);

        StringJoiner schema = new StringJoiner(",\n  ");
        StringJoiner options = new StringJoiner(",\n  ");
        for (int c = 0; c < args.dims; c++) {
            schema.add(String.format("c%d DOUBLE NOT NULL", c));
            options.add(String.format("'fields.c%d.min' = '%s'", c, -SPREAD));
            options.add(String.format("'fields.c%d.max' = '%s'", c, SPREAD));
        }
        env.executeSql(
                String.format(
                        "CREATE TABLE Source (%n  %s%n) WITH (%n  'connector' = 'datagen',%n"
                                + "  'number-of-rows' = '%d',%n  %s%n)",
                        schema, args.rows, options));
        env.executeSql(sourceTable(args));

        System.out.printf("writing %,d rows x %d dims to %s%n", args.rows, args.dims, args.data);
        long start = System.nanoTime();
        env.executeSql("INSERT INTO Points SELECT * FROM Source").await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    private static String sourceTable(Args args) {
        StringJoiner schema = new StringJoiner(",\n  ");
        for (int c = 0; c < args.dims; c++) {
            schema.add(String.format("c%d DOUBLE NOT NULL", c));
        }
        return String.format(
                "CREATE TABLE Points (%n  %s%n) WITH (%n  'connector' = 'filesystem',%n"
                        + "  'path' = '%s',%n  'format' = '%s'%n)",
                schema, args.data, args.format);
    }

    private static StreamTableEnvironment environment(Args args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setRuntimeMode(RuntimeExecutionMode.BATCH);
        if (args.parallelism > 0) {
            env.setParallelism(args.parallelism);
        }
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        if (args.offload) {
            tEnv.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
            tEnv.getConfig()
                    .getConfiguration()
                    .setString(
                            "table.exec.accelerator.fuse-aggregate",
                            Boolean.toString(args.fuseAggregate));
        }
        return tEnv;
    }

    /** One number per run, which moves if anything about the result does. */
    private static String digest(List<Row> rows) {
        double total = 0.0;
        for (Row row : rows) {
            for (int f = 0; f < row.getArity(); f++) {
                Object value = row.getField(f);
                if (value instanceof Number) {
                    total += ((Number) value).doubleValue();
                }
            }
        }
        return String.format("digest=%.6e", total);
    }

    /**
     * Checks two runs agree.
     *
     * <p>Not bit-for-bit: a parallel SUM combines partials in whatever order subtasks finish, and
     * across arms the device accumulates differently again. A cluster assignment, though, is an
     * integer and has to match exactly -- if a point lands in a different cluster the arms have
     * computed different distances, which is a wrong answer and not a rounding difference.
     */
    private static void agree(List<Row> first, List<Row> second) {
        if (first.size() != second.size()) {
            throw new IllegalStateException(
                    "runs produced " + first.size() + " and " + second.size() + " rows");
        }
        double worst = 0.0;
        for (int r = 0; r < first.size(); r++) {
            Row a = first.get(r);
            Row b = second.get(r);
            for (int f = 0; f < a.getArity(); f++) {
                Object left = a.getField(f);
                Object right = b.getField(f);
                if (left instanceof Integer || left instanceof Long) {
                    if (!left.equals(right)) {
                        throw new IllegalStateException(
                                "row "
                                        + r
                                        + " field "
                                        + f
                                        + ": "
                                        + left
                                        + " != "
                                        + right
                                        + ", which is an assignment or a count and cannot round");
                    }
                } else if (left instanceof Number) {
                    double x = ((Number) left).doubleValue();
                    double y = ((Number) right).doubleValue();
                    worst = Math.max(worst, Math.abs(x - y) / Math.max(1.0, Math.abs(x)));
                }
            }
        }
        if (worst > 1e-3) {
            throw new IllegalStateException(
                    "runs disagree by " + worst + ", far beyond floating-point reassociation");
        }
    }

    private static final class Args {
        private String data = "/tmp/flink-kmeans";
        private String format = "csv";
        private String stage = "assign";
        private String argmin = "sign";
        private int rows = 2_000_000;
        private int dims = 16;
        private int clusters = 32;
        private int parallelism = 1;
        private int runs = 6;
        private int iterations = 20;
        private boolean device;
        private boolean offload;
        private boolean fuseAggregate = true;
        private boolean generate;
        private boolean printSql;

        static Args parse(String[] argv) {
            Args args = new Args();
            for (int i = 0; i < argv.length; i++) {
                String flag = argv[i];
                if ("--data".equals(flag)) {
                    args.data = argv[++i];
                } else if ("--format".equals(flag)) {
                    args.format = argv[++i];
                } else if ("--stage".equals(flag)) {
                    args.stage = argv[++i];
                } else if ("--argmin".equals(flag)) {
                    args.argmin = argv[++i];
                } else if ("--rows".equals(flag)) {
                    args.rows = Integer.parseInt(argv[++i]);
                } else if ("--dims".equals(flag)) {
                    args.dims = Integer.parseInt(argv[++i]);
                } else if ("--clusters".equals(flag)) {
                    args.clusters = Integer.parseInt(argv[++i]);
                } else if ("--parallelism".equals(flag)) {
                    args.parallelism = Integer.parseInt(argv[++i]);
                } else if ("--runs".equals(flag)) {
                    args.runs = Integer.parseInt(argv[++i]);
                } else if ("--iterations".equals(flag)) {
                    args.iterations = Integer.parseInt(argv[++i]);
                } else if ("--device".equals(flag)) {
                    args.device = Boolean.parseBoolean(argv[++i]);
                } else if ("--offload".equals(flag)) {
                    args.offload = Boolean.parseBoolean(argv[++i]);
                } else if ("--fuse-aggregate".equals(flag)) {
                    args.fuseAggregate = Boolean.parseBoolean(argv[++i]);
                } else if ("--generate".equals(flag)) {
                    args.generate = true;
                } else if ("--print-sql".equals(flag)) {
                    args.printSql = true;
                } else {
                    throw new IllegalArgumentException("unknown flag " + flag);
                }
            }
            if (!"assign".equals(args.stage)
                    && !"iterate".equals(args.stage)
                    && !"train".equals(args.stage)) {
                throw new IllegalArgumentException("--stage must be assign, iterate or train");
            }
            if (!"sign".equals(args.argmin) && !"case".equals(args.argmin)) {
                throw new IllegalArgumentException("--argmin must be sign or case");
            }
            return args;
        }
    }
}
