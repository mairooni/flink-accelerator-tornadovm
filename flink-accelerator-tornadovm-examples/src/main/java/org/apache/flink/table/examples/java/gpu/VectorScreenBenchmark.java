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

import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.gpu.metrics.BenchmarkRun;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * Candidate generation for vector search: keep the rows near any of a batch of probe points.
 *
 * <h2>Its relation to {@link KMeansBenchmark}, which came first</h2>
 *
 * <p>The arithmetic here is k-means' assignment step: the squared distance to each of several fixed
 * points and the smallest of them. That is deliberate — {@code KMeansBenchmark} established that
 * this shape is the one the device is good at, and repeating it means the two can be read against
 * each other instead of each standing alone.
 *
 * <p>What this adds is the one thing k-means has no use for: a {@code WHERE} on the computed
 * distance. An assignment step keeps every row, so the accelerated {@code Calc} is a projection and
 * the device compaction has nothing to compact. A screen keeps a few. That single difference is
 * what makes the fused path reachable, and it is not a k-means stage, which is why it is a separate
 * query rather than a third {@code --stage} on that one.
 *
 * <h2>Why this workload and not a more arithmetic-heavy one</h2>
 *
 * <p>The accelerated filter refuses, by default, any predicate containing a device transcendental,
 * because CUDA's math library is not required to agree with {@code java.lang.Math} and a
 * disagreement in the last ulp changes <em>which rows the query returns</em>. That refusal decides
 * what a showcase can be, and the reasoning is worth stating because it is not obvious:
 *
 * <ul>
 *   <li>the fused device compaction exists only when the filter and the projection are in one
 *       {@code Calc}, which happens only when the filter reads a value the projection computes — a
 *       filter on raw columns is pushed below the projection by the planner;
 *   <li>so a workload that exercises compaction <em>under the default configuration</em> must
 *       filter on a value that is expensive to compute and exactly computable.
 * </ul>
 *
 * <p>Squared distance is exactly that, and it is the standard formulation rather than a way around
 * the rule: nearest-neighbour code compares squared distances precisely so it does not pay for a
 * square root it does not need. Every operation here is {@code -}, {@code *}, {@code +}, {@code
 * LEAST} and a comparison, all of which are exact under IEEE 754, so the two arms are expected to
 * agree <em>bit for bit</em> and correctness is a strict equality rather than a tolerance.
 *
 * <h2>The workload</h2>
 *
 * <p>A table of {@code --dim}-dimensional vectors. A batch of {@code --queries} probe points. Keep
 * a row if its squared distance to the <em>nearest</em> probe is below {@code --radius} squared,
 * and report the survivors. This is the first stage of a search: a cheap screen that hands a small
 * candidate set to something expensive.
 *
 * <p>{@code --queries} is a real parameter of that stage — a batch of probes is screened in one
 * pass over the data precisely because the pass is what costs — and it is the dial that raises
 * arithmetic per byte without reading the input more than once. It is swept and reported, not fixed
 * at a flattering value.
 *
 * <p>Everything is ordinary SQL. No hint, no option and no syntax from which a device could be
 * inferred; the accelerated arm differs only in a deployment setting.
 *
 * <h2>What it reports</h2>
 *
 * <p>{@code COUNT(*)} and {@code SUM(id)} are exact integers over the survivors and settle
 * membership. {@code MIN}, {@code MAX} and {@code SUM} of the squared distance settle the values.
 * Here all five are expected to agree exactly; a disagreement in the last three while the first two
 * hold would mean the device contracted a multiply-add into an FMA, which is worth knowing and is
 * why the columns are reported separately rather than as one checksum.
 *
 * <h2>Usage</h2>
 *
 * <pre>
 *   flink run VectorScreenBenchmark.jar --generate --rows 10000000 --dim 8 --data /tmp/vectors
 *   flink run VectorScreenBenchmark.jar --data /tmp/vectors --queries 8 --radius 0.35 --gpu true
 * </pre>
 */
public final class VectorScreenBenchmark {

    public static void main(String[] args) throws Exception {
        final Args parsed = Args.parse(args);

        if (parsed.generate) {
            generate(parsed);
            return;
        }
        if (parsed.convertTo != null) {
            convert(parsed);
            return;
        }
        if (parsed.dump != null) {
            dump(parsed);
            return;
        }
        if (!Files.exists(Paths.get(parsed.data))) {
            System.out.printf("%s does not exist yet%n", parsed.data);
            return;
        }

        System.out.printf(
                "vector screen  dim=%d  queries=%d  radius=%s  gpu=%s  parallelism=%d%n",
                parsed.dim, parsed.queries, parsed.radius, parsed.gpu, parsed.parallelism);

        final BenchmarkRun timings = new BenchmarkRun("wall time", parsed.runs > 1 ? 1 : 0);
        Row first = null;
        for (int run = 1; run <= parsed.runs; run++) {
            final long start = System.nanoTime();
            final Row seen = query(parsed);
            final long elapsed = System.nanoTime() - start;
            timings.record(elapsed);
            System.out.printf("run %2d  %10.0f ms  %s%n", run, elapsed / 1e6, seen);
            if (first == null) {
                first = seen;
            } else {
                agreeOnMembership(first, seen);
            }
        }
        System.out.print(timings.summary());
    }

    /**
     * Two runs of one arm must screen the same rows. They need not produce the same total.
     *
     * <p>Measured, not assumed: ten repetitions of the <em>CPU</em> arm over an eight-split input
     * produced ten different values of {@code SUM(d2)} and one value of {@code COUNT(*)} and {@code
     * SUM(id)}. Splits are assigned to a subtask in whatever order they are handed out, so the
     * order of a floating-point sum varies between runs of the same binary on the same data. Any
     * check that demanded a reproducible total would therefore fail on the CPU, before a device was
     * involved at all — which is why membership is the assertion and the total is an observation.
     */
    private static void agreeOnMembership(Row first, Row later) {
        final Object[] firstExact = {first.getField(0), first.getField(1)};
        final Object[] laterExact = {later.getField(0), later.getField(1)};
        if (!Arrays.equals(firstExact, laterExact)) {
            throw new IllegalStateException(
                    "the same arm screened different rows on two runs:\n  "
                            + first
                            + "\n  "
                            + later);
        }
    }

    /**
     * The screen's threshold, typed the same way the distances are.
     *
     * <p>A bare {@code 0.09} is a {@code DECIMAL(3,2)}, so comparing a {@code DOUBLE} distance to
     * it is a coercion of exactly the kind the provider refuses. The {@code E0} makes it the DOUBLE
     * it was always meant to be.
     */
    /**
     * One double, written so SQL types it the way the caller means.
     *
     * <p>{@code Double.toString} switches to exponent form below 1e-3, and appending {@code E0} to
     * {@code 2.25E-6} produces {@code 2.25E-6E0}, which is not a number. That did not fail: the
     * query parsed, matched nothing and reported zero rows, which is the worst way for a benchmark
     * to be wrong. The plain-string expansion of the double is exact and never carries an exponent,
     * so appending the suffix is always valid.
     */
    private static String literal(double x) {
        if (!Boolean.getBoolean("vectorscreen.doubleLiterals")) {
            return Double.toString(x);
        }
        return new java.math.BigDecimal(x).toPlainString() + "E0";
    }

    private static String threshold(Args args) {
        return literal(args.radius * args.radius);
    }

    private static Row query(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        env.executeSql(vectors(args.data, args.format, args.dim));

        if (args.gpu) {
            env.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
            if (args.maxAccelParallelism > 0) {
                env.getConfig()
                        .getConfiguration()
                        .setString(
                                "table.exec.accelerator.max-parallelism",
                                Integer.toString(args.maxAccelParallelism));
            }
        }

        final String query =
                "SELECT COUNT(*) AS hits,\n"
                        + "       SUM(id) AS id_sum,\n"
                        + "       MIN(d2) AS min_d2,\n"
                        + "       MAX(d2) AS max_d2,\n"
                        + "       SUM(d2) AS total_d2\n"
                        + "FROM (\n"
                        + "  SELECT id, "
                        + nearestSquaredDistance(args.dim, args.queries)
                        + " AS d2\n"
                        + "  FROM Vectors\n"
                        + ")\n"
                        + "WHERE d2 < "
                        + threshold(args);

        if (args.explain) {
            System.out.println(env.explainSql(query));
        }

        try (CloseableIterator<Row> rows = env.sqlQuery(query).execute().collect()) {
            if (!rows.hasNext()) {
                throw new IllegalStateException("query returned no rows");
            }
            return rows.next();
        }
    }

    /**
     * Squared distance to the nearest of {@code queries} probe points.
     *
     * <p>Written out rather than expressed with an array type because the accelerated path works on
     * flat columns, which is also how a vector column is stored in every columnar format worth
     * measuring.
     */
    private static String nearestSquaredDistance(int dim, int queries) {
        if (queries == 1) {
            return squaredDistance(dim, 0);
        }
        final StringBuilder sb = new StringBuilder("LEAST(");
        for (int q = 0; q < queries; q++) {
            if (q > 0) {
                sb.append(",\n         ");
            }
            sb.append(squaredDistance(dim, q));
        }
        return sb.append(")").toString();
    }

    private static String squaredDistance(int dim, int query) {
        final StringBuilder sb = new StringBuilder("(");
        for (int i = 0; i < dim; i++) {
            if (i > 0) {
                sb.append(" + ");
            }
            final String delta = "(v" + i + " - " + probe(query, i) + ")";
            sb.append(delta).append(" * ").append(delta);
        }
        return sb.append(")").toString();
    }

    /**
     * A deterministic spread of probe points inside the unit cube.
     *
     * <p>Deterministic because two arms can only be compared if they screen against the same
     * probes, and a spread rather than a cluster because probes that sit on top of each other make
     * {@code LEAST} answer the same thing every time and stop measuring the batch at all.
     */
    private static String probe(int query, int coordinate) {
        // A low-discrepancy pair: the golden-ratio sequence in one index, an irrational rotation
        // in the other, so no two probes share a coordinate and none lands on a grid point.
        final double a = (query + 1) * 0.6180339887498949;
        final double b = (coordinate + 1) * 0.7548776662466927;
        double x = a + b;
        x -= Math.floor(x);
        // A bare decimal in SQL is a DECIMAL literal, not a DOUBLE: the planner types
        // 0.3729116549965876 as DECIMAL(17,16), so every subtraction here is DOUBLE minus DECIMAL
        // and the coercion is part of the expression's semantics. --double-literals casts it, so
        // the two spellings can be compared and the coercion can be ruled in or out.
        // SQL's approximate-numeric literal: 0.5E0 is a DOUBLE, 0.5 is a DECIMAL. Same value,
        // different type, and the type is what decides whether Flink evaluates the expression as
        // double arithmetic. A CAST would do it too but adds nodes the cost model then counts as
        // work the kernel never does, which is how it came to decline a query it had just been
        // measured at 1.99x on.
        return literal(x);
    }

    private static String vectors(String path, String format, int dim) {
        final StringBuilder sb = new StringBuilder("CREATE TABLE Vectors (\n  id INT NOT NULL");
        for (int i = 0; i < dim; i++) {
            sb.append(",\n  v").append(i).append(" DOUBLE NOT NULL");
        }
        return sb.append("\n) WITH (\n")
                .append("  'connector' = 'filesystem',\n")
                .append("  'path' = '")
                .append(path)
                .append("',\n")
                .append("  'format' = '")
                .append(format)
                .append("'\n)")
                .toString();
    }

    private static void generate(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        // Nullable here, NOT NULL in the table it is written to: the same shape the other
        // generators use. datagen produces no nulls without a null-rate, and the accelerated path
        // needs the stored table to declare NOT NULL, which is the one thing review §0.1 allows a
        // query author to have to write.
        final StringBuilder source = new StringBuilder("CREATE TABLE Source (\n  id INT");
        for (int i = 0; i < args.dim; i++) {
            source.append(",\n  v").append(i).append(" DOUBLE");
        }
        source.append("\n) WITH (\n  'connector' = 'datagen',\n")
                .append("  'number-of-rows' = '")
                .append(args.rows)
                .append("',\n")
                .append("  'fields.id.kind' = 'sequence',\n")
                .append("  'fields.id.start' = '0',\n")
                .append("  'fields.id.end' = '")
                .append(args.rows - 1)
                .append("'");
        for (int i = 0; i < args.dim; i++) {
            source.append(",\n  'fields.v")
                    .append(i)
                    .append(".min' = '0.0', 'fields.v")
                    .append(i)
                    .append(".max' = '1.0'");
        }
        env.executeSql(source.append("\n)").toString());
        env.executeSql(vectors(args.data, args.format, args.dim));

        System.out.printf(
                "writing %,d rows of dimension %d to %s%n", args.rows, args.dim, args.data);
        final long start = System.nanoTime();
        env.executeSql("INSERT INTO Vectors SELECT * FROM Source").await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    /** Rewrites one dataset into another format or file count, so neither confounds the other. */
    private static void convert(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        env.executeSql(vectors(args.data, args.format, args.dim));
        String target =
                vectors(args.convertTo, args.convertFormat, args.dim).replace("Vectors", "Copy");
        if (args.convertFiles > 1) {
            target =
                    target.replace(
                            "  'format' = '" + args.convertFormat + "'",
                            "  'format' = '"
                                    + args.convertFormat
                                    + "',\n  'sink.parallelism' = '"
                                    + args.convertFiles
                                    + "'");
        }
        env.executeSql(target);
        final String where = args.subset > 0 ? " WHERE id < " + args.subset : "";
        System.out.printf("converting %s -> %s%s%n", args.data, args.convertTo, where);
        final long start = System.nanoTime();
        env.executeSql("INSERT INTO Copy SELECT * FROM Vectors" + where).await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    /**
     * Writes the surviving rows themselves, so membership can be checked rather than inferred.
     *
     * <p>Aggregates agreeing is not the same fact as the same rows having been selected, and the
     * difference has already been wrong once in this project.
     */
    private static void dump(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        env.executeSql(vectors(args.data, args.format, args.dim));
        if (args.gpu) {
            env.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
            if (args.maxAccelParallelism > 0) {
                env.getConfig()
                        .getConfiguration()
                        .setString(
                                "table.exec.accelerator.max-parallelism",
                                Integer.toString(args.maxAccelParallelism));
            }
        }
        env.executeSql(
                "CREATE TABLE Selected (\n  id INT NOT NULL,\n  d2 DOUBLE NOT NULL\n) WITH (\n"
                        + "  'connector' = 'filesystem',\n  'path' = '"
                        + args.dump
                        + "',\n  'format' = 'csv'\n)");
        System.out.printf("writing selected rows to %s%n", args.dump);
        final long start = System.nanoTime();
        env.executeSql(
                        "INSERT INTO Selected SELECT id, d2 FROM (\n  SELECT id, "
                                + nearestSquaredDistance(args.dim, args.queries)
                                + " AS d2 FROM Vectors\n) WHERE d2 < "
                                + threshold(args))
                .await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    private static TableEnvironment batchEnvironment(Args args) {
        final TableEnvironment env =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        if (args.parallelism > 0) {
            env.getConfig()
                    .getConfiguration()
                    .setString("parallelism.default", Integer.toString(args.parallelism));
        }
        return env;
    }

    /** Flags, parsed rather than positional, because a benchmark is run from a shell loop. */
    private static final class Args {
        private boolean generate;
        private String data = "/tmp/vectors";
        private String format = "csv";
        private long rows = 2_000_000L;
        private int dim = 8;
        private int queries = 8;
        private double radius = 0.35;
        private int parallelism = 0;
        private int runs = 1;
        private boolean gpu;
        private boolean explain;
        private int maxAccelParallelism;
        private String convertTo;
        private String convertFormat = "parquet";
        private int convertFiles = 1;
        private long subset;
        private String dump;

        static Args parse(String[] argv) {
            final Args args = new Args();
            for (int i = 0; i < argv.length; i++) {
                final String flag = argv[i];
                switch (flag) {
                    case "--generate":
                        args.generate = true;
                        break;
                    case "--data":
                        args.data = argv[++i];
                        break;
                    case "--format":
                        args.format = argv[++i];
                        break;
                    case "--rows":
                        args.rows = Long.parseLong(argv[++i]);
                        break;
                    case "--dim":
                        args.dim = Integer.parseInt(argv[++i]);
                        break;
                    case "--queries":
                        args.queries = Integer.parseInt(argv[++i]);
                        break;
                    case "--radius":
                        args.radius = Double.parseDouble(argv[++i]);
                        break;
                    case "--parallelism":
                        args.parallelism = Integer.parseInt(argv[++i]);
                        break;
                    case "--runs":
                        args.runs = Integer.parseInt(argv[++i]);
                        break;
                    case "--gpu":
                        args.gpu = Boolean.parseBoolean(argv[++i]);
                        break;
                    case "--explain":
                        args.explain = true;
                        break;
                    case "--max-accel-parallelism":
                        args.maxAccelParallelism = Integer.parseInt(argv[++i]);
                        break;
                    case "--convert-to":
                        args.convertTo = argv[++i];
                        break;
                    case "--convert-format":
                        args.convertFormat = argv[++i];
                        break;
                    case "--convert-files":
                        args.convertFiles = Integer.parseInt(argv[++i]);
                        break;
                    case "--subset":
                        args.subset = Long.parseLong(argv[++i]);
                        break;
                    case "--dump":
                        args.dump = argv[++i];
                        break;
                    default:
                        throw new IllegalArgumentException("unknown flag: " + flag);
                }
            }
            return args;
        }
    }

    private VectorScreenBenchmark() {}
}
