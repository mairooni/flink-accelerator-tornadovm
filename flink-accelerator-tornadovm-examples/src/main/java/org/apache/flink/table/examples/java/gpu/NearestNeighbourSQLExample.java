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
import org.apache.flink.table.api.TableResult;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Nearest neighbour over wide vectors, as ordinary batch SQL.
 *
 * <pre>
 *   SELECT q.qid, MAX(q.f0*p.f0 + ... + q.f{d-1}*p.f{d-1}) AS best
 *   FROM Queries q, Corpus p
 *   GROUP BY q.qid
 * </pre>
 *
 * <p>For every query vector, its highest inner product against any vector in the corpus. Retrieval,
 * deduplication, entity resolution and recommendation are all this query.
 *
 * <h2>Why this query and not one of the others here</h2>
 *
 * <p>This project has measured four libraries through SQL and found every one of them worth about
 * 1.00x against a kernel that does the same job. That result is real and it is also a result about
 * one regime, which every query measured so far happened to share: the library's own work was a
 * rounding error in a job dominated by reading the input. A GPU does floating point roughly three
 * thousand times faster than a source delivers bytes, so a query has to do thousands of operations
 * per byte read before what the device computes outweighs what it reads — and every shape tried
 * until now does tens.
 *
 * <p>A Gram matrix does {@code d/2}, which is 16 at the widest {@code d} its SQL spelling can
 * reach. A grouped sum does one. This query does {@code nQ·nP / 2(nQ+nP)} — because every query
 * vector meets every corpus vector, so the arithmetic is quadratic in the rows while the bytes read
 * stay linear in them. At sixty thousand vectors a side that is tens of thousands of operations a
 * byte, which is the first time in this project that the device's own speed is the thing being
 * measured.
 *
 * <p>The second precondition is the width. A GEMM's arithmetic intensity is set by its reduction
 * dimension, which here is the number of paired columns, and a library's advantage over a kernel is
 * almost entirely a function of it: measured on this card, {@code cublasSgemm} is 1.57x a tiled
 * kernel at {@code d = 32} and 4.15x at {@code d = 512}, and 0.96x — losing — against a fused
 * kernel at 32 against 67.8x at 512. A dot product is <em>one</em> SQL expression of {@code d}
 * products, so {@code d} can be 256 without Calcite noticing; a Gram matrix needs {@code d(d+1)/2}
 * aggregate calls and planning passes two minutes at {@code d = 64}. That is why the shape that can
 * reach the regime is this one and not that one.
 *
 * <h2>What the query author writes</h2>
 *
 * <p>Standard SQL, and {@code NOT NULL} in the DDL. Nothing names a kernel, a library or a device.
 * The three arms below are deployment properties set on the TaskManager.
 *
 * <pre>
 *   flink run NearestNeighbourSQLExample.jar --generate --data /path
 *   flink run NearestNeighbourSQLExample.jar --data /path     # -Dflink.accelerator.similarity.contraction=cublas|tiled|fused
 * </pre>
 */
public final class NearestNeighbourSQLExample {

    private NearestNeighbourSQLExample() {}

    public static void main(String[] args) throws Exception {
        final Args parsed = Args.parse(args);

        if (parsed.generate) {
            generate(parsed);
            return;
        }

        final EnvironmentSettings settings =
                EnvironmentSettings.newInstance().inBatchMode().build();
        final TableEnvironment env = TableEnvironment.create(settings);

        // Environmental parameters, every one of them. Nothing here is a query-level knob.
        env.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
        env.getConfig()
                .getConfiguration()
                .setString("table.exec.accelerator.approximate-projections", "true");
        env.getConfig()
                .getConfiguration()
                .setString(
                        "table.exec.resource.default-parallelism",
                        Integer.toString(parsed.parallelism));
        // A hash aggregate rather than a sort one, and in two phases. The region substitutes the
        // local half and Flink's own Final combines what the subtasks produced, which is what
        // keeps the answer right at any parallelism.
        env.getConfig().getConfiguration().setString("table.exec.disabled-operators", "SortAgg");
        env.getConfig()
                .getConfiguration()
                .setString("table.optimizer.agg-phase-strategy", "TWO_PHASE");
        env.getConfig()
                .getConfiguration()
                .setString("table.optimizer.multiple-input-enabled", "false");

        env.executeSql(ddl("Queries", parsed.dim, parsed.probeDir(), true));
        env.executeSql(ddl("Corpus", parsed.dim, parsed.corpusDir(), false));

        final String sql =
                "SELECT q.qid, MAX("
                        + dot(parsed.dim)
                        + ") AS best\n"
                        + "FROM Queries q, Corpus p\n"
                        + "GROUP BY q.qid";

        if (parsed.explain) {
            System.out.println(env.explainSql(sql));
        }

        System.out.printf(
                "nearest neighbour: dim=%d probe=%,d in %d files, corpus=%,d, parallelism=%d%n",
                parsed.dim, parsed.probeRows, parsed.files, parsed.corpusRows, parsed.parallelism);
        System.out.printf(
                "pairs=%,d  multiply-adds=%,d%n",
                (long) parsed.probeRows * parsed.corpusRows,
                (long) parsed.probeRows * parsed.corpusRows * parsed.dim);

        final long started = System.nanoTime();
        final TableResult result = env.executeSql(sql);
        // A digest rather than the rows: the answer is one row per query vector, and what has to
        // match across the arms is every one of them, not the first twenty.
        long rows = 0;
        double sum = 0.0;
        double max = Double.NEGATIVE_INFINITY;
        long keys = 0;
        try (CloseableIterator<Row> it = result.collect()) {
            while (it.hasNext()) {
                final Row row = it.next();
                final int qid = (Integer) row.getField(0);
                final float best = (Float) row.getField(1);
                rows++;
                keys += qid;
                sum += best;
                max = Math.max(max, best);
            }
        }
        final double seconds = (System.nanoTime() - started) / 1e9;

        System.out.printf(
                "rows=%d  keyCheck=%d  sum=%.6e  max=%.6e  query=%.2f s%n",
                rows, keys, sum, max, seconds);
    }

    /** The {@code d}-term inner product, which is one expression however wide it is. */
    private static String dot(int d) {
        final StringBuilder sql = new StringBuilder();
        for (int i = 0; i < d; i++) {
            if (i > 0) {
                sql.append(" + ");
            }
            sql.append("q.f").append(i).append(" * p.f").append(i);
        }
        return sql.toString();
    }

    private static String ddl(String name, int d, Path path, boolean withKey) {
        final StringBuilder sql = new StringBuilder("CREATE TABLE ").append(name).append(" (\n");
        if (withKey) {
            sql.append("  qid INT NOT NULL,\n");
        }
        for (int i = 0; i < d; i++) {
            sql.append("  f").append(i).append(" FLOAT NOT NULL");
            sql.append(i == d - 1 ? "\n" : ",\n");
        }
        sql.append(") WITH (\n  'connector' = 'filesystem',\n  'path' = '")
                .append(path.toAbsolutePath())
                .append("',\n  'format' = 'parquet'\n)");
        return sql.toString();
    }

    // ---------------------------------------------------------------------------------------
    // Generation. CSV first and Parquet by conversion, because Flink's own writer is the only
    // one on this classpath -- and one chunk at a time at parallelism 1, because the region
    // allocates its score matrix from the probe file's row count and a plan is rebuilt whenever
    // that changes. Equal files are worth the extra jobs.
    // ---------------------------------------------------------------------------------------

    private static void generate(Args args) throws Exception {
        final Path root = Paths.get(args.data);
        deleteTree(root);
        Files.createDirectories(root);

        final EnvironmentSettings settings =
                EnvironmentSettings.newInstance().inBatchMode().build();
        final TableEnvironment env = TableEnvironment.create(settings);
        env.getConfig()
                .getConfiguration()
                .setString("table.exec.resource.default-parallelism", "1");

        final Path probeDir = args.probeDir();
        final Path corpusDir = args.corpusDir();
        Files.createDirectories(probeDir);
        Files.createDirectories(corpusDir);

        final int perFile = args.probeRows / args.files;
        final Random random = new Random(42);
        int nextId = 0;
        for (int f = 0; f < args.files; f++) {
            final Path csv = root.resolve("probe-csv-" + f);
            nextId = writeCsv(csv, perFile, args.dim, nextId, random);
            convert(env, "P" + f, csv, probeDir, args.dim, true, f);
            deleteTree(csv);
        }
        final Path csv = root.resolve("corpus-csv");
        writeCsv(csv, args.corpusRows, args.dim, -1, random);
        convert(env, "C", csv, corpusDir, args.dim, false, 0);
        deleteTree(csv);

        System.out.printf(
                "generated %,d probe rows in %d files of %,d and %,d corpus rows, dim=%d, at %s%n",
                args.probeRows, args.files, perFile, args.corpusRows, args.dim, root);
    }

    /**
     * @return the next unused identifier, or the one it was given when there is no key column
     */
    private static int writeCsv(Path dir, int rows, int d, int firstId, Random random)
            throws IOException {
        Files.createDirectories(dir);
        int id = firstId;
        try (BufferedWriter out = Files.newBufferedWriter(dir.resolve("data.csv"))) {
            final StringBuilder line = new StringBuilder(d * 12);
            for (int r = 0; r < rows; r++) {
                line.setLength(0);
                if (firstId >= 0) {
                    line.append(id++).append(',');
                }
                for (int i = 0; i < d; i++) {
                    if (i > 0) {
                        line.append(',');
                    }
                    // Unit-ish vectors, so an inner product stays in a range where FP32 is exact
                    // enough for the arms to be compared digit for digit.
                    line.append(String.format("%.6f", random.nextFloat() - 0.5f));
                }
                out.write(line.toString());
                out.newLine();
            }
        }
        return id;
    }

    private static void convert(
            TableEnvironment env,
            String tag,
            Path csv,
            Path into,
            int d,
            boolean withKey,
            int index)
            throws Exception {
        final Path staging = csv.resolveSibling(csv.getFileName() + "-parquet");
        env.executeSql(ddlFor(tag + "Csv", d, csv, withKey, "csv"));
        env.executeSql(ddlFor(tag + "Pq", d, staging, withKey, "parquet"));
        env.executeSql("INSERT INTO " + tag + "Pq SELECT * FROM " + tag + "Csv").await();
        // One file, because the insert ran at parallelism 1. Move it where the table expects it.
        try (Stream<Path> files = Files.list(staging)) {
            final List<Path> written = new ArrayList<>();
            files.filter(p -> !p.getFileName().toString().startsWith(".")).forEach(written::add);
            if (written.size() != 1) {
                throw new IllegalStateException(
                        "expected one parquet file from a parallelism-1 insert, got "
                                + written.size());
            }
            Files.move(
                    written.get(0),
                    into.resolve(String.format("part-%05d.parquet", index)),
                    StandardCopyOption.REPLACE_EXISTING);
        }
        deleteTree(staging);
    }

    private static String ddlFor(String name, int d, Path path, boolean withKey, String format) {
        final StringBuilder sql = new StringBuilder("CREATE TABLE ").append(name).append(" (\n");
        if (withKey) {
            sql.append("  qid INT NOT NULL,\n");
        }
        for (int i = 0; i < d; i++) {
            sql.append("  f").append(i).append(" FLOAT NOT NULL");
            sql.append(i == d - 1 ? "\n" : ",\n");
        }
        sql.append(") WITH (\n  'connector' = 'filesystem',\n  'path' = '")
                .append(path.toAbsolutePath())
                .append("',\n  'format' = '")
                .append(format)
                .append("'\n)");
        return sql.toString();
    }

    private static void deleteTree(Path path) throws IOException {
        if (!Files.exists(path)) {
            return;
        }
        try (Stream<Path> walk = Files.walk(path)) {
            walk.sorted(Comparator.reverseOrder())
                    .forEach(
                            p -> {
                                try {
                                    Files.delete(p);
                                } catch (IOException e) {
                                    throw new RuntimeException(e);
                                }
                            });
        }
    }

    private static final class Args {
        private boolean generate;
        private boolean explain;
        private String data = BenchData.resolve("nn").toString();
        private int dim = 256;
        private int probeRows = 65_536;
        private int files = 32;
        private int corpusRows = 131_072;
        private int parallelism = 1;

        Path probeDir() {
            return Paths.get(data, "queries");
        }

        Path corpusDir() {
            return Paths.get(data, "corpus");
        }

        static Args parse(String[] argv) {
            final Args args = new Args();
            for (int i = 0; i < argv.length; i++) {
                switch (argv[i]) {
                    case "--generate":
                        args.generate = true;
                        break;
                    case "--explain":
                        args.explain = true;
                        break;
                    case "--data":
                        args.data = argv[++i];
                        break;
                    case "--dim":
                        args.dim = Integer.parseInt(argv[++i]);
                        break;
                    case "--probe-rows":
                        args.probeRows = Integer.parseInt(argv[++i]);
                        break;
                    case "--files":
                        args.files = Integer.parseInt(argv[++i]);
                        break;
                    case "--corpus-rows":
                        args.corpusRows = Integer.parseInt(argv[++i]);
                        break;
                    case "--parallelism":
                        args.parallelism = Integer.parseInt(argv[++i]);
                        break;
                    default:
                        throw new IllegalArgumentException("unknown argument " + argv[i]);
                }
            }
            if (args.probeRows % args.files != 0) {
                throw new IllegalArgumentException(
                        "the probe rows must divide evenly into the files: the region sizes its"
                                + " score matrix from a file's row count and rebuilds its plan"
                                + " when that changes");
            }
            return args;
        }
    }
}
