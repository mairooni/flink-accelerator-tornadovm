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

import org.apache.flink.configuration.Configuration;
import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.types.Row;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Iterator;

/**
 * A shuffle hash join, which is the shape a device probe is for.
 *
 * <p>Measured the way {@link SortBenchmark} is, and for the same reasons — a blackhole sink so the
 * job's time is the operator's, and a second arm that drains the result so something checks it.
 *
 * <p>Two things are specific to a join and both are about making the measurement mean anything:
 *
 * <ul>
 *   <li><b>Broadcast is disabled and so is the adaptive join.</b> A broadcast join is out of scope
 *       (M5.0, 1.47x ceiling) and Flink 2.3 wraps a shuffle join in an {@code AdaptiveJoin} by
 *       default, which decides at runtime and is not offloaded. Both are session settings the
 *       cluster operator sets; nothing in the SQL changes between arms.
 *   <li><b>The build side is the smaller table and it is bounded.</b> {@code --build-rows} sizes
 *       it, and past what the staging holds the offload is refused rather than spilled — which is a
 *       thing worth being able to provoke, so the flag exists.
 * </ul>
 */
public final class JoinBenchmark {

    public static void main(String[] args) throws Exception {
        final Args parsed = Args.parse(args);

        if (parsed.generate) {
            generate(parsed);
            return;
        }
        if (!Files.isDirectory(Paths.get(parsed.factData))
                || !Files.isDirectory(Paths.get(parsed.dimData))) {
            System.out.printf("%s or %s does not exist yet%n", parsed.factData, parsed.dimData);
            generate(parsed);
        }

        System.out.printf(
                "fact=%s  dim=%s  rows=%,d  build=%,d  parallelism=%s  gpu=%s  runs=%d"
                        + "  collect=%s  baseline=%s%n",
                parsed.factData,
                parsed.dimData,
                parsed.rows,
                parsed.buildRows,
                parsed.parallelism > 0 ? Integer.toString(parsed.parallelism) : "(default)",
                parsed.gpu,
                parsed.runs,
                parsed.collect,
                parsed.baseline);

        if (parsed.explain) {
            System.out.println(explain(parsed));
            return;
        }

        Row first = null;
        for (int run = 1; run <= parsed.runs; run++) {
            long start = System.nanoTime();
            Row seen = query(parsed);
            double millis = (System.nanoTime() - start) / 1e6;
            System.out.printf("run %2d  %10.0f ms  %s%n", run, millis, seen);
            if (first == null) {
                first = seen;
            } else if (!first.equals(seen)) {
                throw new IllegalStateException(
                        "the two runs disagree: " + first + " against " + seen);
            }
        }
    }

    private static void generate(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        env.executeSql(
                "CREATE TABLE FactSource (\n"
                        + "  id INT,\n"
                        + "  val DOUBLE\n"
                        + ") WITH (\n"
                        + "  'connector' = 'datagen',\n"
                        + "  'number-of-rows' = '"
                        + args.rows
                        + "',\n"
                        + "  'fields.id.kind' = 'sequence',\n"
                        + "  'fields.id.start' = '0',\n"
                        + "  'fields.id.end' = '"
                        + (args.rows - 1)
                        + "',\n"
                        + "  'fields.val.min' = '0.0',\n"
                        + "  'fields.val.max' = '1000.0'\n"
                        + ")");
        env.executeSql(fact(args.factData, args.format));
        env.executeSql(dim(args.dimData, args.format));

        System.out.printf("writing %,d fact rows to %s%n", args.rows, args.factData);
        long start = System.nanoTime();
        // Fact keys are scrambled over twice the build side's range, so half of them match and the
        // probe is not reading its keys in order -- which a binary search would otherwise find
        // suspiciously cache-friendly.
        env.executeSql(
                        "INSERT INTO Fact SELECT\n"
                                + "  CAST(MOD(CAST(id AS BIGINT) * 2654435761, "
                                + (2 * args.buildRows)
                                + ") AS INT),\n"
                                + "  CAST(id AS BIGINT),\n"
                                + "  val\n"
                                + "FROM FactSource")
                .await();
        System.out.printf("writing %,d dim rows to %s%n", args.buildRows, args.dimData);
        env.executeSql(
                        "INSERT INTO Dim SELECT CAST(id AS INT), CAST(id + 1000 AS BIGINT)\n"
                                + "FROM FactSource WHERE id < "
                                + args.buildRows)
                .await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    private static Row query(Args args) throws Exception {
        final TableEnvironment env = session(args);
        if (!args.collect) {
            env.executeSql(sql(args)).await();
            return Row.of("written to the blackhole sink");
        }
        try (org.apache.flink.util.CloseableIterator<Row> rows =
                env.executeSql(sql(args)).collect()) {
            return args.baseline ? one(rows) : summarise(rows);
        }
    }

    private static String explain(Args args) {
        return session(args).explainSql(sql(args));
    }

    private static Row one(Iterator<Row> rows) {
        if (!rows.hasNext()) {
            throw new IllegalStateException("the query returned nothing");
        }
        return rows.next();
    }

    /**
     * Drains the joined rows, checking each pair agrees, and returns a summary.
     *
     * <p>The check is what makes this arm worth running: a join that paired the wrong rows returns
     * the right number of them. {@code label} is {@code key + 1000} and {@code seq} identifies the
     * fact row, so a pair assembled from two different matches fails here.
     */
    private static Row summarise(Iterator<Row> rows) {
        long seen = 0;
        double total = 0.0;
        while (rows.hasNext()) {
            Row row = rows.next();
            int key = ((Number) row.getField(0)).intValue();
            long label = ((Number) row.getField(1)).longValue();
            if (label != key + 1000L) {
                throw new IllegalStateException(
                        "row " + seen + " pairs key " + key + " with label " + label);
            }
            total += ((Number) row.getField(2)).doubleValue();
            seen++;
        }
        return Row.of(seen, total);
    }

    /** The query, in every arm. */
    private static String sql(Args args) {
        if (args.baseline) {
            return args.collect
                    ? "SELECT COUNT(*) AS rows_seen, SUM(val) AS total FROM Fact"
                    : "INSERT INTO Discard SELECT k, CAST(seq AS BIGINT), val FROM Fact";
        }
        String join = "SELECT f.k, d.label, f.val FROM Fact f, Dim d WHERE f.k = d.k";
        return args.collect ? join : "INSERT INTO Discard " + join;
    }

    private static String fact(String path, String format) {
        return table(
                "Fact",
                "k INT NOT NULL,\n  seq BIGINT NOT NULL,\n  val DOUBLE NOT NULL",
                path,
                format);
    }

    private static String dim(String path, String format) {
        return table("Dim", "k INT NOT NULL,\n  label BIGINT NOT NULL", path, format);
    }

    private static String table(String name, String columns, String path, String format) {
        return "CREATE TABLE "
                + name
                + " (\n  "
                + columns
                + "\n) WITH (\n  'connector' = 'filesystem',\n  'path' = '"
                + path
                + "',\n  'format' = '"
                + format
                + "'\n)";
    }

    private static String discard() {
        return "CREATE TABLE Discard (\n"
                + "  k INT NOT NULL,\n"
                + "  label BIGINT NOT NULL,\n"
                + "  val DOUBLE NOT NULL\n"
                + ") WITH ('connector' = 'blackhole')";
    }

    private static TableEnvironment session(Args args) {
        TableEnvironment env = batchEnvironment(args);
        env.executeSql(fact(args.factData, args.format));
        env.executeSql(dim(args.dimData, args.format));
        env.executeSql(discard());
        Configuration conf = env.getConfig().getConfiguration();
        // Both arms, so the plan shape is the same either way: a broadcast join is out of scope
        // and an AdaptiveJoin is not offloaded, so a run that left either on would be comparing
        // two different plans rather than two backends for one.
        conf.setString("table.optimizer.join.broadcast-threshold", "-1");
        conf.setString("table.optimizer.adaptive-broadcast-join.strategy", "NONE");
        conf.setString("table.optimizer.skewed-join-optimization.strategy", "NONE");
        if (args.gpu) {
            conf.setString("table.exec.accelerator.enabled", "true");
            if (args.minSpeedup != null) {
                conf.setString("table.exec.accelerator.min-speedup", args.minSpeedup);
            }
        }
        return env;
    }

    private static TableEnvironment batchEnvironment(Args args) {
        final TableEnvironment env =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        if (args.parallelism > 0) {
            env.getConfig()
                    .getConfiguration()
                    .setString(
                            "table.exec.resource.default-parallelism",
                            Integer.toString(args.parallelism));
        }
        return env;
    }

    /** Command line, in the shape the run scripts use. */
    private static final class Args {
        private String factData;
        private String dimData;
        private int rows = 4_000_000;
        private int buildRows = 1_000_000;
        private int parallelism = -1;
        private int runs = 6;
        private String format = "csv";
        private boolean gpu;
        private boolean generate;
        private boolean explain;
        private boolean baseline;
        private boolean collect = true;
        private String minSpeedup;

        static Args parse(String[] argv) {
            Args args = new Args();
            for (int i = 0; i < argv.length; i++) {
                String flag = argv[i];
                if ("--fact".equals(flag)) {
                    args.factData = argv[++i];
                } else if ("--dim".equals(flag)) {
                    args.dimData = argv[++i];
                } else if ("--rows".equals(flag)) {
                    args.rows = Integer.parseInt(argv[++i]);
                } else if ("--build-rows".equals(flag)) {
                    args.buildRows = Integer.parseInt(argv[++i]);
                } else if ("--parallelism".equals(flag)) {
                    args.parallelism = Integer.parseInt(argv[++i]);
                } else if ("--runs".equals(flag)) {
                    args.runs = Integer.parseInt(argv[++i]);
                } else if ("--format".equals(flag)) {
                    args.format = argv[++i];
                } else if ("--gpu".equals(flag)) {
                    args.gpu = Boolean.parseBoolean(argv[++i]);
                } else if ("--collect".equals(flag)) {
                    args.collect = Boolean.parseBoolean(argv[++i]);
                } else if ("--min-speedup".equals(flag)) {
                    args.minSpeedup = argv[++i];
                } else if ("--generate".equals(flag)) {
                    args.generate = true;
                } else if ("--baseline".equals(flag)) {
                    args.baseline = true;
                } else if ("--explain".equals(flag)) {
                    args.explain = true;
                } else {
                    throw new IllegalArgumentException("unknown argument " + flag);
                }
            }
            if (args.factData == null) {
                args.factData = "/tmp/flink-gpu-fact-" + args.rows + "-" + args.format;
            }
            if (args.dimData == null) {
                args.dimData = "/tmp/flink-gpu-dim-" + args.buildRows + "-" + args.format;
            }
            return args;
        }
    }

    private JoinBenchmark() {}
}
