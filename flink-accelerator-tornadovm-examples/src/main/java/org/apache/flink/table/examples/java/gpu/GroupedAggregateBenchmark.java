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
 * A relational-heavy query, which is the shape a device group-by is for.
 *
 * <p>{@link HaversineBenchmark} is the opposite of one by construction and its {@code --groups} arm
 * proved it: twenty haversines dominate so completely that the {@code GROUP BY} over 2M rows costs
 * 1.4% of the job, and serving it perfectly would win back a percent. Reporting a number from that
 * benchmark would say nothing about the binding.
 *
 * <p>So this query has almost no arithmetic in it. One multiply-add a row makes the projection real
 * without making it the point; everything else the job spends goes into grouping. Against the
 * read-and-count floor the {@code --baseline} arm measures, that is the 60-70% share §T5's CPU arm
 * found for a relational operator, and it is where a group-by on the device has room to show.
 *
 * <p>Three arms, and all three are needed to say anything:
 *
 * <ul>
 *   <li>{@code --baseline} — {@code COUNT(*)} over the same source. What reading the rows costs,
 *       with no grouping at all.
 *   <li>{@code --gpu false} — the grouped query on the CPU.
 *   <li>{@code --gpu true} — the same query with the accelerator enabled.
 * </ul>
 *
 * <p>{@code --min-speedup} exists because the provider's cost model was calibrated on per-row
 * arithmetic, and a query with one multiply-add a row does not clear its floor. Setting it to zero
 * takes the offload anyway, which is how the offload gets measured before the model is asked to
 * predict it. It is a session option like every other here; nothing in the SQL changes between
 * arms.
 */
public final class GroupedAggregateBenchmark {

    public static void main(String[] args) throws Exception {
        final Args parsed = Args.parse(args);

        if (parsed.generate) {
            generate(parsed);
            return;
        }
        if (!Files.isDirectory(Paths.get(parsed.data))) {
            System.out.printf("%s does not exist yet%n", parsed.data);
            generate(parsed);
        }

        System.out.printf(
                "data=%s  parallelism=%s  gpu=%s  runs=%d  groups=%d  baseline=%s%n",
                parsed.data,
                parsed.parallelism > 0 ? Integer.toString(parsed.parallelism) : "(default)",
                parsed.gpu,
                parsed.runs,
                parsed.groups,
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
            } else {
                agree(first, seen);
            }
        }
    }

    /**
     * Checks two runs agree, to within floating-point reassociation.
     *
     * <p>Not bit-for-bit, and the tolerance matters more here than in the haversine benchmark: the
     * device sums a group in whatever order cuDF reduces it and the CPU sums it in row order, so
     * the totals differ in their last bits by construction. Exact equality would fail on correct
     * results.
     */
    private static void agree(Row first, Row second) {
        long rowsFirst = ((Number) first.getField(0)).longValue();
        long rowsSecond = ((Number) second.getField(0)).longValue();
        if (rowsFirst != rowsSecond) {
            throw new IllegalStateException("row counts differ: " + first + " against " + second);
        }
        double a = ((Number) first.getField(1)).doubleValue();
        double b = ((Number) second.getField(1)).doubleValue();
        if (Math.abs(a - b) > Math.abs(a) * 1e-9) {
            throw new IllegalStateException("totals differ: " + first + " against " + second);
        }
    }

    private static void generate(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        env.executeSql(
                "CREATE TABLE Source (\n"
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
        env.executeSql(sales(args.data, args.format));

        System.out.printf("writing %,d rows to %s%n", args.rows, args.data);
        long start = System.nanoTime();
        env.executeSql("INSERT INTO Sales SELECT id, val FROM Source").await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    private static Row query(Args args) throws Exception {
        final TableEnvironment env = session(args);
        try (org.apache.flink.util.CloseableIterator<Row> rows =
                env.executeSql(sql(args)).collect()) {
            return one(rows);
        }
    }

    private static String explain(Args args) {
        TableEnvironment env = session(args);
        return env.explainSql(sql(args));
    }

    private static Row one(Iterator<Row> rows) {
        if (!rows.hasNext()) {
            throw new IllegalStateException("the query returned nothing");
        }
        Row row = rows.next();
        if (rows.hasNext()) {
            throw new IllegalStateException("the query returned more than one row");
        }
        return row;
    }

    /**
     * The query, in both arms.
     *
     * <p>The key is written {@code id - (id / g) * g} rather than {@code MOD(id, g)} for the reason
     * recorded in {@link HaversineBenchmark}: it is what the planner lowers the remainder to, and
     * writing it out keeps the two arms reading the same SQL.
     *
     * <p>Wrapped in an outer {@code COUNT}/{@code SUM} so the job returns one row however many
     * groups there are, which keeps the client transfer out of the measurement. An earlier version
     * of the fusion harness measured exactly that transfer and reported it as a speedup.
     */
    private static String sql(Args args) {
        if (args.baseline) {
            return "SELECT COUNT(*) AS rows_seen, SUM(val) AS total FROM Sales";
        }
        String key = "id - (id / " + args.groups + ") * " + args.groups;
        return "SELECT COUNT(*) AS rows_seen, SUM(total) AS total FROM (\n"
                + "  SELECT "
                + key
                + " AS k, SUM(val * 2.0 + 1.0) AS total\n"
                + "  FROM Sales\n"
                + "  GROUP BY "
                + key
                + "\n)";
    }

    private static String sales(String path, String format) {
        return "CREATE TABLE Sales (\n"
                + "  id INT NOT NULL,\n"
                + "  val DOUBLE NOT NULL\n"
                + ") WITH (\n"
                + "  'connector' = 'filesystem',\n"
                + "  'path' = '"
                + path
                + "',\n"
                + "  'format' = '"
                + format
                + "'\n"
                + ")";
    }

    private static TableEnvironment session(Args args) {
        TableEnvironment env = batchEnvironment(args);
        env.executeSql(sales(args.data, args.format));
        Configuration conf = env.getConfig().getConfiguration();
        // Both arms, so that the plan shape is the same one either way: the local half of the
        // aggregate is what an accelerator can serve, and a one-phase plan has no local half.
        conf.setString("table.optimizer.agg-phase-strategy", "TWO_PHASE");
        if (args.gpu) {
            conf.setString("table.exec.accelerator.enabled", "true");
            // Without this the Calc and the LocalHashAggregate stay separate nodes and only the
            // Calc is offered to a provider, which is the pre-M5.3 arrangement and not what this
            // measures. Fusing them is what produces the node a device group-by can serve.
            conf.setString(
                    "table.exec.accelerator.fuse-aggregate", Boolean.toString(args.fuseAggregate));
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
        private String data;
        private int rows = 4_000_000;
        private int parallelism = -1;
        private int runs = 6;
        private String format = "csv";
        private boolean gpu;
        private boolean generate;
        private boolean explain;
        private boolean baseline;
        private int groups = 1000;
        private String minSpeedup;
        private boolean fuseAggregate = true;

        static Args parse(String[] argv) {
            Args args = new Args();
            for (int i = 0; i < argv.length; i++) {
                String flag = argv[i];
                if ("--data".equals(flag)) {
                    args.data = argv[++i];
                } else if ("--rows".equals(flag)) {
                    args.rows = Integer.parseInt(argv[++i]);
                } else if ("--parallelism".equals(flag)) {
                    args.parallelism = Integer.parseInt(argv[++i]);
                } else if ("--runs".equals(flag)) {
                    args.runs = Integer.parseInt(argv[++i]);
                } else if ("--format".equals(flag)) {
                    args.format = argv[++i];
                } else if ("--gpu".equals(flag)) {
                    args.gpu = Boolean.parseBoolean(argv[++i]);
                } else if ("--groups".equals(flag)) {
                    args.groups = Integer.parseInt(argv[++i]);
                } else if ("--fuse-aggregate".equals(flag)) {
                    args.fuseAggregate = Boolean.parseBoolean(argv[++i]);
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
            if (args.data == null) {
                args.data = "/tmp/flink-gpu-sales-" + args.rows + "-" + args.format;
            }
            return args;
        }
    }

    private GroupedAggregateBenchmark() {}
}
