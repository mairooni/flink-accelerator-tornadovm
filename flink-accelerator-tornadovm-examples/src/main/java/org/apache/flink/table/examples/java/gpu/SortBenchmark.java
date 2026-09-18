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
 * An {@code ORDER BY} over a whole partition, which is the shape a device sort is for.
 *
 * <p>Built on {@link GroupedAggregateBenchmark} and measuring the same way, because the question is
 * the same one: what share of a job an operator occupies, and what serving that share on a device
 * is worth. The three arms are what make an answer possible at all —
 *
 * <ul>
 *   <li>{@code --baseline} — {@code COUNT(*)} over the same source, with no ordering. What reading
 *       the rows costs, which is the floor neither arm below can go under.
 *   <li>{@code --gpu false} — the ordered query on the CPU, which is {@code SortOperator}: binary
 *       rows in managed memory, normalized keys, and a spill when it runs out.
 *   <li>{@code --gpu true} — the same query with the accelerator enabled.
 * </ul>
 *
 * <p>Two shapes, because one cannot do both jobs. {@code --collect true} drains the ordered rows to
 * the client and checks the order on the way through, which is the only arm that can say the result
 * is right — and it is useless as a measurement, because shipping 4M rows to a client costs 6.7
 * seconds against the sort's 1.9 and buries it. {@code --collect false} writes the same ordered
 * rows to a blackhole sink instead: nothing leaves the cluster, the sort survives, and what the job
 * spends is what the sort spends.
 *
 * <p>Run with {@code -Dpipeline.operator-chaining=false} to attribute it. Chained, the sort and the
 * sink are one vertex and the REST API reports their sum; unchained, {@code Sort[n]} stands alone
 * and its duration is the number this benchmark exists to produce. The chain is worth something —
 * §T4 measured how much — so the chained job time is the honest headline and the unchained
 * breakdown is the attribution, and both are reported rather than one standing in for the other.
 *
 * <p>{@code --min-speedup} exists for the reason it does in the grouped benchmark and one more: a
 * sort's work is comparisons, and {@code AcceleratorChoice.repaysFixedCost} prices the CPU at one
 * nanosecond an operation a row, which for a comparison in a real sort is about thirty times too
 * generous. Until that is fixed, the offload has to be asked for before it can be measured.
 */
public final class SortBenchmark {

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
                "data=%s  rows=%,d  parallelism=%s  gpu=%s  runs=%d  collect=%s  baseline=%s%n",
                parsed.data,
                parsed.rows,
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
            } else {
                agree(first, seen);
            }
        }
    }

    /**
     * Checks two runs agree.
     *
     * <p>Exactly, unlike the grouped benchmark's tolerance. A sort reassociates nothing: it moves
     * rows, and two arms that ordered the same input differently is precisely the failure this is
     * here to catch, so any drift at all is a defect rather than floating-point.
     */
    private static void agree(Row first, Row second) {
        if (!first.equals(second)) {
            throw new IllegalStateException(
                    "the two runs disagree: " + first + " against " + second);
        }
    }

    private static void generate(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        // A scrambled key, not a sequence: a sort over already-sorted input measures nothing, and
        // the multiplicative shuffle below is a permutation of [0, rows) so every key appears once.
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
        env.executeSql(events(args.data, args.format));

        System.out.printf("writing %,d rows to %s%n", args.rows, args.data);
        long start = System.nanoTime();
        env.executeSql(
                        "INSERT INTO Events SELECT\n"
                                + "  CAST(MOD(CAST(id AS BIGINT) * 2654435761, "
                                + args.rows
                                + ") AS INT),\n"
                                + "  CAST(id AS BIGINT),\n"
                                + "  val\n"
                                + "FROM Source")
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
            return args.baseline ? one(rows) : drain(rows);
        }
    }

    private static String explain(Args args) {
        return session(args).explainSql(sql(args));
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
     * Drains an ordered stream, checking the order as it goes and returning a summary of it.
     *
     * <p>The check is the reason this exists rather than a plain count: the whole result crosses to
     * the client here, so it is the one arm that can say the rows really came back in order, and a
     * benchmark that measured a sort without ever looking at the order would be measuring a
     * transfer.
     */
    private static Row drain(Iterator<Row> rows) {
        long seen = 0;
        int previous = Integer.MIN_VALUE;
        double total = 0.0;
        while (rows.hasNext()) {
            Row row = rows.next();
            int key = ((Number) row.getField(0)).intValue();
            if (key < previous) {
                throw new IllegalStateException(
                        "row " + seen + " has key " + key + " after " + previous);
            }
            previous = key;
            total += ((Number) row.getField(2)).doubleValue();
            seen++;
        }
        return Row.of(seen, total);
    }

    /** The query, in every arm. See the class comment for why there are two shapes. */
    private static String sql(Args args) {
        if (args.baseline) {
            return args.collect
                    ? "SELECT COUNT(*) AS rows_seen, SUM(val) AS total FROM Events"
                    : "INSERT INTO Discard SELECT k, seq, val FROM Events";
        }
        return args.collect
                ? "SELECT k, seq, val FROM Events ORDER BY k"
                : "INSERT INTO Discard SELECT k, seq, val FROM Events ORDER BY k";
    }

    /**
     * A sink that costs nothing, so the job's time is the sort's.
     *
     * <p>Not a substitute for looking at the result — nothing can check an order it throws away.
     * The {@code --collect} arm is what checks it, over the same data and the same plan up to the
     * sink, and the two arms together say both things.
     */
    private static String discard() {
        return "CREATE TABLE Discard (\n"
                + "  k INT NOT NULL,\n"
                + "  seq BIGINT NOT NULL,\n"
                + "  val DOUBLE NOT NULL\n"
                + ") WITH ('connector' = 'blackhole')";
    }

    /**
     * The table.
     *
     * <p>{@code NOT NULL} throughout, which is the sanctioned exception to the transparency
     * invariant and the same thing every other benchmark here declares. {@code seq} is a {@code
     * BIGINT} deliberately: it is a payload column no kernel could compute with, carried past the
     * device to prove that a sort does not need one to be expressible.
     */
    private static String events(String path, String format) {
        return "CREATE TABLE Events (\n"
                + "  k INT NOT NULL,\n"
                + "  seq BIGINT NOT NULL,\n"
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
        env.executeSql(events(args.data, args.format));
        env.executeSql(discard());
        Configuration conf = env.getConfig().getConfiguration();
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
        private String data;
        private int rows = 4_000_000;
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
            if (args.data == null) {
                args.data = "/tmp/flink-gpu-events-" + args.rows + "-" + args.format;
            }
            return args;
        }
    }

    private SortBenchmark() {}
}
