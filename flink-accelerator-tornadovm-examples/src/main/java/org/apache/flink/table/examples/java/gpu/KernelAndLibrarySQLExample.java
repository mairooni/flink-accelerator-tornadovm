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

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

/**
 * One query with two kinds of work in it, and what the library adds to the kernel.
 *
 * <p>The projection is arithmetic: a score over four columns, compiled from the SQL into a CUDA
 * kernel. The {@code ORDER BY} is not arithmetic at all -- it is a permutation of the whole column,
 * and no expression compiler produces one.
 *
 * <p>So this job has a JIT half and a relational half, and the relational half has somewhere to go
 * only if the cuDF binding is present. Run it twice:
 *
 * <pre>
 *   flink run KernelAndLibrarySQLExample.jar              # kernel + cudf::stable_sorted_order
 *   flink run -Dflink.accelerator.tornadovm.disableCudf=true KernelAndLibrarySQLExample.jar
 * </pre>
 *
 * <p>The second is the all-JIT arm: the score still runs on the device, and the sort falls to
 * Flink's own {@code SortOperator}. Same SQL, same jar, same card. The difference is one library.
 *
 * <h2>Why the sort is the library's case and the grouping is not</h2>
 *
 * <p>Measured rather than assumed, and the two come out opposite ways.
 *
 * <p>A grouped sum over dense keys is reduction-shaped: each row's destination follows from its own
 * key, so {@code atomicAdd(sums, key, value)} does it in one line and beats {@code cudf::groupby}
 * by up to 1.7x. Writing that kernel is the right answer.
 *
 * <p>A sort is not: where a row lands depends on every other row. The best thing a kernel generator
 * could reasonably emit is a bitonic sort -- correct, parallel, and <i>O(n log&sup2;n)</i>
 * comparisons over a few hundred kernel launches. Measured against {@code
 * cudf::stable_sorted_order} over the same data it is <b>4.5x to 9.6x slower</b>, because radix
 * sorting is a handful of linear passes and has had years of tuning behind it.
 *
 * <p>Hence: write the kernel when a row's destination is a function of that row; call the library
 * when it is a function of the whole column.
 *
 * <h2>What the query author writes</h2>
 *
 * <p>Standard SQL, and {@code NOT NULL} in the DDL. Nothing names a kernel, a library or a device.
 */
public final class KernelAndLibrarySQLExample {

    /** Rows to print, so the output fits a slide. */
    private static final int HEAD = 10;

    public static void main(String[] args) throws Exception {

        final long rows = args.length > 0 ? Long.parseLong(args[0]) : 4_000_000L;
        final Path data =
                Paths.get(args.length > 1 ? args[1] : "/tmp/flink-gpu-demo-signals-" + rows);
        writeSignals(data, rows);

        // set up the Table API
        final EnvironmentSettings settings =
                EnvironmentSettings.newInstance().inBatchMode().build();
        final TableEnvironment tableEnv = TableEnvironment.create(settings);

        // ---- the switch, and the whole of it ---------------------------------------------
        //
        // enabled                 offer eligible subtrees to whatever provider is on the
        //                         classpath. Off by default.
        //
        // approximate-projections allow a projection a device may round differently from
        //                         java.lang.Math. It gives up the output VALUE and nothing
        //                         else: a drifting value that could reach a filter, join,
        //                         grouping or sort is refused either way. Here the ORDER BY is
        //                         on an INT the scan read, which nothing rounds.
        //
        // default-parallelism     not an accelerator setting; max-parallelism defaults to 1
        //                         and the offload is declined above it.
        tableEnv.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
        tableEnv.getConfig()
                .getConfiguration()
                .setString("table.exec.accelerator.approximate-projections", "true");
        tableEnv.getConfig()
                .getConfiguration()
                .setString("table.exec.resource.default-parallelism", "1");
        // Four million rows of sorting want more than a MiniCluster's default share.
        tableEnv.getConfig().getConfiguration().setString("taskmanager.memory.managed.size", "2g");
        // ----------------------------------------------------------------------------------

        // NOT NULL is the one thing the query author has to write: a device column has no
        // representation for a null, and the sort key must be dense.
        tableEnv.executeSql(
                "CREATE TABLE Signals (\n"
                        + "  bucket INT NOT NULL,\n"
                        + "  a DOUBLE NOT NULL,\n"
                        + "  b DOUBLE NOT NULL,\n"
                        + "  c DOUBLE NOT NULL\n"
                        + ") WITH (\n"
                        + "  'connector' = 'filesystem',\n"
                        + "  'path' = '"
                        + data
                        + "',\n"
                        + "  'format' = 'csv'\n"
                        + ")");

        // execute a Flink SQL job and print the result locally
        //
        // No LIMIT, and that is not an oversight. A LIMIT here sinks below the projection --
        // sorting by a column the scan already has does not need the score computed first -- so
        // the Calc sees ten rows instead of four million and is declined with "too few rows to
        // repay setting a device up". The whole table is ordered and only the head is printed.
        final String sql =
                // the ordering, which is the library's half
                "SELECT bucket, score\n"
                        + "FROM (\n"
                        // the arithmetic, which is the kernel's half
                        + "  SELECT bucket, "
                        + score()
                        + " AS score\n"
                        + "  FROM Signals\n"
                        + ")\n"
                        + "ORDER BY bucket";
        if (System.getenv("EXPLAIN") != null) {
            // EXPLAIN=1 prints the plan and the accelerator's verdict for every node, which is
            // how to find out why a half that should have offloaded did not.
            final String plan = tableEnv.explainSql(sql);
            System.out.println(plan.substring(plan.indexOf("== Optimized Execution Plan ==")));
        }

        long seen = 0;
        int previous = Integer.MIN_VALUE;
        try (org.apache.flink.util.CloseableIterator<org.apache.flink.types.Row> ordered =
                tableEnv.executeSql(sql).collect()) {
            while (ordered.hasNext()) {
                final org.apache.flink.types.Row row = ordered.next();
                final int bucket = (Integer) row.getField(0);
                // The ordering is the result here, so it is checked rather than assumed: a sort
                // that returns the right rows in the wrong order is the failure to catch.
                if (bucket < previous) {
                    throw new IllegalStateException(
                            "row " + seen + " has bucket " + bucket + " after " + previous);
                }
                if (seen < HEAD) {
                    System.out.println(row);
                }
                previous = bucket;
                seen++;
            }
        }
        System.out.printf("%,d rows, ordered%n", seen);
    }

    /**
     * Enough arithmetic per row that the projection is worth a kernel.
     *
     * <p>Written without {@code RADIANS} or {@code MOD} because the generator maps a fixed set of
     * functions onto {@code TornadoMath} and neither is in it; multiplying by a literal is.
     */
    private static String score() {
        return "SQRT(a * a + b * b + c * c)\n"
                + "         * (1.0 + SIN(a * 0.5) * COS(b * 0.25))\n"
                + "         + EXP(-(a * a + b * b) * 0.001) * LN(1.0 + c * c)";
    }

    /**
     * Writes the input table, once.
     *
     * <p>Plain Java rather than the {@code datagen} connector: {@code datagen} costs about 100
     * microseconds a row, so four million rows take minutes and the audience watches a
     * progress-free terminal.
     */
    private static void writeSignals(Path directory, long rows) throws IOException {
        if (Files.isDirectory(directory)) {
            return;
        }
        Files.createDirectories(directory);
        // Seeded, so every run of this example sorts exactly the same rows.
        final Random random = new Random(20260926L);
        try (BufferedWriter out = Files.newBufferedWriter(directory.resolve("signals.csv"))) {
            final StringBuilder line = new StringBuilder(64);
            for (long i = 0; i < rows; i++) {
                line.setLength(0);
                line.append(random.nextInt(1_000_000))
                        .append(',')
                        .append(random.nextDouble() * 10.0)
                        .append(',')
                        .append(random.nextDouble() * 10.0)
                        .append(',')
                        .append(random.nextDouble() * 10.0)
                        .append('\n');
                out.write(line.toString());
            }
        }
    }

    private KernelAndLibrarySQLExample() {}
}
