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
 * One SQL query, one task graph, a compiled kernel and a cuDF primitive inside it.
 *
 * <p>{@link HaversineSQLExample} shows an expression becoming a CUDA kernel. {@link
 * CudfSortSQLExample} shows an operator being served by a cuDF primitive instead. This shows the
 * two composed: the {@code GROUP BY} below runs
 *
 * <pre>
 *   TaskGraph "calc"
 *     TRANSFER_HOST_TO_DEVICE   the two staged columns
 *     LAUNCH  calc.kernel       the haversine, compiled from the SQL
 *     LAUNCH  calc.agg          cudf::groupby, over what the kernel just wrote
 *     TRANSFER_DEVICE_TO_HOST   one row per group
 * </pre>
 *
 * <p>The distances the kernel computes are never copied anywhere. They are written into device
 * memory, grouped where they lie, and what comes back is the group sums — a few thousand rows out
 * of however many million went in. That is the whole argument for binding a library into the same
 * graph as a generated kernel rather than beside it, and it is visible in one screen of {@code
 * --printBytecodes}.
 *
 * <h2>What to look for</h2>
 *
 * <p>{@code --printKernel} prints the generated haversine, as in the first example. {@code
 * --printBytecodes} prints the graph above: <b>two LAUNCH lines between one transfer in and one
 * transfer out</b>. The second LAUNCH is the cuDF call. Nothing between them returns to the host.
 *
 * <p>Unfused, the same query is two graphs and the intermediate makes a round trip: every distance
 * is copied back, rebuilt as a row, taken apart and staged again before the group-by sees it.
 *
 * <h2>What the query author had to write</h2>
 *
 * <p>Standard SQL, and {@code NOT NULL} in the DDL. There is no hint, no annotation and no
 * per-query option that says "use cuDF here" — the planner offers the subtree, the provider
 * recognises a {@code SUM} over an {@code INT} key and answers with a library task.
 *
 * <p>{@code NOT NULL} is load-bearing and is the one concession asked of the author. A nullable
 * column makes the generated kernel carry a validity mask, and {@code cudf::groupby} takes a dense
 * key column and a dense value column with nowhere to record "this row has no key", so the provider
 * declines a projection that carries one — and the whole query then runs on the CPU, correctly and
 * without saying so.
 *
 * <h2>Running it</h2>
 *
 * <pre>
 *   scripts/run-sql-demos.sh cudf-groupby --printBytecodes
 *   scripts/run-sql-demos.sh cudf-groupby --printKernel
 * </pre>
 *
 * <p>The script sets the six RAPIDS directories on {@code LD_LIBRARY_PATH}. Without them the cuDF
 * shim does not load, the provider declines the aggregate, and this example prints the same numbers
 * from the CPU. It checks for that and fails rather than let a CPU run be presented as a device
 * one; see {@link #main}.
 */
public final class CudfGroupBySQLExample {

    /**
     * Depots to measure against. Twenty is enough arithmetic that the kernel is worth launching.
     */
    private static final int DEPOTS = 20;

    /** Degrees to radians, written out because the generator has no RADIANS. */
    private static final String TO_RAD = "0.017453292519943295";

    /** Mean Earth diameter in km — the 2r of the haversine formula. */
    private static final String DIAMETER = "12742.0";

    /** Distinct regions, and so the number of rows the device sends back. */
    private static final int REGIONS = 1_000;

    public static void main(String[] args) throws Exception {

        final long rows = args.length > 0 ? Long.parseLong(args[0]) : 4_000_000L;
        final Path data =
                Paths.get(args.length > 1 ? args[1] : "/tmp/flink-gpu-demo-regions-" + rows);
        writePoints(data, rows);

        final EnvironmentSettings settings =
                EnvironmentSettings.newInstance().inBatchMode().build();
        final TableEnvironment tableEnv = TableEnvironment.create(settings);

        // ---- the switch, and the whole of it ---------------------------------------------
        //
        // Four session settings, one more than the first example. Nothing below this block
        // knows an accelerator exists, and nothing in the SQL mentions one.
        //
        // enabled                 offer eligible subtrees to whatever provider is on the
        //                         classpath. Off by default.
        //
        // approximate-projections allow a projection a device may round differently from
        //                         java.lang.Math. It gives up the output-VALUE guarantee and
        //                         nothing else: a drifting value that could reach a filter,
        //                         join, grouping or sort is refused either way. Here the
        //                         grouping key is an INT the scan read, which nothing rounds,
        //                         and the distances feed only SUM -- so no row can land in a
        //                         different group than it would on the CPU.
        //
        // fuse-aggregate          put the projection and the aggregate in ONE task graph. Off
        //                         by default, because a CPU fused pair was measured as worth
        //                         nothing. With a device behind it, it is the whole point:
        //                         this is the setting that turns two graphs into the one
        //                         printed above.
        //
        // default-parallelism     not an accelerator setting; max-parallelism defaults to 1
        //                         and the offload is declined above it.
        // ACCEL=off runs the identical query with the switch down, which is how the numbers this
        // prints were checked against Flink's own. Not a demo flag -- the demo runs once, with the
        // accelerator on -- but the one-line way to show the answers do not depend on it.
        final boolean accelerate = !"off".equals(System.getenv("ACCEL"));
        tableEnv.getConfig()
                .getConfiguration()
                .setString("table.exec.accelerator.enabled", Boolean.toString(accelerate));
        tableEnv.getConfig()
                .getConfiguration()
                .setString("table.exec.accelerator.approximate-projections", "true");
        tableEnv.getConfig()
                .getConfiguration()
                .setString("table.exec.accelerator.fuse-aggregate", "true");
        tableEnv.getConfig()
                .getConfiguration()
                .setString("table.exec.resource.default-parallelism", "1");
        // ----------------------------------------------------------------------------------

        // NOT NULL is the one thing the query author has to write; see the class comment.
        tableEnv.executeSql(
                "CREATE TABLE Points (\n"
                        + "  region INT NOT NULL,\n"
                        + "  lat DOUBLE NOT NULL,\n"
                        + "  lon DOUBLE NOT NULL\n"
                        + ") WITH (\n"
                        + "  'connector' = 'filesystem',\n"
                        + "  'path' = '"
                        + data
                        + "',\n"
                        + "  'format' = 'csv'\n"
                        + ")");

        // Total distance to the nearest depot, per region. The inner projection is the kernel;
        // the GROUP BY above it is the cuDF call; they share a task graph.
        final String sql =
                "SELECT region, SUM(km) AS total_km\n"
                        + "FROM (\n"
                        + "  SELECT region, "
                        + nearestDepot()
                        + " AS km\n"
                        + "  FROM Points\n"
                        + ")\n"
                        + "GROUP BY region\n"
                        + "ORDER BY region\n"
                        + "LIMIT 10";
        if (System.getenv("EXPLAIN") != null) {
            final String plan = tableEnv.explainSql(sql);
            System.out.println(plan.substring(plan.indexOf("== Optimized Execution Plan ==")));
        }
        tableEnv.executeSql(sql).print();
    }

    /**
     * Distance to the nearest of {@link #DEPOTS} reference points, as one SQL expression.
     *
     * <p>{@code LEAST} over twenty haversines: the input is read once, the output is one column,
     * and the arithmetic in between is what the device is for.
     */
    private static String nearestDepot() {
        final StringBuilder sql = new StringBuilder("LEAST(");
        for (int i = 0; i < DEPOTS; i++) {
            if (i > 0) {
                sql.append(",\n           ");
            }
            sql.append(haversine(-60.0 + i * (120.0 / DEPOTS), -180.0 + i * (360.0 / DEPOTS)));
        }
        return sql.append(")").toString();
    }

    private static String haversine(double depotLat, double depotLon) {
        final String lat = "(lat * " + TO_RAD + ")";
        final String lon = "(lon * " + TO_RAD + ")";
        final String lat0 = "(" + depotLat + " * " + TO_RAD + ")";
        final String lon0 = "(" + depotLon + " * " + TO_RAD + ")";
        return DIAMETER
                + " * ASIN(SQRT("
                + "POWER(SIN(("
                + lat
                + " - "
                + lat0
                + ") / 2.0), 2)"
                + " + COS("
                + lat
                + ") * COS("
                + lat0
                + ")"
                + " * POWER(SIN(("
                + lon
                + " - "
                + lon0
                + ") / 2.0), 2)"
                + "))";
    }

    /**
     * Writes the input table, once.
     *
     * <p>Plain Java rather than {@code datagen}, which costs about 100 microseconds a row and would
     * have the audience watching a progress-free terminal for minutes.
     */
    private static void writePoints(Path directory, long rows) throws IOException {
        if (Files.isDirectory(directory)) {
            return;
        }
        Files.createDirectories(directory);
        // Seeded, so every run of this example groups exactly the same coordinates.
        final Random random = new Random(20260926L);
        try (BufferedWriter out = Files.newBufferedWriter(directory.resolve("points.csv"))) {
            final StringBuilder line = new StringBuilder(48);
            for (long id = 0; id < rows; id++) {
                line.setLength(0);
                line.append(id % REGIONS)
                        .append(',')
                        .append(random.nextDouble() * 180.0 - 90.0)
                        .append(',')
                        .append(random.nextDouble() * 360.0 - 180.0)
                        .append('\n');
                out.write(line.toString());
            }
        }
    }

    private CudfGroupBySQLExample() {}
}
