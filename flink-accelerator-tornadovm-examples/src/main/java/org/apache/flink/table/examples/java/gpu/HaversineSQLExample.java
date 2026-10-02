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
 * A Flink SQL job whose expression is compiled to a CUDA kernel and run on an accelerator.
 *
 * <p>The query asks how far each of two million coordinates is from its nearest depot. It is
 * ordinary Flink SQL — no hint, no annotation, no accelerator type, no syntax that a device could
 * be inferred from. The only thing in this file that is not in {@code WordCountSQLExample} is the
 * block marked <em>the switch</em>: three session settings, which a cluster operator sets and a
 * query author never sees.
 *
 * <h2>Seeing that it really ran there</h2>
 *
 * <p>Nothing in the output says where the query executed, and nothing should: the claim being made
 * is that the answer does not depend on it. Ask TornadoVM instead, by adding {@code
 * -Dtornado.printKernel=true} to the JVM — {@code scripts/run-sql-demos.sh haversine --printKernel}
 * does that — and the CUDA it generated for this query is printed before the result:
 *
 * <pre>
 *   extern "C" __global__ void evaluate(...)
 *   {
 *     ...
 *     d_40  =  sin(d_39);
 *     d_41  =  pow(d_40, 2.0);
 *     d_45  =  asin(d_44);
 *     d_46  =  d_45 * 12742.0;
 *   }
 * </pre>
 *
 * <p>That function does not exist anywhere in this repository. It was generated from the SQL at job
 * start, for this query and no other: Flink lowers the projection to an accelerator IR, and the
 * provider compiles that IR to CUDA. {@link CudfSortSQLExample} is the other half of the
 * integration, where a whole operator is served by a library instead of by a generated kernel.
 *
 * <h2>Why this query rather than a simpler one</h2>
 *
 * <p>Moving a row to a device costs about as much whatever is then done to it, so an accelerator is
 * offered only expressions with enough arithmetic per byte read to earn the trip. One haversine is
 * two {@code SIN}, two {@code COS}, two {@code POWER}, a {@code SQRT} and an {@code ASIN}; twenty
 * of them — "which of my twenty depots is nearest" — is about 360 weighted operations over sixteen
 * bytes of input. {@code val * 2.0 + 1.0} over the same table is three operations, and the planner
 * is right to refuse it.
 *
 * <p>The aggregate at the top keeps the output to one row while still forcing every distance to be
 * computed. It is not offloaded; only the projection beneath it is.
 *
 * <h2>The one thing the query author has to write</h2>
 *
 * <p>{@code NOT NULL} in the DDL. A device kernel has no representation for a null, so the planner
 * refuses any expression with a nullable operand rather than guessing. Without it the query is
 * still correct and simply never reaches the accelerator.
 *
 * <h2>Running it</h2>
 *
 * <pre>
 *   scripts/run-sql-demos.sh haversine --printKernel
 *   scripts/run-sql-demos.sh haversine --rows 8000000
 * </pre>
 *
 * <p>The script assembles what is a property of the deployment rather than of the query: the
 * provider on the classpath, TornadoVM's JVM arguments, and the two flags that make device
 * arithmetic agree with Flink's bit for bit. If {@code --printKernel} prints no kernel, run again
 * with {@code VERBOSE=1} and the accelerator will say why it declined.
 */
public final class HaversineSQLExample {

    /** Reference points the query measures to, spread deterministically over the globe. */
    private static final int DEPOTS = 20;

    /** Degrees to radians. {@code RADIANS} is not in the kernel generator's function set. */
    private static final String TO_RAD = "0.017453292519943295";

    /** Mean Earth diameter in km — the 2r of the haversine formula. */
    private static final String DIAMETER = "12742.0";

    public static void main(String[] args) throws Exception {

        // `--cpu` anywhere in the arguments runs the identical query with the accelerator
        // switched off, which is how the demo shows the same answer from the other plan. It is a
        // demo affordance and not a query option: the switch it flips is the session setting a
        // cluster operator owns, and the SQL below does not change.
        boolean accelerate = true;
        int parallelism = 1;
        final java.util.List<String> positional = new java.util.ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            if ("--cpu".equals(args[i])) {
                accelerate = false;
            } else if ("--parallelism".equals(args[i])) {
                parallelism = Integer.parseInt(args[++i]);
            } else {
                positional.add(args[i]);
            }
        }
        final long rows = !positional.isEmpty() ? Long.parseLong(positional.get(0)) : 2_000_000L;
        final Path data =
                Paths.get(positional.size() > 1 ? positional.get(1)
                                                : "/tmp/flink-gpu-demo-points-" + rows);
        writePoints(data, rows);

        // set up the Table API
        final EnvironmentSettings settings =
                EnvironmentSettings.newInstance().inBatchMode().build();
        final TableEnvironment tableEnv = TableEnvironment.create(settings);

        // ---- the switch, and the whole of it ---------------------------------------------
        //
        // Three session settings. Nothing below this block knows an accelerator exists.
        //
        // enabled                 offer eligible subtrees to whatever provider is on the
        //                         TaskManager's classpath. Off by default.
        //
        // approximate-projections allow a projection whose value a device may round differently
        //                         from java.lang.Math. SIN and the rest come from the device's own
        //                         math library, which is not required to agree with Flink's to the
        //                         last bit, so this is off by default. It gives up the output-VALUE
        //                         guarantee and nothing else: an offload whose drifting value could
        //                         reach a filter, join, grouping or sort is refused either way, so
        //                         which rows a query returns never depends on it.
        //
        // default-parallelism     not an accelerator setting, but needed here, because
        //                         table.exec.accelerator.max-parallelism defaults to 1 and the
        //                         offload is declined above it. That default is a live workaround
        //                         for an intermittent CUDA launch failure at higher parallelism on
        //                         one card, not a statement about what accelerators can do.
        tableEnv.getConfig().getConfiguration()
                .setString("table.exec.accelerator.enabled", Boolean.toString(accelerate));
        tableEnv.getConfig()
                .getConfiguration()
                .setString("table.exec.accelerator.approximate-projections", "true");
        tableEnv.getConfig()
                .getConfiguration()
                .setString("table.exec.resource.default-parallelism", Integer.toString(parallelism));
        // ----------------------------------------------------------------------------------

        // NOT NULL is the one thing the query author has to write; see the class comment.
        tableEnv.executeSql(
                "CREATE TABLE Points (\n"
                        + "  id INT NOT NULL,\n"
                        + "  lat DOUBLE NOT NULL,\n"
                        + "  lon DOUBLE NOT NULL\n"
                        + ") WITH (\n"
                        + "  'connector' = 'filesystem',\n"
                        + "  'path' = '"
                        + data
                        + "',\n"
                        + "  'format' = 'csv'\n"
                        + ")");

        // execute a Flink SQL job and print the result locally
        tableEnv.executeSql(
                        // how far away the nearest depot is, over every point
                        "SELECT COUNT(*) AS points,\n"
                                + "       MIN(km) AS nearest_km,\n"
                                + "       MAX(km) AS farthest_km\n"
                                + "FROM (\n"
                                // the distance itself, which is the part that is compiled
                                + "  SELECT "
                                + nearestDepot()
                                + " AS km\n"
                                + "  FROM Points\n"
                                + ")")
                .print();
    }

    /**
     * Distance to the nearest of {@link #DEPOTS} reference points, as one SQL expression.
     *
     * <p>{@code LEAST} over twenty haversines: the input is read once, the output is still one
     * column, and the arithmetic in between is what the device is for. Written without {@code
     * RADIANS} because the generator maps a fixed set of functions onto {@code TornadoMath} and
     * that is not one of them, whereas multiplying by a literal is.
     */
    private static String nearestDepot() {
        final StringBuilder sql = new StringBuilder("LEAST(");
        for (int i = 0; i < DEPOTS; i++) {
            if (i > 0) {
                sql.append(",\n           ");
            }
            // A deterministic spread rather than a table of real cities: every run has to compute
            // the same numbers.
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
     * <p>Plain Java rather than the {@code datagen} connector, and that is a demo decision rather
     * than a stylistic one: {@code datagen} costs about 100 microseconds a row, so two million rows
     * take three minutes to produce and the audience watches a progress-free terminal. This writes
     * the same two million rows in well under a second.
     */
    private static void writePoints(Path directory, long rows) throws IOException {
        if (Files.isDirectory(directory)) {
            return;
        }
        Files.createDirectories(directory);
        // Seeded, so every run of this example is over exactly the same coordinates.
        final Random random = new Random(20260925L);
        try (BufferedWriter out = Files.newBufferedWriter(directory.resolve("points.csv"))) {
            final StringBuilder line = new StringBuilder(48);
            for (long id = 0; id < rows; id++) {
                line.setLength(0);
                line.append(id)
                        .append(',')
                        .append(random.nextDouble() * 180.0 - 90.0)
                        .append(',')
                        .append(random.nextDouble() * 360.0 - 180.0)
                        .append('\n');
                out.write(line.toString());
            }
        }
    }

    private HaversineSQLExample() {}
}
