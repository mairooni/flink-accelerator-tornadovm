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
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Two analytical stages over one Parquet table, which is the shape that exercises all three.
 *
 * <p>{@link HaversineBenchmark} has one accelerated stage sitting directly on the source, so its
 * staging is always columnar and the row-major path is never taken. This query has two, with a
 * shuffle between them, and the second stage is where a real pipeline spends most of its life:
 *
 * <ol>
 *   <li><b>Reduce.</b> Per-vehicle centroid — {@code AVG(lat), AVG(lon) GROUP BY vehicle}. Reads
 *       the Parquet source directly, so the columnar bulk gather applies (<b>E</b>), and the
 *       projection fuses with the local aggregate into one task graph (<b>F</b>).
 *   <li><b>Shuffle.</b> Flink hashes by vehicle. Whatever the source produced, what comes out the
 *       other side is {@code BinaryRowData}: a serialised, fixed-width, row-major record.
 *   <li><b>Score.</b> Distance from each centroid to the nearest depot, then a total. Its input is
 *       row-major binary rows, which is the one layout the device transpose exists for (<b>C</b>).
 * </ol>
 *
 * <p>Both projections read only {@code DOUBLE} columns, which is what lets the block path apply:
 * the block is a {@code DoubleArray} and a narrower field would need a bit reinterpretation on the
 * way out. That is a property of this query rather than a trick — a centroid is a pair of doubles.
 *
 * <p>{@code --vehicles} sets how many rows reach the second stage, and it is the parameter that
 * matters. At a thousand vehicles the second stage is a rounding error and this measures the first;
 * at two million it is half the job.
 */
public final class FleetCentroidBenchmark {

    private static final String TO_RAD = "0.017453292519943295";
    private static final String DIAMETER = "12742.0";

    public static void main(String[] args) throws Exception {
        final Args parsed = Args.parse(args);

        System.out.printf(
                "data=%s  gpu=%s  vehicles=%d  depots=%d  runs=%d  block-staging=%s%n",
                parsed.data,
                parsed.gpu,
                parsed.vehicles,
                parsed.depots,
                parsed.runs,
                System.getProperty("flink.accelerator.tornadovm.blockStaging", "false"));

        final List<Double> times = new ArrayList<>();
        Row result = null;
        for (int run = 1; run <= parsed.runs; run++) {
            final long start = System.nanoTime();
            result = run(parsed);
            final double millis = (System.nanoTime() - start) / 1e6;
            times.add(millis);
            System.out.printf("  run %2d  %9.1f ms%n", run, millis);
        }

        System.out.println("result: " + result);
        if (times.size() > 1) {
            final List<Double> warm = new ArrayList<>(times.subList(1, times.size()));
            Collections.sort(warm);
            System.out.printf(
                    "first %.1f ms   warm median %.1f ms   (n=%d, first excluded)%n",
                    times.get(0), warm.get(warm.size() / 2), warm.size());
        }
    }

    private static Row run(Args args) throws Exception {
        final TableEnvironment env =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        final var conf = env.getConfig().getConfiguration();
        if (args.parallelism > 0) {
            conf.setString(
                    "table.exec.resource.default-parallelism", Integer.toString(args.parallelism));
        }
        // Both halves have to be visible to the planner as two phases, or the shuffle this query
        // is built around does not appear.
        conf.setString("table.optimizer.agg-phase-strategy", "TWO_PHASE");

        env.executeSql(
                "CREATE TABLE Points (\n"
                        + "  id INT NOT NULL,\n"
                        + "  lat DOUBLE NOT NULL,\n"
                        + "  lon DOUBLE NOT NULL\n"
                        + ") WITH (\n"
                        + "  'connector' = 'filesystem',\n"
                        + "  'path' = '"
                        + args.data
                        + "',\n"
                        + "  'format' = '"
                        + args.format
                        + "'\n"
                        + ")");

        if (args.gpu) {
            conf.setString("table.exec.accelerator.enabled", "true");
            conf.setString("table.exec.accelerator.approximate-projections", "true");
            conf.setString("table.exec.accelerator.fuse-aggregate", Boolean.toString(args.fuse));
            conf.setString("table.exec.accelerator.min-speedup", "0.1");
        }

        // MOD has no accelerator IR and one unsupported operator refuses the whole projection, so
        // the key is written as the subtraction it expands to.
        final String vehicle = "(id - (id / " + args.vehicles + ") * " + args.vehicles + ")";

        final StringBuilder least = new StringBuilder("LEAST(");
        for (int i = 0; i < args.depots; i++) {
            if (i > 0) {
                least.append(", ");
            }
            least.append(
                    haversine(
                            -60.0 + i * (120.0 / args.depots), -180.0 + i * (360.0 / args.depots)));
        }
        least.append(")");

        // A sort between the stages, when asked for. Flink's SortOperator emits BinaryRowData --
        // fixed-width, row-major, one contiguous run a row -- which is the layout the device
        // transpose exists for and the one an aggregate's JoinedRowData is not.
        // A sort with a LIMIT survives the optimiser, where a bare subquery ORDER BY does not --
        // Calcite drops an inner sort whose order nothing observes. Flink's SortOperator emits
        // BinaryRowData, so this is the shape that hands the projection row-major binary rows.
        if (args.topn > 0) {
            final String scored = "SELECT lat, lon FROM Points ORDER BY lat LIMIT " + args.topn;
            final String sql =
                    "SELECT COUNT(*) AS vehicles, SUM(km) AS total_km FROM (\n"
                            + "  SELECT "
                            + least
                            + " AS km FROM (\n    "
                            + scored
                            + "\n  )\n"
                            + ")";
            if (args.explain) {
                System.out.println(env.explainSql(sql));
            }
            try (CloseableIterator<Row> it = env.sqlQuery(sql).execute().collect()) {
                return it.next();
            }
        }

        final String inner =
                args.order
                        ? "SELECT vehicle, lat, lon FROM (\n"
                                + "      SELECT "
                                + vehicle
                                + " AS vehicle, AVG(lat) AS lat, AVG(lon) AS lon\n"
                                + "      FROM Points GROUP BY "
                                + vehicle
                                + "\n"
                                + "    ) ORDER BY vehicle"
                        : "SELECT "
                                + vehicle
                                + " AS vehicle, AVG(lat) AS lat, AVG(lon) AS lon\n"
                                + "    FROM Points GROUP BY "
                                + vehicle;

        final String sql =
                "SELECT COUNT(*) AS vehicles, SUM(km) AS total_km FROM (\n"
                        + "  SELECT "
                        + least
                        + " AS km FROM (\n    "
                        + inner
                        + "\n  )\n"
                        + ")";

        if (args.explain) {
            System.out.println(env.explainSql(sql));
        }
        try (CloseableIterator<Row> it = env.sqlQuery(sql).execute().collect()) {
            return it.next();
        }
    }

    private static String haversine(double lat0Deg, double lon0Deg) {
        final String lat = "(lat * " + TO_RAD + ")";
        final String lon = "(lon * " + TO_RAD + ")";
        final String lat0 = "(" + lat0Deg + " * " + TO_RAD + ")";
        final String lon0 = "(" + lon0Deg + " * " + TO_RAD + ")";
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
                + ") / 2.0), 2)))";
    }

    private static final class Args {
        private String data = "/tmp/flink-gpu-points-8m-parquet";
        private String format = "parquet";
        private int parallelism = 1;
        private int runs = 1;
        private int vehicles = 2_000_000;
        private int depots = 20;
        private boolean gpu;
        private boolean fuse;
        private boolean order;
        private int topn;
        private boolean explain;

        static Args parse(String[] argv) {
            final Args args = new Args();
            for (int i = 0; i < argv.length; i++) {
                final String flag = argv[i];
                if ("--data".equals(flag)) {
                    args.data = argv[++i];
                } else if ("--format".equals(flag)) {
                    args.format = argv[++i];
                } else if ("--parallelism".equals(flag)) {
                    args.parallelism = Integer.parseInt(argv[++i]);
                } else if ("--runs".equals(flag)) {
                    args.runs = Integer.parseInt(argv[++i]);
                } else if ("--vehicles".equals(flag)) {
                    args.vehicles = Integer.parseInt(argv[++i]);
                } else if ("--depots".equals(flag)) {
                    args.depots = Integer.parseInt(argv[++i]);
                } else if ("--gpu".equals(flag)) {
                    args.gpu = Boolean.parseBoolean(argv[++i]);
                } else if ("--fuse".equals(flag)) {
                    args.fuse = Boolean.parseBoolean(argv[++i]);
                } else if ("--order".equals(flag)) {
                    args.order = Boolean.parseBoolean(argv[++i]);
                } else if ("--topn".equals(flag)) {
                    args.topn = Integer.parseInt(argv[++i]);
                } else if ("--explain".equals(flag)) {
                    args.explain = true;
                } else {
                    throw new IllegalArgumentException("unknown argument " + flag);
                }
            }
            return args;
        }
    }

    private FleetCentroidBenchmark() {}
}
