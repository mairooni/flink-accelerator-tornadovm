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
import org.apache.flink.util.CloseableIterator;

import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * End-to-end benchmark: great-circle distance over a Parquet table, with and without GPU offload.
 *
 * <h2>Why haversine</h2>
 *
 * <p>The measurement this project lacked was an end-to-end one. Earlier attempts used {@code
 * datagen}, which costs roughly 100 µs per row and buries the operator no matter how fast the
 * kernel is, and an expression of two flops, which is far below the cost floor and has nothing for
 * the GPU to win back.
 *
 * <p>The format is a parameter. {@code parquet} is what the operator wants -- it hands over {@code
 * ColumnarRowData}, the layout the columnar gather exists for -- but Flink's Parquet format needs
 * Hadoop on the cluster classpath, which the distribution does not ship. {@code csv} is the default
 * because it works against an unmodified distribution; it costs more per row and yields row data
 * rather than columnar, so a run with it measures the source more than a Parquet run does.
 *
 * <p>Haversine fixes both halves. It is a real query — "how far is each point from here" over a
 * table of coordinates — and it is arithmetically heavy: two {@code SIN}, two {@code COS}, two
 * {@code POWER}, a {@code SQRT} and an {@code ASIN}, which the planner's estimator weighs at well
 * above the default floor of 96. Reading from Parquet instead of generating rows removes the source
 * from the critical path and hands the operator {@code ColumnarRowData}, the layout the columnar
 * gather exists for.
 *
 * <h2>Why the query aggregates</h2>
 *
 * <p>Collecting one row per input would make the client transfer dominate the wall time and measure
 * nothing useful. Aggregating keeps the output at a single row while still forcing every distance
 * to be computed. The aggregate itself stays on the CPU; only the {@code Calc} is offloaded.
 *
 * <h2>Usage</h2>
 *
 * <pre>
 *   # once: write the input table (about three minutes for 2M rows -- datagen is the slow part)
 *   flink run examples/table/HaversineBenchmark.jar --generate --rows 2000000 --data /tmp/points
 *
 *   # then: measure
 *   flink run examples/table/HaversineBenchmark.jar --data /tmp/points --depots 20 --gpu true
 * </pre>
 *
 * <p>Or let {@code scripts/run-haversine.sh} in {@code flink-table-gpu-runtime} do all of it, which
 * is the supported path: it starts the cluster, generates the input if it is missing, and runs both
 * sides.
 */
public final class HaversineBenchmark {

    /** Degrees to radians; {@code RADIANS} is not in the generator's function set. */
    private static final String TO_RAD = "0.017453292519943295";

    /** Mean Earth radius in km, doubled — the 2r of the haversine formula. */
    private static final String DIAMETER = "12742.0";

    /** Reference point a single-depot run measures from: Manchester. */
    private static final double LAT0 = 53.4808;

    private static final double LON0 = -2.2426;

    public static void main(String[] args) throws Exception {
        final Args parsed = Args.parse(args);

        if (parsed.generate) {
            generate(parsed);
            return;
        }

        // Generate on demand rather than failing inside the file source, which reports only
        // "Could not enumerate file splits" and says nothing about what should have created the
        // directory. Running this class straight from an IDE, with no arguments, should work.
        if (!Files.isDirectory(Paths.get(parsed.data))) {
            System.out.printf("%s does not exist yet%n", parsed.data);
            generate(parsed);
        }

        System.out.printf(
                "data=%s  parallelism=%s  gpu=%s  runs=%d  depots=%d%n",
                parsed.data,
                parsed.parallelism > 0 ? Integer.toString(parsed.parallelism) : "(default)",
                parsed.gpu,
                parsed.runs,
                parsed.depots);
        if (parsed.baseline) {
            System.out.println("baseline: reading and counting only, no haversine");
        }

        Row result = null;
        for (int run = 1; run <= parsed.runs; run++) {
            long start = System.nanoTime();
            Row seen = query(parsed);
            double millis = (System.nanoTime() - start) / 1e6;
            System.out.printf("run %2d  %10.0f ms  %s%n", run, millis, seen);
            if (result == null) {
                result = seen;
            } else {
                agree(result, seen);
            }
        }
    }

    /**
     * Checks two runs agree, to within floating-point reassociation.
     *
     * <p>Not bit-for-bit: {@code SUM} over a parallel plan combines partial sums in whatever order
     * the subtasks finish, and floating-point addition is not associative, so the last bits of the
     * total move between runs at parallelism above one. The device reassociates differently again.
     * A tolerance is the honest check here; exact equality would fail on correct results and say
     * nothing about the ones that matter.
     */
    private static void agree(Row first, Row second) {
        if (!first.getField(0).equals(second.getField(0))) {
            throw new IllegalStateException(
                    "runs saw different row counts: " + first + " then " + second);
        }
        double a = ((Number) first.getField(1)).doubleValue();
        double b = ((Number) second.getField(1)).doubleValue();
        double relative = a == 0.0 ? Math.abs(b) : Math.abs((a - b) / a);
        if (relative > 1e-12) {
            throw new IllegalStateException(
                    "runs disagree by "
                            + relative
                            + " relative: "
                            + first
                            + " then "
                            + second
                            + "; that is far beyond floating-point reassociation");
        }
    }

    /** Writes the input table once, so the benchmark itself never pays for generating rows. */
    private static void generate(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        env.executeSql(
                "CREATE TABLE Source (\n"
                        + "  id INT,\n"
                        + "  lat DOUBLE,\n"
                        + "  lon DOUBLE\n"
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
                        + "  'fields.lat.min' = '-90.0',\n"
                        + "  'fields.lat.max' = '90.0',\n"
                        + "  'fields.lon.min' = '-180.0',\n"
                        + "  'fields.lon.max' = '180.0'\n"
                        + ")");
        env.executeSql(points(args.data, args.format));

        System.out.printf("writing %,d rows to %s%n", args.rows, args.data);
        long start = System.nanoTime();
        env.executeSql("INSERT INTO Points SELECT id, lat, lon FROM Source").await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    private static Row query(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        env.executeSql(points(args.data, args.format));

        if (args.gpu) {
            env.getConfig().getConfiguration().setString("table.exec.gpu-offload.enabled", "true");
            if (!args.fuseAggregate) {
                // The unfused arm: the Calc still runs on the device, the SUM above it still runs
                // on the CPU, and the projected rows travel between them. That is what fusing is
                // measured against, and both arms have to be one session apart at most.
                env.getConfig()
                        .getConfiguration()
                        .setString("table.exec.gpu-offload.fuse-aggregate", "false");
            }
            if (args.requireDevice) {
                // Asks the scheduler for a slot that declares the resource, rather than taking
                // whatever slot arrives and finding out on the TaskManager. Requesting one means
                // specifying the whole slot: Flink rejects a slot sharing group that names an
                // external resource without also naming cpu cores and task heap.
                Configuration conf = env.getConfig().getConfiguration();
                conf.setString("table.exec.gpu-offload.require-device", "true");
                conf.setString("table.exec.gpu-offload.device.cpu-cores", args.deviceCpuCores);
                conf.setString("table.exec.gpu-offload.device.task-heap", args.deviceTaskHeap);
            }
        }

        // Reading and counting, with almost no arithmetic: everything the job costs that is not the
        // expression. Subtracting it from a full run is what separates the operator from the
        // source, and so what says whether a speedup on the operator can show up end to end.
        //
        // It sums lat + lon rather than lat alone so that both columns are read: the filesystem
        // source pushes projection down, and a baseline touching one column would understate what
        // the real query pays to read its input.
        final String query =
                args.baseline
                        ? "SELECT COUNT(*) AS rows_seen, SUM(lat + lon) AS total_km FROM Points"
                        : "SELECT COUNT(*) AS rows_seen, SUM(km) AS total_km\n"
                                + "FROM (\n"
                                + "  SELECT "
                                + nearest(args.depots)
                                + " AS km\n"
                                + "  FROM Points\n"
                                + ")";

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
     * Distance to the nearest of {@code n} reference points, as a single expression.
     *
     * <p>With {@code n == 1} this is a plain great-circle distance. Above that it is the query a
     * logistics or retail system actually asks -- how far is each address from its nearest depot --
     * and it is the reason this benchmark can show anything end to end. One haversine is about 173
     * weighted ops against a floor of 96, but it is only a fifth of the job's wall time, so Amdahl
     * caps the whole query at about 1.2x however fast the device is. Twenty of them, with the input
     * read exactly once and still a single output column, moves the arithmetic to where the
     * measurement can see it.
     *
     * <p>Written without {@code RADIANS} -- the kernel generator maps a fixed set of functions onto
     * {@code TornadoMath} and that is not among them, whereas multiplying by a literal is.
     */
    private static String nearest(int n) {
        if (n == 1) {
            return haversine(LAT0, LON0);
        }
        StringBuilder sb = new StringBuilder("LEAST(");
        for (int i = 0; i < n; i++) {
            if (i > 0) {
                sb.append(",\n         ");
            }
            // A deterministic spread rather than a table of real cities: the point is the
            // arithmetic, and every run has to compare against the same numbers.
            sb.append(haversine(-60.0 + i * (120.0 / n), -180.0 + i * (360.0 / n)));
        }
        return sb.append(")").toString();
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
                + ") / 2.0), 2)"
                + "))";
    }

    private static String points(String path, String format) {
        return "CREATE TABLE Points (\n"
                + "  id INT,\n"
                + "  lat DOUBLE,\n"
                + "  lon DOUBLE\n"
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
        /**
         * Where the input lives. Left null so the default can depend on {@code rows} and {@code
         * format}, which are parsed from the same command line -- and so that it matches what
         * run-haversine.sh writes, rather than being a second name for the same thing.
         */
        private String data;

        private int rows = 2_000_000;
        private int parallelism = -1;
        private int runs = 10;
        private String format = "csv";
        private boolean gpu;
        private boolean requireDevice;
        private boolean fuseAggregate = true;
        private String deviceCpuCores = "1.0";
        private String deviceTaskHeap = "2g";
        private boolean generate;
        private boolean explain;
        private boolean baseline;
        private int depots = 1;

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
                } else if ("--fuse-aggregate".equals(flag)) {
                    args.fuseAggregate = Boolean.parseBoolean(argv[++i]);
                } else if ("--require-device".equals(flag)) {
                    args.requireDevice = Boolean.parseBoolean(argv[++i]);
                } else if ("--device-cpu-cores".equals(flag)) {
                    args.deviceCpuCores = argv[++i];
                } else if ("--device-task-heap".equals(flag)) {
                    args.deviceTaskHeap = argv[++i];
                } else if ("--gpu".equals(flag)) {
                    args.gpu = Boolean.parseBoolean(argv[++i]);
                } else if ("--generate".equals(flag)) {
                    args.generate = true;
                } else if ("--depots".equals(flag)) {
                    args.depots = Integer.parseInt(argv[++i]);
                } else if ("--baseline".equals(flag)) {
                    args.baseline = true;
                } else if ("--explain".equals(flag)) {
                    args.explain = true;
                } else {
                    throw new IllegalArgumentException("unknown argument " + flag);
                }
            }
            if (args.data == null) {
                args.data = "/tmp/flink-gpu-points-" + args.rows + "-" + args.format;
            }
            return args;
        }
    }

    private HaversineBenchmark() {}
}
