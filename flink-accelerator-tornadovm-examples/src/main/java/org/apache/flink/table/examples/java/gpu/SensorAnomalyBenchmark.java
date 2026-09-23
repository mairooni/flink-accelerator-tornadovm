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

/**
 * Anomaly screening over machine telemetry: score every reading, keep the few that look wrong.
 *
 * <h2>Why a second application, and why this one</h2>
 *
 * <p>Haversine is the project's reference workload and stays that way, but it is a poor vehicle for
 * one question: what the device compaction is worth. Its natural form has no {@code WHERE} at all,
 * and the filter added to it exists to exercise a mechanism rather than to answer a question
 * anybody asked — a "nearest depot within 2000 km" predicate keeps 23% of the rows because that is
 * the number that made the test interesting, not because a user wanted it.
 *
 * <p>Screening is different. It is <em>defined</em> by being selective: the whole point is that
 * almost everything is normal, so a few thousand rows out of millions survive. That is the regime a
 * compaction is for, and it is the regime in which the question "does packing the survivors on the
 * device pay for itself" has a real answer rather than a contrived one.
 *
 * <h2>The workload</h2>
 *
 * <p>A machine reports temperature, vibration, pressure and current. Each channel has an expected
 * value, and two of them are expected to <em>cycle</em> — a machine warms through its duty cycle
 * and vibrates more under load — so the baseline is a sinusoid of the time of day rather than a
 * constant. The score is the Euclidean norm of the four standardised deviations, plus a vibration
 * energy term and a thermal-stress interaction.
 *
 * <p>A real installation does not have one baseline. It has one per <em>operating mode</em> — idle,
 * ramping, loaded — and a reading is anomalous only if it fits none of them, so the score is the
 * smallest deviation across the modes. That is what {@code --modes} sets, and it is also what makes
 * the arithmetic per row adjustable without changing the shape of the query: more modes, more work,
 * same input read once.
 *
 * <p>Everything is ordinary SQL. There is no hint, no option and no syntax that a GPU could be
 * inferred from; the only thing the accelerated arm does differently is a deployment setting.
 *
 * <h2>What it reports, and why the exact columns matter</h2>
 *
 * <p>{@code SUM(score)} is a floating-point sum over survivors, and two arms can disagree on it
 * while having selected exactly the same rows — device transcendentals differ from {@code
 * java.lang.Math} by a few ulp, which moves every score slightly. So the query also returns {@code
 * COUNT(*)} and {@code SUM(sensor_id)}, an exact integer over the survivors, and {@code MIN} and
 * {@code MAX} of the score, which do not depend on the order of summation.
 *
 * <p>Together they separate the two questions that matter: <b>did the arms screen the same
 * readings</b> (count and integer sum, which must agree exactly) and <b>did they compute the same
 * scores</b> (min, max and sum, which may differ in the last few ulp). A benchmark that reported
 * only the floating-point sum could not tell a correctness failure from a rounding difference.
 *
 * <h2>Usage</h2>
 *
 * <pre>
 *   flink run SensorAnomalyBenchmark.jar --generate --rows 2000000 --data /tmp/readings
 *   flink run SensorAnomalyBenchmark.jar --data /tmp/readings --modes 6 --threshold 6.0 --gpu true
 * </pre>
 */
public final class SensorAnomalyBenchmark {

    /** Radians per second of the daily cycle: 2*pi / 86400. */
    private static final String PER_SECOND = "7.272205216643039E-5";

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
        if (parsed.fingerprint) {
            fingerprint(parsed);
            return;
        }
        if (!Files.isDirectory(Paths.get(parsed.data))) {
            System.out.printf("%s does not exist yet%n", parsed.data);
            generate(parsed);
        }

        System.out.printf(
                "data=%s  parallelism=%s  gpu=%s  runs=%d  modes=%d  threshold=%s%n",
                parsed.data,
                parsed.parallelism > 0 ? Integer.toString(parsed.parallelism) : "(default)",
                parsed.gpu,
                parsed.runs,
                parsed.modes,
                parsed.threshold);

        System.out.print(BenchmarkRun.environment());
        if (parsed.provenance) {
            // A full-file CRC, which is a benchmark-only scan of the whole input. Fine when
            // establishing what a dataset is; ruinous inside a timed run, where it would add a
            // second read of every byte to the thing being timed. Off for measurement.
            System.out.print(BenchmarkRun.dataset(parsed.data));
        } else {
            System.out.printf("  dataset:       %s  (provenance scan skipped)%n", parsed.data);
        }
        System.out.print(
                BenchmarkRun.job(
                        parsed.parallelism,
                        262_144,
                        parsed.rows,
                        "modes="
                                + parsed.modes
                                + "  threshold="
                                + parsed.threshold
                                + "  gpu="
                                + parsed.gpu));

        BenchmarkRun timings = new BenchmarkRun("wall time", parsed.runs > 1 ? 1 : 0);

        Row first = null;
        for (int run = 1; run <= parsed.runs; run++) {
            long start = System.nanoTime();
            Row seen = query(parsed);
            long elapsed = System.nanoTime() - start;
            timings.record(elapsed);
            System.out.printf("run %2d  %10.0f ms  %s%n", run, elapsed / 1e6, seen);
            if (first == null) {
                first = seen;
            } else {
                agree(first, seen);
            }
        }
        System.out.print(timings.summary());
    }

    /**
     * Two runs of the same arm must agree exactly, including the floating-point columns.
     *
     * <p>Within one arm there is no source of divergence at all: the same code reads the same bytes
     * in the same order. A difference here is not rounding, it is nondeterminism, and it would make
     * every comparison between arms meaningless.
     */
    private static void agree(Row first, Row later) {
        if (!first.toString().equals(later.toString())) {
            throw new IllegalStateException(
                    "the same arm returned different answers on two runs:\n  "
                            + first
                            + "\n  "
                            + later);
        }
    }

    private static void generate(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        // Random channels around plausible operating points, and a phase that sweeps the duty
        // cycle. A sequence for the id keeps the exact integer checksum meaningful.
        env.executeSql(
                "CREATE TABLE Source (\n"
                        + "  sensor_id INT,\n"
                        + "  ts BIGINT,\n"
                        + "  phase DOUBLE,\n"
                        + "  temp DOUBLE,\n"
                        + "  vibration DOUBLE,\n"
                        + "  pressure DOUBLE,\n"
                        + "  amps DOUBLE\n"
                        + ") WITH (\n"
                        + "  'connector' = 'datagen',\n"
                        + "  'number-of-rows' = '"
                        + args.rows
                        + "',\n"
                        + "  'fields.sensor_id.kind' = 'sequence',\n"
                        + "  'fields.sensor_id.start' = '0',\n"
                        + "  'fields.sensor_id.end' = '"
                        + (args.rows - 1)
                        + "',\n"
                        + "  'fields.ts.min' = '1700000000', 'fields.ts.max' = '1700086400',\n"
                        + "  'fields.phase.min' = '0.0', 'fields.phase.max' = '86400.0',\n"
                        + "  'fields.temp.min' = '5.0', 'fields.temp.max' = '45.0',\n"
                        + "  'fields.vibration.min' = '0.0', 'fields.vibration.max' = '1.6',\n"
                        + "  'fields.pressure.min' = '95.0', 'fields.pressure.max' = '108.0',\n"
                        + "  'fields.amps.min' = '2.5', 'fields.amps.max' = '6.0'\n"
                        + ")");
        env.executeSql(readings(args.data, args.format));

        System.out.printf("writing %,d rows to %s%n", args.rows, args.data);
        long start = System.nanoTime();
        env.executeSql(
                        "INSERT INTO Readings "
                                + "SELECT sensor_id, ts, phase, temp, vibration, pressure, amps "
                                + "FROM Source")
                .await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    /**
     * Rewrites the same rows into another format, so two formats can be compared on one dataset.
     *
     * <p>Generating twice does not do it. {@code datagen}'s random fields are not seedable, so a
     * second generation produces a different draw — different checksum, different anomaly count —
     * and a CSV-against-Parquet comparison built that way confounds the format with the data. This
     * reads one dataset and writes it out again, so the two differ in encoding and in nothing else.
     */
    private static void convert(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        env.executeSql(readings(args.data, args.format));
        // sink.parallelism, not parallelism.default: the source is one file and therefore one
        // split, so the whole pipeline chains at parallelism 1 and the sink writes one file
        // whatever the job parallelism says. Setting the sink's own parallelism forces a
        // redistribution in front of it, which is what produces N files -- and on a local
        // filesystem N files is N splits for whoever reads it back.
        String target = readings(args.convertTo, args.convertFormat).replace("Readings", "Copy");
        if (args.convertFiles > 1) {
            target =
                    target.replace(
                            "  'format' = '" + args.convertFormat + "'",
                            "  'format' = '"
                                    + args.convertFormat
                                    + "',\n"
                                    + "  'sink.parallelism' = '"
                                    + args.convertFiles
                                    + "'");
        }
        env.executeSql(target);
        System.out.printf(
                "converting %s (%s) -> %s (%s)%n",
                args.data, args.format, args.convertTo, args.convertFormat);
        long start = System.nanoTime();
        // sensor_id is a dense sequence from the generator, so a subset is exactly reproducible
        // and its provenance is one predicate rather than a second unseeded generation.
        final String where = args.subset > 0 ? " WHERE sensor_id < " + args.subset : "";
        env.executeSql("INSERT INTO Copy SELECT * FROM Readings" + where).await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    /**
     * Writes the selected rows themselves, so membership can be checked instead of inferred.
     *
     * <p>A count and an integer id-sum agreeing is evidence and not proof: two different sets of
     * the same size can share both. The only way to establish that two arms screened the same
     * readings is to compare the readings, so this emits one line per survivor — its id and its
     * score at full precision — and leaves the comparison to whatever reads the files.
     */
    private static void dump(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        env.executeSql(readings(args.data, args.format));
        if (args.gpu) {
            env.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
        }
        env.executeSql(
                "CREATE TABLE Selected (sensor_id INT NOT NULL, score DOUBLE NOT NULL) WITH (\n"
                        + "  'connector' = 'filesystem',\n"
                        + "  'path' = '"
                        + args.dump
                        + "',\n"
                        + "  'format' = 'csv'\n"
                        + ")");
        System.out.printf("writing selected rows to %s%n", args.dump);
        long start = System.nanoTime();
        // Unfiltered dumps every score, so the boundary can be examined from both sides and the
        // error reported over the whole evaluated population rather than over the survivors,
        // which are a biased sample of it.
        env.executeSql(
                        "INSERT INTO Selected\n"
                                + "SELECT sensor_id, score FROM (\n"
                                + "  SELECT sensor_id, "
                                + score(args.modes)
                                + " AS score\n"
                                + "  FROM Readings\n"
                                + ")"
                                + (args.dumpAll ? "" : "\nWHERE score > " + args.threshold))
                .await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    /**
     * A logical fingerprint of the table, independent of how it is stored.
     *
     * <p>A file checksum answers "are these the same bytes", which is the wrong question once the
     * same rows have been written as CSV and as Parquet, or as one file and as eight. Every column
     * here is exact and order-independent — a count, an integer sum, and the extremes of each
     * measured channel — so two representations of one logical dataset must produce identical
     * output whatever their layout, and any difference is a difference in the data.
     *
     * <p>It is deliberately not a sum of the doubles: that would depend on the order the rows were
     * added, which is exactly what changes when a dataset is re-partitioned.
     */
    private static void fingerprint(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        env.executeSql(readings(args.data, args.format));
        final String q =
                "SELECT COUNT(*) AS rows_seen, SUM(sensor_id) AS id_sum,\n"
                        + "  MIN(phase) AS min_phase, MAX(phase) AS max_phase,\n"
                        + "  MIN(temp) AS min_temp, MAX(temp) AS max_temp,\n"
                        + "  MIN(vibration) AS min_vib, MAX(vibration) AS max_vib,\n"
                        + "  MIN(pressure) AS min_p, MAX(pressure) AS max_p,\n"
                        + "  MIN(amps) AS min_a, MAX(amps) AS max_a,\n"
                        + "  MIN(ts) AS min_ts, MAX(ts) AS max_ts\n"
                        + "FROM Readings";
        try (CloseableIterator<Row> rows = env.sqlQuery(q).execute().collect()) {
            System.out.printf("fingerprint %s (%s): %s%n", args.data, args.format, rows.next());
        }
    }

    private static Row query(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        env.executeSql(readings(args.data, args.format));

        if (args.gpu) {
            env.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
            if (args.maxAccelParallelism > 0) {
                // Raising the containment ceiling deliberately, for measurement. The default of 1
                // exists because parallelism above it has failed intermittently with a CUDA launch
                // error whose cause is not established; a benchmark that silently ran on the CPU
                // instead would be measuring nothing.
                env.getConfig()
                        .getConfiguration()
                        .setString(
                                "table.exec.accelerator.max-parallelism",
                                Integer.toString(args.maxAccelParallelism));
            }
        }

        // Ordinary SQL. The WHERE is the screen; the exact columns beside SUM(score) are what let
        // two arms be compared on membership rather than only on a floating-point total.
        final String query =
                "SELECT COUNT(*) AS anomalies,\n"
                        + "       SUM(sensor_id) AS id_sum,\n"
                        + "       MIN(score) AS min_score,\n"
                        + "       MAX(score) AS max_score,\n"
                        + "       SUM(score) AS total_score\n"
                        + "FROM (\n"
                        + "  SELECT sensor_id, "
                        + score(args.modes)
                        + " AS score\n"
                        + "  FROM Readings\n"
                        + ")\n"
                        + "WHERE score > "
                        + args.threshold;

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
     * The smallest deviation across the operating modes a machine might be in.
     *
     * <p>One mode is a plain deviation score. Several is the question a screening system actually
     * asks — a reading is anomalous only if it fits <em>none</em> of the ways the machine is
     * allowed to behave — and it is also how the arithmetic per row is raised without reading the
     * input more than once, which is the only way an end-to-end measurement can see the operator
     * rather than the source.
     */
    private static String score(int modes) {
        if (modes == 1) {
            return deviation(20.0, 8.0, 0.5, 0.2, 101.3, 4.2);
        }
        StringBuilder sb = new StringBuilder("LEAST(");
        for (int m = 0; m < modes; m++) {
            if (m > 0) {
                sb.append(",\n         ");
            }
            // A deterministic spread of operating points, so every run scores against the same
            // baselines and two arms can be compared at all.
            sb.append(
                    deviation(
                            18.0 + m * 2.0,
                            6.0 + m * 0.5,
                            0.4 + m * 0.05,
                            0.15 + m * 0.02,
                            100.0 + m * 0.8,
                            3.6 + m * 0.3));
        }
        return sb.append(")").toString();
    }

    /**
     * Standardised distance from one operating baseline.
     *
     * <p>Temperature and vibration are expected to cycle with the duty cycle, so their baselines
     * are sinusoids of the phase rather than constants; pressure and current are expected to hold.
     * The vibration energy term is what separates a machine that is merely off-baseline from one
     * that is shaking, and the thermal-stress term couples two channels that a single-channel
     * threshold would never catch together.
     */
    private static String deviation(
            double tempMean,
            double tempSwing,
            double vibMean,
            double vibSwing,
            double pressureMean,
            double ampsMean) {
        final String w = "(phase * " + PER_SECOND + ")";
        final String tempBase = "(" + tempMean + " + " + tempSwing + " * SIN(" + w + "))";
        final String vibBase = "(" + vibMean + " + " + vibSwing + " * COS(" + w + "))";
        return "SQRT("
                + "POWER((temp - "
                + tempBase
                + ") / 2.5, 2)"
                + " + POWER((vibration - "
                + vibBase
                + ") / 0.15, 2)"
                + " + POWER((pressure - "
                + pressureMean
                + ") / 1.8, 2)"
                + " + POWER((amps - "
                + ampsMean
                + ") / 0.35, 2)"
                + ")"
                + " + 0.5 * LN(1.0 + vibration * vibration)"
                + " + 0.01 * ABS(temp - pressure * 0.2)";
    }

    private static String readings(String path, String format) {
        return "CREATE TABLE Readings (\n"
                + "  sensor_id INT NOT NULL,\n"
                + "  ts BIGINT NOT NULL,\n"
                + "  phase DOUBLE NOT NULL,\n"
                + "  temp DOUBLE NOT NULL,\n"
                + "  vibration DOUBLE NOT NULL,\n"
                + "  pressure DOUBLE NOT NULL,\n"
                + "  amps DOUBLE NOT NULL\n"
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
                    .setString("parallelism.default", Integer.toString(args.parallelism));
        }
        return env;
    }

    private static final class Args {
        private String data = "/tmp/flink-gpu-readings";
        private int rows = 2_000_000;
        private int parallelism = -1;
        private int runs = 10;
        private String format = "csv";
        private boolean gpu;
        private boolean generate;
        private boolean explain;
        private int modes = 6;

        /** Chosen so that a small, screening-shaped fraction of readings survives. */
        private double threshold = 6.0;

        /** Where to rewrite this dataset, for a matched cross-format comparison. */
        private String convertTo;

        private String convertFormat = "parquet";

        /** How many files the converted copy should be written as. */
        private int convertFiles = 1;

        /** Where to write the selected rows themselves, for exact membership checking. */
        private String dump;

        /** Dump every score rather than only the survivors, for boundary analysis. */
        private boolean dumpAll;

        /** Keep only {@code sensor_id < subset} when converting, for reproducible smaller sizes. */
        private long subset;

        /**
         * Whether to CRC the whole input before running.
         *
         * <p>On for establishing what a dataset is, off for anything timed: it is a benchmark-only
         * full scan and inside a measured run it would time itself.
         */
        private boolean provenance = true;

        /** Print a layout-independent logical fingerprint of the table and exit. */
        private boolean fingerprint;

        /** Raises table.exec.accelerator.max-parallelism for measurement; 0 leaves the default. */
        private int maxAccelParallelism;

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
                } else if ("--generate".equals(flag)) {
                    args.generate = true;
                } else if ("--modes".equals(flag)) {
                    args.modes = Integer.parseInt(argv[++i]);
                } else if ("--threshold".equals(flag)) {
                    args.threshold = Double.parseDouble(argv[++i]);
                } else if ("--convert-to".equals(flag)) {
                    args.convertTo = argv[++i];
                } else if ("--convert-format".equals(flag)) {
                    args.convertFormat = argv[++i];
                } else if ("--convert-files".equals(flag)) {
                    args.convertFiles = Integer.parseInt(argv[++i]);
                } else if ("--dump".equals(flag)) {
                    args.dump = argv[++i];
                } else if ("--dump-all".equals(flag)) {
                    args.dumpAll = true;
                } else if ("--subset".equals(flag)) {
                    args.subset = Long.parseLong(argv[++i]);
                } else if ("--no-provenance".equals(flag)) {
                    args.provenance = false;
                } else if ("--fingerprint".equals(flag)) {
                    args.fingerprint = true;
                } else if ("--max-accel-parallelism".equals(flag)) {
                    args.maxAccelParallelism = Integer.parseInt(argv[++i]);
                } else if ("--explain".equals(flag)) {
                    args.explain = true;
                } else {
                    throw new IllegalArgumentException("unknown argument " + flag);
                }
            }
            return args;
        }
    }

    private SensorAnomalyBenchmark() {}
}
