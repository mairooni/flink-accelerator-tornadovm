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

package org.apache.flink.table.examples.java.gpu.llm;

import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Locale;

/**
 * The half of the pipeline both arms share: the data, the SQL, and the prompt.
 *
 * <h2>The application</h2>
 *
 * <p>A fleet of machines reports telemetry. Every reading is scored against each of the operating
 * modes the machine is allowed to be in, and it is anomalous only if it fits none of them. The
 * survivors are rolled up per machine, and the roll-up — a few dozen lines — is handed to a
 * language model, which writes the maintenance note a human would otherwise have written.
 *
 * <p>That shape is the reason the comparison is worth making. The scoring reads millions of rows
 * and does hundreds of floating-point operations on each, which is the regime where a GPU wins; the
 * note is a few hundred tokens, which is the regime where the inference engine decides. An
 * application with both in it is where "the preprocessing is faster" and "the engines are at
 * parity" have to be settled against one number.
 *
 * <h2>Why the score uses no transcendentals</h2>
 *
 * <p>Every operation in it — add, subtract, multiply, divide, square root, absolute value, minimum
 * — is correctly rounded in IEEE 754. A device and a CPU must therefore produce <em>bit-identical
 * scores</em>, so the two arms can be compared on the text they generated and not merely on the
 * time they took, and a divergence is a defect rather than the last-ulp drift that {@code SIN} and
 * {@code LN} would licence.
 *
 * <p>It also removes an obstacle that would otherwise decide the experiment for us. A projection
 * whose value can drift is refused when that value reaches a filter, a grouping or a sort, and this
 * one reaches all three. Written in exact arithmetic it is eligible on its own merits, and the
 * arithmetic per row is raised by the number of modes rather than by reaching for a function whose
 * device implementation is free to differ.
 *
 * <h2>Nothing in the query is about a GPU</h2>
 *
 * <p>The SQL is the same text in both arms, character for character. The only difference is a
 * deployment setting, {@code table.exec.accelerator.enabled}, which a cluster operator sets and a
 * query author never sees. {@code NOT NULL} in the DDL is the one thing the author writes, and it
 * is there because a kernel has no representation for a null, not because a GPU is involved.
 */
public final class TelemetryTriage {

    private TelemetryTriage() {}

    /** Seconds in a day; the phase column sweeps one duty cycle. */
    private static final String INVERSE_DAY = "1.1574074074074073E-5";

    // ------------------------------------------------------------------------------------------
    // the query
    // ------------------------------------------------------------------------------------------

    /**
     * Distance from one operating baseline, in exact arithmetic.
     *
     * <p>Temperature and vibration are expected to follow the duty cycle, so their baselines are a
     * function of the phase rather than constants. The cycle is a quadratic hump rather than a
     * sinusoid for the reason in the class comment: {@code u * (1 - u) * 4} is exact, and {@code
     * SIN} is not.
     *
     * <p>It is a <em>squared</em> distance, and deliberately. Writing {@code SQRT(...)} is the
     * natural spelling and it is refused, because Calcite rewrites {@code SQRT(x)} to {@code
     * POWER(x, 0.5)} and {@code POWER} is in the family the planner will not let a device evaluate
     * when the value reaches a filter — which here it does. Squaring the threshold instead leaves
     * an ordering that is identical, arithmetic that is still exactly rounded, and a projection the
     * accelerator will take without being asked to relax anything. The explain output names the
     * refusal precisely enough to find this, which is why the {@code GPU Offload} section exists.
     */
    private static String deviation(int mode) {
        final String u = "(phase * " + INVERSE_DAY + ")";
        final String cycle = "(" + u + " * (1.0 - " + u + ") * 4.0)";
        final String tempBase =
                "(" + (18.0 + mode * 2.0) + " + " + (6.0 + mode * 0.5) + " * " + cycle + ")";
        final String vibBase =
                "(" + (0.4 + mode * 0.05) + " + " + (0.15 + mode * 0.02) + " * " + cycle + ")";
        final String dt = "((temp - " + tempBase + ") / 2.5)";
        final String dv = "((vibration - " + vibBase + ") / 0.15)";
        final String dp = "((pressure - " + (100.0 + mode * 0.8) + ") / 1.8)";
        final String da = "((amps - " + (3.6 + mode * 0.3) + ") / 0.35)";
        final String st = "((temp - pressure * 0.2) / 4.0)";
        return "(" + dt + " * " + dt + " + " + dv + " * " + dv + " + " + dp + " * " + dp + " + "
                + da + " * " + da + " + " + st + " * " + st + ")";
    }

    /** The smallest deviation across the modes; more modes is more arithmetic on the same read. */
    public static String score(int modes) {
        if (modes == 1) {
            return deviation(0);
        }
        StringBuilder sb = new StringBuilder("LEAST(");
        for (int m = 0; m < modes; m++) {
            if (m > 0) {
                sb.append(",\n           ");
            }
            sb.append(deviation(m));
        }
        return sb.append(")").toString();
    }

    /**
     * Score every reading, keep the anomalies, roll them up per machine, and write the note.
     *
     * <p>The {@code Calc} that computes the score is the offload candidate. The aggregate above it
     * keeps the output small, and the projection above <em>that</em> — the one holding {@code
     * triage} — is a second {@code Calc} which the accelerator will not take and is not meant to:
     * an aggregate between them is what stops the planner merging the two and dragging a UDF into
     * the kernel's way.
     */
    public static String digestQuery(Args args) {
        return "SELECT triage(LISTAGG(line, '"
                + TriageFunction.SEPARATOR
                + "')) AS note\n"
                + "FROM (\n"
                + "  SELECT CONCAT(\n"
                + "           'machine ', LPAD(CAST(machine_id AS STRING), 3, '0'),\n"
                + "           ': ', CAST(anomalies AS STRING), ' anomalous readings, worst score ',\n"
                + "           CAST(ROUND(worst, 2) AS STRING),\n"
                + "           ', mean ', CAST(ROUND(mean_score, 2) AS STRING)) AS line\n"
                + "  FROM (\n"
                + "    SELECT machine_id,\n"
                + "           COUNT(*) AS anomalies,\n"
                + "           MAX(score) AS worst,\n"
                + "           AVG(score) AS mean_score\n"
                + "    FROM (\n"
                + "      SELECT machine_id, "
                + score(args.modes)
                + " AS score\n"
                + "      FROM Readings\n"
                + "    )\n"
                + where(args)
                + "    GROUP BY machine_id\n"
                + "  )\n"
                + ")";
    }

    /**
     * The same query without the model, for measuring the preprocessing on its own.
     *
     * <p>It returns the digest as rows rather than as a note, which is what makes the two halves
     * separable: run this and the difference between the arms is the scan and the score, with no
     * inference in it at all.
     */
    public static String digestOnlyQuery(Args args) {
        return "SELECT machine_id,\n"
                + "       COUNT(*) AS anomalies,\n"
                + "       MAX(score) AS worst,\n"
                + "       AVG(score) AS mean_score\n"
                + "FROM (\n"
                + "  SELECT machine_id, "
                + score(args.modes)
                + " AS score\n"
                + "  FROM Readings\n"
                + ")\n"
                + where(args).stripLeading()
                + "GROUP BY machine_id";
    }

    /**
     * The screen, or nothing.
     *
     * <p>{@code --no-where} exists because a predicate over a computed expression changes the plan
     * in a way that has nothing to do with the device: the optimizer records it on the source as
     * well as on the Calc, and the score is then evaluated twice per row, once on a CPU that no
     * accelerator can relieve. Being able to take the filter out is how that effect is told apart
     * from the offload itself.
     */
    private static String where(Args args) {
        return args.noWhere ? "" : "    WHERE score > " + args.threshold + "\n";
    }

    /** A checksum of the digest that two arms must agree on exactly. */
    public static String digestFingerprintQuery(Args args) {
        return "SELECT COUNT(*) AS machines, SUM(anomalies) AS anomalies,\n"
                + "       SUM(machine_id) AS id_sum, MAX(worst) AS worst\n"
                + "FROM (\n"
                + digestOnlyQuery(args)
                + "\n)";
    }

    // ------------------------------------------------------------------------------------------
    // the prompt
    // ------------------------------------------------------------------------------------------

    /**
     * The digest, wrapped in the instruction that makes it a maintenance task.
     *
     * <p>Deliberately fixed: both arms send the same characters, so the prompt-token count is a
     * property of the data and the difference between the arms is the engine.
     */
    public static String prompt(String[] lines) {
        StringBuilder sb = new StringBuilder(512);
        sb.append(
                "You are a maintenance engineer. The overnight screen of a machine fleet flagged "
                        + "the readings below. Each line gives a machine, how many of its readings "
                        + "were anomalous, the worst deviation score and the mean.\n\n");
        for (String line : lines) {
            sb.append("- ").append(line).append('\n');
        }
        sb.append(
                "\nWrite a short triage note: which machines to inspect first and why, in at most "
                        + "six sentences. Do not invent readings that are not listed.");
        return sb.toString();
    }

    /** The engine's accounting, as one line the client can find in the returned note. */
    public static String report(
            String engine,
            boolean reused,
            long loadNanos,
            int promptChars,
            TriageEngine.Completion c) {
        return String.format(
                Locale.ROOT,
                "@@TRIAGE engine=%s model_load_ms=%.1f resident=%s prompt_chars=%d "
                        + "prompt_tokens=%d generated_tokens=%d prefill_ms=%.1f decode_ms=%.1f "
                        + "inference_wall_ms=%.1f decode_tok_s=%.2f%n",
                engine,
                loadNanos / 1e6,
                reused,
                promptChars,
                c.promptTokens,
                c.generatedTokens,
                c.prefillNanos / 1e6,
                c.decodeNanos / 1e6,
                c.wallNanos / 1e6,
                c.decodeNanos > 0 ? c.generatedTokens / (c.decodeNanos / 1e9) : 0.0);
    }

    // ------------------------------------------------------------------------------------------
    // the data
    // ------------------------------------------------------------------------------------------

    /** The DDL both arms read. {@code NOT NULL} is the one thing the query author writes. */
    public static String readings(String path) {
        return "CREATE TABLE Readings (\n"
                + "  machine_id INT NOT NULL,\n"
                + "  sensor_id INT NOT NULL,\n"
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
                + "  'format' = 'parquet'\n"
                + ")";
    }

    /** Writes the dataset once, so both arms read the same bytes. */
    public static void generate(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args, false);
        env.executeSql(
                "CREATE TABLE Source (\n"
                        + "  machine_id INT,\n"
                        + "  sensor_id INT,\n"
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
                        // The default is 10,000 rows a second, which turns an eight-million-row
                        // dataset into a fourteen-minute sleep. Nothing here is rate limited on
                        // purpose.
                        + "  'rows-per-second' = '10000000',\n"
                        + "  'fields.machine_id.min' = '0', 'fields.machine_id.max' = '"
                        + (args.machines - 1)
                        + "',\n"
                        + "  'fields.sensor_id.kind' = 'sequence',\n"
                        + "  'fields.sensor_id.start' = '0',\n"
                        + "  'fields.sensor_id.end' = '"
                        + (args.rows - 1)
                        + "',\n"
                        + "  'fields.phase.min' = '0.0', 'fields.phase.max' = '86400.0',\n"
                        + "  'fields.temp.min' = '5.0', 'fields.temp.max' = '45.0',\n"
                        + "  'fields.vibration.min' = '0.0', 'fields.vibration.max' = '1.6',\n"
                        + "  'fields.pressure.min' = '95.0', 'fields.pressure.max' = '108.0',\n"
                        + "  'fields.amps.min' = '2.5', 'fields.amps.max' = '6.0'\n"
                        + ")");
        env.executeSql(readings(args.data));
        System.out.printf("writing %,d rows to %s%n", args.rows, args.data);
        long start = System.nanoTime();
        env.executeSql("INSERT INTO Readings SELECT * FROM Source").await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    /**
     * A batch environment, accelerated or not.
     *
     * <p>{@code table.exec.accelerator.enabled} is the <em>only</em> difference between the two
     * arms' configuration, and it is a deployment setting rather than anything the query says.
     */
    public static TableEnvironment batchEnvironment(Args args, boolean accelerated) {
        final TableEnvironment env =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        if (args.parallelism > 0) {
            env.getConfig()
                    .getConfiguration()
                    .setString(
                            "table.exec.resource.default-parallelism",
                            Integer.toString(args.parallelism));
        }
        // Set for both arms, so the plan shape is the same on each side of the comparison.
        //
        // With it on, the optimizer records the predicate on the source *and* leaves it on the
        // Calc, so the score is evaluated twice per row: once by the CPU inside the scan and once
        // by whatever runs the Calc. The second evaluation is the one an accelerator can take, so
        // offloading it removes half of the work and the arms come out level -- which is what the
        // first sweep measured, and it says nothing about the device. A pushed-down predicate over
        // a computed expression cannot prune a Parquet row group anyway: there are no statistics
        // for a value that is not a column.
        env.getConfig()
                .getConfiguration()
                .setString(
                        "table.optimizer.source.predicate-pushdown-enabled",
                        Boolean.toString(args.pushdown));
        if (accelerated) {
            env.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
            if (args.maxAccelParallelism > 0) {
                env.getConfig()
                        .getConfiguration()
                        .setString(
                                "table.exec.accelerator.max-parallelism",
                                Integer.toString(args.maxAccelParallelism));
            }
        }
        return env;
    }

    /** Runs one arm end to end and prints what came back. */
    public static void run(Args args, boolean accelerated, TriageFunction triage) throws Exception {
        if (!Files.isDirectory(Paths.get(args.data))) {
            System.out.printf("%s does not exist yet%n", args.data);
            generate(args);
        }
        final TableEnvironment env = batchEnvironment(args, accelerated);
        env.executeSql(readings(args.data));
        env.createTemporarySystemFunction("triage", triage);

        final String query = args.digestOnly ? digestFingerprintQuery(args) : digestQuery(args);
        if (args.explain) {
            System.out.println(env.explainSql(query));
        }

        // The clock stops after the last row has been read, not after collect() returns.
        //
        // collect() hands back an iterator as soon as the job is submitted, so timing it alone
        // measures submission and planning -- about two seconds here, the same in both arms, and
        // entirely independent of what the query does. Measured that way, a twenty-fold difference
        // in execution was invisible.
        long start = System.nanoTime();
        StringBuilder out = new StringBuilder();
        try (CloseableIterator<Row> rows = env.sqlQuery(query).execute().collect()) {
            while (rows.hasNext()) {
                out.append(rows.next().toString()).append(System.lineSeparator());
            }
        }
        long elapsed = System.nanoTime() - start;
        System.out.printf("%n==== result ====%n%s%n", out);
        System.out.printf("query wall time: %.0f ms%n", elapsed / 1e6);
    }

    // ------------------------------------------------------------------------------------------
    // arguments
    // ------------------------------------------------------------------------------------------

    /** Everything both arms take, parsed the same way, so a flag cannot mean two things. */
    public static final class Args {
        public String data = "/home/mary/gpu-bench-data/flink-llm/readings";
        public String model =
                "/home/mary/Projects/GPULlama3-Beehive/GPULlama3.java/Qwen3-0.6B-f16.gguf";
        public String binary = "/home/mary/Projects/llama.cpp/build/bin/llama-server";
        public String libraryPath =
                "/home/mary/gpu-bench-data/flink-llm/cuda13/targets/x86_64-linux/lib";
        public long rows = 8_000_000L;
        public int machines = 48;
        public int modes = 8;
        public String threshold = "150.0";
        public int parallelism = 1;
        public int maxAccelParallelism = 0;
        public int contextLength = 4096;
        // Wide enough to hold the whole digest prompt in one chunk. A prompt that spills
        // into a second chunk leaves jitllm's native attention path; see JitllmEngine.
        public int promptBatch = 2048;
        public int maxNewTokens = 256;
        public int port = 18080;
        public boolean deviceSampling = false;
        public boolean nativeLibraries = true;
        public boolean cold = false;
        public boolean generate = false;
        public boolean explain = false;
        public boolean digestOnly = false;
        public boolean pushdown = false;
        public boolean noWhere = false;

        public static Args parse(String[] argv) {
            Args a = new Args();
            for (int i = 0; i < argv.length; i++) {
                switch (argv[i]) {
                    case "--data" -> a.data = argv[++i];
                    case "--model" -> a.model = argv[++i];
                    case "--binary" -> a.binary = argv[++i];
                    case "--library-path" -> a.libraryPath = argv[++i];
                    case "--rows" -> a.rows = Long.parseLong(argv[++i]);
                    case "--machines" -> a.machines = Integer.parseInt(argv[++i]);
                    case "--modes" -> a.modes = Integer.parseInt(argv[++i]);
                    case "--threshold" -> a.threshold = argv[++i];
                    case "--parallelism" -> a.parallelism = Integer.parseInt(argv[++i]);
                    case "--accel-parallelism" ->
                            a.maxAccelParallelism = Integer.parseInt(argv[++i]);
                    case "--context" -> a.contextLength = Integer.parseInt(argv[++i]);
                    case "--prompt-batch" -> a.promptBatch = Integer.parseInt(argv[++i]);
                    case "--max-new-tokens" -> a.maxNewTokens = Integer.parseInt(argv[++i]);
                    case "--port" -> a.port = Integer.parseInt(argv[++i]);
                    case "--device-sampling" -> a.deviceSampling = Boolean.parseBoolean(argv[++i]);
                    case "--native" -> a.nativeLibraries = Boolean.parseBoolean(argv[++i]);
                    case "--cold" -> a.cold = true;
                    case "--generate" -> a.generate = true;
                    case "--explain" -> a.explain = true;
                    case "--digest-only" -> a.digestOnly = true;
                    case "--pushdown" -> a.pushdown = Boolean.parseBoolean(argv[++i]);
                    case "--no-where" -> a.noWhere = true;
                    default -> throw new IllegalArgumentException("unknown flag " + argv[i]);
                }
            }
            return a;
        }

        /** What was asked for, printed before anything is timed. */
        public String banner(String arm) {
            return String.format(
                    Locale.ROOT,
                    "arm=%s  data=%s  rows=%,d  machines=%d  modes=%d  threshold=%s%n"
                            + "model=%s  context=%d  prompt-batch=%d  max-new-tokens=%d  cold=%s%n",
                    arm,
                    data,
                    rows,
                    machines,
                    modes,
                    threshold,
                    model,
                    contextLength,
                    promptBatch,
                    maxNewTokens,
                    cold);
        }
    }
}
