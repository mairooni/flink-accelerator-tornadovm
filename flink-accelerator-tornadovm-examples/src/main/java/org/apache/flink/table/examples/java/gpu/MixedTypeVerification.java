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

/**
 * Does an offloaded kernel read a column that is not a {@code DOUBLE}?
 *
 * <p>Until the staging path carried the declared type it did not. Every column was fetched with
 * {@code getDouble}, so a four-byte field returned eight bytes of whatever followed it, and an
 * {@code INT} holding 42 reached the device as 2.08e-322. Nothing caught it: the thirteen-shape
 * coverage probe and the correctness run that compared 239,994 values against the CPU plan both
 * declare a single numeric column, and it is a {@code DOUBLE}.
 *
 * <p>So this runs the same heavy expression over an {@code INT}, a {@code FLOAT}, a {@code DOUBLE}
 * and all three at once, with offload on and off, over one file both arms read. Agreement is the
 * result; the timings are incidental and too small to mean anything.
 *
 * <pre>{@code
 * bin/flink run MixedTypeVerification.jar --generate --rows 2000000
 * bin/flink run MixedTypeVerification.jar
 * }</pre>
 *
 * <p>The expression needs a transcendental to clear the cost floor. That is not incidental to the
 * bug: {@code i * 2} weighs 1 against a floor of 96 and stays on the CPU, so the broken path only
 * opened once an expression was worth offloading in the first place.
 */
public final class MixedTypeVerification {

    /**
     * Same shape for each column, so the arms differ only in which one they read.
     *
     * <p>Every literal carries {@code E0}, and that is not cosmetic. Without it {@code 8.0} is a
     * {@code DECIMAL} in Flink SQL, so {@code i / 8.0} is {@code DECIMAL(17, 6)}, the estimator
     * refuses the expression as not expressible, and <em>both</em> arms run on the CPU. The first
     * run of this verifier did exactly that and reported the {@code INT} case as agreeing -- which
     * it did, for a reason that had nothing to do with staging.
     */
    private static final String HEAVY =
            "EXP(%1$s / 100.0E0) * LN(ABS(%1$s) + 1.0E0) + SIN(%1$s) * COS(%1$s)"
                    + " + POWER(%1$s / 4.0E0, 3.0E0)";

    private MixedTypeVerification() {}

    public static void main(String[] args) throws Exception {
        Args parsed = Args.parse(args);
        if (parsed.generate) {
            generate(parsed);
            return;
        }

        String[][] cases = {
            {"int", String.format(HEAVY, "i")},
            {"float", String.format(HEAVY, "f")},
            {"double", String.format(HEAVY, "d")},
            {
                "mixed",
                "EXP(i / 100.0E0) * LN(ABS(f) + 1.0E0) + SIN(d) * COS(i)"
                        + " + POWER(f / 4.0E0, 3.0E0)"
            },
            {"passthrough-float", String.format(HEAVY, "d") + " + f"},
        };

        int disagreements = 0;
        for (String[] probe : cases) {
            Row cpu = run(parsed, probe[1], false);
            Row gpu = run(parsed, probe[1], true);
            disagreements += report(probe[0], cpu, gpu);
        }
        if (disagreements > 0) {
            throw new IllegalStateException(disagreements + " of " + cases.length + " disagree");
        }
        System.out.println("\nall " + cases.length + " agree");
    }

    /**
     * Compares the two arms.
     *
     * <p>A tolerance rather than equality, because the device reassociates a {@code SUM} it did not
     * compute in the same order. The tolerance is irrelevant to what this is looking for: a misread
     * column does not differ in the last bits, it differs by three hundred orders of magnitude or
     * is not finite at all.
     */
    private static int report(String label, Row cpu, Row gpu) {
        double a = ((Number) cpu.getField(1)).doubleValue();
        double b = ((Number) gpu.getField(1)).doubleValue();
        double relative = a == 0.0 ? Math.abs(b) : Math.abs((a - b) / a);
        boolean ok = Double.isFinite(a) && Double.isFinite(b) && relative <= 1e-9;
        System.out.printf(
                "%-18s cpu=%-24s gpu=%-24s rel=%-12.3e %s%n",
                label, a, b, relative, ok ? "agree" : "DISAGREE");
        if (!cpu.getField(0).equals(gpu.getField(0))) {
            System.out.printf("%-18s row counts differ: %s vs %s%n", label, cpu, gpu);
            return 1;
        }
        return ok ? 0 : 1;
    }

    private static Row run(Args args, String expression, boolean gpu) throws Exception {
        final TableEnvironment env = environment(args);
        env.executeSql(mixed(args.data, args.format));
        if (gpu) {
            env.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
        }
        String query =
                "SELECT COUNT(*) AS rows_seen, SUM(h) AS total\n"
                        + "FROM (SELECT "
                        + expression
                        + " AS h FROM Mixed)";
        if (gpu) {
            // Checked on every run, not only under --explain. An agreement between the arms is
            // evidence of nothing unless this arm actually reached a device, and the way that goes
            // wrong is silent: the estimator declines, the query runs on the CPU, and the two
            // arms match each other perfectly.
            String plan = env.explainSql(query);
            String section = plan.substring(plan.indexOf("== GPU Offload =="));
            if (args.explain) {
                System.out.println(section);
            }
            if (!section.contains("GPU  subtree")) {
                throw new IllegalStateException(
                        "this arm did not offload, so comparing it proves nothing:\n" + section);
            }
        }
        try (CloseableIterator<Row> rows = env.sqlQuery(query).execute().collect()) {
            if (!rows.hasNext()) {
                throw new IllegalStateException("query returned no rows");
            }
            return rows.next();
        }
    }

    /** Writes the input once, so both arms read identical values rather than fresh random ones. */
    private static void generate(Args args) throws Exception {
        final TableEnvironment env = environment(args);
        env.executeSql(
                "CREATE TABLE Source (\n"
                        + "  i INT,\n"
                        + "  f FLOAT,\n"
                        + "  d DOUBLE\n"
                        + ") WITH (\n"
                        + "  'connector' = 'datagen',\n"
                        + "  'number-of-rows' = '"
                        + args.rows
                        + "',\n"
                        // Bounded well inside every width, so the arms are compared on staging
                        // rather than on what a float does at its limits.
                        + "  'fields.i.min' = '1',\n"
                        + "  'fields.i.max' = '1000',\n"
                        + "  'fields.f.min' = '0.5',\n"
                        + "  'fields.f.max' = '20.0',\n"
                        + "  'fields.d.min' = '0.5',\n"
                        + "  'fields.d.max' = '20.0'\n"
                        + ")");
        env.executeSql(mixed(args.data, args.format));
        System.out.printf("writing %,d rows to %s%n", args.rows, args.data);
        env.executeSql("INSERT INTO Mixed SELECT i, f, d FROM Source").await();
        System.out.println("done");
    }

    private static String mixed(String path, String format) {
        return "CREATE TABLE Mixed (\n"
                + "  i INT,\n"
                + "  f FLOAT,\n"
                + "  d DOUBLE\n"
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

    private static TableEnvironment environment(Args args) {
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

    private static final class Args {
        private boolean generate;
        private boolean explain;
        private long rows = 2_000_000;
        private int parallelism = 1;
        private String data = "/tmp/flink-gpu-mixed";
        private String format = "csv";

        static Args parse(String[] argv) {
            Args args = new Args();
            for (int i = 0; i < argv.length; i++) {
                String flag = argv[i];
                if ("--generate".equals(flag)) {
                    args.generate = true;
                } else if ("--explain".equals(flag)) {
                    args.explain = true;
                } else if ("--rows".equals(flag)) {
                    args.rows = Long.parseLong(argv[++i]);
                } else if ("--parallelism".equals(flag)) {
                    args.parallelism = Integer.parseInt(argv[++i]);
                } else if ("--data".equals(flag)) {
                    args.data = argv[++i];
                } else if ("--format".equals(flag)) {
                    args.format = argv[++i];
                } else {
                    throw new IllegalArgumentException("unknown flag " + flag);
                }
            }
            return args;
        }
    }
}
