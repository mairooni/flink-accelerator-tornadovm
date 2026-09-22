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

import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Stress-repricing an option book in SQL: transcendental arithmetic per row, nothing else.
 *
 * <pre>{@code
 * SELECT COUNT(*), SUM(pv)
 * FROM (SELECT <call price under n volatility scenarios, summed> AS pv FROM Book)
 * }</pre>
 *
 * <p>Written to answer one question: can a <em>single-pass</em> query a person would actually write
 * reach the speedups the iterative benchmarks show? {@link org.apache.flink.table.gpu.examples} has
 * two kinds of result. The iterative ones reach 16x to 40x, but only because the CPU arm resubmits
 * a job per pass -- their one-iteration figure is 1.40x. The single-pass ones are transparent and
 * modest: {@link HaversineBenchmark} is 11.9x at 20 depots. This benchmark is the second kind, on a
 * workload nobody has to be talked into caring about.
 *
 * <h2>Why this shape and not one option</h2>
 *
 * <p>One Black-Scholes per row is Amdahl-capped at about 1.2x, for the same reason one haversine
 * is: the expression is a fifth of the job and the source read is the rest. A risk system does not
 * price one option per position either. It reprices the book under a set of shocks, so {@code
 * --scenarios n} sums the call price under {@code n} multiplicative volatility shocks -- one input
 * read, one output column, {@code n} times the arithmetic. That is a stress run, and it is exactly
 * the knob {@code --depots} is on the haversine query.
 *
 * <h2>The normal CDF is branchless on purpose</h2>
 *
 * <p>The accelerator IR has no {@code CASE}, and one unsupported operator refuses the whole
 * projection. Every published rational approximation to {@code N(x)} branches on the sign of {@code
 * x} to use the symmetry {@code N(-x) = 1 - N(x)}, so none of them can be written here. The tanh
 * form below has no branch:
 *
 * <pre>{@code N(x) ~= 0.5 * (1 + tanh(0.7988 * x * (1 + 0.04417 * x^2)))}</pre>
 *
 * <p>It is accurate to about 1e-4 absolute, which is worse than a risk system would accept and
 * irrelevant to what is being measured: <em>both arms evaluate the same expression</em>, so the
 * comparison is exact even where the approximation is not. {@code TANH} is already in the IR's
 * vocabulary, so this adds nothing to the kernel generator -- which is the point, since the rule
 * here is that vocabulary follows measurement rather than leading it.
 *
 * <p>It reads four columns against the haversine query's two. Staging is per column per row, so
 * this pays twice the transfer for comparable arithmetic, and the gap between the two results is a
 * direct reading of the staging term in the eligibility rule.
 *
 * <pre>{@code
 * flink run examples/table/BlackScholesBenchmark.jar --generate --rows 2000000
 * flink run examples/table/BlackScholesBenchmark.jar --gpu false --scenarios 20
 * flink run examples/table/BlackScholesBenchmark.jar --gpu true  --scenarios 20
 * }</pre>
 */
public final class BlackScholesBenchmark {

    /** One point on the discount curve. A book is valued at one date against one curve. */
    private static final String RATE = "0.025";

    /** The tanh approximation's two constants, written once. */
    private static final String CDF_A = "0.7988";

    private static final String CDF_B = "0.04417";

    public static void main(String[] args) throws Exception {
        final Args parsed = Args.parse(args);

        if (parsed.generate) {
            generate(parsed);
            return;
        }

        // Generate on demand rather than failing inside the file source, which reports only
        // "Could not enumerate file splits" and says nothing about what should have created the
        // directory.
        if (!Files.isDirectory(Paths.get(parsed.data))) {
            System.out.printf("%s does not exist yet%n", parsed.data);
            generate(parsed);
        }

        System.out.printf(
                "data=%s  parallelism=%s  gpu=%s  runs=%d  scenarios=%d%n",
                parsed.data,
                parsed.parallelism > 0 ? Integer.toString(parsed.parallelism) : "(default)",
                parsed.gpu,
                parsed.runs,
                parsed.scenarios);
        if (parsed.baseline) {
            System.out.println("baseline: reading and counting only, no pricing");
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
     * the subtasks finish, and floating-point addition is not associative. The device reassociates
     * differently again. A tolerance is the honest check; exact equality would fail on correct
     * results and say nothing about the ones that matter.
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

    /** Writes the book once, so the benchmark itself never pays for generating rows. */
    private static void generate(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        env.executeSql(
                "CREATE TABLE Source (\n"
                        + "  id INT,\n"
                        + "  spot DOUBLE,\n"
                        + "  strike DOUBLE,\n"
                        + "  tau DOUBLE,\n"
                        + "  vol DOUBLE\n"
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
                        // Ranges a real book spans, and all strictly positive: tau and vol are
                        // divided by, so a zero would be an infinity in one arm and a NaN in the
                        // other and the agreement check would be measuring the generator.
                        + "  'fields.spot.min' = '20.0',\n"
                        + "  'fields.spot.max' = '400.0',\n"
                        + "  'fields.strike.min' = '20.0',\n"
                        + "  'fields.strike.max' = '400.0',\n"
                        + "  'fields.tau.min' = '0.05',\n"
                        + "  'fields.tau.max' = '3.0',\n"
                        + "  'fields.vol.min' = '0.08',\n"
                        + "  'fields.vol.max' = '0.75'\n"
                        + ")");
        env.executeSql(book(args.data, args.format));

        System.out.printf("writing %,d rows to %s%n", args.rows, args.data);
        long start = System.nanoTime();
        env.executeSql("INSERT INTO Book SELECT id, spot, strike, tau, vol FROM Source").await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    private static Row query(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        env.executeSql(book(args.data, args.format));

        if (args.gpu) {
            env.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
        }

        // Reading and summing, with almost no arithmetic: everything the job costs that is not the
        // expression. Subtracting it from a full run separates the operator from the source, and so
        // says whether a speedup on the operator can show up end to end. It touches all four
        // columns because the filesystem source pushes projection down, and a baseline reading one
        // would understate what the real query pays for its input.
        final String query;
        if (args.baseline) {
            query =
                    "SELECT COUNT(*) AS rows_seen, SUM(spot + strike + tau + vol) AS total_pv\n"
                            + "FROM Book";
        } else {
            query =
                    "SELECT COUNT(*) AS rows_seen, SUM(pv) AS total_pv\n"
                            + "FROM (\n"
                            + "  SELECT "
                            + stressed(args.scenarios)
                            + " AS pv\n"
                            + "  FROM Book\n"
                            + ")";
        }

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
     * The book's value summed over {@code n} volatility scenarios.
     *
     * <p>A deterministic ladder of shocks rather than a sampled one: the point is the arithmetic,
     * and every run has to compare against the same numbers. With {@code n == 1} this is the
     * unshocked price, which is the Amdahl-capped case kept only so the knob starts where a reader
     * expects it to.
     */
    private static String stressed(int n) {
        if (n == 1) {
            return call("vol");
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            if (i > 0) {
                sb.append("\n       + ");
            }
            // Spread over roughly a halving to a doubling of implied vol, which is the range a
            // stress run covers.
            double shock = 0.5 + i * (1.5 / (n - 1));
            sb.append(call("(vol * " + shock + ")"));
        }
        return sb.toString();
    }

    /** A Black-Scholes call price, with the volatility term left open so a shock can be applied. */
    private static String call(String vol) {
        final String sqrtT = "SQRT(tau)";
        final String d1 =
                "((LN(spot / strike) + ("
                        + RATE
                        + " + 0.5 * "
                        + vol
                        + " * "
                        + vol
                        + ") * tau) / ("
                        + vol
                        + " * "
                        + sqrtT
                        + "))";
        final String d2 = "(" + d1 + " - " + vol + " * " + sqrtT + ")";
        return "(spot * " + cdf(d1) + " - strike * EXP(-" + RATE + " * tau) * " + cdf(d2) + ")";
    }

    /**
     * The standard normal CDF, branchlessly.
     *
     * <p>{@code x} is substituted three times rather than bound to a name, because SQL has no
     * let-binding inside an expression. Calcite's {@code RexProgram} eliminates the common
     * subexpression, so the op count the provider sees counts it once -- which matters, since the
     * ceiling is on operations per row and this expression is repeated per scenario.
     */
    private static String cdf(String x) {
        return "(0.5 * (1.0 + TANH("
                + CDF_A
                + " * "
                + x
                + " * (1.0 + "
                + CDF_B
                + " * "
                + x
                + " * "
                + x
                + "))))";
    }

    private static String book(String path, String format) {
        return "CREATE TABLE Book (\n"
                + "  id INT,\n"
                + "  spot DOUBLE NOT NULL,\n"
                + "  strike DOUBLE NOT NULL,\n"
                + "  tau DOUBLE NOT NULL,\n"
                + "  vol DOUBLE NOT NULL\n"
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
        private String data;

        private int rows = 2_000_000;
        private int parallelism = -1;
        private int runs = 10;
        private String format = "csv";
        private boolean gpu;
        private boolean generate;
        private boolean explain;
        private boolean baseline;
        private int scenarios = 20;

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
                } else if ("--scenarios".equals(flag)) {
                    args.scenarios = Integer.parseInt(argv[++i]);
                } else if ("--baseline".equals(flag)) {
                    args.baseline = true;
                } else if ("--explain".equals(flag)) {
                    args.explain = true;
                } else {
                    throw new IllegalArgumentException("unknown argument " + flag);
                }
            }
            if (args.data == null) {
                args.data = "/tmp/flink-gpu-book-" + args.rows + "-" + args.format;
            }
            return args;
        }
    }

    private BlackScholesBenchmark() {}
}
