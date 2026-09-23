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

    /**
     * Constants as SQL, typed the way the expression means them.
     *
     * <p>A bare {@code 0.025} is a {@code DECIMAL(4,3)}, and the provider declines arithmetic with
     * a {@code DECIMAL} operand because Flink does not evaluate it as double arithmetic. So the
     * whole pricing expression is refused when written the ordinary way, and this benchmark cannot
     * reach the device at all. {@code -Dblackscholes.doubleLiterals=true} writes the same values as
     * SQL approximate-numeric literals ({@code 0.025E0}), which are {@code DOUBLE}.
     *
     * <p>Opt-in rather than the default, so the ordinary spelling stays measurable and the
     * difference between them is itself a reportable number.
     */
    private static String lit(String value) {
        return Boolean.getBoolean("blackscholes.doubleLiterals") ? value + "E0" : value;
    }

    /** Abramowitz &amp; Stegun 26.2.17. Published absolute error bound 7.5e-8 on N(x). */
    private static final String AS_P = "0.2316419";

    private static final String AS_B1 = "0.319381530";

    private static final String AS_B2 = "-0.356563782";

    private static final String AS_B3 = "1.781477937";

    private static final String AS_B4 = "-1.821255978";

    private static final String AS_B5 = "1.330274429";

    /** 1/sqrt(2*pi), the standard normal density's scale. */
    private static final String INV_SQRT_2PI = "0.3989422804014327";

    /** One point on the discount curve. A book is valued at one date against one curve. */
    private static final String RATE_VALUE = "0.025";

    public static void main(String[] args) throws Exception {
        final Args parsed = Args.parse(args);

        if (parsed.generate) {
            generate(parsed);
            return;
        }
        if (parsed.validate > 0) {
            validate(parsed);
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
            if (run == 1 && !parsed.baseline) {
                printScenarios(seen, parsed.scenarios);
            }
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
        // Every scenario, not just the first: two runs agreeing on one column and differing on
        // another is exactly the failure a single-column check would report as success.
        for (int i = 1; i < first.getArity(); i++) {
            double a = ((Number) first.getField(i)).doubleValue();
            double b = ((Number) second.getField(i)).doubleValue();
            double relative = a == 0.0 ? Math.abs(b) : Math.abs((a - b) / a);
            if (relative > 1e-12) {
                throw new IllegalStateException(
                        "runs disagree on scenario "
                                + (i - 1)
                                + " by "
                                + relative
                                + " relative: "
                                + first
                                + " then "
                                + second
                                + "; that is far beyond floating-point reassociation");
            }
        }
    }

    /**
     * The scenario table: what each scenario is, what the book is worth under it, and the change.
     *
     * <p>Reported per scenario rather than as one total. A single summed number cannot be checked
     * against anything a risk report would contain, and summing scenarios together exists only to
     * make the arithmetic happen — which is a benchmark shape, not an application.
     *
     * <p>The baseline is scenario 0 by construction (shock 1.0, the unshocked book), so "change
     * from baseline" is defined rather than relative to whatever came first.
     */
    private static void printScenarios(Row row, int scenarios) {
        final double baseline = ((Number) row.getField(1)).doubleValue();
        System.out.printf(
                "%n  %-8s %-12s %20s %20s %12s%n",
                "scenario", "vol shock", "portfolio value", "change vs baseline", "change %");
        for (int i = 0; i < scenarios; i++) {
            final double value = ((Number) row.getField(i + 1)).doubleValue();
            final double delta = value - baseline;
            System.out.printf(
                    "  %-8d %-12.4f %20.4f %20.4f %11.4f%%%n",
                    i,
                    shock(i, scenarios),
                    value,
                    delta,
                    baseline == 0.0 ? 0.0 : 100.0 * delta / baseline);
        }
        System.out.printf(
                "  baseline is scenario 0 (vol shock 1.0), %,d positions%n%n",
                ((Number) row.getField(0)).longValue());
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

    /**
     * Individual prices, for checking the formula rather than the total.
     *
     * <p>A scenario total is a sum of ten million prices: an error in one of them is invisible in
     * it, and an error in all of them is indistinguishable from a different summation order. So the
     * formula is validated per option, against an independent reference computed outside Flink from
     * libm's {@code erfc} — not against the other arm, which would only show that two evaluations
     * of the same approximation agree.
     *
     * <p>Prints the inputs beside the price so the reference can be recomputed from the same
     * numbers, with no dependence on how the book was generated.
     */
    private static void validate(Args args) throws Exception {
        final TableEnvironment env = batchEnvironment(args);
        env.executeSql(book(args.data, args.format));
        if (args.gpu) {
            env.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
        }
        final String sql =
                "SELECT id, spot, strike, tau, vol, "
                        + call("vol")
                        + " AS price FROM Book WHERE id < "
                        + args.validate;
        try (CloseableIterator<Row> rows = env.sqlQuery(sql).execute().collect()) {
            while (rows.hasNext()) {
                final Row row = rows.next();
                System.out.printf(
                        "PRICE %s %s %s %s %s %s%n",
                        row.getField(0),
                        row.getField(1),
                        row.getField(2),
                        row.getField(3),
                        row.getField(4),
                        row.getField(5));
            }
        }
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
                    "SELECT COUNT(*) AS rows_seen,\n       "
                            + scenarioSums(args.scenarios)
                            + "\nFROM (\n  SELECT "
                            + scenarioColumns(args.scenarios)
                            + "\n  FROM Book\n)";
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
    /**
     * The volatility shock applied in scenario {@code i}, with scenario 0 the baseline.
     *
     * <p>Scenario 0 is deliberately the unshocked book — shock 1.0 — so "change from baseline" is a
     * defined quantity rather than a difference from whichever scenario happened to be first. The
     * rest spread from a halving to a doubling of implied volatility, which is the range a stress
     * run covers, and are a pure function of the scenario index so a rerun reproduces them.
     */
    private static double shock(int i, int n) {
        if (i == 0 || n == 1) {
            return 1.0;
        }
        if (n == 2) {
            return 0.5;
        }
        // The shocked scenarios span a halving to a doubling of implied volatility. Spread over
        // n-2 intervals rather than n-1 so that none of them lands back on 1.0: a stress table
        // listing the baseline twice, once as "scenario 0" and once as a shock, is not a stress
        // table anyone would read.
        return 0.5 + 1.5 * (i - 1) / (double) (n - 2);
    }

    /** One priced column per scenario, so each scenario's value survives to the output. */
    private static String scenarioColumns(int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            if (i > 0) {
                sb.append(",\n         ");
            }
            sb.append(call("(vol * " + lit(Double.toString(shock(i, n))) + ")"))
                    .append(" AS pv")
                    .append(i);
        }
        return sb.toString();
    }

    /** {@code SUM(pv0) AS s0, SUM(pv1) AS s1, ...}: a portfolio value for each scenario. */
    private static String scenarioSums(int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            if (i > 0) {
                sb.append(",\n       ");
            }
            sb.append("SUM(pv").append(i).append(") AS s").append(i);
        }
        return sb.toString();
    }

    /** A Black-Scholes call price, with the volatility term left open so a shock can be applied. */
    private static String call(String vol) {
        final String sqrtT = "SQRT(tau)";
        final String d1 =
                "((LN(spot / strike) + ("
                        + lit(RATE_VALUE)
                        + " + "
                        + lit("0.5")
                        + " * "
                        + vol
                        + " * "
                        + vol
                        + ") * tau) / ("
                        + vol
                        + " * "
                        + sqrtT
                        + "))";
        final String d2 = "(" + d1 + " - " + vol + " * " + sqrtT + ")";
        return "(spot * "
                + cdf(d1)
                + " - strike * EXP(-"
                + lit(RATE_VALUE)
                + " * tau) * "
                + cdf(d2)
                + ")";
    }

    /**
     * The standard normal CDF: Abramowitz &amp; Stegun 26.2.17, branch-free.
     *
     * <h2>Why this replaced a tanh approximation</h2>
     *
     * <p>The previous formula, {@code 0.5*(1 + TANH(0.7988 x (1 + 0.04417 x^2)))}, was chosen
     * because it fitted the accelerator's function vocabulary. Measured against {@code erfc} from
     * libm over x in [-9, 9]:
     *
     * <pre>
     *   max |N_approx - N| :  tanh 1.40e-04      A&amp;S 7.45e-08
     *   relative error at x = -4 :  tanh 42%     A&amp;S 4.7e-04
     *   relative error at x = -6 :  tanh 98%     A&amp;S 3.6e-03
     * </pre>
     *
     * <p>A deep out-of-the-money option's price <em>is</em> that tail probability, so the tanh form
     * priced them wrong by tens of percent. Surveyed over the supported domain, the worst call
     * price error was <b>3.0e-04 of spot</b> with tanh and <b>1.4e-07 of spot</b> with this. The
     * mathematics was not chosen to stay inside the whitelist; it happens to fit, needing only
     * {@code ABS}, {@code SIGN}, {@code EXP} and arithmetic.
     *
     * <h2>How the branch is avoided</h2>
     *
     * <p>26.2.17 gives the upper tail for x &ge; 0 and relies on symmetry for x &lt; 0, which is
     * ordinarily a conditional. Writing {@code t = N(-|x|)} and folding with {@code SIGN(x)}:
     *
     * <pre>
     *   N(x) = 0.5 (1 + SIGN(x)) (1 - t) + 0.5 (1 - SIGN(x)) t
     * </pre>
     *
     * <p>which is {@code 1 - t} for x &gt; 0, {@code t} for x &lt; 0, and — since {@code SIGN(0)}
     * is 0 — exactly {@code 0.5} at zero. No conditional is needed, so no new IR feature is needed
     * either, and the same expression is evaluated by both arms.
     */
    private static String cdf(String x) {
        final String u = "ABS(" + x + ")";
        final String t =
                "(" + lit("1.0") + " / (" + lit("1.0") + " + " + lit(AS_P) + " * " + u + "))";
        // Horner, so the polynomial costs five multiply-adds rather than five powers.
        final String poly =
                "("
                        + t
                        + " * ("
                        + lit(AS_B1)
                        + " + "
                        + t
                        + " * ("
                        + lit(AS_B2)
                        + " + "
                        + t
                        + " * ("
                        + lit(AS_B3)
                        + " + "
                        + t
                        + " * ("
                        + lit(AS_B4)
                        + " + "
                        + t
                        + " * "
                        + lit(AS_B5)
                        + ")))))";
        final String tail =
                "("
                        + lit(INV_SQRT_2PI)
                        + " * EXP("
                        + lit("-0.5")
                        + " * "
                        + u
                        + " * "
                        + u
                        + ") * "
                        + poly
                        + ")";
        final String sign = "SIGN(" + x + ")";
        return "("
                + lit("0.5")
                + " * ("
                + lit("1.0")
                + " + "
                + sign
                + ") * ("
                + lit("1.0")
                + " - "
                + tail
                + ") + "
                + lit("0.5")
                + " * ("
                + lit("1.0")
                + " - "
                + sign
                + ") * "
                + tail
                + ")";
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

        /** Print this many individual prices and exit, for external numerical validation. */
        private int validate;

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
                } else if ("--validate".equals(flag)) {
                    args.validate = Integer.parseInt(argv[++i]);
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
