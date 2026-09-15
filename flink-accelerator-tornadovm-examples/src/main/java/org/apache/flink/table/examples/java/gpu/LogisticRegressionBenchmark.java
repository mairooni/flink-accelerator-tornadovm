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

import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.gpu.operator.LogisticRegressionEngine;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.StringJoiner;

/**
 * Logistic regression by gradient descent: vanilla Flink SQL against a device that iterates on
 * resident data.
 *
 * <p>Every earlier benchmark measured one pass over the input, and every one of them turned out to
 * be bound by reading it. {@link HaversineBenchmark} escaped that only because {@code --depots}
 * adds arithmetic without adding input, which is why it reached 12.7x where {@link
 * FeatureGramBenchmark} reached 2.4x -- the Gram matrix has no such knob, since widening the table
 * adds bytes and arithmetic together.
 *
 * <p>Iteration is that knob, and it is the one that matters for a device: read once, keep the data
 * resident, and pay only arithmetic for every pass after the first.
 *
 * <h2>The two arms are not doing the same thing, deliberately</h2>
 *
 * <p>The CPU arm is Flink as it runs normally. Each iteration is an ordinary SQL query with the
 * current weights inlined as literals, so the source is read once per iteration -- because that is
 * what a Flink job does. There is no way to write "keep this table on the workers and iterate over
 * it" in Flink SQL.
 *
 * <p>The device arm reads once into a resident buffer and iterates there. So the ratio between them
 * mixes two effects: arithmetic that is faster, and I/O that is not repeated. Both are real, and a
 * user would get both, but they are different claims. Running with {@code --iterations 1} isolates
 * the first: at one pass neither arm re-reads anything, so what is left is the arithmetic.
 *
 * <h2>What the device is doing</h2>
 *
 * <pre>{@code
 * z = X w           cublasSgemv     library
 * r = sigmoid(z)-y  generated       kernel      <- reads the library's output, feeds its input
 * g = X' r          cublasSgemv     library
 * w = w - lr*g      generated       kernel
 * }</pre>
 *
 * <p>The middle step is the reason this benchmark exists. cuBLAS has no sigmoid and a generated
 * kernel has no tiled GEMV, so the pass needs both, and the intermediate between them never returns
 * to the host.
 *
 * <pre>{@code
 * bin/flink run LogisticRegressionBenchmark.jar --generate --rows 500000 --features 32
 * bin/flink run LogisticRegressionBenchmark.jar --gpu false --iterations 20
 * bin/flink run LogisticRegressionBenchmark.jar --gpu true  --iterations 20
 * }</pre>
 */
public final class LogisticRegressionBenchmark {

    private static final double LEARNING_RATE = 0.1;

    public static void main(String[] args) throws Exception {
        final Args parsed = Args.parse(args);

        if (parsed.generate) {
            generate(parsed);
            return;
        }
        if (!Files.isDirectory(Paths.get(parsed.data))) {
            System.out.printf("%s does not exist yet%n", parsed.data);
            generate(parsed);
        }

        System.out.printf(
                "data=%s  rows=%,d  features=%d  iterations=%d  parallelism=%d  gpu=%s"
                        + "  contraction=%s  runs=%d%n",
                parsed.data,
                parsed.rows,
                parsed.features,
                parsed.iterations,
                parsed.parallelism,
                parsed.gpu,
                parsed.contraction,
                parsed.runs);

        double[] first = null;
        for (int run = 1; run <= parsed.runs; run++) {
            long start = System.nanoTime();
            double[] weights = parsed.gpu ? deviceTrain(parsed) : sqlTrain(parsed);
            double millis = (System.nanoTime() - start) / 1e6;
            System.out.printf("run %2d  %10.0f ms  norm=%.6e%n", run, millis, norm(weights));
            if (first == null) {
                first = weights;
            } else {
                agree(first, weights);
            }
        }
    }

    /**
     * Gradient descent as a user would write it without a device: one query per iteration.
     *
     * <p>The weights go in as literals because there is nowhere else to put them -- a Flink SQL
     * query is a fixed plan, so changing weights means a new query. That also means the source is
     * read again on every pass, which is the honest cost of doing this in SQL.
     */
    private static double[] sqlTrain(Args args) throws Exception {
        final StreamTableEnvironment env = environment(args);
        env.executeSql(sourceTable(args));

        double[] weights = new double[args.features];
        for (int iteration = 0; iteration < args.iterations; iteration++) {
            StringJoiner dot = new StringJoiner(" + ");
            for (int f = 0; f < args.features; f++) {
                dot.add(String.format("%.9e * c%d", weights[f], f));
            }
            StringJoiner sums = new StringJoiner(", ");
            for (int f = 0; f < args.features; f++) {
                sums.add(String.format("SUM(r * c%d)", f));
            }
            // The residual is computed once per row in a Calc, then contracted by the aggregate --
            // the same two-operator shape the device arm runs as a GEMV, a kernel and a GEMV.
            String sql =
                    String.format(
                            "SELECT %s FROM (SELECT %s, 1.0 / (1.0 + EXP(-(%s))) - label AS r "
                                    + "FROM Points)",
                            sums, columnList(args.features), dot);

            try (CloseableIterator<Row> it = env.executeSql(sql).collect()) {
                Row row = it.next();
                for (int f = 0; f < args.features; f++) {
                    double gradient = ((Number) row.getField(f)).doubleValue();
                    weights[f] -= LEARNING_RATE * gradient / args.rows;
                }
            }
        }
        return weights;
    }

    /** The same descent, with the data resident on the device for every pass. */
    private static double[] deviceTrain(Args args) throws Exception {
        final StreamTableEnvironment env = environment(args);
        env.executeSql(sourceTable(args));

        DataStream<double[]> trained =
                env.toDataStream(env.from("Points"))
                        .transform(
                                "logistic",
                                PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO,
                                new LogisticOperator(
                                        args.rows,
                                        args.features,
                                        args.iterations,
                                        LEARNING_RATE,
                                        args.contraction));

        try (CloseableIterator<double[]> it = trained.executeAndCollect()) {
            return it.next();
        }
    }

    /**
     * Buffers the whole partition, then iterates on the device.
     *
     * <p>Parallelism has to be 1. Gradient descent needs every partition's gradient summed before
     * the weights move, and Flink batch has no way to iterate a dataflow, so more than one subtask
     * would be descending on its own share of the data and calling it the answer.
     */
    private static final class LogisticOperator extends AbstractStreamOperator<double[]>
            implements OneInputStreamOperator<Row, double[]>, BoundedOneInput {

        private static final long serialVersionUID = 1L;

        private final int rows;
        private final int features;
        private final int iterations;
        private final double learningRate;
        private final String contraction;
        private transient LogisticRegressionEngine engine;

        private LogisticOperator(
                int rows, int features, int iterations, double learningRate, String contraction) {
            this.rows = rows;
            this.features = features;
            this.iterations = iterations;
            this.learningRate = learningRate;
            this.contraction = contraction;
        }

        @Override
        public void open() throws Exception {
            super.open();
            engine =
                    new LogisticRegressionEngine(rows, features, (float) learningRate, contraction);
        }

        @Override
        public void processElement(StreamRecord<Row> element) {
            Row row = element.getValue();
            for (int f = 0; f < features; f++) {
                engine.set(f, ((Number) row.getField(f)).doubleValue());
            }
            engine.rowComplete(((Number) row.getField(features)).doubleValue());
        }

        @Override
        public void endInput() {
            engine.open();
            engine.train(iterations);
            // A diagnostic line, so an end-to-end tie can be explained rather than guessed at.
            LOG.info(
                    "logistic: {} rows, {} iterations, contraction={}, loop {} ms",
                    engine.staged(),
                    iterations,
                    contraction,
                    String.format("%.1f", engine.trainMillis()));
            output.collect(new StreamRecord<>(engine.weights()));
        }

        @Override
        public void close() throws Exception {
            if (engine != null) {
                engine.close();
            }
            super.close();
        }
    }

    /** Writes the input once: {@code features} columns of signal, plus a 0/1 label. */
    private static void generate(Args args) throws Exception {
        final StreamTableEnvironment env = environment(args);

        StringJoiner schema = new StringJoiner(",\n  ");
        StringJoiner options = new StringJoiner(",\n  ");
        for (int f = 0; f < args.features; f++) {
            schema.add(String.format("c%d DOUBLE", f));
            options.add(String.format("'fields.c%d.min' = '-1.0'", f));
            options.add(String.format("'fields.c%d.max' = '1.0'", f));
        }
        schema.add("label DOUBLE");
        options.add("'fields.label.min' = '0.0'");
        options.add("'fields.label.max' = '1.0'");

        env.executeSql(
                String.format(
                        "CREATE TABLE Source (%n  %s%n) WITH (%n  'connector' = 'datagen',%n"
                                + "  'number-of-rows' = '%d',%n  %s%n)",
                        schema, args.rows, options));
        env.executeSql(sourceTable(args));

        System.out.printf(
                "writing %,d rows x %d features to %s%n", args.rows, args.features, args.data);
        long start = System.nanoTime();
        env.executeSql("INSERT INTO Points SELECT * FROM Source").await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    private static String columnList(int features) {
        StringJoiner columns = new StringJoiner(", ");
        for (int f = 0; f < features; f++) {
            columns.add("c" + f);
        }
        return columns.toString();
    }

    private static String sourceTable(Args args) {
        StringJoiner schema = new StringJoiner(",\n  ");
        for (int f = 0; f < args.features; f++) {
            schema.add(String.format("c%d DOUBLE", f));
        }
        schema.add("label DOUBLE");
        return String.format(
                "CREATE TABLE Points (%n  %s%n) WITH (%n  'connector' = 'filesystem',%n"
                        + "  'path' = '%s',%n  'format' = 'csv'%n)",
                schema, args.data);
    }

    private static StreamTableEnvironment environment(Args args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setRuntimeMode(RuntimeExecutionMode.BATCH);
        env.setParallelism(args.parallelism);
        return StreamTableEnvironment.create(env);
    }

    /** One number that moves if anything about the fitted weights moves. */
    private static double norm(double[] weights) {
        double sum = 0.0;
        for (double w : weights) {
            sum += w * w;
        }
        return Math.sqrt(sum);
    }

    /**
     * Checks two runs agree.
     *
     * <p>Loose, because the arms differ in precision by construction: the device accumulates in
     * FP32 and Flink in FP64. Gradient descent tolerates that better than a covariance matrix does,
     * being a fixed point iteration rather than a running sum, but the weights still will not match
     * to the last digit and should not be expected to.
     */
    private static void agree(double[] first, double[] second) {
        double worst = 0.0;
        for (int i = 0; i < first.length; i++) {
            double scale = Math.max(1e-6, Math.abs(first[i]));
            worst = Math.max(worst, Math.abs(first[i] - second[i]) / scale);
        }
        if (worst > 1e-2) {
            throw new IllegalStateException("runs disagree by " + worst);
        }
    }

    private static final class Args {
        private String data;
        private int rows = 500_000;
        private int features = 32;
        private int iterations = 20;
        private int parallelism = 1;
        private int runs = 3;
        private boolean gpu;
        private boolean generate;

        /**
         * Which contraction to use. Only the GEMVs change, so this isolates what cuBLAS
         * contributes: {@code library} is cuBLAS, {@code generated} the naive kernel this project's
         * generator would emit, {@code reduce} TornadoVM's own {@code @Reduce}.
         */
        private String contraction = "library";

        static Args parse(String[] argv) {
            Args args = new Args();
            for (int i = 0; i < argv.length; i++) {
                String flag = argv[i];
                if ("--data".equals(flag)) {
                    args.data = argv[++i];
                } else if ("--rows".equals(flag)) {
                    args.rows = Integer.parseInt(argv[++i]);
                } else if ("--features".equals(flag)) {
                    args.features = Integer.parseInt(argv[++i]);
                } else if ("--iterations".equals(flag)) {
                    args.iterations = Integer.parseInt(argv[++i]);
                } else if ("--parallelism".equals(flag)) {
                    args.parallelism = Integer.parseInt(argv[++i]);
                } else if ("--runs".equals(flag)) {
                    args.runs = Integer.parseInt(argv[++i]);
                } else if ("--gpu".equals(flag)) {
                    args.gpu = Boolean.parseBoolean(argv[++i]);
                } else if ("--contraction".equals(flag)) {
                    String mode = argv[++i];
                    if (!"library".equals(mode)
                            && !"generated".equals(mode)
                            && !"reduce".equals(mode)) {
                        throw new IllegalArgumentException(
                                "--contraction must be library, generated or reduce, not " + mode);
                    }
                    args.contraction = mode;
                } else if ("--generate".equals(flag)) {
                    args.generate = true;
                } else {
                    throw new IllegalArgumentException("unknown argument " + flag);
                }
            }
            if (args.data == null) {
                args.data = "/tmp/flink-gpu-logistic-" + args.rows + "-" + args.features;
            }
            return args;
        }
    }

    private LogisticRegressionBenchmark() {}
}
