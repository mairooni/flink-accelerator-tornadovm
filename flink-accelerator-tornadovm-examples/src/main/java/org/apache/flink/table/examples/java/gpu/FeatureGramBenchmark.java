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
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.gpu.operator.FeatureGramEngine;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.StringJoiner;

/**
 * A query where mixing a generated kernel with a library call is the right answer, measured end to
 * end against the same query on vanilla Flink.
 *
 * <pre>{@code
 * SELECT SUM(f0*f0), SUM(f0*f1), ..., SUM(fd*fd)
 * FROM (SELECT SIN(c0) AS f0, ..., SIN(cd) AS fd FROM t)
 * }</pre>
 *
 * <p>This is the Gram matrix of a feature map: the shape underneath linear regression's normal
 * equations, PCA, and any kernel method that needs {@code X'X} over transformed inputs.
 *
 * <h2>Why this query and not the haversine one</h2>
 *
 * <p>{@link HaversineBenchmark} is elementwise. Every output depends on one row, so the whole
 * expression fuses into a single generated kernel and a BLAS has nothing to contribute -- the only
 * primitive it could offer is BLAS-1, which would cost an extra pass over global memory to do less.
 *
 * <p>Here the aggregate contracts <em>over rows</em>, which no {@code Calc} can express, and that
 * contraction is exactly a rank-{@code k} update. So the two halves want different tools: the
 * feature map is transcendental arithmetic per element, and the contraction wants tiling and
 * shared-memory reuse. {@link FeatureGramEngine} puts both in one task graph, so the features are
 * written to a device buffer that cuBLAS reads in place and the intermediate never returns to the
 * host.
 *
 * <h2>What the comparison is, and is not</h2>
 *
 * <p>The CPU arm is the query above, run by Flink exactly as a user would write it. The GPU arm
 * computes the same matrix through {@code toDataStream} and a custom operator, because the planner
 * cannot yet offload an aggregate -- recognising a set of {@code SUM(fi*fj)} as {@code A'A} is
 * planner work that does not exist.
 *
 * <p>That makes the GPU arm's plumbing <em>worse</em> than a real integration's: it pays for {@code
 * Row} conversion the SQL path avoids. Any margin it shows is therefore a floor, not a ceiling. It
 * also means the number to watch is not the ratio alone but whether the ratio justifies building
 * the planner side at all.
 *
 * <p>Precision differs between the arms by construction. The device path is FP32 because the
 * binding has no FP64 GEMM; Flink's is FP64. The run prints the largest relative disagreement so
 * that cost is visible rather than assumed.
 *
 * <pre>{@code
 * bin/flink run FeatureGramBenchmark.jar --generate --rows 1000000 --cols 16
 * bin/flink run FeatureGramBenchmark.jar --gpu false --runs 6
 * bin/flink run FeatureGramBenchmark.jar --gpu true  --runs 6
 * }</pre>
 */
public final class FeatureGramBenchmark {

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
                "data=%s  rows=%,d  cols=%d  parallelism=%s  gpu=%s  offload=%s  fused=%s  runs=%d  batch=%,d%n",
                parsed.data,
                parsed.rows,
                parsed.cols,
                parsed.parallelism > 0 ? Integer.toString(parsed.parallelism) : "(default)",
                parsed.gpu,
                parsed.offload,
                parsed.fuseAggregate,
                parsed.runs,
                parsed.batch);

        double[] first = null;
        for (int run = 1; run <= parsed.runs; run++) {
            long start = System.nanoTime();
            // Three ways to compute the same matrix. cpuGram is the SQL one, and it is also the
            // offloaded one: the query it builds -- SUM(fi*fj) over a Calc of SIN(ci) -- is exactly
            // the shape the planner fuses, so --offload turns the same arm into a device arm
            // without changing a character of the SQL.
            double[] seen = parsed.gpu ? gpuGram(parsed) : cpuGram(parsed);
            double millis = (System.nanoTime() - start) / 1e6;
            System.out.printf(
                    "run %2d  %10.0f ms  trace=%.6e  entries=%d%n",
                    run, millis, trace(seen, parsed.cols), seen.length);
            if (first == null) {
                first = seen;
            } else {
                agree(first, seen);
            }
        }
    }

    /**
     * The query, as a user would write it: a feature map in a subquery, a Gram matrix over it.
     *
     * <p>Only the upper triangle is selected. {@code A'A} is symmetric, and asking for all {@code
     * d^2} entries would make the CPU arm do twice the work for no extra information -- which would
     * flatter the device rather than measure it.
     */
    private static double[] cpuGram(Args args) throws Exception {
        final StreamTableEnvironment env = environment(args);
        env.executeSql(sourceTable(args));

        StringJoiner select = new StringJoiner(", ");
        for (int i = 0; i < args.cols; i++) {
            for (int j = i; j < args.cols; j++) {
                select.add(String.format("SUM(f%d * f%d)", i, j));
            }
        }
        StringJoiner features = new StringJoiner(", ");
        for (int c = 0; c < args.cols; c++) {
            features.add(String.format("SIN(c%d) AS f%d", c, c));
        }
        String sql =
                String.format(
                        "SELECT %s FROM (SELECT %s FROM Points)", select.toString(), features);

        try (CloseableIterator<Row> it = env.executeSql(sql).collect()) {
            Row row = it.next();
            double[] upper = new double[row.getArity()];
            for (int i = 0; i < upper.length; i++) {
                upper[i] = ((Number) row.getField(i)).doubleValue();
            }
            return upper;
        }
    }

    /**
     * The same matrix, computed on the device by a generated kernel feeding a cuBLAS GEMM.
     *
     * <p>Each subtask contracts its own rows and emits one partial matrix; the client sums them.
     * That is what a local aggregate does, and it is correct for the same reason: {@code A'A} over
     * a partitioned {@code A} is the sum of the partials.
     */
    private static double[] gpuGram(Args args) throws Exception {
        final StreamTableEnvironment env = environment(args);
        env.executeSql(sourceTable(args));

        Table points = env.from("Points");
        DataStream<double[]> partials =
                env.toDataStream(points)
                        .transform(
                                "feature-gram",
                                PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO,
                                new FeatureGramOperator(args.cols, args.batch));

        double[] full = new double[args.cols * args.cols];
        try (CloseableIterator<double[]> it = partials.executeAndCollect()) {
            while (it.hasNext()) {
                double[] partial = it.next();
                for (int i = 0; i < full.length; i++) {
                    full[i] += partial[i];
                }
            }
        }

        // Same upper triangle, same order, so the two arms are directly comparable.
        double[] upper = new double[args.cols * (args.cols + 1) / 2];
        int at = 0;
        for (int i = 0; i < args.cols; i++) {
            for (int j = i; j < args.cols; j++) {
                upper[at++] = full[j * args.cols + i];
            }
        }
        return upper;
    }

    /** Buffers rows, runs the hybrid graph per batch, emits one partial matrix per subtask. */
    private static final class FeatureGramOperator extends AbstractStreamOperator<double[]>
            implements OneInputStreamOperator<Row, double[]>, BoundedOneInput {

        private static final long serialVersionUID = 1L;

        private final int columns;
        private final int batchSize;
        private transient FeatureGramEngine engine;

        private FeatureGramOperator(int columns, int batchSize) {
            this.columns = columns;
            this.batchSize = batchSize;
        }

        @Override
        public void open() throws Exception {
            super.open();
            engine = new FeatureGramEngine(columns, batchSize);
            engine.open();
        }

        @Override
        public void processElement(StreamRecord<Row> element) {
            Row row = element.getValue();
            for (int c = 0; c < columns; c++) {
                engine.set(c, ((Number) row.getField(c)).doubleValue());
            }
            engine.rowComplete();
        }

        @Override
        public void endInput() {
            engine.flush();
            LOG.info(
                    "feature-gram: {} rows in {} batches on this subtask",
                    engine.rowCount(),
                    engine.batchCount());
            output.collect(new StreamRecord<>(engine.total()));
        }

        @Override
        public void close() throws Exception {
            if (engine != null) {
                engine.close();
            }
            super.close();
        }
    }

    /** Writes the input once, so neither arm pays for generating rows. */
    private static void generate(Args args) throws Exception {
        final StreamTableEnvironment env = environment(args);

        StringJoiner schema = new StringJoiner(",\n  ");
        StringJoiner options = new StringJoiner(",\n  ");
        for (int c = 0; c < args.cols; c++) {
            schema.add(String.format("c%d DOUBLE", c));
            // A few radians either side of zero, so SIN is exercised across its range rather than
            // over a slice where it is nearly linear.
            options.add(String.format("'fields.c%d.min' = '-3.0'", c));
            options.add(String.format("'fields.c%d.max' = '3.0'", c));
        }
        env.executeSql(
                String.format(
                        "CREATE TABLE Source (%n  %s%n) WITH (%n  'connector' = 'datagen',%n"
                                + "  'number-of-rows' = '%d',%n  %s%n)",
                        schema, args.rows, options));
        env.executeSql(sourceTable(args));

        System.out.printf("writing %,d rows x %d columns to %s%n", args.rows, args.cols, args.data);
        long start = System.nanoTime();
        env.executeSql("INSERT INTO Points SELECT * FROM Source").await();
        System.out.printf("done in %.0f ms%n", (System.nanoTime() - start) / 1e6);
    }

    private static String sourceTable(Args args) {
        StringJoiner schema = new StringJoiner(",\n  ");
        for (int c = 0; c < args.cols; c++) {
            schema.add(String.format("c%d DOUBLE", c));
        }
        return String.format(
                "CREATE TABLE Points (%n  %s%n) WITH (%n  'connector' = 'filesystem',%n"
                        + "  'path' = '%s',%n  'format' = 'csv'%n)",
                schema, args.data);
    }

    private static StreamTableEnvironment environment(Args args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setRuntimeMode(RuntimeExecutionMode.BATCH);
        if (args.parallelism > 0) {
            env.setParallelism(args.parallelism);
        }
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        if (args.offload) {
            tEnv.getConfig().getConfiguration().setString("table.exec.gpu-offload.enabled", "true");
            if (!args.fuseAggregate) {
                tEnv.getConfig()
                        .getConfiguration()
                        .setString("table.exec.gpu-offload.fuse-aggregate", "false");
            }
        }
        return tEnv;
    }

    /** Sum of the diagonal, a single number that moves if anything about the matrix is wrong. */
    private static double trace(double[] upper, int cols) {
        double sum = 0.0;
        int at = 0;
        for (int i = 0; i < cols; i++) {
            sum += upper[at];
            at += cols - i;
        }
        return sum;
    }

    /**
     * Checks two runs agree.
     *
     * <p>Not bit-for-bit, for the reason {@link HaversineBenchmark} gives: a parallel SUM combines
     * partials in whatever order subtasks finish. Across arms the gap is larger again, because the
     * device path accumulates in FP32.
     */
    private static void agree(double[] first, double[] second) {
        double worst = 0.0;
        for (int i = 0; i < first.length; i++) {
            double scale = Math.max(1.0, Math.abs(first[i]));
            worst = Math.max(worst, Math.abs(first[i] - second[i]) / scale);
        }
        if (worst > 1e-3) {
            throw new IllegalStateException(
                    "runs disagree by " + worst + ", far beyond floating-point reassociation");
        }
    }

    private static final class Args {
        private String data;
        private int rows = 1_000_000;
        private int cols = 16;
        private int parallelism = 1;
        private int runs = 6;
        private int batch = 262_144;
        private boolean gpu;
        private boolean offload;
        private boolean fuseAggregate = true;
        private boolean generate;

        static Args parse(String[] argv) {
            Args args = new Args();
            for (int i = 0; i < argv.length; i++) {
                String flag = argv[i];
                if ("--data".equals(flag)) {
                    args.data = argv[++i];
                } else if ("--rows".equals(flag)) {
                    args.rows = Integer.parseInt(argv[++i]);
                } else if ("--cols".equals(flag)) {
                    args.cols = Integer.parseInt(argv[++i]);
                } else if ("--parallelism".equals(flag)) {
                    args.parallelism = Integer.parseInt(argv[++i]);
                } else if ("--runs".equals(flag)) {
                    args.runs = Integer.parseInt(argv[++i]);
                } else if ("--batch".equals(flag)) {
                    args.batch = Integer.parseInt(argv[++i]);
                } else if ("--offload".equals(flag)) {
                    args.offload = Boolean.parseBoolean(argv[++i]);
                } else if ("--fuse-aggregate".equals(flag)) {
                    args.fuseAggregate = Boolean.parseBoolean(argv[++i]);
                } else if ("--gpu".equals(flag)) {
                    args.gpu = Boolean.parseBoolean(argv[++i]);
                } else if ("--generate".equals(flag)) {
                    args.generate = true;
                } else {
                    throw new IllegalArgumentException("unknown argument " + flag);
                }
            }
            if (args.data == null) {
                args.data = "/tmp/flink-gpu-features-" + args.rows + "-" + args.cols;
            }
            return args;
        }
    }

    private FeatureGramBenchmark() {}
}
