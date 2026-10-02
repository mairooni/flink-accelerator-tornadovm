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

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.StringJoiner;

/**
 * The case for mixing a generated kernel with a library, where the mixture is the point.
 *
 * <p>{@link CudfGroupBySQLExample} puts a kernel and a cuDF call in one task graph, and the library
 * half is a grouped sum. This is the same composition over a shape where the library has something
 * a kernel generator cannot match, and where the fusion earns its keep rather than merely working:
 *
 * <pre>
 *   SELECT SUM(f0*f0), SUM(f0*f1), ..., SUM(fd*fd)
 *   FROM (SELECT SIN(c0) AS f0, ..., SIN(cd) AS fd FROM Points)
 * </pre>
 *
 * <p>That is the Gram matrix {@code A'A} of a feature map -- the shape under linear regression's
 * normal equations, under PCA, and under any kernel method. The two halves want different tools and
 * the query author writes neither:
 *
 * <ul>
 *   <li>the <b>feature map</b> is transcendental arithmetic, one output per input, which is exactly
 *       what {@code AccelKernelGenerator} compiles from the IR;
 *   <li>the <b>contraction</b> is a rank-{@code k} update over every row, which wants tiling and
 *       shared-memory reuse -- a GEMM, and cuBLAS has had decades of tuning that no expression
 *       compiler is going to reproduce.
 * </ul>
 *
 * <h2>Why this one and not the sort</h2>
 *
 * <p>A sorted column is as large as the column that went in, so a device sort hands back everything
 * it was given and the job's time goes on moving rows rather than on ordering them. Here the
 * contraction is what makes the output small: {@code d(d+1)/2} numbers come back no matter how many
 * rows went in. The intermediate -- the whole feature matrix, which is larger than the input --
 * never leaves the device at all, because the generated kernel writes into the buffer cuBLAS then
 * reads in place.
 *
 * <p>Run it with {@code --printBytecodes} and that is the whole argument in four lines:
 *
 * <pre>
 *   TRANSFER_HOST_TO_DEVICE   the staged columns
 *   LAUNCH task gram.features         the feature map, compiled from this SQL
 *   LAUNCH task gram.gemm[cublasDgemm]  cuBLAS, over what the kernel just wrote
 *   TRANSFER_DEVICE_TO_HOST   one matrix
 * </pre>
 *
 * <h2>Why the width matters</h2>
 *
 * <p>The arithmetic grows as {@code d squared} while what crosses the bus grows as {@code d}, so
 * the wider the feature map the better the device does. Below {@link
 * org.apache.flink.table.gpu.codegen.GpuGramSpec#MIN_FEATURES} features the two arms are inside
 * each other's noise and the offload is declined.
 *
 * <h2>What the query author writes</h2>
 *
 * <p>Standard SQL, and {@code NOT NULL} in the DDL. Nothing names cuBLAS, a kernel, or a device.
 */
public final class GramMatrixSQLExample {

    /** Size of the top-left block printed, so the output fits a slide. */
    private static final int SHOW = 3;

    public static void main(String[] args) throws Exception {

        final long rows = args.length > 0 ? Long.parseLong(args[0]) : 1_000_000L;
        final int cols = args.length > 1 ? Integer.parseInt(args[1]) : 64;
        final Path data =
                Paths.get(
                        args.length > 2
                                ? args[2]
                                : "/home/mary/gpu-bench-data/features-" + rows + "-" + cols);
        writeFeatures(data, rows, cols);

        final EnvironmentSettings settings =
                EnvironmentSettings.newInstance().inBatchMode().build();
        final TableEnvironment tableEnv = TableEnvironment.create(settings);

        // ---- the switch, and the whole of it ---------------------------------------------
        //
        // enabled                 offer eligible subtrees to whatever provider is on the
        //                         classpath. Off by default.
        //
        // approximate-projections SIN is a function a device may round differently from
        //                         java.lang.Math. Without this the feature map is refused,
        //                         and refusing it costs the whole query: the Gram matrix is
        //                         only recognised in a Calc fused into the aggregate, so no
        //                         feature map on the device means no GEMM either.
        //
        // fuse-aggregate          the fusion that puts the two halves in one node, which is
        //                         what lets them share one device buffer.
        tableEnv.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
        tableEnv.getConfig()
                .getConfiguration()
                .setString("table.exec.accelerator.approximate-projections", "true");
        tableEnv.getConfig()
                .getConfiguration()
                .setString("table.exec.accelerator.fuse-aggregate", "true");
        tableEnv.getConfig()
                .getConfiguration()
                .setString("table.exec.resource.default-parallelism", "1");
        tableEnv.getConfig().getConfiguration().setString("taskmanager.memory.managed.size", "2g");
        // ----------------------------------------------------------------------------------

        // NOT NULL is the one thing the query author writes for the device: a staged column
        // has no representation for a null.
        final StringJoiner columns = new StringJoiner(",\n  ");
        for (int c = 0; c < cols; c++) {
            columns.add("c" + c + " DOUBLE NOT NULL");
        }
        tableEnv.executeSql(
                "CREATE TABLE Points (\n  "
                        + columns
                        + "\n) WITH (\n"
                        + "  'connector' = 'filesystem',\n"
                        + "  'path' = '"
                        + data
                        + "',\n"
                        + "  'format' = 'csv'\n)");

        // Only the upper triangle: A'A is symmetric, and asking for all d^2 entries would make
        // the CPU arm do twice the work for no extra information.
        final StringJoiner sums = new StringJoiner(", ");
        for (int i = 0; i < cols; i++) {
            for (int j = i; j < cols; j++) {
                sums.add(String.format("SUM(f%d * f%d)", i, j));
            }
        }
        final StringJoiner features = new StringJoiner(", ");
        for (int c = 0; c < cols; c++) {
            features.add(String.format("SIN(c%d) AS f%d", c, c));
        }
        final String sql = "SELECT " + sums + "\nFROM (SELECT " + features + " FROM Points)";

        if (System.getenv("EXPLAIN") != null) {
            final String plan = tableEnv.explainSql(sql);
            System.out.println(plan.substring(plan.indexOf("== GPU Offload ==")));
        }

        final double[] upper;
        try (CloseableIterator<Row> it = tableEnv.executeSql(sql).collect()) {
            final Row row = it.next();
            upper = new double[row.getArity()];
            for (int i = 0; i < upper.length; i++) {
                upper[i] = ((Number) row.getField(i)).doubleValue();
            }
        }

        System.out.printf(
                "%n%,d rows x %d features -> %d matrix entries%n", rows, cols, upper.length);
        System.out.printf("trace = %.6e%n%n", trace(upper, cols));
        System.out.printf("top-left %dx%d of A'A:%n", SHOW, SHOW);
        for (int i = 0; i < Math.min(SHOW, cols); i++) {
            final StringBuilder line = new StringBuilder("  ");
            for (int j = 0; j < Math.min(SHOW, cols); j++) {
                final int a = Math.min(i, j);
                final int b = Math.max(i, j);
                line.append(String.format("%14.4f", upper[triangleIndex(cols, a, b)]));
            }
            System.out.println(line);
        }
    }

    /** Where entry {@code (i, j)} of the symmetric matrix lands in the upper-triangle row. */
    private static int triangleIndex(int d, int i, int j) {
        return i * d - (i * (i - 1)) / 2 + (j - i);
    }

    /** Sum of the diagonal: one number that moves if anything about the matrix is wrong. */
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
     * Writes the input table, once.
     *
     * <p>Plain Java rather than the {@code datagen} connector, for the reason {@link
     * KernelAndLibrarySQLExample} gives: {@code datagen} costs about 100 microseconds a row.
     */
    private static void writeFeatures(Path directory, long rows, int cols) throws IOException {
        if (Files.isDirectory(directory)) {
            return;
        }
        Files.createDirectories(directory);
        final Random random = new Random(20260929L);
        try (BufferedWriter out = Files.newBufferedWriter(directory.resolve("points.csv"))) {
            final StringBuilder line = new StringBuilder(cols * 20);
            for (long i = 0; i < rows; i++) {
                line.setLength(0);
                for (int c = 0; c < cols; c++) {
                    if (c > 0) {
                        line.append(',');
                    }
                    line.append(random.nextDouble() * 6.0);
                }
                line.append('\n');
                out.write(line.toString());
            }
        }
    }

    private GramMatrixSQLExample() {}
}
