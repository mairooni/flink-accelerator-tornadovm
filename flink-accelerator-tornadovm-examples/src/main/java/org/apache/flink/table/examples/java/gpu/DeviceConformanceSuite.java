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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Runs one SQL statement on the CPU or on a device and prints its answer bit for bit.
 *
 * <h2>Why the output is in bits</h2>
 *
 * <p>Conformance questions are about values that print the same and are not the same. {@code 0.0}
 * and {@code -0.0} have the same decimal form; so do two NaNs with different payloads; so do two
 * doubles differing in the last ulp once a formatter has rounded them. Printing {@code
 * doubleToRawLongBits} removes the formatter from the comparison entirely, so a difference cannot
 * hide in it and no tolerance has to be chosen.
 *
 * <p>NaN is the one exception, and deliberately: Flink's SQL semantics say a NaN is a NaN, and
 * nothing in the language distinguishes payloads. So NaNs are canonicalised to a single token
 * rather than compared as bits — asserting identical payload bits would be asserting something
 * Flink does not promise.
 *
 * <h2>Why the special values are computed rather than stored</h2>
 *
 * <p>Infinities, NaNs and signed zeros are not written into the input file. They are produced by
 * the query, from ordinary finite columns: {@code z/z} for NaN, {@code a/z} for an infinity, {@code
 * neg * z} for a negative zero, {@code big * big} for overflow. That is how they arise in a real
 * query, it avoids depending on whether a CSV reader accepts the token {@code NaN}, and it puts the
 * special value inside the kernel rather than in the data it loads.
 *
 * <h2>Usage</h2>
 *
 * <pre>
 *   flink run DeviceConformanceSuite.jar --generate --rows 2000000 --data /tmp/conf
 *   flink run DeviceConformanceSuite.jar --data /tmp/conf --gpu true --sql "SELECT ..."
 * </pre>
 */
public final class DeviceConformanceSuite {

    public static void main(String[] args) throws Exception {
        String data = "/tmp/conf";
        String sql = null;
        String dump = null;
        boolean schema = false;
        int vectorDim = 0;
        boolean gpu = false;
        boolean generate = false;
        boolean explain = false;
        long rows = 2_000_000L;
        int parallelism = 1;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--data":
                    data = args[++i];
                    break;
                case "--sql":
                    sql = args[++i];
                    break;
                case "--gpu":
                    gpu = Boolean.parseBoolean(args[++i]);
                    break;
                case "--generate":
                    generate = true;
                    break;
                case "--explain":
                    explain = true;
                    break;
                case "--dump":
                    dump = args[++i];
                    break;
                case "--schema":
                    schema = true;
                    break;
                case "--vectors":
                    // Point at the vector dataset instead of the conformance one, so the shape
                    // that actually diverges can be taken apart with hand-written SQL.
                    vectorDim = Integer.parseInt(args[++i]);
                    break;
                case "--rows":
                    rows = Long.parseLong(args[++i]);
                    break;
                case "--parallelism":
                    parallelism = Integer.parseInt(args[++i]);
                    break;
                default:
                    throw new IllegalArgumentException("unknown flag: " + args[i]);
            }
        }

        if (generate) {
            generate(data, rows);
            return;
        }

        final TableEnvironment env =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        env.getConfig()
                .getConfiguration()
                .setString("parallelism.default", Integer.toString(parallelism));
        if (gpu) {
            env.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
        }
        env.executeSql(vectorDim > 0 ? vectorDdl(data, vectorDim) : ddl(data));
        if (explain) {
            System.out.println(env.explainSql(sql));
        }
        if (schema) {
            // The resolved type of the expression, which is the thing the planner actually
            // reasoned about. Reading it beats inferring it from the SQL text: a bare decimal is
            // not a DOUBLE, and what the arithmetic around it resolves to is the whole question.
            System.out.println("SCHEMA " + env.sqlQuery(sql).getResolvedSchema());
            return;
        }

        if (dump != null) {
            // The whole population, not a summary: MIN and MAX only see the extremes, and a
            // disagreement in the middle of two million rows is exactly what a summary hides.
            // CSV round-trips a double exactly through Double.toString, which the earlier
            // VectorScreen dumps confirmed by resolving differences of a single ulp.
            env.executeSql(
                    "CREATE TABLE Dumped (\n  id INT NOT NULL,\n  v DOUBLE\n)"
                            + " WITH (\n  'connector' = 'filesystem',\n  'path' = '"
                            + dump
                            + "',\n  'format' = 'csv'\n)");
            env.executeSql("INSERT INTO Dumped " + sql).await();
            System.out.println("dumped to " + dump);
            return;
        }

        final List<String> out = new ArrayList<>();
        try (CloseableIterator<Row> it = env.sqlQuery(sql).execute().collect()) {
            while (it.hasNext()) {
                out.add(canonical(it.next()));
            }
        }
        // Sorted, because row order is not part of a SQL answer without ORDER BY and two arms may
        // legitimately produce a different one.
        Collections.sort(out);
        for (String line : out) {
            System.out.println("ROW " + line);
        }
        System.out.println("ROWS " + out.size());
    }

    /** Every field as an exact token: doubles and floats by their bits, NaN canonicalised. */
    private static String canonical(Row row) {
        final StringBuilder sb = new StringBuilder();
        for (int i = 0; i < row.getArity(); i++) {
            if (i > 0) {
                sb.append('|');
            }
            final Object value = row.getField(i);
            if (value == null) {
                sb.append("NULL");
            } else if (value instanceof Double) {
                final double d = (Double) value;
                sb.append(Double.isNaN(d) ? "NaN" : "d" + Double.doubleToRawLongBits(d));
            } else if (value instanceof Float) {
                final float f = (Float) value;
                sb.append(Float.isNaN(f) ? "NaN" : "f" + Float.floatToRawIntBits(f));
            } else {
                sb.append(value);
            }
        }
        return sb.toString();
    }

    private static String vectorDdl(String path, int dim) {
        final StringBuilder sb = new StringBuilder("CREATE TABLE Conf (\n  id INT NOT NULL");
        for (int i = 0; i < dim; i++) {
            sb.append(",\n  v").append(i).append(" DOUBLE NOT NULL");
        }
        return sb.append("\n) WITH (\n  'connector' = 'filesystem',\n  'path' = '")
                .append(path)
                .append("',\n  'format' = 'parquet'\n)")
                .toString();
    }

    private static String ddl(String path) {
        return "CREATE TABLE Conf (\n"
                + "  id INT NOT NULL,\n"
                + "  a DOUBLE NOT NULL,\n" // ordinary magnitudes
                + "  b DOUBLE NOT NULL,\n" // ordinary magnitudes
                + "  c DOUBLE NOT NULL,\n" // exactly -(a*b) on the crafted rows
                + "  z DOUBLE NOT NULL,\n" // zero on every row
                + "  neg DOUBLE NOT NULL,\n" // negative on every row
                + "  big DOUBLE NOT NULL,\n" // near the top of the range
                + "  tiny DOUBLE NOT NULL,\n" // subnormal on the crafted rows
                + "  f FLOAT NOT NULL,\n"
                + "  n DOUBLE\n" // nullable on purpose: must be declined
                + ") WITH (\n"
                + "  'connector' = 'filesystem',\n"
                + "  'path' = '"
                + path
                + "',\n"
                + "  'format' = 'csv',\n"
                // Without this an empty field is handed to Double.parseDouble and the source dies
                // with NumberFormatException before the accelerator is reached at all.
                + "  'csv.null-literal' = ''\n"
                + ")";
    }

    /**
     * Ordinary rows, with crafted ones seeded through them.
     *
     * <p>Written directly rather than through Flink so the file contains exactly the doubles
     * intended: a round trip through a generator and a writer is another place for a value to be
     * rounded, and the cancellation cases only work if {@code c} is bit-exactly minus the rounded
     * product of {@code a} and {@code b}.
     */
    private static void generate(String dir, long rows) throws Exception {
        final Path out = Paths.get(dir);
        Files.createDirectories(out);
        final double[][] crafted = {
            {1.0000000001, 1.0000000003},
            {3.0000000000000004, 7.000000000000001},
            {1.4142135623730951, 1.4142135623730951},
            {0.1, 0.3},
            {1e8 + 1, 1e8 + 3},
            {2.718281828459045, 3.141592653589793},
            {1.7976931348623157e16, 1.0000000000000002},
        };
        final Random random = new Random(20260923L);
        try (BufferedWriter w =
                Files.newBufferedWriter(out.resolve("part-0.csv"), StandardCharsets.UTF_8)) {
            for (long i = 0; i < rows; i++) {
                final double a;
                final double b;
                final double tiny;
                if (i < crafted.length) {
                    a = crafted[(int) i][0];
                    b = crafted[(int) i][1];
                    tiny = Double.MIN_VALUE * (i + 1); // subnormal
                } else {
                    a = random.nextDouble() * 4.0 - 2.0;
                    b = random.nextDouble() * 4.0 - 2.0;
                    tiny = Double.MIN_NORMAL * random.nextDouble();
                }
                final double c = -(a * b);
                w.write(
                        i
                                + ","
                                + repr(a)
                                + ","
                                + repr(b)
                                + ","
                                + repr(c)
                                + ",0.0,-1.5,"
                                + repr(1.5e308)
                                + ","
                                + repr(tiny)
                                + ","
                                + (float) (a * 0.5f)
                                + ",");
                // n: empty every third row, so the nullable column really is nullable.
                if (i % 3 != 0) {
                    w.write(repr(a));
                }
                w.write('\n');
            }
        }
        System.out.printf("wrote %,d rows to %s%n", rows, out.resolve("part-0.csv"));
    }

    /** Seventeen significant digits round-trips every double exactly. */
    private static String repr(double d) {
        return Double.toString(d);
    }

    private DeviceConformanceSuite() {}
}
