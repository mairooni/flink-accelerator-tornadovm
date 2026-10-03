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
import org.apache.flink.table.api.TableResult;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/**
 * Log screening as ordinary batch SQL: how many lines satisfy every one of k regular expressions.
 *
 * <pre>
 *   SELECT COUNT(*) FROM Logs
 *   WHERE REGEXP(line, 'p1') AND REGEXP(line, 'p2') AND ... AND REGEXP(line, 'pk')
 * </pre>
 *
 * <h2>Why k matters more than anything else here</h2>
 *
 * <p>A device reads this file about 11x faster than one Flink slot and matches a pattern against it
 * hundreds of times faster, but the read is a fixed cost and the matching is the marginal one. At
 * one pattern the matching is a few milliseconds under a much larger read and the end-to-end ratio
 * is the reader's; at eight it is most of the query and the ratio is the library's. Each extra
 * pattern costs this device about 6.6 ms and four CPU cores about 271 ms.
 *
 * <p>That is the same lever as the Gram matrix's feature count and the nearest-neighbour join's
 * pair count, and it is the only one this project has found that moves a device result: arithmetic
 * per byte read.
 *
 * <h2>What the query author writes</h2>
 *
 * <p>Standard SQL, and {@code NOT NULL} in the DDL. Nothing names a kernel, a library or a device.
 *
 * <pre>
 *   flink run GrokSQLExample.jar --data /path/to/logs --patterns 8
 * </pre>
 */
public final class GrokSQLExample {

    /**
     * Patterns that every well-formed access-log line satisfies.
     *
     * <p>Chosen to match rather than to reject, so that {@code AND} cannot short-circuit: a pattern
     * that rejects most rows lets the CPU skip the rest of the conjunction, and the device has no
     * equivalent. Measuring against an arm that skips most of its work would flatter this one.
     */
    /**
     * Patterns that most lines fail, most-selective first.
     *
     * <p>The opposite measurement to {@link #PATTERNS}, and the one that tests whether the result
     * survives a predicate anybody would actually write. SQL's {@code AND} short-circuits, so a
     * first pattern that rejects three rows in four lets the CPU skip the other seven on those
     * rows; the device has no equivalent and evaluates all of them on all rows. This is therefore
     * the device's worst case and the CPU's best.
     */
    private static final String[] SELECTIVE = {
        "(GET|POST) /api/v[0-9]+/[a-z]+",
        "Firefox/1[0-9]+",
        "/api/v[0-9]+/(users|orders)",
        "\" (500|404) ",
        "curl/8",
        "python-requests",
        "\"DELETE /",
        "/static/app\\.js",
    };

    private static final String[] PATTERNS = {
        "10\\.0\\.[0-9]+\\.[0-9]+ -",
        "\\[[0-9]{2}/[A-Z][a-z]{2}/[0-9]{4}",
        "01/Oct/2026:16:0[0-9]",
        "\\+0000\\]",
        "\"(GET|POST|PUT|DELETE|HEAD) /",
        "HTTP/1\\.1\" [2-5][0-9]{2}",
        "[0-9]+ [0-9]+ \"-\"",
        "\" \"[A-Za-z]",
    };

    private GrokSQLExample() {}

    public static void main(String[] args) throws Exception {
        String data = BenchData.resolve("logs").toString();
        int patterns = 8;
        int parallelism = 1;
        boolean explain = false;
        boolean selective = false;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--data":
                    data = args[++i];
                    break;
                case "--patterns":
                    patterns = Integer.parseInt(args[++i]);
                    break;
                case "--parallelism":
                    parallelism = Integer.parseInt(args[++i]);
                    break;
                case "--selective":
                    selective = true;
                    break;
                case "--explain":
                    explain = true;
                    break;
                default:
                    throw new IllegalArgumentException("unknown argument " + args[i]);
            }
        }

        final EnvironmentSettings settings =
                EnvironmentSettings.newInstance().inBatchMode().build();
        final TableEnvironment env = TableEnvironment.create(settings);
        env.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
        env.getConfig()
                .getConfiguration()
                .setString(
                        "table.exec.resource.default-parallelism", Integer.toString(parallelism));
        // Nothing else. An ungrouped COUNT(*) already plans as a two-phase hash aggregate, so
        // this shape needs none of the planner hints the nearest-neighbour join example sets --
        // checked rather than assumed, by running it both ways.

        env.executeSql(
                "CREATE TABLE Logs (\n  line STRING NOT NULL\n) WITH (\n"
                        + "  'connector' = 'filesystem',\n  'path' = '"
                        + data
                        + "',\n  'format' = 'parquet'\n)");

        final StringBuilder where = new StringBuilder();
        for (int i = 0; i < patterns; i++) {
            if (i > 0) {
                where.append("\n  AND ");
            }
            where.append("REGEXP(line, '")
                    .append((selective ? SELECTIVE : PATTERNS)[i % PATTERNS.length])
                    .append("')");
        }
        final String sql = "SELECT COUNT(*) AS matched FROM Logs WHERE " + where;

        if (explain) {
            System.out.println(env.explainSql(sql));
        }
        System.out.printf(
                "grok: patterns=%d parallelism=%d selective=%s data=%s%n",
                patterns, parallelism, selective, data);

        final long started = System.nanoTime();
        final TableResult result = env.executeSql(sql);
        Row row = null;
        try (CloseableIterator<Row> it = result.collect()) {
            while (it.hasNext()) {
                row = it.next();
            }
        }
        System.out.printf(
                "matched=%s  query=%.2f s%n",
                row == null ? "none" : row.getField(0), (System.nanoTime() - started) / 1e9);
    }
}
