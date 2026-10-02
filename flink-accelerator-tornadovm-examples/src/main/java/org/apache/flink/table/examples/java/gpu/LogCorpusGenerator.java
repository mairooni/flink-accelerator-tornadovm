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

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Writes the access-log corpus {@link GrokSQLExample} reads, as single-column Parquet.
 *
 * <pre>
 *   java -cp &lt;flink lib&gt;:&lt;this jar&gt;:$HADOOP_CLASSPATH \
 *        org.apache.flink.table.examples.java.gpu.LogCorpusGenerator /path/to/data 16000000 64000000
 * </pre>
 *
 * <p>One directory per row count — {@code logs16}, {@code logs64} — each holding files of
 * {@value #ROWS_PER_FILE} lines.
 *
 * <h2>Why fixed-size files rather than one big one</h2>
 *
 * <p>The device region sizes its buffers from a file's row count and rebuilds its execution plan
 * when that changes. Equal files mean one plan for the whole partition, and §T56 measured the
 * difference: the first draft of the region allocated per file and spent half its time there.
 *
 * <h2>Why CSV first and Parquet by conversion</h2>
 *
 * <p>Flink's own writer is the only Parquet writer on this classpath, and it writes through a
 * table sink. Generating the text in plain Java and converting is both faster and fewer
 * dependencies than the {@code datagen} connector, which costs about 100 microseconds a row.
 *
 * <p>Each chunk converts at parallelism 1 so it lands as exactly one file, which is then moved
 * into place under a name the reader sorts predictably.
 */
public final class LogCorpusGenerator {

    /** Lines per Parquet file. 500k keeps a file's score buffers inside a consumer card. */
    public static final int ROWS_PER_FILE = 500_000;

    private static final String[] VERBS = {"GET", "POST", "PUT", "DELETE", "HEAD"};
    private static final String[] PATHS = {
        "/api/v1/users", "/api/v2/orders", "/api/v3/items", "/static/app.js",
        "/health", "/api/v1/search", "/images/logo.png", "/api/v11/metrics"
    };
    private static final String[] AGENTS = {
        "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/129.0",
        "curl/8.5.0",
        "python-requests/2.31.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    };
    private static final int[] STATUSES = {200, 200, 200, 201, 301, 404, 500};

    private LogCorpusGenerator() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            throw new IllegalArgumentException(
                    "usage: LogCorpusGenerator <data-root> <rows>...  e.g. /path/to/data 16000000 64000000");
        }
        final Path root = Paths.get(args[0]);
        Files.createDirectories(root);

        final EnvironmentSettings settings =
                EnvironmentSettings.newInstance().inBatchMode().build();
        final TableEnvironment env = TableEnvironment.create(settings);
        env.getConfig().getConfiguration().setString("table.exec.resource.default-parallelism", "1");

        for (int a = 1; a < args.length; a++) {
            final long rows = Long.parseLong(args[a]);
            final Path out = root.resolve("logs" + (rows / 1_000_000));
            if (Files.isDirectory(out) && Files.list(out).findAny().isPresent()) {
                System.out.printf("%s already present, skipping%n", out);
                continue;
            }
            Files.createDirectories(out);
            final int files = (int) (rows / ROWS_PER_FILE);
            final Random random = new Random(42 + a);
            System.out.printf("writing %,d lines into %d files under %s%n", rows, files, out);
            for (int f = 0; f < files; f++) {
                final Path csv = root.resolve("staging-csv-" + f);
                writeChunk(csv, ROWS_PER_FILE, random);
                convert(env, "C" + a + "_" + f, csv, out, f);
                deleteTree(csv);
                if ((f + 1) % 16 == 0) {
                    System.out.printf("  %d/%d files%n", f + 1, files);
                }
            }
        }
        System.out.println("done");
    }

    private static void writeChunk(Path dir, int rows, Random random) throws IOException {
        Files.createDirectories(dir);
        try (BufferedWriter w = Files.newBufferedWriter(dir.resolve("data.csv"))) {
            final StringBuilder line = new StringBuilder(200);
            for (int i = 0; i < rows; i++) {
                line.setLength(0);
                line.append("10.0.").append(random.nextInt(512) / 256).append('.')
                    .append(random.nextInt(256))
                    .append(" - - [01/Oct/2026:16:0").append(i % 10).append(":00 +0000] \"")
                    .append(VERBS[random.nextInt(VERBS.length)]).append(' ')
                    .append(PATHS[random.nextInt(PATHS.length)]).append(" HTTP/1.1\" ")
                    .append(STATUSES[random.nextInt(STATUSES.length)]).append(' ')
                    .append(120 + random.nextInt(89_880)).append(' ')
                    .append(1 + random.nextInt(4000)).append(" \"-\" \"")
                    .append(AGENTS[random.nextInt(AGENTS.length)]).append('"');
                w.write(line.toString());
                w.newLine();
            }
        }
    }

    private static void convert(TableEnvironment env, String tag, Path csv, Path into, int index)
            throws Exception {
        final Path staging = csv.resolveSibling(csv.getFileName() + "-pq");
        env.executeSql(ddl(tag + "Csv", csv, "csv"));
        env.executeSql(ddl(tag + "Pq", staging, "parquet"));
        env.executeSql("INSERT INTO " + tag + "Pq SELECT * FROM " + tag + "Csv").await();
        try (Stream<Path> files = Files.list(staging)) {
            final List<Path> written = new ArrayList<>();
            files.filter(p -> !p.getFileName().toString().startsWith(".")).forEach(written::add);
            if (written.size() != 1) {
                throw new IllegalStateException(
                        "expected one Parquet file from a parallelism-1 insert, got " + written.size());
            }
            Files.move(written.get(0), into.resolve(String.format("part-%05d.parquet", index)),
                    StandardCopyOption.REPLACE_EXISTING);
        }
        deleteTree(staging);
    }

    private static String ddl(String name, Path path, String format) {
        final StringBuilder ddl = new StringBuilder("CREATE TABLE ").append(name)
                .append(" (\n  line STRING NOT NULL\n) WITH (\n  'connector' = 'filesystem',\n")
                .append("  'path' = '").append(path.toAbsolutePath()).append("',\n")
                .append("  'format' = '").append(format).append("'");
        if ("csv".equals(format)) {
            // A log line contains commas and double quotes, so the defaults would split it and
            // then try to unescape it. One column, a delimiter that cannot occur, and no quote
            // character at all: the line survives the round trip exactly as written.
            ddl.append(",\n  'csv.field-delimiter' = U&'\\0001',\n")
               .append("  'csv.disable-quote-character' = 'true'");
        }
        return ddl.append("\n)").toString();
    }

    private static void deleteTree(Path path) throws IOException {
        if (!Files.exists(path)) {
            return;
        }
        try (Stream<Path> walk = Files.walk(path)) {
            walk.sorted(Comparator.reverseOrder()).forEach(p -> {
                try {
                    Files.delete(p);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });
        }
    }
}
