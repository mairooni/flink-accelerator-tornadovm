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

/**
 * A Flink SQL {@code ORDER BY} executed by RAPIDS cuDF.
 *
 * <p>The query sorts four million events by key. It is ordinary Flink SQL — no hint, no annotation,
 * no accelerator type. The only thing in this file that is not in {@code WordCountSQLExample} is
 * the block marked <em>the switch</em>: three session settings, which a cluster operator sets and a
 * query author never sees.
 *
 * <h2>Seeing that it really ran there</h2>
 *
 * <p>{@code -Dtornado.printKernel=true}, which is what {@link HaversineSQLExample} is verified
 * with, prints <b>nothing</b> here — and that absence is the point of this example rather than a
 * gap in it. There is no generated kernel to print. Flink's {@code SortOperator} is replaced by one
 * that hands the whole operation to a library that already implements it: {@code
 * cudf::stable_sorted_order}, reached through TornadoVM's {@code tornado-cudf} binding as a library
 * task in a task graph.
 *
 * <p>What shows that is {@code -Dtornado.print.bytecodes=true} — {@code scripts/run-sql-demos.sh
 * cudf-sort --printBytecodes} — which prints what TornadoVM's interpreter actually did:
 *
 * <pre>
 *   bc: TRANSFER_HOST_TO_DEVICE_ALWAYS IntArray
 *           on [NVIDIA CUDA] -- NVIDIA GeForce RTX 4070 Laptop GPU
 *   bc: LAUNCH task - sort.order[sortedOrder]
 *           on [NVIDIA CUDA] -- NVIDIA GeForce RTX 4070 Laptop GPU
 *   bc: TRANSFER_DEVICE_TO_HOST_ALWAYS_BLOCKING IntArray
 *           on [NVIDIA CUDA] -- NVIDIA GeForce RTX 4070 Laptop GPU
 * </pre>
 *
 * <p>The key column goes up, {@code sortedOrder} is launched on the card, and a permutation comes
 * back. The two halves of the integration compose: a library task and a generated kernel can sit in
 * one task graph and read each other's device buffers without a round trip through the host.
 *
 * <h2>What crosses the bus, and what does not</h2>
 *
 * <p>Only the key column — {@code cudf::sorted_order} returns a permutation rather than sorted
 * data, so 16 MB goes up and 16 MB comes back instead of the whole 80 MB table. Applying that
 * permutation, and building the output rows, stays on the host.
 *
 * <h2>The one thing the query author has to write</h2>
 *
 * <p>{@code NOT NULL} on the sort key. The shim builds a device column with no null mask, so a null
 * key would order as whatever its bits happened to be; the provider refuses a nullable key rather
 * than guessing. Note what is <em>not</em> needed here: {@code approximate-projections}, which
 * {@link HaversineSQLExample} sets. A sort computes no values — it moves rows — so there is no
 * rounding for a device to differ about.
 *
 * <h2>Running it</h2>
 *
 * <p>This one needs more than a card: {@code libtornado-cudf.so} must be built against RAPIDS
 * libcudf, and its dependencies must be on {@code LD_LIBRARY_PATH}. The script assembles all of it.
 *
 * <pre>
 *   scripts/run-sql-demos.sh cudf-sort --printBytecodes
 *   scripts/run-sql-demos.sh cudf-sort --rows 8000000
 * </pre>
 *
 * <p><b>Do not go below about two million rows.</b> Setting a device up costs a fixed ~300 ms, and
 * the accelerator refuses work that cannot repay it — at one million rows this query is declined
 * with {@code 1225000 rows do not repay 300 ms of setup}, runs correctly on {@code SortOperator},
 * and prints no {@code LAUNCH}. That is the cost model working, but on a stage it looks like a
 * broken demo. Measured on an RTX 4070 Laptop: declined at 1M, taken at 2M and at the 4M default.
 * {@link HaversineSQLExample} has far more arithmetic per row and clears the same floor at 500,000.
 *
 * <p>Without the shim the provider declines, the query runs on {@code SortOperator}, and the answer
 * is still right — which is worth showing once, because nothing about the query depends on the
 * device being there. Run with {@code VERBOSE=1} to see the accelerator say so.
 */
public final class CudfSortSQLExample {

    /** Rows printed before the rest are drained; the whole result is far too long to show. */
    private static final int HEAD = 20;

    public static void main(String[] args) throws Exception {

        final long rows = args.length > 0 ? Long.parseLong(args[0]) : 4_000_000L;
        final Path data =
                Paths.get(args.length > 1 ? args[1] : "/tmp/flink-gpu-demo-events-" + rows);
        writeEvents(data, rows);

        // set up the Table API
        final EnvironmentSettings settings =
                EnvironmentSettings.newInstance().inBatchMode().build();
        final TableEnvironment tableEnv = TableEnvironment.create(settings);

        // ---- the switch, and the whole of it ---------------------------------------------
        //
        // Three session settings. Nothing below this block knows an accelerator exists.
        //
        // enabled             offer eligible subtrees to whatever provider is on the
        //                     TaskManager's classpath. Off by default.
        //
        // managed.size        not an accelerator setting. The staging buffers are the slot's share
        //                     of managed memory, and a sort has to hold a whole partition rather
        //                     than a batch of it. Set here only because this runs in an in-process
        //                     MiniCluster, whose default share is smaller than four million rows
        //                     need; on a cluster it lives in config.yaml. If it is too small the
        //                     provider declines the partition rather than spilling, and says so.
        //
        // default-parallelism also not an accelerator setting, but needed, because
        //                     table.exec.accelerator.max-parallelism defaults to 1 and the offload
        //                     is declined above it. That default is a live workaround for an
        //                     intermittent CUDA launch failure at higher parallelism on one card.
        tableEnv.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
        tableEnv.getConfig().getConfiguration().setString("taskmanager.memory.managed.size", "2g");
        tableEnv.getConfig()
                .getConfiguration()
                .setString("table.exec.resource.default-parallelism", "1");
        // ----------------------------------------------------------------------------------

        // NOT NULL on k is the one thing the query author has to write; see the class comment.
        // seq is a BIGINT payload the sort carries but never looks at, which is the point of it: a
        // device sort does not need every column to be a key, or even to leave the host.
        tableEnv.executeSql(
                "CREATE TABLE Events (\n"
                        + "  k INT NOT NULL,\n"
                        + "  seq BIGINT NOT NULL,\n"
                        + "  val DOUBLE NOT NULL\n"
                        + ") WITH (\n"
                        + "  'connector' = 'filesystem',\n"
                        + "  'path' = '"
                        + data
                        + "',\n"
                        + "  'format' = 'csv'\n"
                        + ")");

        // execute a Flink SQL job and print the result locally -- the head of it, because the whole
        // result is four million rows and printing them would take longer than ordering them did.
        // Every row is still drained, and the order of every one is checked on the way past: a sort
        // reassociates nothing, so anything out of order is a defect rather than rounding.
        long seen = 0;
        int previous = Integer.MIN_VALUE;
        try (CloseableIterator<Row> ordered =
                tableEnv.executeSql("SELECT k, seq, val FROM Events ORDER BY k").collect()) {
            while (ordered.hasNext()) {
                final Row row = ordered.next();
                final int key = (Integer) row.getField(0);
                if (key < previous) {
                    throw new IllegalStateException(
                            "row " + seen + " has key " + key + " after " + previous);
                }
                if (seen < HEAD) {
                    System.out.println(row);
                }
                previous = key;
                seen++;
            }
        }
        System.out.printf("...%n%,d rows, every one in order.%n", seen);
    }

    /**
     * Writes the input table, once.
     *
     * <p>The key is scrambled rather than sequential, because a sort over already-sorted input
     * demonstrates nothing. The multiplier is odd, so {@code id * 2654435761 mod rows} is a
     * permutation of {@code [0, rows)} and every key appears exactly once — which is also what
     * makes the printed head recognisable as a sort rather than as the input order.
     *
     * <p>Plain Java rather than the {@code datagen} connector, and that is a demo decision rather
     * than a stylistic one: {@code datagen} costs about 100 microseconds a row, so four million
     * rows take six minutes to produce. This writes them in about a third of a second.
     */
    private static void writeEvents(Path directory, long rows) throws IOException {
        if (Files.isDirectory(directory)) {
            return;
        }
        Files.createDirectories(directory);
        try (BufferedWriter out = Files.newBufferedWriter(directory.resolve("events.csv"))) {
            final StringBuilder line = new StringBuilder(48);
            for (long id = 0; id < rows; id++) {
                line.setLength(0);
                line.append(id * 2654435761L % rows)
                        .append(',')
                        .append(id)
                        .append(',')
                        .append(id * 0.25)
                        .append('\n');
                out.write(line.toString());
            }
        }
    }

    private CudfSortSQLExample() {}
}
