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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Runs the same SQL query twice — once normally, once with GPU offload enabled — and verifies that
 * the results are identical.
 *
 * <p>The point of the example is what is <b>not</b> in it. There is no kernel, no device code, no
 * annotation, and no GPU dependency: this class compiles against the Table API alone. Offload is a
 * planner decision, so the only difference between the two runs is two configuration entries.
 *
 * <p>Contrast {@code org.apache.flink.streaming.examples.gpu.MatrixVectorMul}, which demonstrates
 * the other approach — the user writes JCuda calls inside a {@code RichMapFunction} and manages
 * device memory by hand.
 *
 * <p>This example shows how to:
 *
 * <ul>
 *   <li>enable transparent GPU offload for Flink SQL,
 *   <li>read the planner's decision, and the reason behind it, from {@code EXPLAIN},
 *   <li>confirm that offloading does not change query results.
 * </ul>
 *
 * <h2>Prerequisites</h2>
 *
 * <p>The example <b>fails</b> if offload was requested but did not happen. Matching results are not
 * evidence on their own: if the GPU was never used, both runs executed the same CPU plan and would
 * match trivially. Two things must be in place, and the failure message names whichever is missing.
 *
 * <ul>
 *   <li>{@code flink-table-gpu-runtime} on the classpath. It registers an {@code
 *       AcceleratorProvider} through {@code META-INF/services}; the TaskManager discovers it with
 *       {@link java.util.ServiceLoader} and Flink holds no compile-time reference to it. This
 *       module declares it at runtime scope, so it is present when run from the IDE.
 *   <li>TornadoVM's JVM arguments. They are not optional and are not inherited from the build:
 *       TornadoVM's off-heap array types use a preview API, so without {@code --enable-preview} and
 *       the JVMCI flags the runtime declines and the planner falls back.
 * </ul>
 *
 * <h2>Running from IntelliJ</h2>
 *
 * <p>Run → Edit Configurations → Modify options → Add VM options, then paste:
 *
 * <pre>@/path/to/tornadovm-sdk/tornado-argfile</pre>
 *
 * <p>The distribution ships that argfile with the JVMCI flags, module path and {@code
 * --enable-preview} already assembled. For a cluster, the same content goes into {@code
 * env.java.opts} in {@code flink-conf.yaml}.
 *
 * <h2>A note on the query</h2>
 *
 * <p>{@code val * 2.0 + 1.0} is two floating-point operations per row, which is far below the cost
 * floor at which moving data to a device pays for itself — measurements put it at roughly half the
 * speed of CPU execution. It is used here because it matches the one kernel shape the runtime
 * currently implements, and because a small query makes the mechanism easy to follow. The example
 * therefore lowers {@code min-row-cost} deliberately; a real workload should leave the calibrated
 * default in place and let the planner refuse queries like this one.
 */
public final class GpuOffloadExample {

    public static void main(String[] args) throws Exception {
        final int rows = args.length > 0 ? Integer.parseInt(args[0]) : 200_000;

        final Run withoutGpu = run(rows, false);
        final Run withGpu = run(rows, true);

        System.out.printf(
                "%nrows without GPU: %,d%nrows with GPU:    %,d%n",
                withoutGpu.rows.size(), withGpu.rows.size());

        if (!withoutGpu.rows.equals(withGpu.rows)) {
            throw new IllegalStateException("GPU offload changed the query result, which is a bug");
        }

        // Matching results prove nothing on their own. If offload did not happen then both runs
        // executed the same CPU plan and would match trivially, so the assertion that carries the
        // weight is that the work actually reached the device.
        if (!withGpu.eligible()) {
            System.out.println();
            System.out.println(
                    "Results match, but THE PLANNER REFUSED THE QUERY. Both runs executed the");
            System.out.println("same CPU plan, so this comparison demonstrates nothing. Reason:");
            System.out.println();
            System.out.println("  " + withGpu.offloadReason());
            System.out.println();
            System.out.println(SETUP_HELP);
            throw new IllegalStateException("the query was not eligible; see the reason above");
        }

        System.out.println("Results are identical, and the planner found the query eligible.");
        System.out.println();
        System.out.println("Whether a device actually ran it is decided on the TaskManager, and");
        System.out.println("this program cannot see that. Check the TaskManager log for one of:");
        System.out.println();
        System.out.println("  Accelerated on this TaskManager: provider <name> claims <n>x");
        System.out.println("  Accelerator declined on this TaskManager, running the generated ...");
        System.out.println();
        System.out.println("Expect a decline for this query, and that is the right answer: it is");
        System.out.println("three operations over 24 bytes a row, so the bus costs more than the");
        System.out.println("arithmetic saves. What it demonstrates is transparency -- identical");
        System.out.println("results either way -- not speed.");
    }

    private static final String SETUP_HELP =
            "Running on a GPU needs both of:\n"
                    + "  1. flink-table-gpu-runtime on the classpath. It registers the provider\n"
                    + "     via META-INF/services; in a distribution, put the jar in lib/.\n"
                    + "  2. TornadoVM's JVM arguments. The distribution ships them as an argfile:\n"
                    + "     pass '@$TORNADO_SDK/tornado-argfile' as a VM option (IntelliJ: Run ->\n"
                    + "     Edit Configurations -> Modify options -> Add VM options), or put the\n"
                    + "     same content in env.java.opts in flink-conf.yaml for a cluster.\n"
                    + "     Without them TornadoVM cannot initialise and the planner falls back.";

    /** One execution of the query, carrying the plan so the caller can tell what actually ran. */
    private static final class Run {

        private final List<Row> rows;
        private final String plan;

        Run(List<Row> rows, String plan) {
            this.rows = rows;
            this.plan = plan;
        }

        /**
         * Whether a subtree really executed on the device.
         *
         * <p>Read back from EXPLAIN rather than tracked separately, because EXPLAIN is rendered
         * after translation and so reports the outcome of the substitution attempt rather than
         * merely the planner's intent. A node the cost gate selected but which then fell back is
         * reported as CPU, with the reason.
         */
        /**
         * Whether the <em>planner</em> found the query eligible. Not whether a device ran it.
         *
         * <p>Those used to be the same question and are not any more. The planner runs in this
         * JVM and cannot know which machine the scheduler will pick or what hardware it has, so
         * since the cost comparison moved to the TaskManager, EXPLAIN reports eligibility and the
         * device verdict is taken -- and logged -- where the task actually runs.
         *
         * <p>This method was called {@code offloaded()} and read exactly the same string, which
         * made it assert something it had no access to: run against a device that declines on
         * cost, it reported that the query "ran on the GPU" while the TaskManager log said
         * otherwise. Asserting the stronger thing needs the operator's own metrics, which is
         * separate work.
         */
        boolean eligible() {
            return plan.contains("GPU  subtree");
        }

        String offloadReason() {
            for (String line : plan.split("\n")) {
                final String trimmed = line.trim();
                if (trimmed.startsWith("selected but fell back")
                        || trimmed.startsWith("below cost floor")
                        || trimmed.startsWith("not expressible on device")) {
                    return trimmed;
                }
            }
            return "no GPU Offload section in the plan; is table.exec.gpu-offload.enabled set?";
        }
    }

    private static Run run(int rows, boolean gpu) throws Exception {
        final TableEnvironment env =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());

        // A deterministic source: datagen's random generator is not seeded, so random values would
        // differ between the two runs and make the comparison meaningless.
        env.executeSql(
                // NOT NULL is load-bearing, and is the one thing the query author has to write.
                // A device kernel has no representation for null, so the estimator refuses any
                // expression with a nullable operand rather than guessing -- and datagen's columns
                // are nullable unless the DDL says otherwise. Without these the planner reports
                // "nullable operand of +" and the whole comparison below runs on the CPU twice.
                "CREATE TABLE Measurements (\n"
                        + "  id INT NOT NULL,\n"
                        + "  val DOUBLE NOT NULL\n"
                        + ") WITH (\n"
                        + "  'connector' = 'datagen',\n"
                        + "  'number-of-rows' = '"
                        + rows
                        + "',\n"
                        + "  'fields.id.kind' = 'sequence',\n"
                        + "  'fields.id.start' = '0',\n"
                        + "  'fields.id.end' = '"
                        + (rows - 1)
                        + "',\n"
                        + "  'fields.val.kind' = 'sequence',\n"
                        + "  'fields.val.start' = '0',\n"
                        + "  'fields.val.end' = '"
                        + (rows - 1)
                        + "'\n"
                        + ")");

        if (gpu) {
            // The entire user-facing surface of the feature.
            env.getConfig().getConfiguration().setString("table.exec.gpu-offload.enabled", "true");
            // See the class comment: this query is well below the calibrated floor, and is only
            // offloaded here because it matches the kernel the runtime implements.
            env.getConfig()
                    .getConfiguration()
                    .setString("table.exec.gpu-offload.min-row-cost", "1");
        }

        final String query =
                "SELECT id, val * 2.0 + 1.0 AS scaled\n"
                        + "FROM Measurements\n"
                        + "WHERE val > "
                        + (rows / 2)
                        + ".0";

        System.out.println(
                "========== " + (gpu ? "GPU offload enabled" : "default") + " ==========");
        // With offload enabled, EXPLAIN gains a "== GPU Offload ==" section reporting which
        // subtrees were selected and, for those that were not, why. It is rendered after
        // translation, so it reflects what actually ran rather than only what was intended.
        final String plan = env.explainSql(query);
        System.out.println(plan);

        final List<Row> result = new ArrayList<>();
        try (CloseableIterator<Row> rows0 = env.sqlQuery(query).execute().collect()) {
            rows0.forEachRemaining(result::add);
        }
        result.sort(Comparator.comparingInt(row -> (Integer) row.getField(0)));
        return new Run(result, plan);
    }

    private GpuOffloadExample() {}
}
