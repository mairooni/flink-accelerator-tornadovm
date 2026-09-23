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

package org.apache.flink.table.gpu.operator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cudf.Cudf;

/**
 * The two per-row kernels that turn a filter's mask into compacted columns, on the device.
 *
 * <h2>Why this exists at all</h2>
 *
 * <p>A generated kernel writes a 0/1 mask beside its projections and stops there: computing a
 * predicate per row is a map, and moving the survivors together is not — it needs a prefix sum and
 * a scatter, which no {@code @Parallel} loop states. So the host used to copy every projected row
 * back and skip the ones the mask rejected while building output rows.
 *
 * <p>cuDF can do the part a kernel cannot. {@code Cudf.selectedIndices} is a stream compaction and
 * returns the surviving positions, and given those, gathering is a map again. What is left over is
 * two trivial kernels: one to widen the mask into the byte-per-row column cuDF wants, and one to
 * gather.
 *
 * <p><b>Packing the survivors is not by itself a saving,</b> and it is worth being exact about why.
 * A {@code transferToHost} inside a task graph moves a whole buffer, and the packed columns are
 * sized to the batch because nothing knows in advance how many rows will survive — so for a while
 * this moved precisely as many bytes as never packing at all: measured at 0.750 MiB out for 65,536
 * rows at 10% selectivity, with the compaction on and with it off, the same figure to three
 * decimals. The saving comes from fetching only the prefix afterwards, which is {@code
 * GeneratedKernelEngine.fetchSurvivors}; with it, the same workload moves 0.075 MiB.
 *
 * <p>Hand-written rather than generated, because neither depends on the query: the widen is the
 * same for every filter and the gather is the same for every column of a given width. Generating
 * them would mean compiling per operator what could be compiled once.
 *
 * <h2>Reading the count from a buffer, not from a parameter</h2>
 *
 * <p>{@code gather*} is launched over the whole batch and does nothing above the survivor count,
 * which it reads from the buffer cuDF wrote. It cannot be launched over the count instead: the
 * count is only known on the device, after the compaction, and a launch size fixed on the host
 * would either be the batch (this) or a stale figure from the previous batch (wrong).
 *
 * <p>The guard therefore costs one buffer read a thread and leaves most threads idle on a selective
 * filter. That is the correct trade here — the alternative is a host round trip between the
 * compaction and the gather, which is the thing this exists to remove.
 */
public final class DeviceCompaction {

    private static final Logger LOG = LoggerFactory.getLogger(DeviceCompaction.class);

    private DeviceCompaction() {}

    /**
     * Whether a cuDF compaction can actually run on this machine's device, probed by running one.
     *
     * <h2>Why the probe is the real thing</h2>
     *
     * <p>The obvious probe is {@code CudfLibraryProvider.isAvailable()}, which is what the provider
     * uses to decide whether to offer a grouped aggregate, and it is too weak for this question. It
     * answers whether the {@code tornado-cudf} module is present and its class loads — and on a
     * host where {@code libtornado-cudf.so} is built but its RAPIDS dependencies do not resolve, it
     * says yes. The failure then arrives from the interpreter as
     *
     * <pre>Library `rapids/cudf` is not supported on device: [NVIDIA CUDA] -- ...</pre>
     *
     * <p>out of {@code execute()}, which is after the batch has been staged and consumed and where
     * nothing can recover it. That is the one place this must not be discovered, so the probe here
     * builds a one-row compaction and runs it. It costs one tiny graph per JVM and it cannot drift
     * from what the engine needs, because it is what the engine does.
     *
     * <p>Standalone rather than a dry run of the operator's own plan: a failure half way through a
     * plan that is about to be used leaves it in a state nothing here can reason about, and the
     * answer is wanted before the plan is built rather than after.
     */
    public static boolean available() {
        return ENABLED && AVAILABLE;
    }

    /**
     * A switch for measurement, off the query path entirely.
     *
     * <p>Whether compacting on the device beats draining on the host is an empirical question, and
     * it cannot be answered without running the same workload both ways. Nothing else in this
     * project can turn the path off: it engages whenever a filter, a computed column and a working
     * cuDF coincide.
     *
     * <p>Deliberately a <em>provider</em> setting and not a Flink one. It is read from a system
     * property or an environment variable on the machine the operator runs on, so it is invisible
     * to the query author — no hint, no option, no SQL — which is the project's transparency
     * invariant. Default on; set {@code -Dtornadovm.accelerator.compaction=false} or {@code
     * TORNADOVM_ACCELERATOR_COMPACTION=false} to compare against the host drain.
     */
    private static final boolean ENABLED = enabled();

    private static boolean enabled() {
        String property = System.getProperty("tornadovm.accelerator.compaction");
        if (property == null) {
            property = System.getenv("TORNADOVM_ACCELERATOR_COMPACTION");
        }
        if (property == null) {
            return true;
        }
        boolean on = !("false".equalsIgnoreCase(property) || "off".equalsIgnoreCase(property));
        LOG.info(
                "device compaction {} by configuration ({})",
                on ? "enabled" : "disabled",
                property);
        return on;
    }

    private static final boolean AVAILABLE = probe();

    private static boolean probe() {
        try {
            IntArray mask = new IntArray(1);
            IntArray rows = new IntArray(1);
            ByteArray bytes = new ByteArray(1);
            IntArray indices = new IntArray(1);
            IntArray survivors = new IntArray(1);
            mask.set(0, 1);
            rows.set(0, 1);
            bytes.init((byte) 0);
            indices.set(0, -1);
            survivors.set(0, -1);

            TaskGraph graph =
                    new TaskGraph("cudf-probe")
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION,
                                    mask,
                                    rows,
                                    bytes,
                                    indices,
                                    survivors)
                            .task("widen", DeviceCompaction::widenMask, mask, bytes, rows)
                            .libraryTask(
                                    "select",
                                    Cudf::selectedIndices,
                                    1,
                                    1,
                                    bytes,
                                    indices,
                                    survivors)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, indices, survivors);
            GridScheduler scheduler = new GridScheduler();
            scheduler.addWorkerGrid("cudf-probe.widen", new WorkerGrid1D(1));
            try (TornadoExecutionPlan plan =
                    new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(scheduler)) {
                plan.execute();
            }
            // Asserted rather than assumed: a shim that runs and returns nothing would be worse
            // than one that throws, because the compaction would silently emit no rows.
            return survivors.get(0) == 1 && indices.get(0) == 0;
        } catch (Throwable unusable) {
            LOG.info(
                    "cuDF compaction is not available here, filters drain on the host: {}",
                    String.valueOf(unusable));
            return false;
        }
    }

    /**
     * The mask, one byte a row, which is the column layout cuDF's boolean expects.
     *
     * <p>The generated kernel writes {@code IntArray} because that is what every other buffer it
     * touches is, and a second integer column costs three bytes a row more than this one. Widening
     * here rather than narrowing there keeps the generator unaware that a compaction exists.
     *
     * <p>It also clears the tail. {@code Cudf.selectedIndices} captures its row count when the task
     * graph is built, so it always looks at the whole batch; the kernel, which re-reads the live
     * count, leaves the mask past that count holding whatever the previous batch put there. A
     * partition's short last batch would otherwise emit rows from the one before it — a wrong
     * answer no row count catches, because the caller reads the count the compaction wrote.
     */
    public static void widenMask(IntArray mask, ByteArray bytes, IntArray rows) {
        for (@Parallel int i = 0; i < mask.getSize(); i++) {
            bytes.set(i, i < rows.get(0) ? (byte) mask.get(i) : 0);
        }
    }

    /** Gathers a {@code DOUBLE} column at the surviving positions. */
    public static void gatherDoubles(
            IntArray indices, IntArray survivors, DoubleArray source, DoubleArray compacted) {
        for (@Parallel int i = 0; i < compacted.getSize(); i++) {
            if (i < survivors.get(0)) {
                compacted.set(i, source.get(indices.get(i)));
            }
        }
    }

    /** Gathers a {@code FLOAT} column at the surviving positions. */
    public static void gatherFloats(
            IntArray indices, IntArray survivors, FloatArray source, FloatArray compacted) {
        for (@Parallel int i = 0; i < compacted.getSize(); i++) {
            if (i < survivors.get(0)) {
                compacted.set(i, source.get(indices.get(i)));
            }
        }
    }

    /** Gathers an {@code INT} column at the surviving positions. */
    public static void gatherInts(
            IntArray indices, IntArray survivors, IntArray source, IntArray compacted) {
        for (@Parallel int i = 0; i < compacted.getSize(); i++) {
            if (i < survivors.get(0)) {
                compacted.set(i, source.get(indices.get(i)));
            }
        }
    }
}
