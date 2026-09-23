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

package org.apache.flink.table.gpu.metrics;

/**
 * The gather / copy-in / kernel / copy-out breakdown that is P1's exit criterion.
 *
 * <p>Shaped after Table 3 of "Enabling Transparent Acceleration of Big Data Frameworks Using
 * Heterogeneous Hardware" (PVLDB 15(13):3869-3882), where the equivalent breakdown showed copy-in
 * at 77.4% and kernel at 4.7% of accelerated time. The point of collecting these separately is that
 * they discriminate between different failures: copy-in dominant means device residency is the fix,
 * gather dominant means the input tier is wrong, kernel dominant means the batch is too small.
 *
 * <p>Host times are measured here; device times come from TornadoVM's profiler, so an execution
 * plan must have been built {@code .withProfiler(ProfilerMode.SILENT)} for them to be populated.
 * Values are nanoseconds unless named otherwise.
 */
public final class OffloadMetrics {

    private long batches;
    private long rowsIn;
    private long rowsOut;

    private long gatherNanos;
    private long drainNanos;
    private long executeWallNanos;

    private long copyInNanos;
    private long kernelNanos;
    private long copyOutNanos;
    private long compileNanos;

    private long bytesCopyIn;
    private long bytesCopyOut;
    private long onDemandBytesOut;

    /** Records one batch. {@code result} may be null when the profiler is disabled. */
    public void recordBatch(
            int rows,
            int emitted,
            long gather,
            long executeWall,
            long drain,
            org.apache.flink.table.gpu.operator.GeneratedKernelEngine.DeviceProfile profile) {
        batches++;
        rowsIn += rows;
        rowsOut += emitted;
        gatherNanos += gather;
        executeWallNanos += executeWall;
        drainNanos += drain;
        addProfile(profile);
    }

    /**
     * Folds in device numbers that did not come from {@link #recordBatch}.
     *
     * <p><b>Not for a second task graph in the same plan.</b> {@code TornadoProfilerResult}
     * delegates to the <em>executor</em>, so every graph of one execution plan answers with the
     * same plan-wide cumulative totals; adding two of them counts everything twice. That was done
     * here and inflated the fused path's reported kernel time and byte counts by a factor of two
     * until a controlled measurement caught it — the give-away was that the "first graph only"
     * figure equalled the whole non-compacted plan's, for a graph that contains no {@code
     * transferToHost} at all.
     */
    public void addProfile(
            org.apache.flink.table.gpu.operator.GeneratedKernelEngine.DeviceProfile profile) {
        if (profile == null) {
            return;
        }
        copyInNanos += profile.copyInNanos();
        kernelNanos += profile.kernelNanos();
        copyOutNanos += profile.copyOutNanos();
        compileNanos += profile.compileNanos();
        bytesCopyIn += profile.bytesIn();
        bytesCopyOut += profile.bytesOut();
    }

    public long getBatches() {
        return batches;
    }

    public long getRowsIn() {
        return rowsIn;
    }

    public long getRowsOut() {
        return rowsOut;
    }

    public long getGatherNanos() {
        return gatherNanos;
    }

    public long getDrainNanos() {
        return drainNanos;
    }

    public long getExecuteWallNanos() {
        return executeWallNanos;
    }

    public long getCopyInNanos() {
        return copyInNanos;
    }

    public long getKernelNanos() {
        return kernelNanos;
    }

    public long getCopyOutNanos() {
        return copyOutNanos;
    }

    public long getCompileNanos() {
        return compileNanos;
    }

    public long getBytesCopyIn() {
        return bytesCopyIn;
    }

    /**
     * Bytes fetched by an on-demand partial transfer, which the runtime's profiler never sees.
     *
     * <p>The compaction copies out a prefix after the graph has run, sized from a count only the
     * device knew. That transfer re-enters the runtime and resets the per-execution counters, so
     * the profiler reports neither the bytes nor the time. The bytes at least are exactly known --
     * they are what the transfer asked for -- so they are counted here and added into the copy-out
     * total, with the report naming how much of it came this way.
     */
    public void addOnDemandBytesOut(long bytes) {
        onDemandBytesOut += bytes;
        bytesCopyOut += bytes;
    }

    public long getOnDemandBytesOut() {
        return onDemandBytesOut;
    }

    public long getBytesCopyOut() {
        return bytesCopyOut;
    }

    /** Sum of the four segments the breakdown is meant to attribute. */
    public long getAttributedNanos() {
        return gatherNanos + copyInNanos + kernelNanos + copyOutNanos + drainNanos;
    }

    /**
     * Renders the breakdown as a fixed-width table. Percentages are of the attributed total, not of
     * wall time — the difference between the two is TornadoVM dispatch overhead and is reported
     * separately so it cannot hide inside another segment.
     */
    public String report(String label) {
        long total = Math.max(1, getAttributedNanos());
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%n=== %s ===%n", label));
        sb.append(
                String.format(
                        "batches=%d  rows_in=%d  rows_out=%d (%.1f%% selectivity)%n",
                        batches, rowsIn, rowsOut, rowsIn == 0 ? 0.0 : 100.0 * rowsOut / rowsIn));
        sb.append(String.format("%-12s %12s %8s%n", "segment", "ms", "share"));
        sb.append(row("gather", gatherNanos, total));
        sb.append(transfer("copy-in", copyInNanos, bytesCopyIn, total));
        sb.append(row("kernel", kernelNanos, total));
        sb.append(transfer("copy-out", copyOutNanos, bytesCopyOut, total));
        sb.append(row("drain", drainNanos, total));
        sb.append(String.format("%-12s %12.3f%n", "attributed", total / 1e6));
        sb.append(
                String.format(
                        "%-12s %12.3f   (execute() wall, incl. dispatch)%n",
                        "execute", executeWallNanos / 1e6));
        sb.append(
                String.format(
                        "%-12s %12.3f   (once per task, not per batch)%n",
                        "compile", compileNanos / 1e6));
        sb.append(
                String.format(
                        "bytes in=%.3f MiB  out=%.3f MiB%s%n",
                        bytesCopyIn / 1048576.0,
                        bytesCopyOut / 1048576.0,
                        onDemandBytesOut == 0
                                ? ""
                                : String.format(
                                        "  (of which %.3f MiB fetched on demand as a prefix)",
                                        onDemandBytesOut / 1048576.0)));
        return sb.toString();
    }

    private static String row(String name, long nanos, long total) {
        return String.format("%-12s %12.3f %7.1f%%%n", name, nanos / 1e6, 100.0 * nanos / total);
    }

    /**
     * A transfer segment, which has to distinguish "took no time" from "was not timed".
     *
     * <p>TornadoVM accumulates {@code COPY_OUT_TIME} only when the copy produced an event to wait
     * on. A copy issued without dependency tracking returns {@code -1} instead of an event — see
     * {@code TornadoVMInterpreter.transferDeviceToHost*}, which guards the timing on {@code
     * readEvent != -1} while counting the bytes unconditionally. So a transfer that demonstrably
     * happened can report zero nanoseconds.
     *
     * <p>Printing that as {@code 0.000 ms} invites exactly the wrong conclusion — that moving the
     * data was free, on a path where the review's whole argument is that the interconnect decides
     * these benchmarks. Bytes moved with no time against them are therefore reported as untimed,
     * which is a smaller claim and a true one.
     */
    private static String transfer(String name, long nanos, long bytes, long total) {
        if (nanos == 0 && bytes > 0) {
            return String.format(
                    "%-12s %12s %7s   (%.2f MiB moved; the runtime produced no event to time)%n",
                    name, "untimed", "-", bytes / 1048576.0);
        }
        return row(name, nanos, total);
    }
}
