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

import uk.ac.manchester.tornado.api.TornadoProfilerResult;

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

    /** Records one batch. {@code result} may be null when the profiler is disabled. */
    public void recordBatch(
            int rows,
            int emitted,
            long gather,
            long executeWall,
            long drain,
            TornadoProfilerResult result) {
        batches++;
        rowsIn += rows;
        rowsOut += emitted;
        gatherNanos += gather;
        executeWallNanos += executeWall;
        drainNanos += drain;
        if (result != null) {
            copyInNanos += result.getDeviceWriteTime();
            kernelNanos += result.getDeviceKernelTime();
            copyOutNanos += result.getDeviceReadTime();
            compileNanos += result.getCompileTime();
            bytesCopyIn += result.getTotalBytesCopyIn();
            bytesCopyOut += result.getTotalBytesCopyOut();
        }
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
        sb.append(row("copy-in", copyInNanos, total));
        sb.append(row("kernel", kernelNanos, total));
        sb.append(row("copy-out", copyOutNanos, total));
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
                        "bytes in=%.2f MiB  out=%.2f MiB%n",
                        bytesCopyIn / 1048576.0, bytesCopyOut / 1048576.0));
        return sb.toString();
    }

    private static String row(String name, long nanos, long total) {
        return String.format("%-12s %12.3f %7.1f%%%n", name, nanos / 1e6, 100.0 * nanos / total);
    }
}
