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

import org.apache.flink.core.memory.MemorySegmentFactory;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.gpu.gather.RowGather;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;

import java.util.Arrays;

/**
 * What the row-major to column-major conversion costs, three ways.
 *
 * <p>Every offloaded operator measured on this project has been dominated not by the device but by
 * the seam either side of it — §T12 put 616 ms of drain against 15.7 ms of ordering. The staging
 * half of that seam is {@link RowGather}: one virtual call, one cast, one typed accessor and one
 * setter <em>per field per row</em>, measured at about 78 ns a field. This asks whether a device
 * pass beats it.
 *
 * <h2>The three arms</h2>
 *
 * <ul>
 *   <li><b>host</b> — what runs today. {@code RowGather} per column, reading {@code BinaryRowData}
 *       into device-capable {@code DoubleArray}s, then one transfer per column.
 *   <li><b>parallel</b> — one bulk transfer of the row block, then a {@code @Parallel} kernel where
 *       thread <i>i</i> reads row <i>i</i> whole and writes one element into each column.
 *   <li><b>kernelcontext</b> — the same, but the block is staged through shared memory first: each
 *       thread group loads a tile of rows with fully coalesced reads, barriers, then writes columns
 *       with fully coalesced writes.
 * </ul>
 *
 * <p>All three end in the same place — four column-major {@code DoubleArray}s resident on the
 * device — so the comparison is like for like. The host arm's transfers are counted; leaving them
 * out would flatter it by exactly the thing the device arms are paying.
 *
 * <h2>Why every column is a DOUBLE</h2>
 *
 * <p>Not a simplification for convenience — a limitation worth recording. A {@code BinaryRowData}
 * holds every fixed-width field in an eight-byte slot, so a mixed row is a block of slots whose
 * interpretation varies by column. Reading that on a device needs to reinterpret a slot's bits as a
 * double, and the CUDA backend has no intrinsic for {@code Double.longBitsToDouble}. So a mixed row
 * block cannot be deinterleaved by one kernel today, whatever the access pattern. An all-double
 * block can, and it is the case that matters most: the haversine's hot columns are two doubles.
 *
 * <h2>Running it</h2>
 *
 * <pre>
 *   java @$TORNADO_SDK/tornado-argfile -cp &lt;examples&gt;:&lt;provider&gt;:&lt;flink lib&gt; \
 *        org.apache.flink.table.examples.java.gpu.LayoutGatherSweep [rows] [repeats]
 * </pre>
 */
public final class LayoutGatherSweep {

    /** Columns per row. Four doubles is a realistic staged width and keeps the stride round. */
    private static final int COLUMNS = 4;

    /** Eight-byte slots per row: the null bit set, then one per field. */
    private static final int SLOTS = 1 + COLUMNS;

    /** Rows per thread group in the shared-memory arm. */
    private static final int TILE = 64;

    private LayoutGatherSweep() {}

    // ---------------------------------------------------------------------------------------
    // The kernels
    // ---------------------------------------------------------------------------------------

    /**
     * Thread per row: read the row whole, write one element into each column.
     *
     * <p>The reads are already coalesced in the way that matters — thread <i>i</i> reads a
     * contiguous run and the runs are adjacent, so a warp covers one contiguous span. What this
     * leaves to the cache is the reuse <em>within</em> a row, since the four loads revisit the same
     * span four times.
     */
    private static void deinterleaveParallel(
            DoubleArray rows,
            DoubleArray c0,
            DoubleArray c1,
            DoubleArray c2,
            DoubleArray c3,
            int n) {
        for (@Parallel int i = 0; i < n; i++) {
            int base = i * SLOTS;
            c0.set(i, rows.get(base + 1));
            c1.set(i, rows.get(base + 2));
            c2.set(i, rows.get(base + 3));
            c3.set(i, rows.get(base + 4));
        }
    }

    /**
     * The same, staged through shared memory, which makes that reuse explicit rather than
     * cache-dependent.
     *
     * <p>The load loop is strided by the group size, so consecutive threads read consecutive
     * elements and the whole tile arrives in fully coalesced transactions. After the barrier each
     * thread owns one row of the tile and reads its four slots out of shared memory, where a
     * strided access costs nothing.
     */
    private static void deinterleaveShared(
            KernelContext context,
            DoubleArray rows,
            DoubleArray c0,
            DoubleArray c1,
            DoubleArray c2,
            DoubleArray c3,
            int n) {
        double[] tile = context.allocateDoubleLocalArray(TILE * SLOTS);

        int groupStartRow = context.groupIdx * TILE;
        int base = groupStartRow * SLOTS;
        for (int j = context.localIdx; j < TILE * SLOTS; j += context.localGroupSizeX) {
            int source = base + j;
            tile[j] = source < n * SLOTS ? rows.get(source) : 0.0;
        }
        context.localBarrier();

        int row = groupStartRow + context.localIdx;
        if (row < n) {
            int slot = context.localIdx * SLOTS;
            c0.set(row, tile[slot + 1]);
            c1.set(row, tile[slot + 2]);
            c2.set(row, tile[slot + 3]);
            c3.set(row, tile[slot + 4]);
        }
    }

    // ---------------------------------------------------------------------------------------

    public static void main(String[] args) {
        final int rows = args.length > 0 ? Integer.parseInt(args[0]) : 4_000_000;
        final int repeats = args.length > 1 ? Integer.parseInt(args[1]) : 7;

        System.out.printf(
                "%,d rows x %d DOUBLE columns, %d-byte stride, %d repeats%n%n",
                rows, COLUMNS, SLOTS * Long.BYTES, repeats);

        final BinaryRowData[] sample = new BinaryRowData[1];
        final DoubleArray block = buildRowBlock(rows, sample);

        final double[] host = new double[repeats];
        final double[] parallel = new double[repeats];
        final double[] shared = new double[repeats];

        for (int r = 0; r < repeats; r++) {
            host[r] = timeHost(rows, sample[0]);
            parallel[r] = timeParallel(rows, block);
            shared[r] = timeShared(rows, block);
        }

        report("host RowGather + transfer", host, rows);
        report("device @Parallel", parallel, rows);
        report("device KernelContext", shared, rows);

        System.out.printf(
                "%nspeedup over host: @Parallel %.2fx, KernelContext %.2fx%n",
                median(host) / median(parallel), median(host) / median(shared));
    }

    /**
     * The row block, laid out exactly as {@code BinaryRowData} lays out a fixed-width row.
     *
     * <p>Built once and shared by every arm, so all three read the same bytes. The host arm reads
     * it through a {@code BinaryRowData} pointed at each row in turn, which is what an operator
     * receives; the device arms read the same memory as a flat array of slots.
     */
    private static DoubleArray buildRowBlock(int rows, BinaryRowData[] sampleOut) {
        final int stride = SLOTS * Long.BYTES;
        final DoubleArray block = new DoubleArray(rows * SLOTS);
        for (int i = 0; i < rows; i++) {
            int base = i * SLOTS;
            block.set(base, 0.0); // the null bit set, unread by either arm
            for (int c = 0; c < COLUMNS; c++) {
                block.set(base + 1 + c, i * 0.25 + c);
            }
        }

        // A BinaryRowData over the same bytes, for the host arm. Flink's MemorySegment rather than
        // the foreign one, because that is what BinaryRowData.pointTo takes.
        final byte[] copy = new byte[rows * stride];
        MemorySegmentFactory.wrap(copy)
                .put(
                        0,
                        block.getSegment().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE),
                        0,
                        rows * stride);
        final BinaryRowData row = new BinaryRowData(COLUMNS);
        row.pointTo(MemorySegmentFactory.wrap(copy), 0, stride);
        sampleOut[0] = row;
        return block;
    }

    /** What runs today: a gather per column, then one transfer per column. */
    private static double timeHost(int rows, BinaryRowData sample) {
        final DoubleArray[] columns = newColumns(rows);
        final RowGather[] gathers = new RowGather[COLUMNS];
        for (int c = 0; c < COLUMNS; c++) {
            final DoubleArray target = columns[c];
            gathers[c] = RowGather.forColumn(sample, c, GpuValueType.DOUBLE, target::set);
        }

        final int stride = SLOTS * Long.BYTES;
        final org.apache.flink.core.memory.MemorySegment segment = sample.getSegments()[0];

        final long start = System.nanoTime();
        for (int i = 0; i < rows; i++) {
            sample.pointTo(segment, i * stride, stride);
            for (int c = 0; c < COLUMNS; c++) {
                gathers[c].accept(sample, i);
            }
        }
        // The columns still have to reach the device, which the device arms pay for too.
        final TaskGraph graph =
                new TaskGraph("host")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                columns[0],
                                columns[1],
                                columns[2],
                                columns[3])
                        .task("noop", LayoutGatherSweep::touch, columns[0], rows)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, columns[0]);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return (System.nanoTime() - start) / 1e6;
    }

    /** A kernel that does nothing but force the transfers to be real. */
    private static void touch(DoubleArray a, int n) {
        for (@Parallel int i = 0; i < n; i++) {
            a.set(i, a.get(i));
        }
    }

    private static double timeParallel(int rows, DoubleArray block) {
        final DoubleArray[] c = newColumns(rows);
        final TaskGraph graph =
                new TaskGraph("parallel")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, block)
                        .task(
                                "deinterleave",
                                LayoutGatherSweep::deinterleaveParallel,
                                block,
                                c[0],
                                c[1],
                                c[2],
                                c[3],
                                rows)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, c[0]);
        final long start = System.nanoTime();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return (System.nanoTime() - start) / 1e6;
    }

    private static double timeShared(int rows, DoubleArray block) {
        final DoubleArray[] c = newColumns(rows);
        final KernelContext context = new KernelContext();
        final WorkerGrid worker = new WorkerGrid1D(((rows + TILE - 1) / TILE) * TILE);
        worker.setLocalWork(TILE, 1, 1);
        final GridScheduler scheduler = new GridScheduler("shared.deinterleave", worker);

        final TaskGraph graph =
                new TaskGraph("shared")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, block)
                        .task(
                                "deinterleave",
                                LayoutGatherSweep::deinterleaveShared,
                                context,
                                block,
                                c[0],
                                c[1],
                                c[2],
                                c[3],
                                rows)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, c[0]);
        final long start = System.nanoTime();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return (System.nanoTime() - start) / 1e6;
    }

    private static DoubleArray[] newColumns(int rows) {
        final DoubleArray[] columns = new DoubleArray[COLUMNS];
        for (int c = 0; c < COLUMNS; c++) {
            columns[c] = new DoubleArray(rows);
        }
        return columns;
    }

    private static void report(String name, double[] times, int rows) {
        final double median = median(times);
        System.out.printf(
                "%-28s median %8.1f ms   min %8.1f   max %8.1f   %6.1f ns/row   %5.1f ns/field%n",
                name,
                median,
                Arrays.stream(times).min().orElse(0),
                Arrays.stream(times).max().orElse(0),
                median * 1e6 / rows,
                median * 1e6 / rows / COLUMNS);
    }

    /** Median of the runs after the first, which pays kernel compilation. */
    private static double median(double[] times) {
        final double[] warm = Arrays.copyOfRange(times, 1, times.length);
        Arrays.sort(warm);
        return warm[warm.length / 2];
    }
}
