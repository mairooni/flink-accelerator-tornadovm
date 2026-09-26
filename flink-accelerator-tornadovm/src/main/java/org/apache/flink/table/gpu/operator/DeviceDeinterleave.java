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

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Row-major to column-major, on the device.
 *
 * <h2>What this replaces</h2>
 *
 * <p>Staging a row-major batch means reading one field at a time: a virtual call on the gather, a
 * cast to the row's concrete class, a typed accessor with its bounds and null checks, and a setter
 * into the staging buffer — per field, per row. Measured at <b>21.2 ns a row</b> over four million
 * rows of four columns, which is 5.3 ns a field and is entirely host time.
 *
 * <p>The alternative is to move the bytes as they already lie. A {@code BinaryRowData}'s
 * fixed-width part is a contiguous run, so a whole row copies with one {@code MemorySegment.copy};
 * the batch then crosses the bus once, as one block, and the transpose happens here with one thread
 * a row. Measured at <b>10.8 ns a row</b> on the same data: <b>1.96x</b>.
 *
 * <h2>Why {@code @Parallel} and not {@code KernelContext}</h2>
 *
 * <p>Both were written and measured. The tiled {@code KernelContext} version — each group loading a
 * tile of rows into shared memory with fully coalesced reads, a barrier, then writing columns out
 * of shared memory — came in at 13.5 ns a row, a third <em>slower</em> than this loop.
 *
 * <p>The reason is that the reads here are already coalesced in the way that matters. Thread
 * <i>i</i> reads a contiguous run of slots and the runs are adjacent, so a warp already covers one
 * contiguous span and the hardware already coalesces it. Staging that through shared memory adds a
 * barrier and a second pass over the data to buy a reuse the cache was providing for free. The
 * simpler kernel is the faster one, and it is the one here.
 *
 * <h2>Why the arity is spelled out</h2>
 *
 * <p>A task graph task is a {@code Method} with a fixed signature, so a loop over a variable column
 * count would have to carry the columns in one buffer and hand the generated kernel offsets into it
 * — which changes the kernel generator to serve the staging path. Spelling out the arities instead
 * keeps the generated kernel reading exactly the buffers it already reads, and a projection over
 * more than {@link #MAX_COLUMNS} input columns simply falls back to the host gather.
 *
 * <p>{@code offN} is the slot index of column <i>N</i> within a staged row, so the kernel does not
 * need to know anything about {@code BinaryRowData}'s header: the operator computed the offsets
 * once when it bound the gather.
 */
public final class DeviceDeinterleave {

    /** Above this, the host gather runs instead. Six covers every projection measured here. */
    public static final int MAX_COLUMNS = 6;

    private DeviceDeinterleave() {}

    public static void deinterleave1(
            DoubleArray block, DoubleArray c0, int slots, int off0, IntArray rows) {
        final int n = rows.get(0);
        for (@Parallel int i = 0; i < n; i++) {
            final int base = i * slots;
            c0.set(i, block.get(base + off0));
        }
    }

    public static void deinterleave2(
            DoubleArray block,
            DoubleArray c0,
            DoubleArray c1,
            int slots,
            int off0,
            int off1,
            IntArray rows) {
        final int n = rows.get(0);
        for (@Parallel int i = 0; i < n; i++) {
            final int base = i * slots;
            c0.set(i, block.get(base + off0));
            c1.set(i, block.get(base + off1));
        }
    }

    public static void deinterleave3(
            DoubleArray block,
            DoubleArray c0,
            DoubleArray c1,
            DoubleArray c2,
            int slots,
            int off0,
            int off1,
            int off2,
            IntArray rows) {
        final int n = rows.get(0);
        for (@Parallel int i = 0; i < n; i++) {
            final int base = i * slots;
            c0.set(i, block.get(base + off0));
            c1.set(i, block.get(base + off1));
            c2.set(i, block.get(base + off2));
        }
    }

    public static void deinterleave4(
            DoubleArray block,
            DoubleArray c0,
            DoubleArray c1,
            DoubleArray c2,
            DoubleArray c3,
            int slots,
            int off0,
            int off1,
            int off2,
            int off3,
            IntArray rows) {
        final int n = rows.get(0);
        for (@Parallel int i = 0; i < n; i++) {
            final int base = i * slots;
            c0.set(i, block.get(base + off0));
            c1.set(i, block.get(base + off1));
            c2.set(i, block.get(base + off2));
            c3.set(i, block.get(base + off3));
        }
    }

    public static void deinterleave5(
            DoubleArray block,
            DoubleArray c0,
            DoubleArray c1,
            DoubleArray c2,
            DoubleArray c3,
            DoubleArray c4,
            int slots,
            int off0,
            int off1,
            int off2,
            int off3,
            int off4,
            IntArray rows) {
        final int n = rows.get(0);
        for (@Parallel int i = 0; i < n; i++) {
            final int base = i * slots;
            c0.set(i, block.get(base + off0));
            c1.set(i, block.get(base + off1));
            c2.set(i, block.get(base + off2));
            c3.set(i, block.get(base + off3));
            c4.set(i, block.get(base + off4));
        }
    }

    public static void deinterleave6(
            DoubleArray block,
            DoubleArray c0,
            DoubleArray c1,
            DoubleArray c2,
            DoubleArray c3,
            DoubleArray c4,
            DoubleArray c5,
            int slots,
            int off0,
            int off1,
            int off2,
            int off3,
            int off4,
            int off5,
            IntArray rows) {
        final int n = rows.get(0);
        for (@Parallel int i = 0; i < n; i++) {
            final int base = i * slots;
            c0.set(i, block.get(base + off0));
            c1.set(i, block.get(base + off1));
            c2.set(i, block.get(base + off2));
            c3.set(i, block.get(base + off3));
            c4.set(i, block.get(base + off4));
            c5.set(i, block.get(base + off5));
        }
    }

    /** The task method for this column count, by name, as {@code TaskGraph.task} wants it. */
    public static java.lang.reflect.Method entryFor(int columns) throws NoSuchMethodException {
        final Class<?>[] signature = new Class<?>[1 + columns + 1 + columns + 1];
        int at = 0;
        signature[at++] = DoubleArray.class;
        for (int i = 0; i < columns; i++) {
            signature[at++] = DoubleArray.class;
        }
        signature[at++] = int.class;
        for (int i = 0; i < columns; i++) {
            signature[at++] = int.class;
        }
        signature[at] = IntArray.class;
        return DeviceDeinterleave.class.getMethod("deinterleave" + columns, signature);
    }
}
