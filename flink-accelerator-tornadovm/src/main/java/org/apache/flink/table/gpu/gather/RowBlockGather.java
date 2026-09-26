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

package org.apache.flink.table.gpu.gather;

import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.binary.BinaryRowData;

import java.nio.ByteBuffer;

/**
 * Copies whole rows into one contiguous block, for a device-side transpose.
 *
 * <p>Tier 2, bulk. Where {@link BinaryGather} reads one field at a time — a virtual call, a cast, a
 * typed accessor and a setter, per field, per row — this copies the row's whole fixed-width part in
 * one segment copy and leaves the transpose to {@code DeviceDeinterleave}, which does it with one
 * thread a row.
 *
 * <p>Measured over four million rows of four {@code DOUBLE} columns: the field-by-field host path
 * costs 21.2 ns a row, this plus the device kernel costs 10.8 ns a row. The trade is one memcpy of
 * {@code slots * 8} bytes against one accessor call per column, so it widens with the row.
 *
 * <h2>What is copied, and why the header comes too</h2>
 *
 * <p>The whole fixed-width region — null bits included — rather than only the fields the projection
 * reads. A contiguous run is one call; a chosen subset of slots is one call per run, which is the
 * per-field cost this exists to avoid. The device kernel is handed each column's slot index, so it
 * skips the header without the host having to.
 *
 * <p>This is only valid while every field the projection reads is a fixed-width primitive, which
 * {@link #canCopy} insists on. A {@code BinaryRowData} holding a string keeps a pointer in the
 * fixed part and the bytes after it, and copying the pointer would hand the device an offset into
 * memory it cannot address.
 */
public final class RowBlockGather {

    /** {@code BinaryRowData} gives every fixed-width field an eight-byte slot. */
    public static final int SLOT_BYTES = 8;

    private final ByteBuffer block;
    private final int rowBytes;

    private long bulkRows;

    /**
     * @param block the staging block. This is {@code DoubleArray.getSegment()}, which is already
     *     sliced past TornadoVM's array header, so element zero is at byte zero — adding the header
     *     again overruns the buffer by exactly its size on the last row of a batch.
     * @param slots eight-byte slots in one staged row, from {@link #slotsFor}
     */
    public RowBlockGather(ByteBuffer block, int slots) {
        this.block = block.duplicate();
        this.rowBytes = slots * SLOT_BYTES;
    }

    /**
     * Eight-byte slots in a row of this arity, header included.
     *
     * <p>{@code BinaryRowData} lays out {@code ((arity + 71) / 64) * 8} bytes of null bits and then
     * one eight-byte slot a field, so this is that arithmetic in slots rather than bytes.
     */
    public static int slotsFor(int arity) {
        return ((arity + 71) / 64) + arity;
    }

    /** The slot a field occupies, which is what the device kernel is given per column. */
    public static int slotOf(int arity, int field) {
        return ((arity + 71) / 64) + field;
    }

    /**
     * Whether this row can be copied wholesale.
     *
     * <p>Single-segment because a row spanning two segments needs two copies, and one copy is the
     * whole point; the row is then left to the per-column gathers, which do not care.
     */
    public static boolean canCopy(RowData row, int arity) {
        if (!(row instanceof BinaryRowData binary)) {
            return false;
        }
        return binary.getSegments().length == 1
                && binary.getArity() == arity
                && binary.getSizeInBytes() >= slotsFor(arity) * SLOT_BYTES;
    }

    /** Copies this row's fixed-width part to {@code position} in the block. */
    public void accept(RowData row, int position) {
        final BinaryRowData binary = (BinaryRowData) row;
        block.limit(block.capacity());
        block.position(position * rowBytes);
        binary.getSegments()[0].get(binary.getOffset(), block, rowBytes);
        bulkRows++;
    }

    public long bulkRows() {
        return bulkRows;
    }

    public String tier() {
        return "tier2-binary-block(" + bulkRows + " rows, device transpose)";
    }
}
