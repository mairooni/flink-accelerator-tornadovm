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

import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.writer.BinaryRowWriter;
import org.apache.flink.table.gpu.codegen.GpuValueType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import javax.annotation.Nullable;

import java.nio.ByteBuffer;

/**
 * A run of rows held column by column, with one column on the device and the rest beside it.
 *
 * <p>What the sort and the join both need and neither should own. Both hold rows for longer than a
 * batch, both send exactly one column to a device — the key — and both emit by gathering the rest
 * back out in an order the device chose. The shapes differ only in what they do with the
 * permutation, and that is the part that stays in the operators.
 *
 * <h2>Why the key is a TornadoVM array and nothing else is</h2>
 *
 * <p>Because the key is the only column a device sees. Holding the payload in TornadoVM's arrays
 * too would buy nothing and cost the header and the indirection; holding the key in a {@link
 * ByteBuffer} would mean copying it into one before every launch. So the key is an {@link IntArray}
 * over the same arena the rest comes from, and the rest is plain buffers.
 *
 * <h2>Validity</h2>
 *
 * <p>One {@code int} a row, a bit a column, which is the layout the kernel path already uses for
 * its input nulls: four bytes a row whatever the column count, against four bytes per column the
 * obvious way. A row wider than 32 fields is refused rather than given a second word.
 */
final class StagedColumns {

    /** Fields a row may have, because validity is one {@code int} a row. */
    static final int MAX_FIELDS = 32;

    private final RowType rowType;
    private final int keyField;
    private final int capacity;

    private final IntArray keys;
    private final ByteBuffer[] columns;
    private final @Nullable ByteBuffer validity;
    private final int[] widths;
    private final LogicalType[] types;

    private StagedColumns(
            RowType rowType,
            int keyField,
            int capacity,
            IntArray keys,
            ByteBuffer[] columns,
            @Nullable ByteBuffer validity,
            int[] widths,
            LogicalType[] types) {
        this.rowType = rowType;
        this.keyField = keyField;
        this.capacity = capacity;
        this.keys = keys;
        this.columns = columns;
        this.validity = validity;
        this.widths = widths;
        this.types = types;
    }

    /**
     * Allocates room for {@code capacity} rows, from the arena where there is one.
     *
     * @param keyField the one column a device reads, or -1 where none is — a running total sends
     *     its values in a buffer of their own and wants every field of the row held host-side
     * @param staging where buffers come from, or null to allocate on the heap
     */
    static StagedColumns allocate(
            RowType rowType,
            int keyField,
            int capacity,
            @Nullable GeneratedKernelEngine.Staging staging) {
        if ((long) capacity * Long.BYTES > Integer.MAX_VALUE - 64) {
            // Every buffer here is indexed by an int, so a capacity this large would overflow the
            // offset arithmetic rather than fail to allocate.
            throw new IllegalStateException(
                    "a staging capacity of " + capacity + " rows overflows an int-indexed buffer");
        }
        int fields = rowType.getFieldCount();
        int[] widths = new int[fields];
        LogicalType[] types = new LogicalType[fields];
        ByteBuffer[] columns = new ByteBuffer[fields];
        boolean anyNullable = false;
        for (int i = 0; i < fields; i++) {
            types[i] = rowType.getTypeAt(i);
            widths[i] = widthOf(types[i]);
            anyNullable |= types[i].isNullable();
            if (i != keyField) {
                columns[i] = buffer(widths[i] * capacity, staging);
            }
        }
        IntArray keys = allocateInts(capacity, staging);
        ByteBuffer validity = anyNullable ? buffer(4 * capacity, staging) : null;
        return new StagedColumns(
                rowType, keyField, capacity, keys, columns, validity, widths, types);
    }

    int capacity() {
        return capacity;
    }

    /** The key column, which is the only thing here a device ever reads. */
    IntArray keys() {
        return keys;
    }

    int keyAt(int position) {
        return keys.get(position);
    }

    RowType rowType() {
        return rowType;
    }

    /** Copies every field of one row into slot {@code position}. */
    void stage(RowData row, int position) {
        if (validity != null) {
            int bits = 0;
            for (int f = 0; f < widths.length; f++) {
                if (row.isNullAt(f)) {
                    bits |= 1 << f;
                }
            }
            validity.putInt(position * 4, bits);
        }
        for (int f = 0; f < widths.length; f++) {
            if (row.isNullAt(f)) {
                continue;
            }
            if (f == keyField) {
                keys.set(position, row.getInt(f));
                continue;
            }
            int at = position * widths[f];
            switch (types[f].getTypeRoot()) {
                case INTEGER:
                    columns[f].putInt(at, row.getInt(f));
                    break;
                case BIGINT:
                    columns[f].putLong(at, row.getLong(f));
                    break;
                case FLOAT:
                    columns[f].putFloat(at, row.getFloat(f));
                    break;
                default:
                    columns[f].putDouble(at, row.getDouble(f));
                    break;
            }
        }
    }

    /**
     * Writes the staged row at {@code position} into an output row, starting at {@code firstField}.
     *
     * <p>The offset is what makes this serve a join as well as a sort: a joined row is one side's
     * fields followed by the other's, and neither side knows where it starts.
     */
    void writeInto(BinaryRowWriter writer, int firstField, int position) {
        int bits = validity == null ? 0 : validity.getInt(position * 4);
        for (int f = 0; f < widths.length; f++) {
            int out = firstField + f;
            if ((bits & (1 << f)) != 0) {
                writer.setNullAt(out);
                continue;
            }
            if (f == keyField) {
                writer.writeInt(out, keys.get(position));
                continue;
            }
            int at = position * widths[f];
            switch (types[f].getTypeRoot()) {
                case INTEGER:
                    writer.writeInt(out, columns[f].getInt(at));
                    break;
                case BIGINT:
                    writer.writeLong(out, columns[f].getLong(at));
                    break;
                case FLOAT:
                    writer.writeFloat(out, columns[f].getFloat(at));
                    break;
                default:
                    writer.writeDouble(out, columns[f].getDouble(at));
                    break;
            }
        }
    }

    /** Copies a live row into a binary row, for the host path where nothing is staged. */
    static void materialise(RowData row, RowType rowType, BinaryRowWriter writer, int firstField) {
        for (int f = 0; f < rowType.getFieldCount(); f++) {
            int out = firstField + f;
            if (row.isNullAt(f)) {
                writer.setNullAt(out);
                continue;
            }
            switch (rowType.getTypeAt(f).getTypeRoot()) {
                case INTEGER:
                    writer.writeInt(out, row.getInt(f));
                    break;
                case BIGINT:
                    writer.writeLong(out, row.getLong(f));
                    break;
                case FLOAT:
                    writer.writeFloat(out, row.getFloat(f));
                    break;
                default:
                    writer.writeDouble(out, row.getDouble(f));
                    break;
            }
        }
    }

    /** Bytes one row costs here: every column, plus the validity word when there is one. */
    static int bytesPerRow(RowType rowType) {
        int bytes = 0;
        boolean anyNullable = false;
        for (LogicalType field : rowType.getChildren()) {
            bytes += widthOf(field);
            anyNullable |= field.isNullable();
        }
        return bytes + (anyNullable ? 4 : 0);
    }

    /** Whether every column can be held at a fixed width for as long as this needs to hold it. */
    static boolean canHold(RowType rowType) {
        if (rowType.getFieldCount() == 0 || rowType.getFieldCount() > MAX_FIELDS) {
            return false;
        }
        for (LogicalType field : rowType.getChildren()) {
            if (widthOf(field) == 0) {
                return false;
            }
        }
        return true;
    }

    /** Bytes one value of this type occupies, or 0 for a type this cannot hold. */
    static int widthOf(LogicalType type) {
        switch (type.getTypeRoot()) {
            case INTEGER:
            case FLOAT:
                return 4;
            case BIGINT:
            case DOUBLE:
                return 8;
            default:
                return 0;
        }
    }

    private static ByteBuffer buffer(int bytes, @Nullable GeneratedKernelEngine.Staging staging) {
        // A heap buffer where there is no arena, rather than a direct one: a direct buffer counts
        // against MaxDirectMemorySize, which is the accounting M3.1 exists to stop depending on.
        return staging == null ? ByteBuffer.allocate(bytes) : staging.allocate(bytes);
    }

    private static IntArray allocateInts(
            int capacity, @Nullable GeneratedKernelEngine.Staging staging) {
        if (staging == null) {
            return (IntArray) GeneratedKernel.allocate(GpuValueType.INT, capacity);
        }
        return (IntArray)
                GeneratedKernel.allocateOn(
                        GpuValueType.INT,
                        capacity,
                        staging.allocate(GeneratedKernel.sizeOf(GpuValueType.INT, capacity)));
    }

    /** A bare {@link IntArray} of the same provenance, for a buffer with no row behind it. */
    static IntArray ints(int capacity, @Nullable GeneratedKernelEngine.Staging staging) {
        return allocateInts(capacity, staging);
    }
}
