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

import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.writer.BinaryRowWriter;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The host half of the device transpose, checked without a device.
 *
 * <h2>What is actually under test</h2>
 *
 * <p>{@link RowBlockGather} copies rows in whole and {@code DeviceDeinterleave} reads fields back
 * out at slot indices the gather computed. Nothing checks that those two agree — the kernel trusts
 * {@link RowBlockGather#slotOf} and a wrong index reads a neighbouring field, which is a plausible
 * number rather than a failure. So the contract between them is what these tests assert: write real
 * {@code BinaryRowData} rows through the gather, then read the block back at exactly the slots the
 * kernel would use, and require the original values.
 *
 * <p>That makes this a host-only test of a device contract, which is the point: it fails on a
 * machine with no GPU, and it fails for the reason a GPU run would have been wrong.
 *
 * <h2>Why the last row of a batch gets its own test</h2>
 *
 * <p>Because the first version of this code was wrong there and nowhere else. It added TornadoVM's
 * array header to the write offset, but the buffer the engine hands over is already sliced past the
 * header, so every write was displaced by exactly that much. Every row but the last still landed
 * inside the block — reading plausible, wrong values — and the last one overran it. A test whose
 * row count was not a full batch would have passed.
 */
class RowBlockGatherTest {

    /** Two doubles, which is the shape the fleet query stages and the narrowest worth blocking. */
    private static final int ARITY = 2;

    @Test
    @DisplayName("BinaryRowData's own layout decides the slots, so read them back and compare")
    void slotsPointAtTheFieldsTheyName() {
        final int rows = 4;
        final int slots = RowBlockGather.slotsFor(ARITY);
        final ByteBuffer block = block(rows, slots);
        final RowBlockGather gather = new RowBlockGather(block, slots);

        for (int i = 0; i < rows; i++) {
            gather.accept(row(i + 0.5, i + 100.25), i);
        }

        final int latSlot = RowBlockGather.slotOf(ARITY, 0);
        final int lonSlot = RowBlockGather.slotOf(ARITY, 1);
        for (int i = 0; i < rows; i++) {
            assertThat(slot(block, i * slots + latSlot))
                    .withFailMessage("row %d field 0 read back wrong; slotOf is out of step", i)
                    .isEqualTo(i + 0.5);
            assertThat(slot(block, i * slots + lonSlot))
                    .withFailMessage("row %d field 1 read back wrong; slotOf is out of step", i)
                    .isEqualTo(i + 100.25);
        }
    }

    @Test
    @DisplayName("the last row of a full block stays inside it, which is where the header bug was")
    void theLastRowOfABatchLandsInsideTheBlock() {
        final int rows = 64;
        final int slots = RowBlockGather.slotsFor(ARITY);
        final ByteBuffer block = block(rows, slots);
        final RowBlockGather gather = new RowBlockGather(block, slots);

        // Every position the engine will ever write, including the one at the very end. An offset
        // that is too large by any amount throws here rather than corrupting a neighbour.
        for (int i = 0; i < rows; i++) {
            gather.accept(row(i, -i), i);
        }

        final int last = rows - 1;
        assertThat(slot(block, last * slots + RowBlockGather.slotOf(ARITY, 0)))
                .isEqualTo((double) last);
        assertThat(slot(block, last * slots + RowBlockGather.slotOf(ARITY, 1)))
                .isEqualTo((double) -last);
        assertThat(gather.bulkRows()).isEqualTo(rows);
    }

    @Test
    @DisplayName("the slot arithmetic tracks BinaryRowData's own, not a copy of it")
    void theNullBitHeaderIsNotMistakenForData() {
        // Checked against Flink's own layout rather than against numbers written down here. The
        // header is not just null bits -- BinaryRowData reserves eight bits for the row kind on
        // top of them -- so arity 64 needs two words, not one, and hand-computed expectations get
        // that wrong. This assertion cannot drift from the real layout, because it asks it.
        for (int arity : new int[] {1, 2, 3, 7, 56, 57, 63, 64, 65, 128}) {
            final int headerSlots =
                    BinaryRowData.calculateBitSetWidthInBytes(arity) / RowBlockGather.SLOT_BYTES;
            assertThat(RowBlockGather.slotsFor(arity))
                    .withFailMessage(
                            "arity %d: block stride %d slots, but a row occupies %d",
                            arity, RowBlockGather.slotsFor(arity), headerSlots + arity)
                    .isEqualTo(headerSlots + arity);
            for (int field = 0; field < arity; field++) {
                assertThat(RowBlockGather.slotOf(arity, field))
                        .withFailMessage("arity %d field %d lands on the wrong slot", arity, field)
                        .isEqualTo(headerSlots + field);
            }
        }

        // And the one the kernel is handed for the fleet query, spelled out: two fields need one
        // word of header, so field 0 is at slot 1. Reading slot 0 would return the bitmap as a
        // double, silently.
        assertThat(RowBlockGather.slotsFor(2)).isEqualTo(3);
        assertThat(RowBlockGather.slotOf(2, 0)).isEqualTo(1);
    }

    @Test
    @DisplayName("only a single-segment binary row of the expected arity is copied")
    void refusesWhatItCannotCopyWholesale() {
        assertThat(RowBlockGather.canCopy(row(1.0, 2.0), ARITY)).isTrue();

        // Not a binary row: there is no contiguous fixed-width part to copy at all.
        assertThat(RowBlockGather.canCopy(GenericRowData.of(1.0, 2.0), ARITY)).isFalse();

        // Arity disagrees with the layout the block was sized and indexed for, so every slot
        // index would be off. Refusing sends the batch down the per-column path instead.
        assertThat(RowBlockGather.canCopy(row(1.0, 2.0), 3)).isFalse();
    }

    private static BinaryRowData row(double lat, double lon) {
        final BinaryRowData row = new BinaryRowData(ARITY);
        final BinaryRowWriter writer = new BinaryRowWriter(row);
        writer.writeDouble(0, lat);
        writer.writeDouble(1, lon);
        writer.complete();
        return row;
    }

    /** Native order, because the block is read back as the device would read it. */
    private static ByteBuffer block(int rows, int slots) {
        return ByteBuffer.allocateDirect(rows * slots * RowBlockGather.SLOT_BYTES)
                .order(ByteOrder.nativeOrder());
    }

    private static double slot(ByteBuffer block, int index) {
        return block.getDouble(index * RowBlockGather.SLOT_BYTES);
    }
}
