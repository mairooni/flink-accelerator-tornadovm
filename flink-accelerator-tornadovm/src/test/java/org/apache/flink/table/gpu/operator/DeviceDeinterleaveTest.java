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

import org.apache.flink.table.gpu.gather.RowBlockGather;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.lang.reflect.Method;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * The deinterleave kernel, run on the host as plain Java.
 *
 * <p>A {@code @Parallel} loop is ordinary Java until TornadoVM compiles it, so the transpose can be
 * checked for what it computes without a device. What that leaves untested is only the compilation,
 * which fails loudly; getting the indexing wrong does not, because a neighbouring slot holds a
 * perfectly plausible double.
 *
 * <p>{@link #signaturesMatchWhatTheEngineBuilds} is the one that earns its place. The engine names
 * the task by reflection and passes arguments positionally, so a signature and its argument list
 * disagreeing is not a compile error — it is a failure at graph-build time on a machine with a GPU,
 * which is the slowest possible place to find it.
 */
class DeviceDeinterleaveTest {

    @Test
    @DisplayName("each column is gathered from its own slot, for every row")
    void transposesARowBlock() {
        final int arity = 2;
        final int rows = 5;
        final int slots = RowBlockGather.slotsFor(arity);

        final DoubleArray block = new DoubleArray(rows * slots);
        block.init(Double.NaN); // so an untouched slot cannot pass as a value
        for (int i = 0; i < rows; i++) {
            block.set(i * slots + RowBlockGather.slotOf(arity, 0), i + 0.5);
            block.set(i * slots + RowBlockGather.slotOf(arity, 1), i + 100.25);
        }

        final DoubleArray lat = new DoubleArray(rows);
        final DoubleArray lon = new DoubleArray(rows);
        final IntArray live = new IntArray(1);
        live.set(0, rows);

        DeviceDeinterleave.deinterleave2(
                block,
                lat,
                lon,
                slots,
                RowBlockGather.slotOf(arity, 0),
                RowBlockGather.slotOf(arity, 1),
                live);

        for (int i = 0; i < rows; i++) {
            assertThat(lat.get(i)).isEqualTo(i + 0.5);
            assertThat(lon.get(i)).isEqualTo(i + 100.25);
        }
    }

    @Test
    @DisplayName("the live row count bounds the loop, so a short batch leaves the tail alone")
    void aShortBatchDoesNotReadTheTail() {
        // The engine reuses buffers across batches and narrows the grid rather than clearing them,
        // so the last batch of a partition is short and whatever the previous batch left behind is
        // still in the block. Reading it would fold stale rows into the answer.
        final int slots = RowBlockGather.slotsFor(1);
        final DoubleArray block = new DoubleArray(4 * slots);
        for (int i = 0; i < 4; i++) {
            block.set(i * slots + RowBlockGather.slotOf(1, 0), i + 1.0);
        }
        final DoubleArray out = new DoubleArray(4);
        out.init(-1.0);
        final IntArray live = new IntArray(1);
        live.set(0, 2);

        DeviceDeinterleave.deinterleave1(block, out, slots, RowBlockGather.slotOf(1, 0), live);

        assertThat(out.get(0)).isEqualTo(1.0);
        assertThat(out.get(1)).isEqualTo(2.0);
        assertThat(out.get(2)).withFailMessage("row 2 is past the live count").isEqualTo(-1.0);
        assertThat(out.get(3)).withFailMessage("row 3 is past the live count").isEqualTo(-1.0);
    }

    @Test
    @DisplayName("every arity resolves, and its signature matches the engine's argument order")
    void signaturesMatchWhatTheEngineBuilds() throws Exception {
        for (int columns = 1; columns <= DeviceDeinterleave.MAX_COLUMNS; columns++) {
            final Method entry = DeviceDeinterleave.entryFor(columns);
            final Class<?>[] parameters = entry.getParameterTypes();

            // block, one DoubleArray a column, slots, one int offset a column, rows.
            assertThat(parameters)
                    .as("arity %d parameter count", columns)
                    .hasSize(1 + columns + 1 + columns + 1);
            assertThat(parameters[0]).isEqualTo(DoubleArray.class);
            for (int c = 0; c < columns; c++) {
                assertThat(parameters[1 + c]).isEqualTo(DoubleArray.class);
            }
            assertThat(parameters[1 + columns]).isEqualTo(int.class);
            for (int c = 0; c < columns; c++) {
                assertThat(parameters[2 + columns + c]).isEqualTo(int.class);
            }
            assertThat(parameters[parameters.length - 1]).isEqualTo(IntArray.class);
        }
    }

    @Test
    @DisplayName("an unsupported column count is refused rather than resolved to the wrong method")
    void refusesAnArityItDoesNotHave() {
        assertThatThrownBy(() -> DeviceDeinterleave.entryFor(DeviceDeinterleave.MAX_COLUMNS + 1))
                .as("the engine checks MAX_COLUMNS first; this is the backstop")
                .isInstanceOf(NoSuchMethodException.class);
    }
}
