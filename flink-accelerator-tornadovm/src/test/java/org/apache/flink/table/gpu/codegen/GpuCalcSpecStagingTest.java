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

package org.apache.flink.table.gpu.codegen;

import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.BooleanType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.FloatType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.table.types.logical.VarCharType;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Which output rows this backend can stage.
 *
 * <p>These used to be planner refusals, asserted through EXPLAIN. They are not: staging a column
 * means holding it in a primitive array of a width this backend has, which is a fact about this
 * backend and nothing at all about the query. The planner now reports such a node as eligible and
 * the provider declines it here, so a different backend -- one with a byte-packed boolean buffer,
 * say -- is free to accept what this one cannot.
 */
class GpuCalcSpecStagingTest {

    private static boolean canStage(int[] layout, LogicalType... fields) {
        GpuKernelSource kernel =
                new GpuKernelSource(
                        "K",
                        "evaluate",
                        "",
                        new int[] {0},
                        new GpuValueType[] {GpuValueType.DOUBLE},
                        new GpuValueType[] {GpuValueType.DOUBLE},
                        false,
                        layout);
        return new GpuCalcSpec(kernel, layout, RowType.of(fields), 1024).canStage();
    }

    private static final int OUT = GpuCalcSpec.COMPUTED;

    @Test
    @DisplayName("the numeric four pass through")
    void numericColumnsStage() {
        assertTrue(
                canStage(
                        new int[] {0, 1, 2, 3, OUT},
                        new BigIntType(),
                        new IntType(),
                        new FloatType(),
                        new DoubleType(),
                        new DoubleType()));
    }

    @Test
    @DisplayName("a BOOLEAN pass-through column has no buffer here")
    void booleanColumnIsRefused() {
        assertFalse(canStage(new int[] {0, OUT}, new BooleanType(), new DoubleType()));
    }

    @Test
    @DisplayName("nor does a variable-width one")
    void varcharColumnIsRefused() {
        assertFalse(canStage(new int[] {0, OUT}, new VarCharType(16), new DoubleType()));
    }

    @Test
    @DisplayName("a computed column must be one the kernel can write")
    void computedColumnMustBeFloatingPoint() {
        assertTrue(canStage(new int[] {OUT}, new DoubleType()));
        assertTrue(canStage(new int[] {OUT}, new FloatType()));
        assertFalse(canStage(new int[] {OUT}, new BigIntType()));
    }

    @Test
    @DisplayName("a layout that does not describe the row is refused rather than trusted")
    void mismatchedLayoutIsRefused() {
        assertFalse(canStage(new int[] {0, OUT}, new DoubleType()));
    }
}
