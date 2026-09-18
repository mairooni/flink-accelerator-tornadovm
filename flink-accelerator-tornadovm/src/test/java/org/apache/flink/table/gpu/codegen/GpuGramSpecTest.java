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

import org.apache.flink.table.accelerator.AccelAggCall;
import org.apache.flink.table.accelerator.AccelAggFunction;
import org.apache.flink.table.accelerator.AccelAggregate;
import org.apache.flink.table.accelerator.AccelCall;
import org.apache.flink.table.accelerator.AccelExpression;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * What counts as a Gram matrix, checked on the host.
 *
 * <p>This is the whole of M5.6's judgement: the planner emits an ungrouped aggregate over a
 * projection and says nothing about what shape it is, so everything that decides whether cuBLAS can
 * serve it is here. It is also the piece with the most ways to be quietly wrong — a triangle read
 * in the wrong order gives a full result with the entries transposed, which no row count sees.
 */
class GpuGramSpecTest {

    private static final LogicalType DOUBLE = new DoubleType(false);

    /** {@code SIN(ci)}, standing in for an arbitrary feature map. */
    private static AccelExpression feature(int i) {
        return new AccelCall(AccelFunction.SIN, List.of(new AccelInputRef(i, DOUBLE)), DOUBLE);
    }

    private static AccelExpression product(AccelExpression a, AccelExpression b) {
        return new AccelCall(AccelFunction.TIMES, Arrays.asList(a, b), DOUBLE);
    }

    /** The canonical shape: upper triangle, row-major, over {@code d} features. */
    private static AccelAggregate gram(int d) {
        return gram(d, false);
    }

    private static AccelAggregate gram(int d, boolean swapOneProduct) {
        List<AccelExpression> products = new ArrayList<>();
        for (int i = 0; i < d; i++) {
            for (int j = i; j < d; j++) {
                boolean swap = swapOneProduct && i == 0 && j == 1;
                products.add(
                        swap ? product(feature(j), feature(i)) : product(feature(i), feature(j)));
            }
        }
        return aggregateOver(products, d);
    }

    private static AccelAggregate aggregateOver(List<AccelExpression> products, int inputCols) {
        LogicalType[] inFields = new LogicalType[inputCols];
        Arrays.fill(inFields, DOUBLE);
        LogicalType[] projFields = new LogicalType[products.size()];
        Arrays.fill(projFields, DOUBLE);
        RowType inputType = RowType.of(inFields);
        RowType projType = RowType.of(projFields);
        AccelProject projection = new AccelProject(products, new AccelInput(inputType), projType);
        List<AccelAggCall> calls = new ArrayList<>();
        for (int i = 0; i < products.size(); i++) {
            calls.add(new AccelAggCall(AccelAggFunction.SUM, i, DOUBLE));
        }
        return new AccelAggregate(new int[0], calls, projection, projType);
    }

    @Test
    void recognisesTheUpperTriangle() {
        GpuGramSpec.Recognition r = GpuGramSpec.recognise(gram(8));
        assertThat(r.recognised()).as(String.valueOf(r.reason())).isTrue();
        assertThat(r.spec().featureCount()).isEqualTo(8);
        assertThat(r.spec().outputType().getFieldCount()).isEqualTo(36);
    }

    @Test
    void acceptsAProductWrittenInEitherOrder() {
        // f1*f0 is f0*f1. Refusing it would refuse a query Calcite is free to emit either way.
        GpuGramSpec.Recognition r = GpuGramSpec.recognise(gram(8, true));
        assertThat(r.recognised()).as(String.valueOf(r.reason())).isTrue();
    }

    @Test
    void refusesAGroupedAggregate() {
        AccelAggregate grouped =
                new AccelAggregate(
                        new int[] {0},
                        gram(8).calls(),
                        gram(8).inputs().get(0),
                        (RowType) gram(8).outputType());
        assertThat(GpuGramSpec.recognise(grouped).reason())
                .isEqualTo("a Gram matrix is an ungrouped aggregate");
    }

    @Test
    void refusesTooFewFeaturesToRepayAGemm() {
        // Four features is ten sums, a perfectly good triangle, and 1.05x at eight (§T15).
        assertThat(GpuGramSpec.recognise(gram(4)).reason()).contains("below the 8 a GEMM repays");
    }

    @Test
    void refusesAColumnCountThatIsNotATriangle() {
        List<AccelExpression> products = new ArrayList<>();
        for (int i = 0; i < 37; i++) {
            products.add(product(feature(0), feature(1)));
        }
        assertThat(GpuGramSpec.recognise(aggregateOver(products, 8)).reason())
                .contains("not d(d+1)/2 for any d");
    }

    @Test
    void refusesAProjectionThatIsNotAllProducts() {
        List<AccelExpression> products = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            for (int j = i; j < 8; j++) {
                products.add(product(feature(i), feature(j)));
            }
        }
        products.set(5, feature(0));
        assertThat(GpuGramSpec.recognise(aggregateOver(products, 8)).reason())
                .isEqualTo("projected column 5 is not a product of two terms");
    }

    @Test
    void refusesTheTriangleInTheWrongOrder() {
        // Every entry present, once each, and two of them swapped. The result would be a complete
        // Gram matrix with two cells transposed, which is exactly the silent failure.
        List<AccelExpression> products = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            for (int j = i; j < 8; j++) {
                products.add(product(feature(i), feature(j)));
            }
        }
        AccelExpression swap = products.get(1);
        products.set(1, products.get(2));
        products.set(2, swap);
        assertThat(GpuGramSpec.recognise(aggregateOver(products, 8)).reason())
                .contains("not the upper triangle's");
    }

    @Test
    void triangleIndexIsRowMajorAndSymmetric() {
        int d = 8;
        int at = 0;
        for (int i = 0; i < d; i++) {
            for (int j = i; j < d; j++) {
                assertThat(GpuGramSpec.triangleIndex(d, i, j)).isEqualTo(at++);
            }
        }
        assertThat(at).isEqualTo(d * (d + 1) / 2);
    }

    @Test
    void widthOfInvertsTheTriangleNumbers() {
        assertThat(GpuGramSpec.widthOf(36)).isEqualTo(8);
        assertThat(GpuGramSpec.widthOf(136)).isEqualTo(16);
        assertThat(GpuGramSpec.widthOf(2080)).isEqualTo(64);
        assertThat(GpuGramSpec.widthOf(37)).isEqualTo(-1);
    }
}
