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
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;
import org.apache.flink.table.types.logical.RowType;

import javax.annotation.Nullable;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A Gram matrix recognised in an ungrouped aggregate: {@code SUM(fi*fj)} for every {@code i <= j}.
 *
 * <h2>Why the recognition is here and not in the planner</h2>
 *
 * <p>Because it is knowledge about a library, and the IR is meant not to carry any. Flink's side
 * already describes this query exactly — an ungrouped {@link AccelAggregate} of {@code d(d+1)/2}
 * sums over a projection whose expressions are pairwise products — and a planner that additionally
 * labelled it "this is a GEMM" would be naming a vendor primitive in a device-neutral IR. That the
 * shape happens to be {@code A'A} is something cuBLAS knows and Flink should not, which is what
 * M0.2 meant by a provider claiming a whole subtree.
 *
 * <p>It also means this needed no IR change at all. The planner already builds the node, already
 * admits an empty grouping set, and already fuses the Calc into it (M4.1). M5.0 predicted M5.6
 * would need a generic "this subtree is one library call" node and would thereby enable M5.2 to
 * M5.4; it needed no node, and those four were served without it.
 *
 * <h2>What the shape has to be</h2>
 *
 * <p>Exactly the upper triangle, once each, in row-major order — {@code (0,0) (0,1) … (0,d-1) (1,1)
 * … (d-1,d-1)}. That is what Calcite produces for the query written in the obvious way, and it is
 * what lets the result be read straight out of a {@code d x d} symmetric matrix without a
 * permutation. A projection carrying the same products in some other order is refused rather than
 * reordered: the reorder is easy and the way to get it wrong is silent.
 */
public final class GpuGramSpec implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Fewest features worth a GEMM.
     *
     * <p>§T15 measured 1.05x at eight features and 12.09x at sixty-four — the speedup grows with
     * the width because the arithmetic does and the transfer does not. Below this the two arms are
     * within each other's noise and the offload is not worth a device context.
     */
    public static final int MIN_FEATURES = 8;

    private final List<AccelExpression> features;
    private final RowType inputType;
    private final RowType outputType;

    private GpuGramSpec(List<AccelExpression> features, RowType inputType, RowType outputType) {
        this.features = features;
        this.inputType = inputType;
        this.outputType = outputType;
    }

    /** The {@code d} expressions whose Gram matrix this is, in column order. */
    public List<AccelExpression> features() {
        return features;
    }

    public int featureCount() {
        return features.size();
    }

    /** The row the features are computed from. */
    public RowType inputType() {
        return inputType;
    }

    /** One row of {@code d(d+1)/2} sums, the upper triangle in row-major order. */
    public RowType outputType() {
        return outputType;
    }

    /** Where entry {@code (i, j)} of the symmetric matrix lands in the output row. */
    public static int triangleIndex(int d, int i, int j) {
        // Rows above i contribute d-0, d-1, ... entries; within row i, column j is at j-i.
        return i * d - (i * (i - 1)) / 2 + (j - i);
    }

    /** Whether a subtree is a Gram matrix, or why it is not. */
    public static Recognition recognise(AccelAggregate aggregate) {
        if (aggregate.grouping().length != 0) {
            return Recognition.no("a Gram matrix is an ungrouped aggregate");
        }
        AccelNode input = aggregate.inputs().get(0);
        if (!(input instanceof AccelProject)) {
            return Recognition.no("the aggregate's input is not a projection");
        }
        AccelProject projection = (AccelProject) input;
        List<AccelExpression> products = projection.projections();
        List<AccelAggCall> calls = aggregate.calls();
        if (calls.size() != products.size()) {
            return Recognition.no(
                    calls.size() + " sums over " + products.size() + " projected columns");
        }
        int d = widthOf(calls.size());
        if (d < 0) {
            return Recognition.no(
                    calls.size() + " sums is not d(d+1)/2 for any d, so not an upper triangle");
        }
        if (d < MIN_FEATURES) {
            return Recognition.no(d + " features is below the " + MIN_FEATURES + " a GEMM repays");
        }
        for (int i = 0; i < calls.size(); i++) {
            AccelAggCall call = calls.get(i);
            if (call.function() != AccelAggFunction.SUM || call.inputField() != i) {
                // Each sum must take the projected column in the same position, or the triangle
                // the operator reads back is not the triangle the query asked for.
                return Recognition.no("sum " + i + " is not SUM over projected column " + i);
            }
            if (call.outputType().getTypeRoot() != LogicalTypeRoot.DOUBLE) {
                return Recognition.no("sum " + i + " is not a DOUBLE");
            }
        }

        // The features, discovered in the order they first appear, which for the upper triangle in
        // row-major order is feature order: (0,0) introduces f0, (0,1) introduces f1, and so on.
        Map<String, Integer> featureIndex = new LinkedHashMap<>();
        List<AccelExpression> features = new ArrayList<>();
        int[][] pairs = new int[products.size()][];
        for (int p = 0; p < products.size(); p++) {
            AccelExpression product = products.get(p);
            if (!(product instanceof AccelCall)
                    || ((AccelCall) product).function() != AccelFunction.TIMES
                    || ((AccelCall) product).operands().size() != 2) {
                return Recognition.no("projected column " + p + " is not a product of two terms");
            }
            List<AccelExpression> operands = ((AccelCall) product).operands();
            int left = indexOf(featureIndex, features, operands.get(0));
            int right = indexOf(featureIndex, features, operands.get(1));
            pairs[p] = new int[] {Math.min(left, right), Math.max(left, right)};
        }
        if (features.size() != d) {
            return Recognition.no(
                    features.size() + " distinct features but " + calls.size() + " products");
        }
        int at = 0;
        for (int i = 0; i < d; i++) {
            for (int j = i; j < d; j++) {
                if (pairs[at][0] != i || pairs[at][1] != j) {
                    return Recognition.no(
                            "column "
                                    + at
                                    + " is f"
                                    + pairs[at][0]
                                    + "*f"
                                    + pairs[at][1]
                                    + ", not the upper triangle's f"
                                    + i
                                    + "*f"
                                    + j);
                }
                at++;
            }
        }
        RowType inputType = projection.inputs().get(0).outputType();
        for (LogicalType column : inputType.getChildren()) {
            if (column.isNullable()) {
                // NOT NULL, as everywhere else here, and with a sharper consequence than usual: a
                // null does not spoil one cell of a Gram matrix, it spoils the row's whole outer
                // product and therefore every cell. The kernel would also carry a validity word
                // per row, which changes its parameter list -- and an engine built for the
                // unvalidated shape then calls it with the wrong arity.
                return Recognition.no("a nullable input column is not contracted");
            }
        }
        return Recognition.yes(new GpuGramSpec(features, inputType, aggregate.outputType()));
    }

    /**
     * Interns an expression by its {@code toString}, which is the IR's structural form.
     *
     * <p>Good enough and deliberately so: {@link AccelExpression} has no {@code equals}, and the
     * rendering is a full structural description — a literal prints its value, an input ref its
     * index, a call its function and operands. Two expressions that print the same compute the same
     * thing. The failure direction is a false <em>difference</em>, which refuses a query rather
     * than mis-computing one.
     */
    private static int indexOf(
            Map<String, Integer> seen, List<AccelExpression> features, AccelExpression expression) {
        String key = expression.toString();
        Integer known = seen.get(key);
        if (known != null) {
            return known;
        }
        int index = features.size();
        seen.put(key, index);
        features.add(expression);
        return index;
    }

    /** The {@code d} with {@code d(d+1)/2 == entries}, or -1 if there is none. */
    static int widthOf(int entries) {
        for (int d = 1; d * (d + 1) / 2 <= entries; d++) {
            if (d * (d + 1) / 2 == entries) {
                return d;
            }
        }
        return -1;
    }

    @Override
    public String toString() {
        return "GpuGramSpec["
                + features.size()
                + " features, "
                + outputType.getFieldCount()
                + " sums]";
    }

    /** Yes with a spec, or no with a reason. */
    public static final class Recognition {
        private final @Nullable GpuGramSpec spec;
        private final @Nullable String reason;

        private Recognition(@Nullable GpuGramSpec spec, @Nullable String reason) {
            this.spec = spec;
            this.reason = reason;
        }

        static Recognition yes(GpuGramSpec spec) {
            return new Recognition(spec, null);
        }

        static Recognition no(String reason) {
            return new Recognition(null, reason);
        }

        public boolean recognised() {
            return spec != null;
        }

        public GpuGramSpec spec() {
            return spec;
        }

        public String reason() {
            return reason;
        }
    }
}
