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
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelJoin;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelScan;
import org.apache.flink.table.types.logical.FloatType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

import javax.annotation.Nullable;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * A nearest-neighbour query recognised as a GEMM with a reduction after it:
 *
 * <pre>
 *   SELECT q.id, MAX(q.f0*p.f0 + ... + q.f{d-1}*p.f{d-1})
 *   FROM Q q JOIN P p ON q.band = p.band
 *   GROUP BY q.id
 * </pre>
 *
 * <h2>Why this shape and not the Gram matrix</h2>
 *
 * <p>Both end in a contraction a linear-algebra library serves, and only one of them can ever show
 * what the library is worth. A Gram matrix over {@code n} rows of {@code d} columns is
 * {@code 2·n·d²} of arithmetic over {@code 4·n·d} bytes, so the GEMM is {@code d/2} operations a
 * byte read — about 16 at the widest {@code d} the Gram's SQL spelling can reach, because that
 * spelling needs {@code d(d+1)/2} aggregate calls and Calcite's planning cost passes two minutes at
 * {@code d = 64}. A device does floating point some three thousand times faster than a source
 * delivers bytes, so at 16 operations a byte the GEMM is a fraction of a percent of the job, and
 * swapping it for a hand-written kernel changes nothing anyone can measure. That is the whole of
 * why every library this project has measured came out near 1.00x.
 *
 * <p>This shape breaks that for one reason: its arithmetic is quadratic in the rows and its input
 * is linear in them. Every probe row meets every build row, so the work is {@code 2·nQ·nP·d} over
 * {@code 4·(nQ+nP)·d} bytes — {@code nQ·nP/(2(nQ+nP))} operations a byte, which at a hundred
 * thousand rows a side is four orders of magnitude past where a Gram matrix tops out. The GEMM
 * becomes the job rather than a rounding error in it, and what tunes the GEMM becomes visible end
 * to end.
 *
 * <p>And the SQL stays small. The dot product is <em>one</em> expression of {@code d} products, not
 * {@code d(d+1)/2} aggregate calls, so {@code d} may be 256 or 512 without Calcite noticing — which
 * matters twice over, because a GEMM's own arithmetic intensity is set by its reduction dimension
 * and that dimension is {@code d}.
 *
 * <h2>What the shape has to be</h2>
 *
 * <p>The projection is the join key's left column, then a left-deep sum of exactly {@code d}
 * products, each pairing one column of the probe row with one column of the build row. The pairs
 * must be in step — the {@code i}-th product multiplies the {@code i}-th admitted column of each
 * side — because that is what makes the expression an inner product between two contiguous vectors
 * rather than an arbitrary bilinear form. A projection carrying the same products in some other
 * order is refused rather than reordered: the reorder is easy and the way to get it wrong is
 * silent. This is the same rule {@link GpuGramSpec} applies, for the same reason.
 *
 * <h2>Why the join is not the thing being offloaded</h2>
 *
 * <p>Nothing here joins. A join produces the pairs and this never does: the GEMM computes every
 * pair's score straight from the two matrices, and the maximum is taken over a tile of scores
 * before the tile is overwritten. The equality in the {@code ON} clause survives only as a
 * requirement that it hold — a single band is a cross join, and more than one band is a block
 * diagonal the region walks a block at a time. That is why the band column must be {@code INT NOT
 * NULL} and dense: it indexes blocks, it is not hashed.
 */
public final class GpuSimilarityJoinSpec implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Fewest paired columns worth a GEMM over a kernel.
     *
     * <p>Measured on this shape rather than assumed. At {@code d = 32} a fused kernel that never
     * materialises the score matrix matches {@code cublasSgemm} to within 4% — it reads both
     * operands straight from global memory and at that width there is little enough traffic that
     * it does not matter. At 128 the GEMM is 3.1x the best tiled kernel and 36x the fused one, and
     * at 512 it is 4.2x and 68x. The gap is the GEMM's reduction dimension and nothing else, so a
     * floor stated in {@code d} is the right shape of floor. 64 is where it is worth the device
     * context; below it the region is still correct and still offered, and the cost model decides.
     */
    public static final int MIN_WIDTH = 64;

    private final int[] probeColumns;
    private final int[] buildColumns;
    private final int groupKeyField;
    private final boolean probeIsLeft;
    private final RowType probeType;
    private final RowType buildType;
    private final RowType outputType;

    private GpuSimilarityJoinSpec(
            int[] probeColumns,
            int[] buildColumns,
            int groupKeyField,
            boolean probeIsLeft,
            RowType probeType,
            RowType buildType,
            RowType outputType) {
        this.probeColumns = probeColumns;
        this.buildColumns = buildColumns;
        this.groupKeyField = groupKeyField;
        this.probeIsLeft = probeIsLeft;
        this.probeType = probeType;
        this.buildType = buildType;
        this.outputType = outputType;
    }

    public int width() {
        return probeColumns.length;
    }

    public int[] probeColumns() {
        return probeColumns.clone();
    }

    public int[] buildColumns() {
        return buildColumns.clone();
    }

    /** The probe column the query groups by, which is the identifier each answer is labelled with. */
    public int groupKeyField() {
        return groupKeyField;
    }

    public boolean probeIsLeft() {
        return probeIsLeft;
    }

    public RowType probeType() {
        return probeType;
    }

    public RowType buildType() {
        return buildType;
    }

    public RowType outputType() {
        return outputType;
    }

    /** What a recognition attempt concluded, and why when it concluded nothing. */
    public static final class Recognition {
        private final @Nullable GpuSimilarityJoinSpec spec;
        private final @Nullable AccelScan probeScan;
        private final @Nullable AccelScan buildScan;
        private final String reason;

        private Recognition(
                @Nullable GpuSimilarityJoinSpec spec,
                @Nullable AccelScan probeScan,
                @Nullable AccelScan buildScan,
                String reason) {
            this.spec = spec;
            this.probeScan = probeScan;
            this.buildScan = buildScan;
            this.reason = reason;
        }

        static Recognition no(String reason) {
            return new Recognition(null, null, null, reason);
        }

        static Recognition yes(GpuSimilarityJoinSpec spec, AccelScan probe, AccelScan build) {
            return new Recognition(spec, probe, build, "");
        }

        public boolean recognised() {
            return spec != null;
        }

        public GpuSimilarityJoinSpec spec() {
            return spec;
        }

        /** The side read by split: one subtask's share of the probe rows. */
        public AccelScan probeScan() {
            return probeScan;
        }

        /** The side read whole on every subtask, which is what the planner chose to broadcast. */
        public AccelScan buildScan() {
            return buildScan;
        }

        public String reason() {
            return reason;
        }
    }

    public static Recognition recognise(AccelAggregate aggregate) {
        if (aggregate.grouping().length != 1) {
            return Recognition.no(
                    "a nearest neighbour is one answer per probe row, so it groups by exactly one"
                            + " key; this groups by "
                            + aggregate.grouping().length);
        }
        final List<AccelAggCall> calls = aggregate.calls();
        if (calls.size() != 1 || calls.get(0).function() != AccelAggFunction.MAX) {
            return Recognition.no("the aggregate is not a single MAX");
        }
        if (aggregate.inputs().size() != 1 || !(aggregate.inputs().get(0) instanceof AccelProject)) {
            return Recognition.no("no projection beneath the aggregate");
        }
        final AccelProject project = (AccelProject) aggregate.inputs().get(0);
        if (project.inputs().size() != 1 || !(project.inputs().get(0) instanceof AccelJoin)) {
            return Recognition.no("the projection does not sit on a join");
        }
        final AccelJoin join = (AccelJoin) project.inputs().get(0);
        // Only a cross join, for now, and the limit is the read binding rather than the shape.
        // `Cudf::readParquet` carries one INT column a read, which the probe side already spends
        // on the identifier the query groups by; an equality would need a second for the band on
        // each side. A banded join is the same region with a mask in the epilogue -- the GEMM
        // computes every pair either way -- and is worth having when the binding carries two keys.
        if (join.probeKeyField() != AccelJoin.NO_KEY
                || join.buildKeyField() != AccelJoin.NO_KEY) {
            return Recognition.no(
                    "the join has an equality, and this region serves every pair or none");
        }
        if (join.inputs().size() != 2
                || !(join.inputs().get(0) instanceof AccelScan)
                || !(join.inputs().get(1) instanceof AccelScan)) {
            return Recognition.no("the join does not read two files");
        }
        // AccelJoin lists the build side first, which is a costing decision about which side has
        // to be resident and says nothing about which side this query labels its answers with.
        // The projection, though, reads the joined row in query order -- left side's columns, then
        // right side's -- so that order is what the column indexes below mean.
        final AccelScan leftScan =
                (AccelScan) join.inputs().get(join.buildIsLeft() ? 0 : 1);
        final AccelScan rightScan =
                (AccelScan) join.inputs().get(join.buildIsLeft() ? 1 : 0);
        if (!"parquet".equals(leftScan.format()) || !"parquet".equals(rightScan.format())) {
            return Recognition.no("the region reads parquet, and these are not both parquet");
        }
        final RowType leftType = leftScan.outputType();
        final RowType rightType = rightScan.outputType();
        final int leftWidth = leftType.getFieldCount();

        final List<AccelExpression> projections = project.projections();
        if (projections.size() != 2) {
            return Recognition.no(
                    "the projection is "
                            + projections.size()
                            + " expressions; a nearest neighbour is the key and the score");
        }
        // The aggregate names its key and its value as indexes into the projection's output.
        final int keyAt = aggregate.grouping()[0];
        final int valueAt = calls.get(0).inputField();
        if (keyAt == valueAt || keyAt > 1 || valueAt > 1) {
            return Recognition.no("the aggregate's key and value are not the projection's two");
        }
        if (!(projections.get(keyAt) instanceof AccelInputRef)) {
            return Recognition.no("the grouping key is computed rather than a column");
        }
        final int keyRef = ((AccelInputRef) projections.get(keyAt)).index();
        // Which side the query labels its answers with is what makes it the probe. It is read by
        // split, one answer per row of it; the other side is the corpus and is read whole on every
        // subtask. Deriving this from the grouping key rather than from which side Flink chose to
        // build is the difference between a region that is selected and one that is not -- the
        // build choice flips with the relative sizes of the two tables and is none of this's
        // business.
        final boolean probeIsLeft = keyRef < leftWidth;
        final AccelScan probeScan = probeIsLeft ? leftScan : rightScan;
        final AccelScan buildScan = probeIsLeft ? rightScan : leftScan;
        final int groupKeyField = probeIsLeft ? keyRef : keyRef - leftWidth;

        final List<AccelExpression> terms = new ArrayList<>();
        if (!flattenSum(projections.get(valueAt), terms)) {
            return Recognition.no("the score is not a sum of products");
        }
        if (terms.size() < 2) {
            return Recognition.no("the score has " + terms.size() + " terms; an inner product has d");
        }
        final int[] leftColumns = new int[terms.size()];
        final int[] rightColumns = new int[terms.size()];
        for (int i = 0; i < terms.size(); i++) {
            if (!(terms.get(i) instanceof AccelCall)) {
                return Recognition.no("term " + i + " of the score is not a product");
            }
            final AccelCall product = (AccelCall) terms.get(i);
            if (product.function() != AccelFunction.TIMES || product.operands().size() != 2) {
                return Recognition.no("term " + i + " of the score is not a two-operand product");
            }
            final AccelExpression a = product.operands().get(0);
            final AccelExpression b = product.operands().get(1);
            if (!(a instanceof AccelInputRef) || !(b instanceof AccelInputRef)) {
                return Recognition.no("term " + i + " multiplies something other than two columns");
            }
            int l = ((AccelInputRef) a).index();
            int r = ((AccelInputRef) b).index();
            if (l > r) {
                final int swap = l;
                l = r;
                r = swap;
            }
            if (l >= leftWidth || r < leftWidth) {
                // Both from one side is a square or a product within a row: real arithmetic, but
                // not an inner product between the two sides, and not a GEMM.
                return Recognition.no("term " + i + " does not pair a column of each side");
            }
            leftColumns[i] = l;
            rightColumns[i] = r - leftWidth;
        }
        // In step, and contiguous. Out of step the expression is still bilinear and still
        // computable, but it is not Q.P' over two contiguous vectors, and gathering it into one
        // would be a permutation this declines to invent.
        for (int i = 1; i < leftColumns.length; i++) {
            if (leftColumns[i] != leftColumns[i - 1] + 1
                    || rightColumns[i] != rightColumns[i - 1] + 1) {
                return Recognition.no(
                        "the products are not over two contiguous column ranges in the same order");
            }
        }

        final int[] probeColumns = probeIsLeft ? leftColumns : rightColumns;
        final int[] buildColumns = probeIsLeft ? rightColumns : leftColumns;
        final RowType probeType = probeScan.outputType();
        final RowType buildType = buildScan.outputType();

        final String typed = admissible(probeType, buildType, probeColumns, buildColumns);
        if (typed != null) {
            return Recognition.no(typed);
        }
        if (!(probeType.getTypeAt(groupKeyField) instanceof IntType)
                || probeType.getTypeAt(groupKeyField).isNullable()) {
            return Recognition.no("the grouping key is not INT NOT NULL");
        }
        final LogicalType score = calls.get(0).outputType();
        if (!(score instanceof FloatType)) {
            return Recognition.no("the score is " + score + ", and this region computes FLOAT");
        }

        return Recognition.yes(
                new GpuSimilarityJoinSpec(
                        probeColumns,
                        buildColumns,
                        groupKeyField,
                        probeIsLeft,
                        probeType,
                        buildType,
                        aggregate.outputType()),
                probeScan,
                buildScan);
    }

    /** A left-deep (or any-shaped) tree of PLUS, flattened to its leaves. */
    private static boolean flattenSum(AccelExpression expression, List<AccelExpression> into) {
        if (expression instanceof AccelCall
                && ((AccelCall) expression).function() == AccelFunction.PLUS) {
            final AccelCall sum = (AccelCall) expression;
            if (sum.operands().size() != 2) {
                return false;
            }
            return flattenSum(sum.operands().get(0), into)
                    && flattenSum(sum.operands().get(1), into);
        }
        into.add(expression);
        return true;
    }

    private static @Nullable String admissible(
            RowType probeType, RowType buildType, int[] probeColumns, int[] buildColumns) {
        for (int i = 0; i < probeColumns.length; i++) {
            final LogicalType p = probeType.getTypeAt(probeColumns[i]);
            final LogicalType b = buildType.getTypeAt(buildColumns[i]);
            if (!(p instanceof FloatType) || !(b instanceof FloatType)) {
                return "the paired columns are " + p + " and " + b + ", and the GEMM is FP32";
            }
            if (p.isNullable() || b.isNullable()) {
                return "a nullable column has no dense representation in the matrix";
            }
        }
        return null;
    }
}
