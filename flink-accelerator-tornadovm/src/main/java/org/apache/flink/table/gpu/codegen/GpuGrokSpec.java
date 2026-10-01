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
import org.apache.flink.table.accelerator.AccelFilter;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelLiteral;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelScan;
import org.apache.flink.table.types.logical.CharType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.table.types.logical.VarCharType;

import javax.annotation.Nullable;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * A multi-pattern regex filter over one string column, counted:
 *
 * <pre>
 *   SELECT COUNT(*) FROM Logs
 *   WHERE REGEXP(line, 'p1') AND REGEXP(line, 'p2') AND ... AND REGEXP(line, 'pk')
 * </pre>
 *
 * <h2>Why this shape earns a library and a single {@code LIKE} does not</h2>
 *
 * <p>Measured rather than assumed, in {@code VERIFY.md} §T54–§T56. cuDF matches a regex at tens of
 * GB/s against {@code java.util.regex}'s hundreds of MB/s a core, so one pattern is worth 138x–233x
 * against one core. But one pattern is also only ~6 ms of device work against a ~72 ms read, which
 * puts the query straight back in the regime §T20 and §T49 describe, where the library's own work
 * is a rounding error and the reader decides the answer.
 *
 * <p>What changes that is the pattern count. Each extra pattern costs the device ~6.6 ms and four
 * CPU cores ~271 ms, so the read is paid once on both sides while the matching diverges by ~41x a
 * pattern. The region is therefore worth selecting in proportion to {@code k}, which is why
 * {@link #MIN_PATTERNS} exists and is not 1.
 *
 * <h2>What the shape has to be</h2>
 *
 * <p>One string column, every predicate a {@code REGEXP} over that same column against a literal
 * pattern, combined with {@code AND}, and a {@code COUNT(*)} above. The patterns must be literals
 * because a regex is compiled; a pattern computed per row is refused rather than compiled per row.
 *
 * <h2>The dialect, which is the real hazard</h2>
 *
 * <p>Flink evaluates {@code REGEXP} with {@code java.util.regex}; cuDF implements its own dialect
 * and rejects patterns using features it lacks. The two agree on the common subset and not
 * everywhere, so a pattern is admitted only when it uses constructs both engines read the same way.
 * {@link #unsupported} is that filter, and it is deliberately conservative: declining a pattern
 * costs a CPU query, and admitting one that means something different costs a wrong answer.
 */
public final class GpuGrokSpec implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Fewest patterns worth the region.
     *
     * <p>At one pattern the device does ~6 ms of matching under a ~72 ms read, and the end-to-end
     * ratio is the reader's rather than the library's. The marginal pattern is where the gap is, so
     * the floor is stated in patterns. Four is where the measured whole-query ratio at parallelism
     * 4 passes 15x (§T56); below it the region is still correct and the cost model may still take
     * it, but it is not what this shape was built for.
     */
    public static final int MIN_PATTERNS = 2;

    /** Constructs where cuDF's regex and {@code java.util.regex} are known to diverge or to fail. */
    private static final String[] REFUSED = {
        "(?=", "(?!", "(?<=", "(?<!",   // lookaround: cuDF has none
        "\\b", "\\B",                     // word boundaries differ on Unicode
        "(?i", "(?m", "(?s", "(?x",      // inline flags: cuDF takes flags per call, not inline
        "\\p{", "\\P{",                  // Unicode property classes differ
        "++", "*+", "?+",                 // possessive quantifiers: Java only
        "\\G", "\\A", "\\Z", "\\z",    // anchors Java has and cuDF does not
    };

    private final int stringField;
    private final List<String> patterns;
    private final RowType outputType;

    private GpuGrokSpec(int stringField, List<String> patterns, RowType outputType) {
        this.stringField = stringField;
        this.patterns = patterns;
        this.outputType = outputType;
    }

    /** Index, into the scan's produced row, of the column every pattern matches against. */
    public int stringField() {
        return stringField;
    }

    public List<String> patterns() {
        return new ArrayList<>(patterns);
    }

    public RowType outputType() {
        return outputType;
    }

    /** What a recognition attempt concluded, and why when it concluded nothing. */
    public static final class Recognition {
        private final @Nullable GpuGrokSpec spec;
        private final @Nullable AccelScan scan;
        private final String reason;

        private Recognition(@Nullable GpuGrokSpec spec, @Nullable AccelScan scan, String reason) {
            this.spec = spec;
            this.scan = scan;
            this.reason = reason;
        }

        static Recognition no(String reason) {
            return new Recognition(null, null, reason);
        }

        static Recognition yes(GpuGrokSpec spec, AccelScan scan) {
            return new Recognition(spec, scan, "");
        }

        public boolean recognised() {
            return spec != null;
        }

        public GpuGrokSpec spec() {
            return spec;
        }

        public AccelScan scan() {
            return scan;
        }

        public String reason() {
            return reason;
        }
    }

    public static Recognition recognise(AccelAggregate aggregate) {
        if (aggregate.grouping().length != 0) {
            return Recognition.no("this region counts the whole partition and does not group");
        }
        final List<AccelAggCall> calls = aggregate.calls();
        if (calls.size() != 1 || calls.get(0).function() != AccelAggFunction.COUNT_STAR) {
            return Recognition.no("the aggregate is not a single COUNT(*)");
        }
        if (aggregate.inputs().size() != 1 || !(aggregate.inputs().get(0) instanceof AccelProject)) {
            return Recognition.no("no projection beneath the aggregate");
        }
        final AccelProject project = (AccelProject) aggregate.inputs().get(0);
        if (project.inputs().size() != 1 || !(project.inputs().get(0) instanceof AccelFilter)) {
            return Recognition.no("the projection does not sit on a filter");
        }
        final AccelFilter filter = (AccelFilter) project.inputs().get(0);
        if (filter.inputs().size() != 1 || !(filter.inputs().get(0) instanceof AccelScan)) {
            return Recognition.no("the filter does not read a file");
        }
        final AccelScan scan = (AccelScan) filter.inputs().get(0);
        if (!"parquet".equals(scan.format())) {
            return Recognition.no("the region reads parquet and this is " + scan.format());
        }

        final List<AccelExpression> terms = new ArrayList<>();
        if (!flattenAnd(filter.condition(), terms)) {
            return Recognition.no("the predicate is not a conjunction");
        }
        final List<String> patterns = new ArrayList<>(terms.size());
        int field = -1;
        for (int i = 0; i < terms.size(); i++) {
            if (!(terms.get(i) instanceof AccelCall)) {
                return Recognition.no("conjunct " + i + " is not a call");
            }
            final AccelCall call = (AccelCall) terms.get(i);
            if (call.function() != AccelFunction.REGEXP || call.operands().size() != 2) {
                return Recognition.no("conjunct " + i + " is not REGEXP(column, literal)");
            }
            if (!(call.operands().get(0) instanceof AccelInputRef)) {
                return Recognition.no("conjunct " + i + " matches a computed value, not a column");
            }
            final int ref = ((AccelInputRef) call.operands().get(0)).index();
            if (field == -1) {
                field = ref;
            } else if (field != ref) {
                return Recognition.no(
                        "the conjuncts match different columns, and this region reads one");
            }
            if (!(call.operands().get(1) instanceof AccelLiteral)) {
                return Recognition.no(
                        "conjunct " + i + " has a computed pattern; a regex is compiled, and one"
                                + " compiled per row is not a thing any engine does");
            }
            final Object value = ((AccelLiteral) call.operands().get(1)).value();
            if (!(value instanceof String)) {
                return Recognition.no("conjunct " + i + "'s pattern is not a character literal");
            }
            final String pattern = (String) value;
            final String refused = unsupported(pattern);
            if (refused != null) {
                return Recognition.no(
                        "pattern " + i + " uses " + refused
                                + ", which this binding's regex and java.util.regex do not agree on");
            }
            patterns.add(pattern);
        }
        if (field < 0 || field >= scan.outputType().getFieldCount()) {
            return Recognition.no("the matched column is not one the scan produces");
        }
        final LogicalType matched = scan.outputType().getTypeAt(field);
        if (!(matched instanceof VarCharType) && !(matched instanceof CharType)) {
            return Recognition.no("the matched column is " + matched + ", and REGEXP reads a string");
        }
        if (matched.isNullable()) {
            return Recognition.no(
                    "a nullable string has no validity mask in this binding; declare it NOT NULL");
        }
        return Recognition.yes(
                new GpuGrokSpec(field, patterns, aggregate.outputType()), scan);
    }

    /** Which refused construct a pattern uses, or null when it uses none of them. */
    public static @Nullable String unsupported(String pattern) {
        for (String construct : REFUSED) {
            if (pattern.contains(construct)) {
                return construct;
            }
        }
        return null;
    }

    private static boolean flattenAnd(AccelExpression expression, List<AccelExpression> into) {
        if (expression instanceof AccelCall
                && ((AccelCall) expression).function() == AccelFunction.AND) {
            // Calcite writes AND n-ary, not as a tree of pairs, so this takes any arity.
            for (AccelExpression operand : ((AccelCall) expression).operands()) {
                if (!flattenAnd(operand, into)) {
                    return false;
                }
            }
            return true;
        }
        into.add(expression);
        return true;
    }
}
