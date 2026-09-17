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

import org.apache.flink.table.accelerator.AccelCall;
import org.apache.flink.table.accelerator.AccelExpression;
import org.apache.flink.table.accelerator.AccelFilter;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelLiteral;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;

import javax.annotation.Nullable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;

/**
 * Generates a TornadoVM kernel, as Java source, from an accelerator IR subtree.
 *
 * <p>This lived in {@code flink-table-planner} and read {@code RexNode}s, which meant core Flink's
 * planner emitted {@code uk.ac.manchester.tornado} imports. No amount of tidying fixes that: it is
 * the wrong direction of dependency, and it is the single thing that would have stopped the
 * extension point being proposed as a FLIP. It now reads Flink's device-neutral IR instead, and
 * lives with the provider that owns the device.
 *
 * <h2>Storage follows the column, arithmetic does not</h2>
 *
 * <p>Each staged column is held at its declared width — {@code IntArray}, {@code FloatArray} or
 * {@code DoubleArray} — and widened to {@code double} on entry to the loop body, so the expression
 * is evaluated in double whatever it reads. Only the buffer and the transfer follow the column.
 * Reading a four-byte field with {@code getDouble} reads eight: an {@code INT} column holding 42
 * once reached the device as 2.08e-322.
 *
 * <p><b>{@code BIGINT} is excluded as an expression input.</b> Values beyond 2^53 do not survive
 * the round trip through a double. A {@code BIGINT} column may still be projected through: it never
 * reaches the device and keeps its exact type.
 *
 * <h2>Filtering does not compact</h2>
 *
 * <p>A condition becomes a 0/1 mask written alongside the projections. Compacting on the device
 * needs a prefix sum and a scatter — two more kernels and another buffer — whereas the host has to
 * walk the results anyway to build output rows.
 */
public final class AccelKernelGenerator {

    /**
     * Words that are legal Java identifiers but reserved in OpenCL C.
     *
     * <p>TornadoVM translates the generated Java into OpenCL, so a name taken from this list fails
     * late, inside the sketcher, with "Java method name corresponds to an OpenCL Token".
     */
    private static final Set<String> OPENCL_RESERVED =
            new HashSet<>(
                    Arrays.asList(
                            "kernel",
                            "global",
                            "local",
                            "constant",
                            "private",
                            "read_only",
                            "write_only",
                            "read_write",
                            "uniform",
                            "pipe",
                            "half",
                            "quad",
                            "complex",
                            "imaginary",
                            "generic"));

    private static final String INDENT = "        ";

    /** Marks an output field produced by the kernel rather than copied from the input. */
    public static final int COMPUTED = -1;

    private AccelKernelGenerator() {}

    /**
     * Generates a kernel for a projection, optionally over a filter, or empty if any part of it
     * cannot be written as one.
     *
     * <p>Returning empty is an ordinary outcome and is how this provider declines a subtree.
     */
    public static Optional<GpuKernelSource> generate(AccelNode subtree, String classNameSuffix) {
        return generate(subtree, classNameSuffix, 0);
    }

    /**
     * As above, optionally packing every computed column into one buffer {@code packedStride} rows
     * apart, column-major.
     *
     * <p>Packing exists so a contraction over <em>k</em> columns is one task rather than
     * <em>k</em>: a task takes a fixed argument list, so k separate output arrays force k separate
     * reductions, and TornadoVM refuses to build a graph past about a hundred tasks.
     */
    public static Optional<GpuKernelSource> generate(
            AccelNode subtree, String classNameSuffix, int packedStride) {

        if (!(subtree instanceof AccelProject)) {
            return Optional.empty();
        }
        if (inputs32Exceeded(subtree)) {
            // Validity is packed into one int a row, so 32 staged columns is the ceiling. Refusing
            // beyond it rather than silently widening: an expression over 33 columns would
            // otherwise have its 33rd read someone else's bit.
            return Optional.empty();
        }

        AccelProject project = (AccelProject) subtree;
        AccelNode input = project.inputs().get(0);
        AccelExpression condition = null;
        if (input instanceof AccelFilter) {
            condition = ((AccelFilter) input).condition();
            input = input.inputs().get(0);
        }
        // Anything deeper than Project-over-Filter-over-Input is a shape this generator has never
        // been asked for; declining is the correct answer, not an error.
        if (!input.inputs().isEmpty()) {
            return Optional.empty();
        }

        NULLABLE_INPUTS.get().clear();
        CSE.set(new Cse());
        final Map<Integer, String> inputs = new LinkedHashMap<>();
        final Map<Integer, GpuValueType> inputTypes = new LinkedHashMap<>();
        final List<String> computed = new ArrayList<>();
        final List<String> computedValidity = new ArrayList<>();
        final List<GpuValueType> outputTypes = new ArrayList<>();
        final int[] layout = new int[project.projections().size()];

        for (int field = 0; field < project.projections().size(); field++) {
            AccelExpression expression = project.projections().get(field);
            if (expression instanceof AccelInputRef) {
                layout[field] = ((AccelInputRef) expression).index();
                continue;
            }
            layout[field] = COMPUTED;
            if (!isDoubleResult(expression.outputType())) {
                return Optional.empty();
            }
            Rendered rendered = render(expression, inputs, inputTypes);
            if (rendered == null) {
                return Optional.empty();
            }
            computed.add(rendered.value);
            computedValidity.add(rendered.valid);
            outputTypes.add(valueTypeOf(expression.outputType()));
        }
        if (computed.isEmpty()) {
            // Nothing to compute; the CPU path already handles pure projection.
            return Optional.empty();
        }

        String renderedCondition = null;
        if (condition != null) {
            Rendered rendered = render(condition, inputs, inputTypes);
            if (rendered == null) {
                return Optional.empty();
            }
            // A comparison with an absent operand is UNKNOWN, and UNKNOWN does not select a row --
            // so the condition's own validity ANDs into the mask rather than being checked
            // separately. That is the whole of three-valued logic for a WHERE clause.
            renderedCondition =
                    rendered.alwaysPresent()
                            ? rendered.value
                            : "(" + rendered.valid + " != 0 && " + rendered.value + ")";
        }
        if (inputs.isEmpty()) {
            // A kernel with no input column has nothing to size the parallel loop by.
            return Optional.empty();
        }

        String className = "GpuCalcKernel$" + classNameSuffix;
        String methodName = "evaluate";
        if (OPENCL_RESERVED.contains(methodName.toLowerCase(Locale.ROOT))) {
            throw new IllegalStateException(
                    "generated method name collides with an OpenCL keyword");
        }

        int[] inputFields = inputs.keySet().stream().mapToInt(Integer::intValue).toArray();
        GpuValueType[] stagedTypes = new GpuValueType[inputFields.length];
        for (int i = 0; i < inputFields.length; i++) {
            stagedTypes[i] = inputTypes.get(inputFields[i]);
        }
        if (packedStride > 0) {
            for (GpuValueType type : outputTypes) {
                if (type != GpuValueType.DOUBLE) {
                    return Optional.empty();
                }
            }
        }
        boolean carriesValidity = carriesValidity(NULLABLE_INPUTS.get());
        String source =
                renderClass(
                        className,
                        methodName,
                        inputs,
                        inputTypes,
                        computed,
                        computedValidity,
                        NULLABLE_INPUTS.get(),
                        outputTypes,
                        renderedCondition,
                        packedStride,
                        CSE.get().declarations());
        return Optional.of(
                new GpuKernelSource(
                        className,
                        methodName,
                        source,
                        inputFields,
                        stagedTypes,
                        outputTypes.toArray(new GpuValueType[0]),
                        renderedCondition != null,
                        layout,
                        packedStride,
                        carriesValidity));
    }

    private static String renderClass(
            String className,
            String methodName,
            Map<Integer, String> inputs,
            Map<Integer, GpuValueType> inputTypes,
            List<String> computed,
            List<String> computedValidity,
            Set<Integer> nullableInputs,
            List<GpuValueType> outputTypes,
            @Nullable String condition,
            int packedStride,
            List<String> subexpressions) {

        Set<String> arrayTypes = new TreeSet<>();
        for (GpuValueType type : inputTypes.values()) {
            arrayTypes.add(type.arrayType());
        }
        if (packedStride > 0) {
            arrayTypes.add(GpuValueType.DOUBLE.arrayType());
        } else {
            for (GpuValueType type : outputTypes) {
                arrayTypes.add(type.arrayType());
            }
        }
        // IntArray unconditionally: the live row count is one, whether or not there is a mask.
        arrayTypes.add(GpuValueType.INT.arrayType());

        StringBuilder sb = new StringBuilder();
        sb.append("import uk.ac.manchester.tornado.api.annotations.Parallel;\n");
        sb.append("import uk.ac.manchester.tornado.api.math.TornadoMath;\n");
        for (String arrayType : arrayTypes) {
            sb.append("import uk.ac.manchester.tornado.api.types.arrays.")
                    .append(arrayType)
                    .append(";\n");
        }
        sb.append("\n");
        sb.append("public final class ").append(className).append(" {\n\n");
        sb.append("    public static void ").append(methodName).append("(");

        List<String> params = new ArrayList<>();
        for (Map.Entry<Integer, String> input : inputs.entrySet()) {
            params.add(inputTypes.get(input.getKey()).arrayType() + " " + input.getValue() + "_in");
        }
        if (packedStride > 0) {
            params.add("DoubleArray out");
        } else {
            for (int i = 0; i < computed.size(); i++) {
                params.add(outputTypes.get(i).arrayType() + " out" + i);
            }
        }
        if (condition != null) {
            params.add("IntArray mask");
        }
        // Validity, when anything in this kernel can be absent. One int a row, one bit a column,
        // rather than one array per column: four bytes a row whatever the column count, against
        // four bytes *per column* the naive way. On a path where every measurement so far is bound
        // by moving the input, that difference is the design.
        boolean carriesValidity = carriesValidity(nullableInputs);
        if (carriesValidity) {
            params.add("IntArray inNulls");
            params.add("IntArray outNulls");
        }
        // Last, so the engine can append it without knowing what precedes it.
        params.add("IntArray rows");
        sb.append(String.join(", ", params)).append(") {\n");

        // Bounded by the rows actually staged, not by the buffer's capacity.
        //
        // Capacity was the obvious bound and it is wrong in both directions. A partial batch --
        // the last of every partition, and every batch of a partition smaller than one -- had the
        // device evaluate the whole expression over the buffer's tail, which is the previous
        // batch's data or, on the first batch, whatever the allocator left. The results were never
        // read, so this was not a wrong answer; it was up to a full batch of arithmetic per
        // partition spent on values that do not exist, and one INF or NAN away from becoming a
        // wrong answer the moment anything downstream reads past the count.
        //
        // The count arrives in a one-element IntArray rather than as an int: a scalar argument is
        // captured when the task graph is built and cannot then change, and this has to change on
        // the last batch. An array is transferred on every execution.
        sb.append("        final int n = rows.get(0);\n");
        sb.append("        for (@Parallel int i = 0; i < n; i++) {\n");
        if (carriesValidity) {
            sb.append(INDENT).append("    final int nulls = inNulls.get(i);\n");
            int bit = 0;
            for (Integer index : inputs.keySet()) {
                // The bit is the column's position among the staged columns, which is the same
                // order the operator writes them in -- so the two cannot drift apart. Only a
                // nullable column gets a variable; a NOT NULL one has nothing to read.
                int position = bit++;
                if (!nullableInputs.contains(index)) {
                    continue;
                }
                // 1 means present, so the stored bit is inverted on the way in. Branchless: read
                // once per row per column and combined with & thereafter.
                sb.append(INDENT)
                        .append("    final int v")
                        .append(index)
                        .append(" = 1 - ((nulls >>> ")
                        .append(position)
                        .append(") & 1);\n");
            }
        }
        for (String var : inputs.values()) {
            sb.append(INDENT)
                    .append("    double ")
                    .append(var)
                    .append(" = ")
                    .append(var)
                    .append("_in.get(i);\n");
        }
        // One statement per distinct operation, in the order they were produced -- which is
        // bottom-up, so each one only mentions names already declared above it.
        for (String declaration : subexpressions) {
            sb.append(INDENT).append("    ").append(declaration).append("\n");
        }
        for (int i = 0; i < computed.size(); i++) {
            if (packedStride > 0) {
                sb.append(INDENT)
                        .append("    out.set(")
                        .append(i * packedStride)
                        .append(" + i, ")
                        .append(computed.get(i))
                        .append(");\n");
                continue;
            }
            GpuValueType outputType = outputTypes.get(i);
            // The kernel computes in double; a narrower column is cast on the way out.
            String open =
                    outputType == GpuValueType.FLOAT
                            ? "(float) ("
                            : outputType == GpuValueType.INT ? "(int) (" : "";
            boolean narrow = !open.isEmpty();
            sb.append(INDENT)
                    .append("    out")
                    .append(i)
                    .append(".set(i, ")
                    .append(open)
                    .append(computed.get(i))
                    .append(narrow ? ")" : "")
                    .append(");\n");
        }
        if (carriesValidity) {
            // The value is written whatever its validity -- computing it costs nothing extra and
            // branching to avoid it would cost divergence -- and the bit says whether to read it.
            // Absent slots hold a benign value rather than whatever was there, so no INF, NAN or
            // denormal arises from arithmetic on a row that does not exist.
            StringBuilder packed = new StringBuilder();
            for (int i = 0; i < computedValidity.size(); i++) {
                if (i > 0) {
                    packed.append(" | ");
                }
                packed.append("((1 - (")
                        .append(computedValidity.get(i))
                        .append(")) << ")
                        .append(i)
                        .append(")");
            }
            sb.append(INDENT)
                    .append("    outNulls.set(i, ")
                    .append(packed.length() == 0 ? "0" : packed.toString())
                    .append(");\n");
        }
        if (condition != null) {
            sb.append(INDENT).append("    if (").append(condition).append(") {\n");
            sb.append(INDENT).append("        mask.set(i, 1);\n");
            sb.append(INDENT).append("    } else {\n");
            sb.append(INDENT).append("        mask.set(i, 0);\n");
            sb.append(INDENT).append("    }\n");
        }
        sb.append("        }\n");
        sb.append("    }\n");
        sb.append("}\n");
        return sb.toString();
    }

    /** Renders one expression, registering any column it reads. Null if it cannot be written. */
    /** More staged columns than one int of validity can describe. */
    private static boolean inputs32Exceeded(AccelNode subtree) {
        java.util.Set<Integer> referenced = new java.util.HashSet<>();
        collectRefs(subtree, referenced);
        return referenced.size() > 32;
    }

    private static void collectRefs(AccelNode node, java.util.Set<Integer> into) {
        if (node instanceof AccelProject) {
            for (AccelExpression e : ((AccelProject) node).projections()) {
                collectRefs(e, into);
            }
        } else if (node instanceof AccelFilter) {
            collectRefs(((AccelFilter) node).condition(), into);
        }
        for (AccelNode input : node.inputs()) {
            collectRefs(input, into);
        }
    }

    private static void collectRefs(AccelExpression expression, java.util.Set<Integer> into) {
        if (expression instanceof AccelInputRef) {
            into.add(((AccelInputRef) expression).index());
        } else if (expression instanceof AccelCall) {
            for (AccelExpression operand : ((AccelCall) expression).operands()) {
                collectRefs(operand, into);
            }
        }
    }

    /** Whether this subtree contains an AND or an OR, whose null semantics are not strict. */
    private static boolean hasThreeValuedLogic(AccelNode node) {
        if (node instanceof AccelProject) {
            for (AccelExpression e : ((AccelProject) node).projections()) {
                if (hasThreeValuedLogic(e)) {
                    return true;
                }
            }
        } else if (node instanceof AccelFilter) {
            if (hasThreeValuedLogic(((AccelFilter) node).condition())) {
                return true;
            }
        }
        for (AccelNode input : node.inputs()) {
            if (hasThreeValuedLogic(input)) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasThreeValuedLogic(AccelExpression expression) {
        if (expression instanceof AccelCall) {
            AccelCall call = (AccelCall) expression;
            if (call.function() == AccelFunction.AND || call.function() == AccelFunction.OR) {
                return true;
            }
            for (AccelExpression operand : call.operands()) {
                if (hasThreeValuedLogic(operand)) {
                    return true;
                }
            }
        }
        return false;
    }

    /** Whether anything in this subtree may be null, values or conditions alike. */
    private static boolean hasNullable(AccelNode node) {
        if (node instanceof AccelProject) {
            for (AccelExpression e : ((AccelProject) node).projections()) {
                if (hasNullable(e)) {
                    return true;
                }
            }
        } else if (node instanceof AccelFilter) {
            if (hasNullable(((AccelFilter) node).condition())) {
                return true;
            }
        }
        for (AccelNode input : node.inputs()) {
            if (hasNullable(input)) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasNullable(AccelExpression expression) {
        if (expression.outputType().isNullable()) {
            return true;
        }
        if (expression instanceof AccelCall) {
            for (AccelExpression operand : ((AccelCall) expression).operands()) {
                if (hasNullable(operand)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * A rendered value, and the expression saying whether it is there.
     *
     * <p>Validity is an {@code int}: 1 present, 0 absent. An {@code int} rather than a {@code
     * boolean} because it is combined with {@code &} far more often than it is branched on, and
     * because the constant {@link #ALWAYS} then folds away in the emitted source — an expression
     * over {@code NOT NULL} columns produces exactly the kernel it produced before validity
     * existed, character for character.
     */
    /**
     * Names every distinct subexpression once, so the kernel is straight-line code.
     *
     * <p>Without this the generator emits a tree, and a tree repeats itself. Calcite expands {@code
     * LEAST(a, b, c, ...)} into nested conditionals that mention each operand several times over --
     * once for a null test, once per comparison, once per branch -- so a twenty-argument {@code
     * LEAST} over an expression of any size produces source that grows far faster than the
     * operation count the cost model sees. Measured 2026-09-17: the haversine benchmark at {@code
     * --depots 20} rendered to over 20 000 characters and javac rejected it, while the planner
     * counted 359 operations. Naming each subexpression as it is produced makes the emitted size
     * proportional to the distinct operations, which is what the cost model already counts.
     *
     * <p>It also reads better, which is not the point but is worth having: every statement is one
     * operation, which is what a reader of a generated kernel wants to see.
     */
    private static final class Cse {
        private final Map<String, String> byExpression = new LinkedHashMap<>();
        private final List<String> declarations = new ArrayList<>();
        private int next;

        /** The name standing for this expression, declaring it the first time it is seen. */
        String name(String expression, String javaType) {
            String existing = byExpression.get(expression);
            if (existing != null) {
                return existing;
            }
            String name = "t" + next++;
            byExpression.put(expression, name);
            declarations.add("final " + javaType + " " + name + " = " + expression + ";");
            return name;
        }

        List<String> declarations() {
            return declarations;
        }
    }

    /**
     * The subexpression names for the kernel being generated.
     *
     * <p>A thread local for the same reason {@link #NULLABLE_INPUTS} is one: rendering is a static
     * recursion and threading a context through every arm of it would say nothing the name does
     * not.
     */
    private static final ThreadLocal<Cse> CSE = ThreadLocal.withInitial(Cse::new);

    private static final class Rendered {
        static final String ALWAYS = "1";

        final String value;
        final String valid;

        Rendered(String value, String valid) {
            this.value = value;
            this.valid = valid;
        }

        boolean alwaysPresent() {
            return ALWAYS.equals(valid);
        }
    }

    /**
     * {@code AND} and {@code OR} under SQL's three-valued logic.
     *
     * <p>Folded left over the operands, two at a time, because the rule is binary and n-ary {@code
     * AND} is just that rule applied repeatedly.
     *
     * <p>For {@code AND}: the result is present when both operands are present, <em>or</em> when
     * either is present and false — because a false decides the answer whatever the other operand
     * turns out to be. The value is the ordinary conjunction of the two, with an absent operand
     * read as true so that it cannot turn a decided false into a true.
     *
     * <p>{@code OR} is the mirror image, with true as the deciding value.
     */
    private static Rendered renderLogical(boolean and, List<Rendered> operands) {
        Rendered result = operands.get(0);
        for (int i = 1; i < operands.size(); i++) {
            result = renderLogicalPair(and, result, operands.get(i));
        }
        return result;
    }

    private static Rendered renderLogicalPair(boolean and, Rendered left, Rendered right) {
        if (left.alwaysPresent() && right.alwaysPresent()) {
            // Nothing can be absent, so this is ordinary boolean logic and emits as such.
            return new Rendered(
                    "(" + left.value + (and ? " && " : " || ") + right.value + ")",
                    Rendered.ALWAYS);
        }
        // "Decided" means this operand alone settles the answer: false for AND, true for OR.
        String leftDecides = decides(and, left);
        String rightDecides = decides(and, right);
        String valid =
                "((("
                        + left.valid
                        + " & "
                        + right.valid
                        + ") != 0 || "
                        + leftDecides
                        + " || "
                        + rightDecides
                        + ") ? 1 : 0)";
        // An absent operand reads as the non-deciding value, so it cannot overturn a decided one.
        String leftValue = neutral(and, left);
        String rightValue = neutral(and, right);
        String value = "(" + leftValue + (and ? " && " : " || ") + rightValue + ")";
        return new Rendered(value, valid);
    }

    /** Whether this operand is present and holds the value that settles the answer on its own. */
    private static String decides(boolean and, Rendered operand) {
        String settling = and ? "!" : "";
        if (operand.alwaysPresent()) {
            return "(" + settling + operand.value + ")";
        }
        return "(" + operand.valid + " != 0 && " + settling + operand.value + ")";
    }

    /** This operand's value, with an absent one read as the value that decides nothing. */
    private static String neutral(boolean and, Rendered operand) {
        if (operand.alwaysPresent()) {
            return operand.value;
        }
        return "("
                + operand.valid
                + " == 0 ? "
                + (and ? "true" : "false")
                + " : "
                + operand.value
                + ")";
    }

    /** {@code a & b & ...}, or the constant 1 when nothing can be absent. */
    private static String andValidity(List<Rendered> operands) {
        List<String> terms = new ArrayList<>();
        for (Rendered r : operands) {
            if (!r.alwaysPresent()) {
                terms.add(r.valid);
            }
        }
        if (terms.isEmpty()) {
            return Rendered.ALWAYS;
        }
        return terms.size() == 1 ? terms.get(0) : "(" + String.join(" & ", terms) + ")";
    }

    /**
     * Whether this kernel needs to move validity at all.
     *
     * <p>False for an expression over {@code NOT NULL} columns, and then the emitted source is
     * character-for-character what it was before validity existed — no extra parameter, no extra
     * transfer, no extra instruction. That is deliberate: declaring {@code NOT NULL} stays the
     * cheaper path rather than merely the older one.
     */
    private static boolean carriesValidity(Set<Integer> nullableInputs) {
        return !nullableInputs.isEmpty();
    }

    /**
     * Staged columns that may be absent, collected while rendering.
     *
     * <p>Not derivable from the projections alone, which is what the first version tried: a
     * predicate over a nullable column with a projection over a {@code NOT NULL} one produces no
     * nullable <em>output</em>, and the kernel still has to read a validity bit to evaluate the
     * predicate. That generated {@code v0} against a signature with no {@code inNulls} in it, and
     * failed to compile -- loudly, which is the one merciful thing about it.
     */
    private static final ThreadLocal<Set<Integer>> NULLABLE_INPUTS =
            ThreadLocal.withInitial(LinkedHashSet::new);

    private static @Nullable Rendered render(
            AccelExpression node, Map<Integer, String> inputs, Map<Integer, GpuValueType> types) {
        if (node instanceof AccelInputRef) {
            AccelInputRef ref = (AccelInputRef) node;
            if (!isDoubleSafeInput(ref.outputType())) {
                return null;
            }
            types.putIfAbsent(ref.index(), valueTypeOf(ref.outputType()));
            String name = inputs.computeIfAbsent(ref.index(), index -> "c" + index);
            if (!ref.outputType().isNullable()) {
                // A column declared NOT NULL carries no validity and costs nothing to read.
                return new Rendered(name, Rendered.ALWAYS);
            }
            NULLABLE_INPUTS.get().add(ref.index());
            return new Rendered(name, "v" + ref.index());
        }
        if (node instanceof AccelLiteral) {
            Object value = ((AccelLiteral) node).value();
            if (value == null) {
                // A literal NULL is representable, but every use of one is a constant this kernel
                // would be computing for no reason; the planner folds them away long before here.
                return null;
            }
            if (value instanceof Boolean) {
                return new Rendered(value.toString(), Rendered.ALWAYS);
            }
            if (!(value instanceof Number)) {
                return null;
            }
            double d = ((Number) value).doubleValue();
            if (!Double.isFinite(d)) {
                return null;
            }
            // Double.toString round-trips exactly, so the constant the kernel sees is the constant
            // the planner folded.
            return new Rendered(Double.toString(d), Rendered.ALWAYS);
        }
        AccelCall call = (AccelCall) node;
        List<Rendered> operands = new ArrayList<>(call.operands().size());
        for (AccelExpression operand : call.operands()) {
            Rendered rendered = render(operand, inputs, types);
            if (rendered == null) {
                return null;
            }
            operands.add(rendered);
        }

        // AND and OR are the one place "absent in, absent out" is wrong, and getting it wrong
        // changes a row count rather than failing. SQL's three-valued truth tables:
        //
        //     FALSE AND UNKNOWN = FALSE        TRUE  OR UNKNOWN = TRUE
        //     TRUE  AND UNKNOWN = UNKNOWN      FALSE OR UNKNOWN = UNKNOWN
        //
        // So an absent operand does not make the result absent when the other operand already
        // decides it. Value and validity have to be computed together, which is why these two are
        // handled here rather than falling through to the strict rule below.
        if (call.function() == AccelFunction.AND || call.function() == AccelFunction.OR) {
            return named(renderLogical(call.function() == AccelFunction.AND, operands), "boolean");
        }

        // IS NULL and IS NOT NULL read a validity bit rather than propagating one, and their own
        // answer is never absent -- which is the whole of what makes them useful in a WHERE clause
        // over a nullable column.
        if (call.function() == AccelFunction.IS_NULL) {
            return named(
                    new Rendered("(" + operands.get(0).valid + " == 0)", Rendered.ALWAYS),
                    "boolean");
        }
        if (call.function() == AccelFunction.IS_NOT_NULL) {
            return named(
                    new Rendered("(" + operands.get(0).valid + " != 0)", Rendered.ALWAYS),
                    "boolean");
        }

        List<String> values = new ArrayList<>(operands.size());
        for (Rendered r : operands) {
            values.add(r.value);
        }
        String rendered = renderCall(call, values);
        if (rendered == null) {
            return null;
        }
        // Every other operator here is strict: absent in, absent out. AND and OR are not, and are
        // handled above, where three-valued logic computes value and validity together.
        return named(
                new Rendered(rendered, andValidity(operands)),
                isBooleanResult(call.outputType()) ? "boolean" : "double");
    }

    /**
     * Gives this subexpression a name, so a second occurrence of it emits the name rather than the
     * expression again. See {@link Cse}.
     *
     * <p>A rendering that is already a bare name -- a staged column, a folded constant, the
     * always-present validity -- is left alone: naming it would add a line and save nothing.
     */
    private static Rendered named(Rendered rendered, String javaType) {
        Cse cse = CSE.get();
        String value = isName(rendered.value) ? rendered.value : cse.name(rendered.value, javaType);
        String valid =
                rendered.alwaysPresent() || isName(rendered.valid)
                        ? rendered.valid
                        : cse.name(rendered.valid, "int");
        return new Rendered(value, valid);
    }

    /** Whether this rendering is already a single identifier or literal. */
    private static boolean isName(String rendered) {
        if (rendered.isEmpty()) {
            return true;
        }
        for (int i = 0; i < rendered.length(); i++) {
            char c = rendered.charAt(i);
            if (!Character.isLetterOrDigit(c) && c != '_' && c != '.') {
                return false;
            }
        }
        return true;
    }

    private static boolean isBooleanResult(org.apache.flink.table.types.logical.LogicalType type) {
        return type.getTypeRoot() == org.apache.flink.table.types.logical.LogicalTypeRoot.BOOLEAN;
    }

    private static @Nullable String renderCall(AccelCall call, List<String> operands) {
        switch (call.function()) {
            case PLUS:
                return infix(operands, "+");
            case MINUS:
                return infix(operands, "-");
            case TIMES:
                return infix(operands, "*");
            case DIVIDE:
                return infix(operands, "/");
            case INT_DIVIDE:
                return truncating(call, operands, false);
            case MOD:
                return truncating(call, operands, true);
            case NEGATE:
                return operands.size() == 1 ? "(-" + operands.get(0) + ")" : null;
            case GREATER_THAN:
                return infix(operands, ">");
            case GREATER_OR_EQUAL:
                return infix(operands, ">=");
            case LESS_THAN:
                return infix(operands, "<");
            case LESS_OR_EQUAL:
                return infix(operands, "<=");
            case EQUALS:
                return infix(operands, "==");
            case NOT_EQUALS:
                return infix(operands, "!=");
            case AND:
                return infix(operands, "&&");
            case OR:
                return infix(operands, "||");
            case NOT:
                return operands.size() == 1 ? "(!" + operands.get(0) + ")" : null;
            case CAST:
                // Every value here is already a double, so a widening numeric cast is a no-op.
                // Narrowing casts must raise on overflow and a kernel cannot, so the planner
                // refuses them before they arrive.
                return operands.size() == 1 ? operands.get(0) : null;
            case LEAST:
            case GREATEST:
                return foldOrdering(call, operands);
            default:
                return renderMath(call, operands);
        }
    }

    /**
     * LEAST and GREATEST, matching Flink's ordering rather than {@code TornadoMath.min}.
     *
     * <p>Flink folds these with {@code Double.compareTo}, whose total order places NaN above every
     * other value: NaN wins GREATEST and loses LEAST. {@code min} and {@code max} propagate it
     * instead, so a kernel using them returned a different answer from the CPU plan for any row
     * where an operand was NaN — which SQRT of a negative produces routinely.
     *
     * <p>One divergence remains and is deliberate: {@code compareTo} orders {@code -0.0} below
     * {@code 0.0} and {@code <} does not. Recovering the sign of a zero needs a reciprocal
     * comparison, which is a division per comparison in every kernel using LEAST.
     */
    private static @Nullable String foldOrdering(AccelCall call, List<String> operands) {
        if (operands.size() < 2 || !isDoubleResult(call.outputType())) {
            return null;
        }
        boolean least = call.function() == org.apache.flink.table.accelerator.AccelFunction.LEAST;
        String folded = operands.get(0);
        for (int i = 1; i < operands.size(); i++) {
            String next = operands.get(i);
            String step =
                    least
                            ? "((" + next + " != " + next + " || " + folded + " < " + next + ") ? "
                                    + folded + " : " + next + ")"
                            : "((" + folded + " != " + folded + " || " + folded + " > " + next
                                    + ") ? " + folded + " : " + next + ")";
            // Named at every step, and this is the step that matters. The accumulator appears
            // three times in the expression above (four for GREATEST), so folding it inline
            // triples the text on each operand: a twenty-argument LEAST reached 861 MB of source
            // before this, and 18.9 MB once the operands themselves were named but the fold was
            // not. Naming each step makes it one statement per operand. Measured 2026-09-17.
            folded = CSE.get().name(step, "double");
        }
        return folded;
    }

    /**
     * Integer division and remainder, truncated toward zero as SQL means them.
     *
     * <p>A kernel computing in double answers 3.5 where SQL answers 3, so the truncation is written
     * out rather than left to the result type. Exact for every operand an integral column can hold
     * below 2^53.
     *
     * <p><b>Only with a non-zero literal divisor.</b> SQL raises on division by zero and a kernel
     * cannot: {@code a / 0.0} is infinity, and {@code (long)} of that is a number. Refusing unless
     * the divisor is a literal that is plainly not zero keeps a wrong answer from being possible,
     * and covers what these are actually used for -- {@code MOD(id, 1000)} and the like. A computed
     * divisor falls back to the CPU, where SQL's error is raised properly.
     */
    private static @Nullable String truncating(
            AccelCall call, List<String> operands, boolean remainder) {
        if (operands.size() != 2 || !nonZeroLiteral(call.operands().get(1))) {
            return null;
        }
        String quotient = "((double) (long) (" + operands.get(0) + " / " + operands.get(1) + "))";
        return remainder
                ? "(" + operands.get(0) + " - (" + quotient + " * " + operands.get(1) + "))"
                : quotient;
    }

    /** Whether this operand is a literal number that is certainly not zero. */
    private static boolean nonZeroLiteral(AccelExpression expression) {
        if (!(expression instanceof AccelLiteral)) {
            return false;
        }
        Object value = ((AccelLiteral) expression).value();
        return value instanceof Number && ((Number) value).doubleValue() != 0.0;
    }

    /**
     * Maps the remaining functions onto {@code TornadoMath}, which is what compiles to a device.
     */
    private static @Nullable String renderMath(AccelCall call, List<String> operands) {
        String fn;
        int arity = 1;
        switch (call.function()) {
            case ABS:
                fn = "abs";
                break;
            case SQRT:
                fn = "sqrt";
                break;
            case EXP:
                fn = "exp";
                break;
            case LN:
                fn = "log";
                break;
            case LOG2:
                fn = "log2";
                break;
            case SIN:
                fn = "sin";
                break;
            case COS:
                fn = "cos";
                break;
            case TAN:
                fn = "tan";
                break;
            case ASIN:
                fn = "asin";
                break;
            case ACOS:
                fn = "acos";
                break;
            case ATAN:
                fn = "atan";
                break;
            case TANH:
                fn = "tanh";
                break;
            case FLOOR:
                fn = "floor";
                break;
            case CEIL:
                fn = "ceil";
                break;
            case SIGN:
                fn = "signum";
                break;
            case POWER:
                fn = "pow";
                arity = 2;
                break;
            case ATAN2:
                fn = "atan2";
                arity = 2;
                break;
            default:
                return null;
        }
        if (operands.size() != arity) {
            return null;
        }
        return "TornadoMath." + fn + "(" + String.join(", ", operands) + ")";
    }

    private static @Nullable String infix(List<String> operands, String op) {
        if (operands.size() < 2) {
            return null;
        }
        // Fully parenthesised: the tree already encodes precedence.
        return "(" + String.join(" " + op + " ", operands) + ")";
    }

    private static GpuValueType valueTypeOf(LogicalType type) {
        switch (type.getTypeRoot()) {
            case INTEGER:
                return GpuValueType.INT;
            case FLOAT:
                return GpuValueType.FLOAT;
            case DOUBLE:
                return GpuValueType.DOUBLE;
            default:
                throw new IllegalStateException("no staging width for " + type);
        }
    }

    private static boolean isDoubleSafeInput(LogicalType type) {
        LogicalTypeRoot root = type.getTypeRoot();
        return root == LogicalTypeRoot.DOUBLE
                || root == LogicalTypeRoot.FLOAT
                || root == LogicalTypeRoot.INTEGER;
    }

    /**
     * Whether this generator will produce a value of this type.
     *
     * <p>{@code DOUBLE} only, and {@code FLOAT} deliberately not. This generator evaluates every
     * expression in double and used to narrow a {@code FLOAT} result once at the end, which is not
     * what Flink does: Flink evaluates float arithmetic in float and rounds at every step. The two
     * agree for a single operation — double carries far more than the 2p+2 bits of mantissa that
     * makes one rounding exact — and diverge for a chain, which is what any expression worth
     * offloading is. Narrowing at the end was therefore a wrong answer that looked like a
     * conversion.
     *
     * <p>Refusing is the honest position until the generator can emit float arithmetic throughout.
     * Flink's estimator refuses a {@code FLOAT} result before it gets here (M2.3), so in practice
     * this is a second line rather than the first; it is here because a provider should not claim a
     * semantics it does not implement, whoever is asking.
     */
    /**
     * Result types a computed column may have.
     *
     * <p>{@code INTEGER} joins {@code DOUBLE} because a grouping key is usually one, and refusing
     * it refused the whole projection beside it -- a `GROUP BY` query offloaded nothing however
     * expressible its arithmetic was. The kernel still computes in double and narrows on the way
     * out, which is exact for every int: the arithmetic that produced it is integral, and integral
     * arithmetic below 2^53 is exact in a double.
     *
     * <p>{@code BIGINT} is deliberately absent. It is not exact above 2^53 and a kernel has no way
     * to say so.
     */
    private static boolean isDoubleResult(LogicalType type) {
        LogicalTypeRoot root = type.getTypeRoot();
        return root == LogicalTypeRoot.DOUBLE || root == LogicalTypeRoot.INTEGER;
    }
}
