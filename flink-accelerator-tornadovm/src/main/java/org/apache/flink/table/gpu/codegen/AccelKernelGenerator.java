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
        if (hasNullable(subtree)) {
            // Until validity travels with the values (M2.11), this generator cannot represent a
            // value that is not there and declines rather than computing on whatever occupies the
            // slot. Flink stopped refusing these in the planner at M2.9 precisely so the decision
            // could be made here; declining means the code-generated operator runs and the answer
            // is unchanged.
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

        final Map<Integer, String> inputs = new LinkedHashMap<>();
        final Map<Integer, GpuValueType> inputTypes = new LinkedHashMap<>();
        final List<String> computed = new ArrayList<>();
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
            String rendered = render(expression, inputs, inputTypes);
            if (rendered == null) {
                return Optional.empty();
            }
            computed.add(rendered);
            outputTypes.add(valueTypeOf(expression.outputType()));
        }
        if (computed.isEmpty()) {
            // Nothing to compute; the CPU path already handles pure projection.
            return Optional.empty();
        }

        String renderedCondition = null;
        if (condition != null) {
            renderedCondition = render(condition, inputs, inputTypes);
            if (renderedCondition == null) {
                return Optional.empty();
            }
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
        String source =
                renderClass(
                        className,
                        methodName,
                        inputs,
                        inputTypes,
                        computed,
                        outputTypes,
                        renderedCondition,
                        packedStride);
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
                        packedStride));
    }

    private static String renderClass(
            String className,
            String methodName,
            Map<Integer, String> inputs,
            Map<Integer, GpuValueType> inputTypes,
            List<String> computed,
            List<GpuValueType> outputTypes,
            @Nullable String condition,
            int packedStride) {

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
        // Last, so the engine can append it without knowing whether a mask is present.
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
        for (String var : inputs.values()) {
            sb.append(INDENT)
                    .append("    double ")
                    .append(var)
                    .append(" = ")
                    .append(var)
                    .append("_in.get(i);\n");
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
            boolean narrow = outputTypes.get(i) == GpuValueType.FLOAT;
            sb.append(INDENT)
                    .append("    out")
                    .append(i)
                    .append(".set(i, ")
                    .append(narrow ? "(float) (" : "")
                    .append(computed.get(i))
                    .append(narrow ? ")" : "")
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

    private static @Nullable String render(
            AccelExpression node, Map<Integer, String> inputs, Map<Integer, GpuValueType> types) {
        if (node instanceof AccelInputRef) {
            AccelInputRef ref = (AccelInputRef) node;
            if (!isDoubleSafeInput(ref.outputType())) {
                return null;
            }
            types.putIfAbsent(ref.index(), valueTypeOf(ref.outputType()));
            return inputs.computeIfAbsent(ref.index(), index -> "c" + index);
        }
        if (node instanceof AccelLiteral) {
            Object value = ((AccelLiteral) node).value();
            if (value instanceof Boolean) {
                return value.toString();
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
            return Double.toString(d);
        }
        AccelCall call = (AccelCall) node;
        List<String> operands = new ArrayList<>(call.operands().size());
        for (AccelExpression operand : call.operands()) {
            String rendered = render(operand, inputs, types);
            if (rendered == null) {
                return null;
            }
            operands.add(rendered);
        }
        return renderCall(call, operands);
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
            folded =
                    least
                            ? "((" + next + " != " + next + " || " + folded + " < " + next + ") ? "
                                    + folded + " : " + next + ")"
                            : "((" + folded + " != " + folded + " || " + folded + " > " + next
                                    + ") ? " + folded + " : " + next + ")";
        }
        return folded;
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
    private static boolean isDoubleResult(LogicalType type) {
        return type.getTypeRoot() == LogicalTypeRoot.DOUBLE;
    }
}
