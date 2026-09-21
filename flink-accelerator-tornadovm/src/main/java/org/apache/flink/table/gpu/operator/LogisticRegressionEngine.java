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

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.annotations.Reduce;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;

/**
 * Logistic regression by gradient descent, iterating entirely on the device.
 *
 * <p>Every benchmark before this one measured a single pass over the data, and every one of them
 * ended up bound by reading its input rather than by arithmetic. That is not a fact about devices,
 * it is a fact about single-pass queries: the work is linear in the bytes, so the ratio between a
 * device and a host is capped by how much of the job is arithmetic in the first place.
 *
 * <p>Iteration breaks that. The data is read once and stays resident; each pass adds arithmetic and
 * no I/O. The iteration count is therefore the knob the feature-map benchmark lacked, and it plays
 * the part {@code --depots} plays for the haversine query.
 *
 * <h2>Why it needs both a kernel and a library</h2>
 *
 * <p>One pass is four steps, and no single tool does all four:
 *
 * <pre>{@code
 * z = X w           cublasSgemv    contraction over features
 * r = sigmoid(z)-y  generated      elementwise transcendental
 * g = X' r          cublasSgemv    contraction over rows
 * w = w - lr*g      generated      elementwise, on features
 * }</pre>
 *
 * <p>cuBLAS has no sigmoid, and a generated kernel has no tiled GEMV. What makes it a hybrid rather
 * than two pipelines is the middle step: it reads the buffer the first library call wrote and
 * writes the buffer the second one reads, on the device, without the intermediate ever returning to
 * the host. On a host implementation that intermediate is an array of {@code rows} floats
 * materialised and re-read twice per iteration.
 *
 * <h2>Residency</h2>
 *
 * <p>{@code X} and {@code y} transfer under {@link DataTransferMode#FIRST_EXECUTION} and the
 * weights stay on the device across every iteration, so executing the plan {@code k} times moves no
 * data at all after the first. The weights come back once, at the end, under {@link
 * DataTransferMode#UNDER_DEMAND}. Nothing else in this project has exercised residency: the Calc
 * operator re-uploads every batch because every batch is different data.
 *
 * <h2>Float, and what it costs</h2>
 *
 * <p>FP32 throughout, because the binding has no FP64 GEMM. Gradient descent is more forgiving of
 * that than a covariance matrix is -- it is a fixed point iteration, so a small error per step is
 * corrected by the next one rather than accumulated -- but the caller should still compare the
 * weights against a double-precision reference rather than assume.
 */
public final class LogisticRegressionEngine implements AutoCloseable {

    private final int rows;
    private final int features;
    private final float learningRate;

    /** Which contraction to use: "library", "generated" or "reduce". */
    private final String contraction;

    /**
     * One single-element array per gradient component, used only by the "reduce" arm.
     *
     * <p>There is one per feature because {@code @Reduce} cannot fuse: {@code ReduceTaskGraph}
     * reads its result from index 0 of the annotated array, so an <em>n</em>-output reduction is
     * <em>n</em> separate kernels over <em>n</em> separate arrays, each re-reading the residual.
     */
    private final FloatArray[] partials;

    /** Column-major {@code rows x features}, so a feature's values are contiguous for cuBLAS. */
    private final FloatArray x;

    private final FloatArray y;
    private final FloatArray weights;

    /** {@code X w}, then overwritten with the residual. Never leaves the device. */
    private final FloatArray z;

    private final FloatArray gradient;

    /**
     * Whether to capture the iteration into a CUDA graph and replay it.
     *
     * <p>Off by default and offered only to be measured. A pass here is four launches against
     * milliseconds of kernel, so there is little launch overhead to remove; the expectation is that
     * it changes nothing, and the point of the flag is to record that rather than assert it. A
     * graph containing a cuDF task could not be captured at all -- the shim ends every entry point
     * with a stream synchronise -- but this engine has only cuBLAS and generated kernels.
     */
    private boolean cudaGraph;

    private TornadoExecutionPlan plan;
    private int staged;
    private double trainMillis;

    public LogisticRegressionEngine(
            int rows, int features, float learningRate, String contraction) {
        this.rows = rows;
        this.features = features;
        this.learningRate = learningRate;
        this.contraction = contraction;
        this.partials = new FloatArray[features];
        if ("reduce".equals(contraction)) {
            for (int f = 0; f < features; f++) {
                partials[f] = new FloatArray(1);
            }
        }
        this.x = new FloatArray(rows * features);
        this.y = new FloatArray(rows);
        this.weights = new FloatArray(features);
        this.z = new FloatArray(rows);
        this.gradient = new FloatArray(features);
    }

    /** Stages one value of the row being built. */
    public void set(int feature, double value) {
        x.set(feature * rows + staged, (float) value);
    }

    /** Ends a row, with its label. */
    public void rowComplete(double label) {
        y.set(staged, (float) label);
        staged++;
    }

    public int staged() {
        return staged;
    }

    /**
     * Builds the plan. Call once, after every row is staged.
     *
     * <p>The graph is one iteration. Running {@code k} of them is {@code k} executions of the same
     * plan, which is what keeps the data resident: only the first moves {@code X} and {@code y}.
     */
    public void open() {
        if ("reduce".equals(contraction)) {
            plan = new TornadoExecutionPlan(reduceGraph().snapshot());
            return;
        }

        TaskGraph graph =
                new TaskGraph("logistic")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, y, weights);

        // z = X w
        if ("library".equals(contraction)) {
            graph =
                    graph.libraryTask(
                            "forward",
                            CuBlas::cublasSgemv,
                            CuBlasOperation.CUBLAS_OP_N.operation(),
                            rows,
                            features,
                            1.0f,
                            x,
                            rows,
                            weights,
                            1,
                            0.0f,
                            z,
                            1);
        } else {
            graph =
                    graph.task(
                            "forward",
                            LogisticRegressionEngine::forward,
                            x,
                            weights,
                            z,
                            rows,
                            features);
        }

        graph = graph.task("residual", LogisticRegressionEngine::residual, z, y, rows);

        // g = X' r
        if ("library".equals(contraction)) {
            graph =
                    graph.libraryTask(
                            "backward",
                            CuBlas::cublasSgemv,
                            CuBlasOperation.CUBLAS_OP_T.operation(),
                            rows,
                            features,
                            1.0f,
                            x,
                            rows,
                            z,
                            1,
                            0.0f,
                            gradient,
                            1);
        } else {
            graph =
                    graph.task(
                            "backward",
                            LogisticRegressionEngine::backward,
                            x,
                            z,
                            gradient,
                            rows,
                            features);
        }

        graph =
                graph.task(
                                "update",
                                LogisticRegressionEngine::update,
                                weights,
                                gradient,
                                learningRate / rows,
                                features)
                        .transferToHost(DataTransferMode.UNDER_DEMAND, weights);
        plan = new TornadoExecutionPlan(graph.snapshot());
        if (cudaGraph) {
            // Returns a wrapper rather than mutating, so the result has to be kept.
            plan = plan.withCUDAGraph();
        }
    }

    /** Selects the CUDA-graph arm. Call before {@link #open()}. */
    public void withCudaGraph(boolean enabled) {
        this.cudaGraph = enabled;
    }

    /**
     * The pure-TornadoVM arm: a generated forward pass and {@code @Reduce} for the contraction.
     *
     * <p><b>This does not run on TornadoVM 6.0.1.</b> It is kept because the way it fails is the
     * answer to whether {@code @Reduce} could serve a SQL aggregate, and because anyone
     * re-attempting it should not have to rediscover the sequence. Four obstacles, in the order
     * they appear:
     *
     * <ol>
     *   <li><b>No fusion.</b> {@code MultipleReductions} says so, and {@code ReduceTaskGraph} reads
     *       its result from index 0 of the annotated array, so an <em>n</em>-output contraction is
     *       <em>n</em> kernels over <em>n</em> single-element arrays, each re-reading the input.
     *   <li><b>The loop bound must come from the array.</b> Bounding by an {@code int} parameter
     *       makes {@code ReduceCodeAnalysis.obtainLoopBoundForPanamaRegions} fail a {@code
     *       requireNonNull}, and the execution dies with a bare {@code NullPointerException} that
     *       names nothing. Use {@code array.getSize()} or a constant.
     *   <li><b>Every task parameter must be declared</b> in a transfer list, including pure
     *       intermediates that the plain graph is happy to leave undeclared.
     *   <li><b>And declaring them is not enough.</b> With all four buffers in one {@code
     *       EVERY_EXECUTION} call it still reports the first parameter of the <em>non-reduce</em>
     *       task as undeclared. Every reduction test TornadoVM ships builds a graph containing
     *       exactly one task, which is the reduction; there is no example of a {@code @Reduce} task
     *       beside an ordinary one, and this graph has thirty-four.
     * </ol>
     *
     * <p>Splitting into two plans would sidestep the last point, but the intermediate would have to
     * round-trip through the host and {@code X} would be re-uploaded for the reduction plan --
     * 66&nbsp;MB a pass, which measures the workaround rather than the tiling.
     *
     * <p>So the conclusion is not that {@code @Reduce} is slow. It is that it cannot express a
     * multi-output contraction inside a larger pipeline, which is the shape every interesting SQL
     * aggregate has. Whether its tiling is fast for a single reduction in a graph of its own is
     * still unmeasured, and a microbenchmark could settle it -- but it would not change this.
     *
     * <p>It differs from the "generated" arm in the backward pass alone, so the gap between those
     * two is attributable to the reduction and nothing else.
     *
     * <p>Three costs come from the same limitation. {@code @Reduce} cannot fuse, so there is one
     * kernel and one single-element array per feature, each re-reading the residual: the scan is
     * paid <em>features</em> times rather than once. There is no way to combine those outputs on
     * the device either, since a task cannot take 32 arrays, so the weights are updated on the host
     * and re-uploaded every pass. And the weights therefore cannot stay resident the way they do in
     * the other two arms. All three are consequences of no fusion rather than free choices.
     */
    private TaskGraph reduceGraph() {
        TaskGraph graph =
                new TaskGraph("logistic")
                        // One transfer call, one mode, everything in it. The reduce rewrite
                        // validates every task parameter and does not appear to carry declarations
                        // across two transferToDevice calls with different modes: with x, y and z
                        // under FIRST_EXECUTION and weights under EVERY_EXECUTION it still reports
                        // weights as undeclared. So the whole set moves every pass, which
                        // re-uploads
                        // 64 MB of X that has not changed -- a cost of the workaround, not of the
                        // algorithm, and one the other two arms do not pay.
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, y, z, weights)
                        .task(
                                "forward",
                                LogisticRegressionEngine::forward,
                                x,
                                weights,
                                z,
                                rows,
                                features)
                        .task("residual", LogisticRegressionEngine::residual, z, y, rows);
        for (int f = 0; f < features; f++) {
            graph =
                    graph.task(
                            "g" + f,
                            LogisticRegressionEngine::reduceColumn,
                            x,
                            z,
                            partials[f],
                            rows,
                            f);
        }
        return graph.transferToHost(DataTransferMode.EVERY_EXECUTION, (Object[]) partials);
    }

    /**
     * One gradient component, as a TornadoVM reduction.
     *
     * <p>The naive loop is what the programmer writes; {@code ReduceTaskGraph} rewrites it into a
     * tree reduction with device-sized partials. That is the tiling a generator cannot express by
     * itself, which is the point of measuring it.
     */
    public static void reduceColumn(
            FloatArray x, FloatArray z, @Reduce FloatArray out, int rows, int f) {
        out.set(0, 0.0f);
        // The bound has to be z.getSize() rather than the rows parameter. ReduceCodeAnalysis
        // derives the reduction's input range from the loop bound to size its partials array, and
        // with a plain int parameter it cannot: obtainLoopBoundForPanamaRegions fails a
        // requireNonNull and the whole execution dies with a bare NullPointerException. Every
        // reduction in TornadoVM's own tests uses either array.getSize() or a constant.
        for (@Parallel int r = 0; r < z.getSize(); r++) {
            out.set(0, out.get(0) + x.get(f * rows + r) * z.get(r));
        }
    }

    /**
     * {@code z = X w}, generated.
     *
     * <p>One thread per row, so the parallelism is the row count -- hundreds of thousands of it.
     * This is the contraction a generated kernel is well suited to.
     */
    public static void forward(FloatArray x, FloatArray w, FloatArray z, int rows, int features) {
        for (@Parallel int r = 0; r < rows; r++) {
            float sum = 0.0f;
            for (int f = 0; f < features; f++) {
                sum += x.get(f * rows + r) * w.get(f);
            }
            z.set(r, sum);
        }
    }

    /**
     * {@code g = X' r}, generated.
     *
     * <p>One thread per <em>feature</em>, each walking every row. That is tens of threads on a
     * device with thousands of cores, and it is the honest comparison rather than a handicap: it is
     * what this project's generator emits from a {@code RexNode} tree, which is a {@code @Parallel}
     * loop over the output and a sequential loop inside it. A tiled parallel reduction is what
     * cuBLAS brings and what the generator has no way to express.
     */
    public static void backward(FloatArray x, FloatArray z, FloatArray g, int rows, int features) {
        for (@Parallel int f = 0; f < features; f++) {
            float sum = 0.0f;
            for (int r = 0; r < rows; r++) {
                sum += x.get(f * rows + r) * z.get(r);
            }
            g.set(f, sum);
        }
    }

    /** Runs {@code iterations} passes. No data crosses the bus after the first. */
    public void train(int iterations) {
        long start = System.nanoTime();
        try {
            if ("reduce".equals(contraction)) {
                float step = learningRate / rows;
                for (int i = 0; i < iterations; i++) {
                    plan.execute();
                    // No device-side combine is possible, so the step happens here.
                    for (int f = 0; f < features; f++) {
                        weights.set(f, weights.get(f) - step * partials[f].get(0));
                    }
                }
                trainMillis = (System.nanoTime() - start) / 1e6;
                return;
            }
            TornadoExecutionResult result = null;
            for (int i = 0; i < iterations; i++) {
                result = plan.execute();
            }
            // UNDER_DEMAND, so the weights come back once here rather than after every pass.
            if (result != null) {
                result.transferToHost(weights);
            }
        } catch (Exception e) {
            throw new IllegalStateException("logistic regression execution failed", e);
        }
        trainMillis = (System.nanoTime() - start) / 1e6;
    }

    /**
     * How long the iterations took, excluding staging.
     *
     * <p>A diagnostic, not the result. End to end is what a user experiences and is what the
     * benchmark reports; this exists only to explain an end-to-end number after the fact -- in
     * particular, to say whether two contraction choices tie because they are equally good or
     * because both are lost in the cost of reading the input.
     */
    public double trainMillis() {
        return trainMillis;
    }

    /**
     * The elementwise step, in place: {@code z := sigmoid(z) - y}.
     *
     * <p>In place because the value is consumed by the next library call from the same buffer.
     * Writing to a second array would cost another {@code rows}-sized allocation and another pass,
     * and buy nothing -- the previous GEMV has finished before this task starts.
     */
    public static void residual(FloatArray z, FloatArray y, int rows) {
        for (@Parallel int r = 0; r < rows; r++) {
            float sigmoid = 1.0f / (1.0f + TornadoMath.exp(-z.get(r)));
            z.set(r, sigmoid - y.get(r));
        }
    }

    /** The weight update, elementwise over features. Tiny, but it keeps the loop on the device. */
    public static void update(FloatArray weights, FloatArray gradient, float step, int features) {
        for (@Parallel int f = 0; f < features; f++) {
            weights.set(f, weights.get(f) - step * gradient.get(f));
        }
    }

    public double[] weights() {
        double[] out = new double[features];
        for (int f = 0; f < features; f++) {
            out[f] = weights.get(f);
        }
        return out;
    }

    @Override
    public void close() throws Exception {
        if (plan != null) {
            plan.close();
            plan = null;
        }
    }
}
