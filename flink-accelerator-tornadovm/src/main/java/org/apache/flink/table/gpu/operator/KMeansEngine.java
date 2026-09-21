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
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cudf.Cudf;
import uk.ac.manchester.tornado.cudf.enums.CudfAggregation;

/**
 * K-means by Lloyd's algorithm, iterating entirely on the device.
 *
 * <p>§T17 measured k-means as batch SQL and got 0.94x to 1.09x. That measured one SQL query per
 * iteration, which is the shape §T16's host arm uses to lose by 75x -- the points are re-read from
 * the source on every pass, because a Flink SQL plan is fixed and new centroids mean a new query.
 * This engine is the other architecture: the points are uploaded once and every pass after the
 * first is arithmetic with no I/O.
 *
 * <h2>One pass, three steps, two tools</h2>
 *
 * <pre>{@code
 * assignment = argmin_j |x_i - c_j|^2   generated   n*k*d, per row
 * sums, counts = group by assignment    cuDF        cross-row, no @Parallel loop states it
 * c_j = sums_j / count_j                generated   k*d, per centroid
 * }</pre>
 *
 * <p>The middle step is why this is a hybrid rather than two pipelines: it reads the buffer the
 * assignment kernel wrote and writes the buffers the update kernel reads, on the device, without
 * the assignment ever returning to the host. On a host implementation that assignment is an array
 * of {@code rows} integers materialised and re-read once per pass.
 *
 * <h2>Why a column of ones</h2>
 *
 * <p>An iteration needs {@code d} coordinate sums <em>and</em> a member count per cluster, and cuDF
 * returns groups in whatever order its hash produced. Two calls would give two key orderings to
 * reconcile. Instead the staged buffer carries {@code d + 1} columns, the last of them all ones, so
 * a single {@code groupAggregate} with {@code SUM} returns the sums and the count together against
 * one set of keys. That is what the multi-column form of the binding exists for.
 *
 * <h2>Residency</h2>
 *
 * <p>The points transfer under {@link DataTransferMode#FIRST_EXECUTION} and the centroids stay on
 * the device across every pass, so executing the plan {@code k} times moves no data after the
 * first. The centroids come back once, at the end, under {@link DataTransferMode#UNDER_DEMAND}.
 *
 * <h2>Empty clusters</h2>
 *
 * <p>A cluster nobody is nearest to produces no group, so its centroid keeps its previous value
 * rather than becoming a division by zero. That is what Lloyd's algorithm does on a host too.
 */
public final class KMeansEngine implements AutoCloseable {

    private final int rows;
    private final int dims;
    private final int clusters;

    /**
     * Column-major {@code rows x (dims + 1)}: a coordinate's values are contiguous, and the last
     * column is all ones so that one SUM yields the member counts as well.
     */
    private final DoubleArray values;

    /**
     * {@code clusters x dims}, a centroid's coordinates contiguous. Read and written every pass.
     */
    private final DoubleArray centroids;

    private final IntArray assignment;
    private final IntArray outKeys;
    private final DoubleArray outSums;
    private final IntArray outGroups;

    private TornadoExecutionPlan plan;
    private double trainMillis;

    public KMeansEngine(int rows, int dims, int clusters) {
        this.rows = rows;
        this.dims = dims;
        this.clusters = clusters;
        this.values = new DoubleArray(rows * (dims + 1));
        this.centroids = new DoubleArray(clusters * dims);
        this.assignment = new IntArray(rows);
        // cuDF writes one row per group and cannot know the count in advance, so both output
        // buffers are sized for the worst case -- every row its own group -- as the binding asks.
        this.outKeys = new IntArray(rows);
        this.outSums = new DoubleArray((dims + 1) * rows);
        this.outGroups = new IntArray(1);
        for (int i = 0; i < rows; i++) {
            values.set(dims * rows + i, 1.0);
        }
    }

    /**
     * Stages one point. Column-major, so coordinate {@code c} of row {@code i} is at {@code
     * c*rows+i}.
     */
    public void setPoint(int row, int coordinate, double value) {
        values.set(coordinate * rows + row, value);
    }

    /** The initial centroids, which the device then owns until {@link #train} returns. */
    public void setCentroid(int cluster, int coordinate, double value) {
        centroids.set(cluster * dims + coordinate, value);
    }

    /**
     * Builds the plan. Call once, after every point is staged.
     *
     * <p>The graph is one iteration; running {@code k} of them is {@code k} executions of the same
     * plan, which is what keeps the points resident.
     */
    public void open() {
        TaskGraph graph =
                new TaskGraph("kmeans")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, values, centroids)
                        .task(
                                "assign",
                                KMeansEngine::assign,
                                values,
                                centroids,
                                assignment,
                                rows,
                                dims,
                                clusters)
                        .libraryTask(
                                "accumulate",
                                Cudf::groupAggregate,
                                rows,
                                dims + 1,
                                CudfAggregation.SUM.code(),
                                assignment,
                                values,
                                outKeys,
                                outSums,
                                outGroups)
                        .task(
                                "update",
                                KMeansEngine::update,
                                outKeys,
                                outSums,
                                outGroups,
                                centroids,
                                rows,
                                dims,
                                clusters)
                        .transferToHost(DataTransferMode.UNDER_DEMAND, centroids);
        plan = new TornadoExecutionPlan(graph.snapshot());
    }

    /** Runs {@code iterations} passes and brings the centroids back once. */
    public void train(int iterations) {
        long start = System.nanoTime();
        TornadoExecutionResult result = null;
        for (int i = 0; i < iterations; i++) {
            result = plan.execute();
        }
        if (result != null) {
            result.transferToHost(centroids);
        }
        trainMillis = (System.nanoTime() - start) / 1e6;
    }

    public double centroid(int cluster, int coordinate) {
        return centroids.get(cluster * dims + coordinate);
    }

    public double trainMillis() {
        return trainMillis;
    }

    /**
     * The nearest centroid to each point, as a cluster index.
     *
     * <p>Squared distance, because the nearest centroid is the same either way and a square root
     * per centroid per row is arithmetic Lloyd's algorithm does not need.
     *
     * <p>The conditional here is ordinary Java that TornadoVM compiles. The accelerator IR's
     * missing conditional, which §T17 hit, is a limit on what a SQL <em>expression</em> can carry
     * -- a hand-written kernel has no such limit, which is part of why this architecture reaches
     * shapes the SQL one cannot.
     */
    public static void assign(
            DoubleArray values,
            DoubleArray centroids,
            IntArray assignment,
            int rows,
            int dims,
            int clusters) {
        for (@Parallel int i = 0; i < rows; i++) {
            double best = Double.MAX_VALUE;
            int nearest = 0;
            for (int j = 0; j < clusters; j++) {
                double distance = 0.0;
                for (int c = 0; c < dims; c++) {
                    double difference = values.get(c * rows + i) - centroids.get(j * dims + c);
                    distance += difference * difference;
                }
                if (distance < best) {
                    best = distance;
                    nearest = j;
                }
            }
            assignment.set(i, nearest);
        }
    }

    /**
     * New centroids from the grouped sums.
     *
     * <p>The loop runs to {@code clusters} rather than to the group count so that the iteration
     * space is a parameter: a bound read out of a device buffer defeats TornadoVM's inference and
     * the graph deoptimises. The group count is read inside instead, as a guard.
     */
    public static void update(
            IntArray outKeys,
            DoubleArray outSums,
            IntArray outGroups,
            DoubleArray centroids,
            int rows,
            int dims,
            int clusters) {
        for (@Parallel int g = 0; g < clusters; g++) {
            if (g < outGroups.get(0)) {
                int key = outKeys.get(g);
                // The ones column, so the member count for this group.
                double members = outSums.get(dims * rows + g);
                if (members > 0.0) {
                    for (int c = 0; c < dims; c++) {
                        centroids.set(key * dims + c, outSums.get(c * rows + g) / members);
                    }
                }
            }
        }
    }

    @Override
    public void close() throws TornadoExecutionPlanException {
        if (plan != null) {
            plan.close();
        }
    }
}
