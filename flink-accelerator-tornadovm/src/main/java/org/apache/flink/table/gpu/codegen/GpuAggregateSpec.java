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

import org.apache.flink.annotation.Internal;
import org.apache.flink.table.types.logical.RowType;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Everything needed to run a {@code Calc} and the ungrouped aggregate above it as one device pass.
 *
 * <p>Travels in the JobGraph and names no TornadoVM type, for the same reason {@link GpuCalcSpec}
 * does: a TaskManager without the GPU module has to be able to deserialize it in order to decide
 * that it cannot serve it.
 *
 * <h2>Why the pair and not the aggregate alone</h2>
 *
 * <p>Offloading the aggregate by itself would be a loss. The contraction is two weighted operations
 * a row against the Calc's hundred-odd, so it earns nothing on its own, and reaching it through a
 * separate operator would mean the Calc writing rows to the host and the aggregate uploading them
 * again. The value is entirely in the intermediate never landing: the kernel writes the projected
 * columns to a device buffer and the contraction reads that buffer in place.
 *
 * <h2>What the contraction is</h2>
 *
 * <p>{@code SUM(e1), ..., SUM(ek)} over the {@code rows x k} matrix the kernel just wrote is {@code
 * ones' * M} -- one GEMV, not <em>k</em> reductions. That is the shape a library serves well and
 * the one TornadoVM's {@code @Reduce} cannot express at all, since it fuses no multiple reductions.
 */
@Internal
public final class GpuAggregateSpec implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Marks an output field that is {@code COUNT(*)} rather than a sum of a staged column. */
    public static final int COUNT_STAR = -1;

    private final GpuKernelSource kernel;
    private final int[] sumSources;
    private final RowType calcOutputType;
    private final RowType outputType;
    private final int batchSize;

    public GpuAggregateSpec(
            GpuKernelSource kernel,
            int[] sumSources,
            RowType calcOutputType,
            RowType outputType,
            int batchSize) {
        if (sumSources.length != outputType.getFieldCount()) {
            throw new IllegalArgumentException(
                    "every aggregate output needs a source: "
                            + outputType.getFieldCount()
                            + " fields against "
                            + sumSources.length
                            + " sources");
        }
        this.kernel = kernel;
        this.sumSources = sumSources;
        this.calcOutputType = calcOutputType;
        this.outputType = outputType;
        this.batchSize = batchSize;
    }

    /** The Calc's kernel, whose outputs the contraction reads. */
    public GpuKernelSource kernel() {
        return kernel;
    }

    /**
     * For each aggregate output field, the Calc output column summed into it, or {@link
     * #COUNT_STAR}.
     */
    public int[] sumSources() {
        return sumSources;
    }

    /** Row type the Calc produces, which is the contraction's input. */
    public RowType calcOutputType() {
        return calcOutputType;
    }

    /** Row type of the partial aggregate this emits: one row, at end of input. */
    public RowType outputType() {
        return outputType;
    }

    public int batchSize() {
        return batchSize;
    }

    @Override
    public String toString() {
        return "GpuAggregateSpec["
                + kernel.className()
                + ", sums="
                + Arrays.toString(sumSources)
                + ", batch="
                + batchSize
                + "]";
    }
}
