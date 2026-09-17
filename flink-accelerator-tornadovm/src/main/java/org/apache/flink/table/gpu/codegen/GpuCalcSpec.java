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

import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;
import org.apache.flink.table.types.logical.RowType;

import javax.annotation.Nullable;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * Everything the runtime needs to execute one offloaded Calc, expressed without reference to
 * Calcite.
 *
 * <p>The kernel arrives as source rather than as a catalogue selection. A fixed catalogue could
 * only match expressions someone had anticipated, because a kernel had to exist as a compiled
 * method before the query did; generating it at run time removes that ceiling.
 *
 * <p>Serializable because it travels inside the operator into the JobGraph: the planner generates
 * on the client, the TaskManager compiles and runs.
 */
public final class GpuCalcSpec implements Serializable {

    private static final long serialVersionUID = 2L;

    /** Marks an output field produced by the kernel rather than copied from the input. */
    public static final int COMPUTED = -1;

    private final @Nullable GpuKernelSource kernel;
    private final int[] outputLayout;
    private final RowType outputType;
    private final int batchSize;

    public GpuCalcSpec(
            GpuKernelSource kernel, int[] outputLayout, RowType outputType, int batchSize) {
        this.kernel = kernel;
        this.outputLayout = outputLayout;
        this.outputType = outputType;
        this.batchSize = batchSize;
    }

    /**
     * The pre-generated TornadoVM kernel, or null.
     *
     * <p>Null is normal. It means only that the legacy generator could not express this Calc — a
     * pure pass-through with a filter, for instance, which it refuses because it has no computed
     * column. That is a limitation of one provider's code generator and says nothing about whether
     * the subtree can be accelerated: a provider reading {@link #subtree()} may well serve it.
     * Gating the whole accelerator path on this was hiding otherwise-offloadable queries.
     */
    public @Nullable GpuKernelSource kernel() {
        return kernel;
    }

    /** Whether the legacy pre-generated kernel is present. */
    public boolean hasKernel() {
        return kernel != null;
    }

    /**
     * For each output field, the input field it is copied from, or {@link #COMPUTED} for one the
     * kernel produces.
     *
     * <p>The kernel's outputs fill the {@code COMPUTED} slots in order, which is the order the
     * projections were generated in.
     *
     * <p>Pass-through columns never reach the device: they are staged host-side and emitted from
     * that buffer. That is both cheaper and the only way to carry a type the kernel cannot
     * represent — a {@code BIGINT} key, for instance, which would lose precision as a double.
     */
    public int[] outputLayout() {
        return outputLayout;
    }

    public RowType outputType() {
        return outputType;
    }

    /** Rows staged before each kernel launch, from {@code table.exec.accelerator.batch-size}. */
    public int batchSize() {
        return batchSize;
    }

    /**
     * Whether the output row can be staged: the kernel writes a {@code DOUBLE}, and every
     * pass-through column is held host-side in a primitive array of a type that buffer supports.
     *
     * <p>This lives on the spec rather than in a provider because it depends on nothing but the
     * spec. Both sides need the same answer: the planner asks so that a node it cannot serve is
     * reported as CPU in {@code EXPLAIN} rather than looking offloaded until the job runs, and the
     * provider asks so that a spec built by some other planner is refused rather than failing
     * inside the operator.
     */
    public boolean canStage() {
        List<LogicalType> fields = outputType.getChildren();
        if (fields.size() != outputLayout.length) {
            return false;
        }
        for (int i = 0; i < outputLayout.length; i++) {
            LogicalTypeRoot root = fields.get(i).getTypeRoot();
            if (outputLayout[i] == COMPUTED) {
                // FLOAT joins DOUBLE now that a computed column is written to a buffer of its own
                // declared width and narrowed inside the kernel. While every output was a
                // DoubleArray this had to refuse it, or the row would carry a Double in a field
                // the type says is a Float.
                if (root != LogicalTypeRoot.DOUBLE && root != LogicalTypeRoot.FLOAT) {
                    return false;
                }
            } else if (root != LogicalTypeRoot.BIGINT
                    && root != LogicalTypeRoot.INTEGER
                    && root != LogicalTypeRoot.FLOAT
                    && root != LogicalTypeRoot.DOUBLE) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        return "GpuCalcSpec["
                + kernel
                + ", layout="
                + Arrays.toString(outputLayout)
                + ", batch="
                + batchSize
                + ", batch="
                + batchSize
                + "]";
    }
}
