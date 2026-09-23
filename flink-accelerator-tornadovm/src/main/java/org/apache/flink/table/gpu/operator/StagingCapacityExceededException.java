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

import org.apache.flink.util.FlinkRuntimeException;

/**
 * More rows arrived than the operator undertook at {@code open()} to be able to hold.
 *
 * <h2>Why this is fatal, when it used to be survivable</h2>
 *
 * <p>Two operators here must hold their whole input: a sort, because a sorted batch is only a
 * sorted run, and a join, because a build side has to be resident to probe against. Both size that
 * undertaking against the staging reservation Flink gave them, both check it at {@code open()}
 * against the planner's estimate, and an estimate is a statistic.
 *
 * <p>When the estimate was low, these used to finish on the host: the staged rows were materialised
 * into an {@code ArrayList} or a {@code HashMap} and the rest joined them there. That answer was
 * correct and the memory was not bounded by anything. The reservation stops applying the moment the
 * overflow starts, so the true bound became the task heap, and a cardinality miss — the one thing
 * this path exists to survive — turned into a TaskManager OOM. Flink's own sorter and hash join
 * spill; these do not, and pretending otherwise made an estimate error worse than it needed to be.
 *
 * <p>So the undertaking is now kept or the task fails. What makes that acceptable rather than
 * merely strict is what happens next: Flink does not offer a retried attempt to an accelerator, so
 * batch failover re-runs this task on the operator that knows how to spill. One attempt is lost and
 * the query finishes. The old behaviour finished too, until the day it did not.
 *
 * <p>Distinguished as its own type so that a reader of a failed job — and, later, anything counting
 * failure causes — can tell this apart from a device error. It is not a device error. The device
 * was working; the estimate was wrong.
 */
public class StagingCapacityExceededException extends FlinkRuntimeException {

    private static final long serialVersionUID = 1L;

    private final long estimatedRows;
    private final long capacity;

    StagingCapacityExceededException(String what, long estimatedRows, long capacity) {
        super(
                what
                        + ": the planner estimated "
                        + estimatedRows
                        + " rows and the staging reservation holds "
                        + capacity
                        + ". Failing rather than buffering the remainder on the ordinary heap, "
                        + "which is bounded by nothing. The retry of this task will not be offered "
                        + "to an accelerator and will run on the operator that spills.");
        this.estimatedRows = estimatedRows;
        this.capacity = capacity;
    }

    /** What the planner predicted, which is the number that was wrong. */
    public long estimatedRows() {
        return estimatedRows;
    }

    /** What the reservation actually held. */
    public long capacity() {
        return capacity;
    }
}
