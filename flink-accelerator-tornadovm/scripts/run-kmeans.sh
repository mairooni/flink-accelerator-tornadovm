#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# One k-means iteration on a standalone cluster, CPU against GPU.
#
# Measures what a user experiences: `flink run` against a real JobManager and TaskManager,
# not a MiniCluster. The input is written once beforehand so the source is not on the
# critical path.
#
# The query has the shape that has paid before -- a lot of arithmetic per row, an output far
# smaller than the input -- and it splits along the seam this integration exists for: the
# assignment is per-row work a generated kernel does, and recomputing the centroids is a
# grouped aggregation that needs cross-row cooperation.
#
# Usage: run-kmeans.sh <flink-dist-dir> [rows] [parallelism] [runs]
#   DIMS and CLUSTERS set the arithmetic: the projection holds about 3*DIMS*CLUSTERS
#   expressions, and the two shapes that paid before sat between 500 and 2000 of them.
#   STAGE is `assign` (the arithmetic alone, one output row) or `iterate` (a whole
#   iteration, CLUSTERS output rows).
#   ARGMIN is `sign` or `case`. Only `iterate` uses it, and the difference between the two
#   is what a missing conditional in the IR costs: `case` is the natural SQL and cannot be
#   offloaded, `sign` says the same thing with arithmetic that can.
#   TORNADOVM_HOME must point at a built TornadoVM SDK; run gpu-cluster-setup.sh first.

set -euo pipefail

FLINK_HOME="${1:-}"
ROWS="${2:-2000000}"
PARALLELISM="${3:-1}"
RUNS="${4:-6}"
DIMS="${DIMS:-16}"
CLUSTERS="${CLUSTERS:-32}"
STAGE="${STAGE:-assign}"
ARGMIN="${ARGMIN:-sign}"
FORMAT="${FORMAT:-csv}"
DATA="${DATA:-/tmp/flink-gpu-kmeans-${ROWS}-${DIMS}d-${FORMAT}}"

if [[ -z "${FLINK_HOME}" || ! -x "${FLINK_HOME}/bin/flink" ]]; then
    echo "usage: $0 <flink-dist-dir> [rows] [parallelism] [runs]" >&2
    exit 1
fi

JAR=$(ls "${FLINK_HOME}"/examples/table/KMeansBenchmark.jar \
         "${FLINK_HOME}"/examples/table/*KMeansBenchmark*.jar 2>/dev/null | head -1 || true)
if [[ -z "${JAR}" ]]; then
    echo "KMeansBenchmark jar not found under ${FLINK_HOME}/examples/table" >&2
    exit 1
fi

"${FLINK_HOME}/bin/start-cluster.sh"
trap '"${FLINK_HOME}/bin/stop-cluster.sh" >/dev/null 2>&1 || true' EXIT

if [[ ! -d "${DATA}" ]]; then
    echo "### generating ${ROWS} rows x ${DIMS} dims into ${DATA}"
    "${FLINK_HOME}/bin/flink" run "${JAR}" --generate --rows "${ROWS}" --dims "${DIMS}" \
        --data "${DATA}" --format "${FORMAT}"
else
    echo "### reusing ${DATA}"
fi

for offload in false true; do
    echo
    echo "############ offload=${offload}  rows=${ROWS}  dims=${DIMS}  clusters=${CLUSTERS}" \
         " stage=${STAGE}  argmin=${ARGMIN}  parallelism=${PARALLELISM} ############"
    "${FLINK_HOME}/bin/flink" run "${JAR}" \
        --data "${DATA}" \
        --rows "${ROWS}" \
        --dims "${DIMS}" \
        --clusters "${CLUSTERS}" \
        --stage "${STAGE}" \
        --argmin "${ARGMIN}" \
        --parallelism "${PARALLELISM}" \
        --runs "${RUNS}" \
        --format "${FORMAT}" \
        --offload "${offload}"
done
