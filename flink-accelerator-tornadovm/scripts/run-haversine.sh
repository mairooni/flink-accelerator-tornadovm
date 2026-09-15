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
#
# End-to-end haversine benchmark on a standalone cluster, CPU against GPU.
#
# Measures what a user actually experiences: `flink run` against a real JobManager and
# TaskManager, not a MiniCluster. The input is written once to Parquet beforehand, so the
# source is not on the critical path -- with datagen it costs ~100 us/row and buries the
# operator no matter how fast the kernel is.
#
# Usage: run-haversine.sh <flink-dist-dir> [rows] [parallelism] [runs]
#   DEPOTS sets how many reference points the nearest-of query compares against (20 by default).
#   One depot is a plain distance, which is only about a fifth of the job's wall time and so
#   caps the whole query near 1.2x however fast the device is.
#   TORNADOVM_HOME must point at a built TornadoVM SDK; run gpu-cluster-setup.sh first.
#   FORMAT selects the input format (csv by default; parquet needs Hadoop on the cluster
#   classpath, which the distribution does not ship).

set -euo pipefail

FLINK_HOME="${1:-}"
# 2M rows takes about three minutes to generate and is enough to show the effect.
# Generation runs through datagen at roughly 100 us/row, so 50M would take over an hour.
ROWS="${2:-2000000}"
PARALLELISM="${3:-1}"
RUNS="${4:-10}"
DEPOTS="${DEPOTS:-20}"
FORMAT="${FORMAT:-csv}"
DATA="${DATA:-/tmp/flink-gpu-points-${ROWS}-${FORMAT}}"

if [[ -z "${FLINK_HOME}" || ! -x "${FLINK_HOME}/bin/flink" ]]; then
    echo "usage: $0 <flink-dist-dir> [rows] [parallelism] [runs]" >&2
    exit 1
fi

JAR=$(ls "${FLINK_HOME}"/examples/table/HaversineBenchmark.jar \
         "${FLINK_HOME}"/examples/table/*HaversineBenchmark*.jar 2>/dev/null | head -1 || true)
if [[ -z "${JAR}" ]]; then
    echo "HaversineBenchmark jar not found under ${FLINK_HOME}/examples/table" >&2
    exit 1
fi

"${FLINK_HOME}/bin/start-cluster.sh"
# stop the cluster however this script exits, so a failed run does not leave one behind
trap '"${FLINK_HOME}/bin/stop-cluster.sh" >/dev/null 2>&1 || true' EXIT

if [[ ! -d "${DATA}" ]]; then
    echo "### generating ${ROWS} rows into ${DATA}"
    "${FLINK_HOME}/bin/flink" run "${JAR}" --generate --rows "${ROWS}" --data "${DATA}" \
        --format "${FORMAT}"
else
    echo "### reusing ${DATA}"
fi

for gpu in false true; do
    echo
    echo "############ gpu=${gpu}  rows=${ROWS}  parallelism=${PARALLELISM}  format=${FORMAT}  depots=${DEPOTS} ############"
    "${FLINK_HOME}/bin/flink" run "${JAR}" \
        --data "${DATA}" \
        --parallelism "${PARALLELISM}" \
        --runs "${RUNS}" \
        --format "${FORMAT}" \
        --depots "${DEPOTS}" \
        --gpu "${gpu}"
done
