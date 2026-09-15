#!/usr/bin/env bash
################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
################################################################################
#
# Does a job planned for the GPU survive a TaskManager that has no GPU module?
#
# The offload decision is made in the client, which cannot know where the task will run. This
# reproduces that mismatch on one machine: the client and JobManager start from the full lib/,
# so the plan is built by a JVM that can reach TornadoVM, while the TaskManager is started with
# FLINK_LIB_DIR pointing at a copy of lib/ with the provider jar removed.
#
# The split matters. Simply uninstalling the module does not reproduce anything, because then
# the client cannot reach a device either and never selects the node -- the plan comes out CPU
# and the job runs normally. Only a client that can offload and a TaskManager that cannot puts
# the two halves of the decision on different machines, which is what a heterogeneous cluster
# does every time it schedules.
#
# Expected outcome depends on which side of GpuOrCpuCalcOperatorFactory the build is on:
#
#   before it   the job FAILS -- "Cannot load user class:
#               org.apache.flink.table.gpu.operator.GpuCalcOperator"
#   after it    the job COMPLETES on the code-generated operator, at CPU speed, and the
#               TaskManager log says which path it took and why
#
# Usage: heterogeneous-fallback.sh <flink-dist-dir> [rows] [runs]
#   DEPOTS sets the number of reference points (20 by default).
#   DATA overrides the input path; the input is generated on first use.
#   TORNADOVM_HOME must point at a built TornadoVM SDK; run gpu-cluster-setup.sh first.

set -uo pipefail

FLINK_HOME="${1:-}"
ROWS="${2:-2000000}"
RUNS="${3:-6}"
DEPOTS="${DEPOTS:-20}"
DATA="${DATA:-/tmp/flink-gpu-points-${ROWS}-csv}"

if [[ -z "${FLINK_HOME}" || ! -x "${FLINK_HOME}/bin/flink" ]]; then
    echo "usage: $0 <flink-dist-dir> [rows] [runs]" >&2
    exit 1
fi

JAR=$(ls "${FLINK_HOME}"/examples/table/HaversineBenchmark.jar \
         "${FLINK_HOME}"/examples/table/*HaversineBenchmark*.jar 2>/dev/null | head -1 || true)
if [[ -z "${JAR}" ]]; then
    echo "HaversineBenchmark jar not found under ${FLINK_HOME}/examples/table" >&2
    exit 1
fi
if ! ls "${FLINK_HOME}"/lib/flink-accelerator-tornadovm-*.jar >/dev/null 2>&1; then
    echo "no provider jar in lib/ -- run gpu-cluster-setup.sh first" >&2
    echo "(with the module absent from the client too, there is nothing to reproduce)" >&2
    exit 1
fi

# The TaskManager's classpath: everything the cluster ships, minus the GPU runtime.
NOGPU="${FLINK_HOME}/lib-nogpu"
rm -rf "${NOGPU}"
mkdir -p "${NOGPU}"
cp "${FLINK_HOME}"/lib/*.jar "${NOGPU}"/
rm -f "${NOGPU}"/flink-accelerator-tornadovm-*.jar

echo "### client lib: $(ls "${FLINK_HOME}"/lib/flink-accelerator-tornadovm-*.jar | wc -l) provider jar(s)"
echo "### TM     lib: $(ls "${NOGPU}"/flink-accelerator-tornadovm-*.jar 2>/dev/null | wc -l) provider jar(s)"

"${FLINK_HOME}/bin/stop-cluster.sh" >/dev/null 2>&1
trap '"${FLINK_HOME}/bin/stop-cluster.sh" >/dev/null 2>&1 || true' EXIT
sleep 2

"${FLINK_HOME}/bin/jobmanager.sh" start >/dev/null 2>&1
FLINK_LIB_DIR="${NOGPU}" "${FLINK_HOME}/bin/taskmanager.sh" start >/dev/null 2>&1
sleep 12

if [[ ! -d "${DATA}" ]]; then
    echo "### generating ${ROWS} rows into ${DATA}"
    "${FLINK_HOME}/bin/flink" run "${JAR}" --generate --rows "${ROWS}" --data "${DATA}" --format csv
fi

echo
echo "############ gpu=true on a TaskManager with no GPU module ############"
"${FLINK_HOME}/bin/flink" run "${JAR}" \
    --data "${DATA}" --format csv --parallelism 1 \
    --runs "${RUNS}" --depots "${DEPOTS}" --gpu true
JOB_EXIT=$?
echo "JOB_EXIT=${JOB_EXIT}"

echo
echo "### what the TaskManager decided"
grep -h -E "GPU offload (active|declined)" "${FLINK_HOME}"/log/*.log 2>/dev/null | sort -u \
    || echo "(no GpuOrCpuCalcOperatorFactory log line -- this build predates it)"

echo
echo "### did it fail on a missing device class?"
grep -h -E "Cannot load user class|ClassNotFoundException: org\.apache\.flink\.table\.gpu" \
    "${FLINK_HOME}"/log/*.log 2>/dev/null | sort -u | head -5 \
    || echo "(no -- the job did not fail on a missing device class)"

exit "${JOB_EXIT}"
