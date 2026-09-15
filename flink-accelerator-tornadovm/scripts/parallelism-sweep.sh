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
# Does the offload path work, and scale, above parallelism 1?
#
# The CPU arm gets another core per subtask; the GPU arm gets another competitor for one
# device. The interesting number is therefore not whether the GPU arm gets faster -- it cannot
# outrun one device -- but whether it stays correct, whether every subtask actually reaches the
# device, and how fast the CPU arm closes the gap.
#
# Input is regenerated as one file per subtask at the widest level tested, so no level is
# limited by how few splits the source can hand out.
#
# Usage: parallelism-sweep.sh <flink-dist-dir>
#   LEVELS  parallelism levels to test (default "1 2 4 8 16 32")
#   RUNS    runs per configuration, the first discarded as warm-up (default 5)
#   DEPOTS  reference points per row (default 20)
#   OUT     results directory (default /tmp/flink-gpu-parallelism)
#   DATA    input path, generated on first use
#   TORNADOVM_HOME must point at a built TornadoVM SDK.

set -uo pipefail

FLINK_HOME="${1:-}"
if [[ -z "${FLINK_HOME}" || ! -x "${FLINK_HOME}/bin/flink" ]]; then
    echo "usage: $0 <flink-dist-dir>" >&2
    exit 1
fi

LEVELS="${LEVELS:-1 2 4 8 16 32}"
RUNS="${RUNS:-5}"
DEPOTS="${DEPOTS:-20}"
OUT="${OUT:-/tmp/flink-gpu-parallelism}"
MAXP=$(echo "${LEVELS}" | tr ' ' '\n' | sort -n | tail -1)
DATA="${DATA:-/tmp/flink-gpu-points-2000000-csv-p${MAXP}}"
mkdir -p "${OUT}"

SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${SCRIPTS}/gpu-cluster-setup.sh" "${FLINK_HOME}" > "${OUT}/setup.txt" 2>&1

# Enough slots for the widest level, and process memory to match. gpu-cluster-setup rewrites
# config.yaml from its pristine copy, so these have to be applied after it, not before. The
# off-heap budget the GPU path needs is set by that script; see its comment for why.
CONF="${FLINK_HOME}/conf/config.yaml"
sed -i "s/^  numberOfTaskSlots: .*/  numberOfTaskSlots: ${MAXP}/" "${CONF}"
sed -i "s/^      size: 1728m/      size: 32768m/" "${CONF}"
grep -E "numberOfTaskSlots|off-heap|size: 32768m" "${CONF}" | tee "${OUT}/cluster.txt"

"${FLINK_HOME}/bin/stop-cluster.sh" >/dev/null 2>&1
trap '"${FLINK_HOME}/bin/stop-cluster.sh" >/dev/null 2>&1 || true' EXIT
sleep 3
"${FLINK_HOME}/bin/start-cluster.sh" >/dev/null 2>&1
sleep 15

J="${FLINK_HOME}/examples/table/HaversineBenchmark.jar"
if [[ ! -d "${DATA}" ]]; then
    echo "### generating ${MAXP}-way input into ${DATA}"
    "${FLINK_HOME}/bin/flink" run "${J}" --generate --rows 2000000 --data "${DATA}" \
        --format csv --parallelism "${MAXP}" 2>&1 | tail -2
fi
echo "input files: $(ls "${DATA}" | wc -l)" | tee -a "${OUT}/cluster.txt"

: > "${OUT}/sweep.txt"
for p in ${LEVELS}; do
    for g in false true; do
        # Read only the lines each run appends. Deleting the log instead would unlink a file
        # log4j still holds open: the TaskManager keeps writing to the dead inode, no file
        # exists on disk, and every later grep silently finds nothing.
        TMLOG=$(ls "${FLINK_HOME}"/log/*taskexecutor*.log 2>/dev/null | head -1)
        MARK=$( [[ -n "${TMLOG}" ]] && wc -c < "${TMLOG}" || echo 0 )
        JMLOG=$(ls "${FLINK_HOME}"/log/*standalonesession*.log 2>/dev/null | head -1)
        JMARK=$( [[ -n "${JMLOG}" ]] && wc -c < "${JMLOG}" || echo 0 )

        echo "### parallelism=$p gpu=$g" | tee -a "${OUT}/sweep.txt"
        "${FLINK_HOME}/bin/flink" run "${J}" --data "${DATA}" --format csv \
            --parallelism "$p" --runs "${RUNS}" --depots "${DEPOTS}" --gpu "$g" 2>&1 \
            | grep -E "^run" | tee -a "${OUT}/sweep.txt"
        echo "EXIT=$?" | tee -a "${OUT}/sweep.txt"

        if [[ "$g" == "true" ]]; then
            # One line per subtask that built an operator. Divided by RUNS this is how many
            # subtasks reached the device per run; it should equal the parallelism.
            TMLOG=$(ls "${FLINK_HOME}"/log/*taskexecutor*.log 2>/dev/null | head -1)
            NEW=$(mktemp)
            [[ -n "${TMLOG}" ]] && tail -c +$((MARK + 1)) "${TMLOG}" > "${NEW}"
            act=$(grep -c "GPU offload active" "${NEW}" 2>/dev/null || true)
            dec=$(grep -c "GPU offload declined" "${NEW}" 2>/dev/null || true)
            echo "subtasks_on_device=${act} subtasks_declined=${dec} (over ${RUNS} runs)" \
                | tee -a "${OUT}/sweep.txt"

            JMLOG=$(ls "${FLINK_HOME}"/log/*standalonesession*.log 2>/dev/null | head -1)
            [[ -n "${JMLOG}" ]] && tail -c +$((JMARK + 1)) "${JMLOG}" >> "${NEW}"
            err=$(grep -c -E "CUDA_ERROR|CL_OUT_OF_RESOURCES|LAUNCH_FAILED|OutOfMemoryError" \
                  "${NEW}" 2>/dev/null || true)
            echo "device_errors=${err}" | tee -a "${OUT}/sweep.txt"
            if [[ "${err}" != "0" ]]; then
                grep -h -E "CUDA_ERROR|LAUNCH_FAILED|OutOfMemoryError" "${NEW}" \
                    | cut -c1-160 | sort -u | head -3 | tee -a "${OUT}/sweep.txt"
            fi
            rm -f "${NEW}"
        fi
    done
done

echo "### done" | tee -a "${OUT}/sweep.txt"
