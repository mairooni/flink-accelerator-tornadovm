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
# Prepares a built Flink distribution to run GPU-offloaded jobs.
#
#   - moves flink-table-gpu-runtime from opt/ to lib/, so it is on the TaskManager classpath
#   - writes the TornadoVM JVM flags into conf/config.yaml
#
# The flags come from TornadoVM's own argfile template rather than being copied by hand: it is
# generated from `tornado --printJavaFlags`, so it stays correct across TornadoVM versions. They
# have to reach every JVM that touches the plan -- the client builds it, the JobManager holds it,
# the TaskManager runs the kernel -- so all three are set. Without them the TaskManager dies
# deserializing the operator with UnsupportedClassVersionError.
#
# It also installs the Parquet SQL format, which the distribution does not ship. The benchmark
# reads its input from Parquet, and an example jar carries only its own class, so the format has
# to come from the cluster classpath.
#
# Usage: gpu-cluster-setup.sh <flink-dist-dir>
#   TORNADOVM_HOME must point at a built TornadoVM SDK.
#   PARQUET_JAR may point at flink-sql-parquet-<version>.jar; otherwise it is looked up in this
#   checkout's flink-formats/flink-sql-parquet/target.

set -euo pipefail

FLINK_HOME="${1:-}"
if [[ -z "${FLINK_HOME}" || ! -d "${FLINK_HOME}/bin" ]]; then
    echo "usage: $0 <flink-dist-dir>" >&2
    exit 1
fi
if [[ -z "${TORNADOVM_HOME:-}" || ! -f "${TORNADOVM_HOME}/tornado-argfile.template" ]]; then
    echo "TORNADOVM_HOME must point at a built TornadoVM SDK" >&2
    exit 1
fi

CONFIG="${FLINK_HOME}/conf/config.yaml"

# opt/ -> lib/. The jar is shipped in opt/ because it needs a JVM launched with the flags below.
GPU_JAR=$(ls "${FLINK_HOME}"/opt/flink-table-gpu-runtime-*.jar 2>/dev/null | head -1 || true)
if [[ -n "${GPU_JAR}" ]]; then
    cp "${GPU_JAR}" "${FLINK_HOME}/lib/"
    echo "installed $(basename "${GPU_JAR}") into lib/"
elif ls "${FLINK_HOME}"/lib/flink-table-gpu-runtime-*.jar >/dev/null 2>&1; then
    echo "flink-table-gpu-runtime already in lib/"
else
    echo "no flink-table-gpu-runtime jar in opt/ or lib/ -- was the dist built from this branch?" >&2
    exit 1
fi

# Swap the planner loader for the planner itself.
#
# flink-table-planner-loader ships in lib/ and carries a shaded copy of the planner. On this branch
# that copy has been observed to lag the built sources -- the shade resolves an older
# flink-table-planner artifact -- so a cluster using it runs a planner without the current offload
# code and silently reports that the expressions could not be generated. The distribution already
# ships the unshaded planner in opt/ for exactly this swap, and it is built from the reactor.
LOADER=$(ls "${FLINK_HOME}"/lib/flink-table-planner-loader-*.jar 2>/dev/null | head -1 || true)
PLANNER=$(ls "${FLINK_HOME}"/opt/flink-table-planner_*.jar 2>/dev/null | head -1 || true)
if [[ -z "${PLANNER}" ]]; then
    if ls "${FLINK_HOME}"/lib/flink-table-planner_*.jar >/dev/null 2>&1; then
        echo "no planner in opt/; keeping the one already in lib/" >&2
    else
        echo "no planner jar in opt/ or lib/" >&2
        exit 1
    fi
else
    if [[ -n "${LOADER}" ]]; then
        mv "${LOADER}" "${FLINK_HOME}/opt/"
        echo "moved $(basename "${LOADER}") out of lib/"
    fi
    # Unconditionally, every run. This used to happen only on the first, when the loader was still
    # in lib/ to be displaced; after that the script reported "already in lib/" and left whatever
    # was there. A rebuilt planner therefore sat in opt/ while the cluster went on running the old
    # one -- which is the same stale-planner failure the loader swap exists to prevent, arriving by
    # a different route, and it silently cost a placement measurement.
    cp "${PLANNER}" "${FLINK_HOME}/lib/"
    echo "installed $(basename "${PLANNER}") into lib/"
fi

# Parquet SQL format. flink-sql-parquet is the shaded jar meant for lib/; the unshaded
# flink-parquet would drag its Hadoop dependencies in behind it.
PARQUET_JAR="${PARQUET_JAR:-}"
if [[ -z "${PARQUET_JAR}" ]]; then
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
    PARQUET_JAR=$(ls "${REPO_ROOT}"/flink-formats/flink-sql-parquet/target/flink-sql-parquet-*.jar 2>/dev/null \
                  | grep -v original | head -1 || true)
fi
if ls "${FLINK_HOME}"/lib/flink-sql-parquet-*.jar >/dev/null 2>&1; then
    echo "flink-sql-parquet already in lib/"
elif [[ -n "${PARQUET_JAR}" && -f "${PARQUET_JAR}" ]]; then
    cp "${PARQUET_JAR}" "${FLINK_HOME}/lib/"
    echo "installed $(basename "${PARQUET_JAR}") into lib/"
else
    # Not fatal: the benchmark reads csv by default, and csv ships in lib/ already. Only a
    # FORMAT=parquet run needs this, and that needs Hadoop on the classpath as well.
    echo "note: no flink-sql-parquet jar found; csv will work, parquet will not" >&2
fi

# Off-heap budget for the staging buffers. Each offloaded subtask holds batch-size doubles per
# input column plus its outputs -- about 6 MiB at the default batch of 262,144 -- and TornadoVM
# allocates its own on top. This is per TaskManager, not per slot, so it has to cover every subtask
# that might run there at once. 4g is enough for 32 subtasks at the default batch size, measured;
# raise it if you raise the batch size or the slot count.
OFF_HEAP="${OFF_HEAP:-4g}"

# Total process memory has to cover that budget and everything else the TaskManager needs, and it
# has to grow faster than the budget does: managed and network memory are *fractions* of the total
# (0.4 and 0.1 by default), so every gigabyte added to the total gives only half a gigabyte back.
# At 8g with a 4g budget the sum comes to 7.72g against 6.95g available and the TaskManager refuses
# to start; 16g leaves comfortable headroom.
#
# The failure is worth recognising because it does not look like a memory problem.
# start-cluster.sh reports it only in the TaskManager's .out file, so what you see is a cluster
# with a JobManager and no workers, a job that is accepted, and a wait for a slot that never
# arrives. If you raise OFF_HEAP, raise this too, and if the TaskManager will not start look for
# "exceed configured Total Flink Memory".
PROCESS_MEM="${PROCESS_MEM:-16g}"

# Whether this TaskManager advertises a GPU to the scheduler.
#
# Only the amount matters here. A driver factory is what an operator would need to ask which device
# it was given; the scheduler needs nothing but the number, and ExternalResourceUtils reads the
# amount independently of whether a driver is configured.
#
# Set DECLARE_GPU=0 for a TaskManager that should not be offered offloaded work -- which is what
# heterogeneous-fallback.sh is testing, and the only case in this repo where the answer is no.
DECLARE_GPU="${DECLARE_GPU:-1}"
GPU_RESOURCE_NAME="${GPU_RESOURCE_NAME:-gpu}"
GPU_RESOURCE_AMOUNT="${GPU_RESOURCE_AMOUNT:-1}"

# Flatten the argfile into one line. Comments and blank lines go; everything else is a JVM flag.
FLAGS=$(TORNADOVM_HOME="${TORNADOVM_HOME}" envsubst < "${TORNADOVM_HOME}/tornado-argfile.template" \
        | grep -vE '^\s*#' | grep -vE '^\s*$' | tr '\n' ' ' | sed 's/  */ /g; s/ $//')

# The distribution's config.yaml already sets env.java.opts.all -- it carries the --add-opens that
# Flink itself needs on Java 17+ -- so the flags are appended to that line rather than written as a
# second env: key, which YAML rejects as a duplicate, and rather than replacing it, which would take
# Flink's own flags away with it.
#
# The pristine file is kept alongside so re-running this script starts from it instead of appending
# twice.
PRISTINE="${CONFIG}.pre-gpu"
if [[ ! -f "${PRISTINE}" ]]; then
    cp "${CONFIG}" "${PRISTINE}"
fi

awk -v flags="${FLAGS}" -v offheap="${OFF_HEAP}" -v process="${PROCESS_MEM}" '
    !flagged && /^      all: / { print $0 " " flags; flagged = 1; next }
    # TornadoVM stages through java.lang.foreign, and those allocations are counted against the
    # JVM direct-memory limit. Flink derives that limit from task.off-heap.size, which defaults to
    # zero because Flink has no idea the operator allocates anything off-heap. At parallelism 1 the
    # framework reserve absorbs it; above about four subtasks per TaskManager it does not, and the
    # job dies with "OutOfMemoryError: Direct buffer memory" -- sometimes indirectly, as a kernel
    # that fails to compile because javac cannot get a buffer either.
    # Only under taskmanager: -- jobmanager has a memory block too, and it is the one that
    # comes first in the file.
    /^taskmanager:/ { in_tm = 1 }
    in_tm && !sized && /^  memory:/ { print "  memory:"; print "    task:"; print "      off-heap:";
                                      print "        size: " offheap; sized = 1; next }
    # And the total the budget has to fit inside. The JobManager block says 1600m, so this line
    # matches the TaskManager alone.
    in_tm && /^      size: 1728m$/ { print "      size: " process; next }
    { print }
' "${PRISTINE}" > "${CONFIG}.tmp"

if ! grep -q -- "--enable-preview" "${CONFIG}.tmp"; then
    rm -f "${CONFIG}.tmp"
    echo "could not find 'env.java.opts.all' in ${PRISTINE} to append to" >&2
    exit 1
fi
if ! grep -q "off-heap" "${CONFIG}.tmp"; then
    rm -f "${CONFIG}.tmp"
    echo "could not find 'taskmanager.memory' in ${PRISTINE} to size off-heap under" >&2
    exit 1
fi
mv "${CONFIG}.tmp" "${CONFIG}"

# Appended rather than woven in: these are dotted top-level keys, which Flink's YAML reads as the
# flat names directly, so they need no place in the nested tree above.
if [[ "${DECLARE_GPU}" == "1" ]]; then
    {
        echo ""
        echo "external-resource.list: ${GPU_RESOURCE_NAME}"
        echo "external-resource.${GPU_RESOURCE_NAME}.amount: ${GPU_RESOURCE_AMOUNT}"
    } >> "${CONFIG}"
    echo "declared external-resource.${GPU_RESOURCE_NAME}.amount=${GPU_RESOURCE_AMOUNT}"
else
    echo "no external resource declared (DECLARE_GPU=0)"
fi

echo "wrote TornadoVM flags to ${CONFIG}"
echo "set taskmanager.memory.task.off-heap.size=${OFF_HEAP}, process.size=${PROCESS_MEM}"
echo "TORNADOVM_HOME=${TORNADOVM_HOME}"
