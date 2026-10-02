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
# Runs HaversineSQLExample or CudfSortSQLExample in one JVM, against an in-process
# MiniCluster -- the way WordCountSQLExample runs, and the way these are meant to be
# read and demonstrated. No cluster to start and no jar to install.
#
#   run-sql-demos.sh haversine  [--printKernel]     [--rows N] [--data DIR]
#   run-sql-demos.sh cudf-sort  [--printBytecodes]  [--rows N] [--data DIR]
#
# --rows            defaults to 2,000,000 for haversine and 4,000,000 for the cuDF demos.
#                   Lower them with care: setting a device up costs a fixed ~300 ms and
#                   the accelerator refuses work that cannot repay it, so cudf-sort is
#                   DECLINED below about two million rows -- it answers correctly on
#                   SortOperator and prints no LAUNCH, which on a stage looks like a broken
#                   demo rather than a cost model working. haversine has far more
#                   arithmetic per row and still offloads at 500,000.
#
# --printKernel     prints the CUDA TornadoVM generated for the query, before the result.
#                   This is the evidence for the haversine example: the function it prints
#                   exists nowhere in this repository and was compiled from the SQL at job
#                   start.
#
# --printBytecodes  prints what TornadoVM's interpreter actually did -- the transfers and
#                   the LAUNCH. This is the evidence for the cuDF example, where
#                   --printKernel prints nothing because there is no generated kernel to
#                   print: the operator is served by cudf::stable_sorted_order, and the
#                   line to look for is
#                       bc: LAUNCH task - sort.order[sortedOrder] on [NVIDIA CUDA] ...
#
# Either flag works on either example. Neither changes what is computed.
#
# EXTRA_JVM         extra JVM flags, passed through verbatim.
#
# VERBOSE=1         turns on the accelerator's own logging. The demos are quiet by default
#                   so the output reads from the back of a room; turn this on when a kernel
#                   or a LAUNCH does not appear and the accelerator will say why it
#                   declined.
#
# What this script exists to assemble, none of which is a property of the query:
#
#   1. TornadoVM's JVM arguments, from the SDK's own argfile.
#   2. -Dtornado.enable.fma=false -Dtornado.cuda.compile.profile=repro. Not optional and
#      not a tuning knob. A GPU will happily compute a*b+c as one fused operation with a
#      single rounding, which is a *different* number from the two the CPU computes; the
#      and the provider no longer checks, so these are a hard deployment obligation.
#      BOTH are needed: enable.fma=false stops Graal emitting a literal fma() call, and
#      --fmad=false (the repro profile) stops NVRTC contracting the separated multiply and
#      add back into one fma.rn.f64 in the PTX. Either alone still fuses.
#   3. RAPIDS libcudf and its dependencies on LD_LIBRARY_PATH, for the cuDF demo.
#      libtornado-cudf.so links against several wheels' worth of shared objects and dlopen
#      resolves them at load time, so a missing one reads exactly like a missing shim.
#   4. The provider jar on the classpath, from a built Flink distribution's lib/.
#
# Environment (all have defaults that match this machine):
#   FLINK_DIST      a built Flink distribution: its lib/ is the classpath, and the
#                   provider jar must be in it (scripts/gpu-cluster-setup.sh puts it there)
#   TORNADO_SDK     a built TornadoVM CUDA SDK. Deliberately not TORNADOVM_HOME: that one
#                   is commonly already exported by sdkman to an unrelated SDK, and this
#                   demo once produced nothing but a log4j linkage error against a
#                   TornadoVM 2.2.0-opencl install that happened to be on PATH. Whatever
#                   is chosen is validated below before the JVM is started.
#   RAPIDS_HOME     the directory holding libcudf/, librmm/, libkvikio/, rapids_logger/,
#                   nvidia/ and libkvikio_cu12.libs from the RAPIDS wheels. Create one with
#                       python3 -m venv /tmp/cudfenv
#                       /tmp/cudfenv/bin/pip install --extra-index-url=https://pypi.nvidia.com libcudf-cu12
#                       RAPIDS_HOME=$(echo /tmp/cudfenv/lib/python3.*/site-packages)
#                   Only the cuDF demo needs it.
#   JAVA_HOME       a JDK 21; TornadoVM's off-heap arrays are a preview API there

set -euo pipefail

usage() {
    echo "usage: $0 <haversine|cudf-sort|cudf-groupby|gram> [--printKernel] [--printBytecodes]" \
         "[--rows N] [--data DIR]" >&2
    exit 1
}

DEMO="${1:-}"
shift || true
case "${DEMO}" in
    haversine)  MAIN=org.apache.flink.table.examples.java.gpu.HaversineSQLExample ;;
    cudf-sort)  MAIN=org.apache.flink.table.examples.java.gpu.CudfSortSQLExample ;;
    cudf-groupby) MAIN=org.apache.flink.table.examples.java.gpu.CudfGroupBySQLExample ;;
    gram)       MAIN=org.apache.flink.table.examples.java.gpu.GramMatrixSQLExample ;;
    *)          usage ;;
esac

ROWS=""
DATA=""
TORNADO_FLAGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --printKernel|--print-kernel)
            TORNADO_FLAGS+=("-Dtornado.printKernel=true")
            ;;
        --printBytecodes|--print-bytecodes)
            TORNADO_FLAGS+=("-Dtornado.print.bytecodes=true")
            ;;
        --rows) ROWS="${2:-}"; shift ;;
        --data) DATA="${2:-}"; shift ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
    shift
done

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${HERE}/../.." && pwd)

JAVA_HOME="${JAVA_HOME:-/home/mary/Projects/JDKs/jdk-21.0.3}"
FLINK_DIST="${FLINK_DIST:-/home/mary/Projects/flink/flink-dist/target/flink-2.3.0-bin/flink-2.3.0}"
RAPIDS_HOME="${RAPIDS_HOME:-$HOME/.local/share/rapids-libcudf}"

# The SDK, resolved rather than assumed. Explicit TORNADO_SDK wins; otherwise the newest
# CUDA distribution under TORNADO_SRC, which is where `make BACKEND=cuda` leaves it.
TORNADO_SRC="${TORNADO_SRC:-/home/mary/Projects/TornadoVM}"
if [[ -z "${TORNADO_SDK:-}" ]]; then
    TORNADO_SDK=$(ls -d "${TORNADO_SRC}"/dist/*cuda*/*cuda* 2>/dev/null | sort | tail -1 || true)
fi
if [[ -z "${TORNADO_SDK}" || ! -f "${TORNADO_SDK}/tornado-argfile" ]]; then
    echo "no TornadoVM CUDA SDK: set TORNADO_SDK, or build one with 'make BACKEND=cuda' in ${TORNADO_SRC}" >&2
    exit 1
fi
# An SDK without the CUDA backend loads, runs, and accelerates nothing; one without the
# cuDF module declines the sort with a reason that sounds like a missing card. Both are
# worth catching here rather than in the middle of a demo.
if ! grep -q 'tornado.drivers.cuda' "${TORNADO_SDK}/tornado-argfile"; then
    echo "${TORNADO_SDK} is not a CUDA build (no tornado.drivers.cuda in its argfile)" >&2
    exit 1
fi
if [[ "${DEMO}" == gram ]] && ! ls "${TORNADO_SDK}"/share/java/tornado/tornado-cublas-*.jar >/dev/null 2>&1; then
    # Same trap as the cuDF one below: without the binding the provider declines the Gram
    # matrix, the query runs on the host, and it still prints the right matrix.
    echo "no tornado-cublas jar in ${TORNADO_SDK}, so ${DEMO} would run on the host and still" \
         "print the right answer. Rebuild TornadoVM with the cuBLAS module." >&2
    exit 1
fi
if [[ "${DEMO}" == cudf-* && ! -f "${TORNADO_SDK}/lib/libtornado-cudf.so" ]]; then
    # Fatal for a demo rather than a note. Without the shim the provider declines and the query
    # runs on the CPU, printing the same numbers -- which is a CPU run presented as a device one.
    echo "${TORNADO_SDK}/lib/libtornado-cudf.so is missing, so ${DEMO} would run on the host" \
         "and still print the right answer. Install RAPIDS libcudf and rebuild TornadoVM with" \
         "'make BACKEND=cuda'." >&2
    exit 1
fi
# Exported because the provider and BenchmarkRun read it to report what they ran against.
TORNADOVM_HOME="${TORNADO_SDK}"
export TORNADOVM_HOME

# The module jar, the one holding every example class -- not one of the many single-class
# jars the build also produces for `flink run`, whose names carry a classifier.
EXAMPLES=$(ls "${REPO}"/flink-accelerator-tornadovm-examples/target/flink-accelerator-tornadovm-examples-*-SNAPSHOT.jar 2>/dev/null \
           | grep -vE -- '-sources|-javadoc|original-' | head -1 || true)

for required in "${JAVA_HOME}/bin/java" "${TORNADO_SDK}/tornado-argfile" "${FLINK_DIST}/lib"; do
    if [[ ! -e "${required}" ]]; then
        echo "not found: ${required}" >&2
        exit 1
    fi
done
if [[ -z "${EXAMPLES}" ]]; then
    echo "no examples jar: run 'mvn install -DskipTests' in ${REPO} first" >&2
    exit 1
fi
# The provider, taken from THIS checkout's target/ and put ahead of the distribution's
# lib/ on the classpath. Not from lib/, and the difference has already cost time: the
# distribution keeps whatever gpu-cluster-setup.sh last copied there, so a provider change
# that has been rebuilt and installed is still invisible to a demo reading lib/. That is
# the trap DEPLOYMENT.md records for clusters, and it applies here for the same reason.
# Deleting a class makes it worse rather than better -- the stale jar still has it.
PROVIDER=$(ls "${REPO}"/flink-accelerator-tornadovm/target/flink-accelerator-tornadovm-*.jar 2>/dev/null \
           | grep -vE -- '-sources|-javadoc|original-' | head -1 || true)
if [[ -z "${PROVIDER}" ]]; then
    echo "no provider jar: run 'mvn install -DskipTests' in ${REPO} first" >&2
    exit 1
fi

# RAPIDS, for the cuDF demo. Six directories rather than the three first tried here:
# libcudf pulls in librapids_logger and libnvcomp, and libkvikio's wheel carries an
# auditwheel-mangled libzstd of its own in a side-car directory. A missing one of these
# fails as "isAvailable() == false" with no further explanation, so check with
#     ldd "${TORNADO_SDK}/lib/libtornado-cudf.so" | grep 'not found'
if [[ -d "${RAPIDS_HOME}" ]]; then
    export LD_LIBRARY_PATH="${RAPIDS_HOME}/libcudf/lib64:${RAPIDS_HOME}/librmm/lib64:\
${RAPIDS_HOME}/libkvikio/lib64:${RAPIDS_HOME}/rapids_logger/lib64:\
${RAPIDS_HOME}/nvidia/libnvcomp/lib64:${RAPIDS_HOME}/libkvikio_cu12.libs:${LD_LIBRARY_PATH:-}"
elif [[ "${DEMO}" == cudf-* ]]; then
    echo "RAPIDS_HOME=${RAPIDS_HOME} does not exist, so cuDF will report itself unavailable and" \
         "${DEMO} will run on the host while printing the right answer. Set RAPIDS_HOME." >&2
    exit 1
fi

# Quiet by default. These are meant to be watched, and the accelerator's own INFO logging
# is a paragraph of provenance per job; the evidence a demo wants is --printKernel or
# --printBytecodes, not a log line. VERBOSE=1 turns the logging back on, which is what to
# do when neither of those printed anything.
DECISION_LEVEL=WARN
PROVIDER_LEVEL=WARN
if [[ -n "${VERBOSE:-}" ]]; then
    DECISION_LEVEL=INFO
    PROVIDER_LEVEL=DEBUG
fi

LOGCONF=$(mktemp -d)
trap 'rm -rf "${LOGCONF}"' EXIT
cat > "${LOGCONF}/log4j2.properties" <<EOF
rootLogger.level = WARN
rootLogger.appenderRef.console.ref = ConsoleAppender
appender.console.name = ConsoleAppender
appender.console.type = CONSOLE
appender.console.target = SYSTEM_OUT
appender.console.layout.type = PatternLayout
appender.console.layout.pattern = %-5p %c{1} - %m%n
# "Accelerated on this TaskManager" / "Accelerator declined on this TaskManager".
logger.decision.name = org.apache.flink.table.runtime.gpu
logger.decision.level = ${DECISION_LEVEL}
# The provider's own reasons, which are where a decline is explained.
logger.provider.name = org.apache.flink.table.gpu
logger.provider.level = ${PROVIDER_LEVEL}
# Teardown noise. The MiniCluster stops as soon as the last row is collected, and the
# fetcher then finds it gone and says so; it is not a failure and not a lost row.
logger.fetcher.name = org.apache.flink.streaming.api.operators.collect
logger.fetcher.level = ERROR
# Start-up noise a MiniCluster emits and a demo cannot act on: no Kerberos tokens, no
# log.file to point the dashboard at, no checkpointing on a batch job. All three are
# WARN, all three are correct, and none of them is about this query.
logger.tokens.name = org.apache.flink.runtime.security.token
logger.tokens.level = ERROR
logger.webmonitor.name = org.apache.flink.runtime.webmonitor
logger.webmonitor.level = ERROR
logger.jobmaster.name = org.apache.flink.runtime.jobmaster
logger.jobmaster.level = ERROR
EOF

exec "${JAVA_HOME}/bin/java" \
    "@${TORNADO_SDK}/tornado-argfile" \
    -Dtornado.enable.fma=false \
    -Dtornado.cuda.compile.profile=repro \
    ${TORNADO_FLAGS[@]+"${TORNADO_FLAGS[@]}"} \
    ${EXTRA_JVM:-} \
    -cp "${LOGCONF}:${EXAMPLES}:${PROVIDER}:${FLINK_DIST}/lib/*" \
    "${MAIN}" ${ROWS:+"${ROWS}"} ${DATA:+"${DATA}"}
