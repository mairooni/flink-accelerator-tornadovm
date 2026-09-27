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
# Prepares a distribution already set up by gpu-cluster-setup.sh to also run the LLM pipeline
# examples. Run that first; this adds only what the language model needs.
#
# Two things, and both are deployment rather than anything a query says:
#
#   1. jitllm into lib/, not into the job jar. A model kept resident across jobs has to be held
#      by a class the parent classloader owns, because Flink discards the user-code classloader
#      when a job finishes and everything it defined goes with it. From lib/ the engine classes
#      outlive the job, so the second query on this TaskManager finds the weights already on the
#      device. See ResidentEngines.
#
#   2. The system properties jitllm reads at class-initialisation time. The batched-prefill path
#      is gated on two of them and they are read when the state buffers are allocated, which is
#      before any code of ours could set them. Left unset, prompt processing runs one token per
#      forward -- 138 tokens a second on this card against 5562 -- and a digest of a few hundred
#      tokens spends seconds in prefill.
#
# Usage: llm-pipeline-setup.sh <flink-dist-dir> <jitllm-jar>

set -euo pipefail

FLINK_HOME="${1:-}"
JITLLM_JAR="${2:-}"
if [[ -z "${FLINK_HOME}" || ! -d "${FLINK_HOME}/bin" ]]; then
    echo "usage: $0 <flink-dist-dir> <jitllm-jar>" >&2
    exit 1
fi
if [[ -z "${JITLLM_JAR}" || ! -f "${JITLLM_JAR}" ]]; then
    echo "usage: $0 <flink-dist-dir> <jitllm-jar>" >&2
    exit 1
fi

CONFIG="${FLINK_HOME}/conf/config.yaml"

rm -f "${FLINK_HOME}"/lib/jitllm-*.jar
cp "${JITLLM_JAR}" "${FLINK_HOME}/lib/"
echo "installed $(basename "${JITLLM_JAR}") into lib/"

# Device memory is one pool per TornadoVM runtime, and in this deployment two things draw on it:
# the offloaded operator's staging buffers and the model's weights and KV cache. Sized for both
# on an 8 GiB card; raise it on a larger one.
#
# PREFILL_BATCH is the width of the batched-prefill MMA path. It must be at least as large as a
# prompt chunk for the path to be worth entering, and it costs device memory proportional to it.
PREFILL_BATCH="${PREFILL_BATCH:-2048}"
DEVICE_MEMORY="${DEVICE_MEMORY:-5GB}"

# jitllm.nativeLibraries and the prefill width are ONE setting, and either one alone does
# nothing. The fast prefill is cuDNN's fused attention, whose causal mask is only correct when
# the query block is the whole prefix -- so only a chunk starting at position zero may use it,
# and every later chunk drops to a JIT paged-attention kernel about twenty times slower.
#
# So: the batch must be wide enough for the whole prompt (one chunk), AND the native path must
# be on. Measured here on a 2048-token prompt, native on against off: 20,905 against 998 tokens
# a second. The default is off, which is why this has to be set.
#
# The width costs what it reserves -- the kernel computes the padding rows too, so prefill takes
# a flat ~90 ms for any prompt that fits. Size it to the prompt, not to the context.
ADDITIONS=(
    "--add-modules jdk.incubator.vector"
    "-Dtornado.device.memory=${DEVICE_MEMORY}"
    "-Dtornado.tvm.maxbytecodesize=65536"
    "-Duse.tornadovm=true"
    "-Dtornado.enable.nativeFunctions=true"
    "-Dtornado.loop.interchange=true"
    "-Dtornado.eventpool.maxwaitevents=32000"
    "-Djitllm.withPrefillDecode=true"
    "-Djitllm.nativeLibraries=true"
    "-Djitllm.prefillBatchSize=${PREFILL_BATCH}"
)

python3 - "${CONFIG}" "${ADDITIONS[@]}" <<'PY'
import sys
config, additions = sys.argv[1], sys.argv[2:]
lines = open(config).read().split("\n")
for i, line in enumerate(lines):
    if line.strip().startswith("all:") and "tornado" in line:
        for addition in additions:
            # Idempotent: the flag's name, not the whole string, so that changing a value
            # replaces rather than appends a second contradictory copy.
            name = addition.split("=", 1)[0]
            if name in line:
                continue
            line += " " + addition
        lines[i] = line
        break
else:
    raise SystemExit("env.java.opts.all not found: run gpu-cluster-setup.sh first")
open(config, "w").write("\n".join(lines))
print("jitllm JVM properties written into conf/config.yaml")
PY

cat <<'NOTE'

Restart the cluster for the properties to take effect. The examples also need Hadoop on the
client and cluster classpath, because they read Parquet:

    export HADOOP_CLASSPATH=$(hadoop classpath)

and llama.cpp's llama-server on PATH or at --binary for the comparison arm.
NOTE
