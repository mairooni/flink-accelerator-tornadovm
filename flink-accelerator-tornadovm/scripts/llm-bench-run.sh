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
# Runs ONE of the two LLM pipeline experiments: start the cluster, run it once, stop the
# cluster. Requires llm-bench-setup.sh to have been run first.
#
#   ./llm-bench-run.sh --gpu     GPU preprocessing + jitllm inside the TaskManager's JVM
#   ./llm-bench-run.sh --cpu     CPU preprocessing + llama.cpp in its own process
#
#   --warm      run once to make the engine resident, then time a second run. Without it the
#               timed run is a COLD one, which is a different and much closer number -- the
#               accelerated arm spends its preprocessing advantage loading the model. The
#               report's headline is the warm figure.
#   --modes N   operating modes per reading, i.e. arithmetic per row (default 8). Above 8 the
#               provider declines the subtree unless its ceiling is raised; see EXPERIMENTS.md.
#   --explain   print the plan and the accelerator's accept/decline reasons, then run.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVFILE="${LLM_BENCH_ENV:-$HOME/gpu-bench-data/flink-llm/llm-bench.env}"
[[ -f "$ENVFILE" ]] || { echo "no environment file at $ENVFILE -- run llm-bench-setup.sh first" >&2; exit 1; }
# shellcheck source=/dev/null
source "$ENVFILE"

ARM="" WARM=0 MODES=8 EXPLAIN=""
JOB_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu|gpu) ARM=Gpu ;;
        --cpu|cpu) ARM=Cpu ;;
        --warm)    WARM=1 ;;
        --modes)   MODES="$2"; shift ;;
        --explain) EXPLAIN="--explain" ;;
        -h|--help) sed -n '18,32p' "$HERE/$(basename "${BASH_SOURCE[0]}")"; exit 0 ;;
        # Anything else is handed to the pipeline, so --max-new-tokens and the
        # rest of TelemetryTriage's flags are reachable without editing this.
        *) JOB_ARGS+=("$1") ;;
    esac
    shift
done
[[ -n "$ARM" ]] || { echo "usage: $(basename "$0") --gpu | --cpu [--warm] [--modes N] [--explain]" >&2; exit 2; }

if [[ "$ARM" == Gpu ]]; then
    LABEL="GPU preprocessing + jitllm (in the TaskManager's JVM)"
else
    LABEL="CPU preprocessing + llama.cpp (separate process)"
fi
CLASS="org.apache.flink.table.examples.java.gpu.llm.${ARM}TriagePipeline"
JAR="$LLM_BENCH_JAR_PREFIX-${ARM}TriagePipeline.jar"
[[ -f "$JAR" ]] || { echo "no jar at $JAR -- re-run llm-bench-setup.sh" >&2; exit 1; }
[[ -d "$LLM_BENCH_DATA" ]] || { echo "no dataset at $LLM_BENCH_DATA -- re-run llm-bench-setup.sh" >&2; exit 1; }

# Always torn down, however this exits, so a failed run does not leave a cluster or a model
# server holding 8 GiB of VRAM behind it.
#
# LLM_BENCH_KEEP_CLUSTER=1 keeps the Flink cluster across the run, for a demo that wants the
# web UI to survive and the earlier jobs to stay listed. The llama-server is killed either
# way: it holds GPU memory and nothing later needs it.
cleanup() {
    [[ "${LLM_BENCH_KEEP_CLUSTER:-0}" == 1 ]] || "$FLINK_HOME/bin/stop-cluster.sh" >/dev/null 2>&1 || true
    pkill -x llama-server 2>/dev/null || true
}
trap cleanup EXIT

printf '\n\033[1m%s\033[0m\n' "$LABEL"
printf '  model %s\n  data  %s\n  modes %s\n\n' \
       "$(basename "$LLM_BENCH_MODEL")" "$LLM_BENCH_DATA" "$MODES"

if [[ "${LLM_BENCH_KEEP_CLUSTER:-0}" == 1 ]] \
   && curl -s -m 2 localhost:8081/overview 2>/dev/null | grep -qE '"slots-available":[1-9]'; then
    echo "==> reusing the cluster already running"
else
    echo "==> starting the cluster"
    cleanup; sleep 2
    "$FLINK_HOME/bin/start-cluster.sh" >/dev/null
fi
# Any free slot will do. Matching "slots-available":1 exactly meant a cluster configured with
# more than one slot never satisfied this and the script waited forever with an idle GPU.
until curl -s localhost:8081/overview 2>/dev/null | grep -qE '"slots-available":[1-9]'; do sleep 2; done

run_once() {
    "$FLINK_HOME/bin/flink" run -c "$CLASS" "$JAR" \
        --data "$LLM_BENCH_DATA" --model "$LLM_BENCH_MODEL" \
        --binary "$LLM_BENCH_BINARY" --modes "$MODES" $EXPLAIN "${JOB_ARGS[@]}" 2>&1
}

if (( WARM )); then
    echo "==> warm-up run (not timed): making the engine resident"
    run_once | grep -E "^@@TRIAGE|model_load_ms" >/dev/null || true
fi

echo "==> running"
START=$(date +%s.%N)
OUT="$(run_once)"
END=$(date +%s.%N)
WALL=$(python3 -c "print(f'{$END-$START:.2f}')")

# Hadoop's shutdown hook trips over Flink's closed user classloader after the result is
# already in hand. It is noise on every run of both arms, and it buries the answer.
clean() { grep -vE "SLF4J|ShutdownHookManager|FlinkUserCodeClassLoaders|org\\.apache\\.hadoop\\.(conf|util)\\.|^\\s+at |^\\s*\\.\\.\\. |Trying to access closed classloader"; }

echo "---- the triage note this query produced ----"
echo "$OUT" | clean | sed -n '/==== result ====/,/query wall time/p' \
    | sed -e 's/^+I\[@@TRIAGE[^]]*$//' -e 's/@@TRIAGE.*decode_tok_s=[0-9.]*//' \
    | grep -vE "^==== result ====|^query wall time|^$" | sed 's/\]$//' | head -30

if [[ -n "$EXPLAIN" ]]; then
    echo; echo "---- accelerator decisions ----"
    echo "$OUT" | sed -n '/== GPU Offload ==/,/^$/p' | cut -c1-160
fi

echo
printf '\033[1m---- %s ----\033[0m\n' "$LABEL"
# Readable rather than raw key=value: this is the slide, not a log line.
echo "$OUT" | grep -oE '@@TRIAGE.*' | tr ' ' '\n' | grep '=' | awk -F= '
    $1=="engine"            { printf "  %-26s %s\n", "engine", $2 }
    $1=="resident"          { printf "  %-26s %s\n", "model already resident", ($2=="true" ? "yes" : "no -- this run loaded it") }
    $1=="model_load_ms"     { printf "  %-26s %.1f s\n", "model load", $2/1000 }
    $1=="prompt_tokens"     { printf "  %-26s %s\n", "prompt tokens", $2 }
    $1=="generated_tokens"  { printf "  %-26s %s\n", "generated tokens", $2 }
    $1=="prefill_ms"        { printf "  %-26s %s ms\n", "prefill", $2 }
    $1=="decode_ms"         { printf "  %-26s %s ms\n", "decode", $2 }
    $1=="decode_tok_s"      { printf "  %-26s %s tokens/s\n", "decode throughput", $2 }
    $1=="inference_wall_ms" { printf "  %-26s %s ms\n", "inference, end to end", $2 }'

printf '  \033[1mflink run wall time: %s s\033[0m%s\n' "$WALL" \
       "$( ((WARM)) && echo '  (warm: the engine was already resident)' || echo '  (COLD: this run loaded the model)')"

# Where the offload decision was actually taken -- the plan only says a subtree is eligible.
#
# The provider's "claims NNx over CPU" is an estimate from a static cost model, used only to
# decide whether offloading is worth the setup. It is not measured, it excludes the read, the
# JVM and planning, and printed beside a wall-clock figure it invites exactly the wrong
# comparison -- so the claim and the per-phase breakdown are kept out of the demo output.
# LLM_BENCH_SHOW_PHASES=1 brings them back for debugging.
if [[ "$ARM" == Gpu ]]; then
    LOG="$FLINK_HOME/log/flink-$(whoami)-taskexecutor-0-$(hostname).log"
    if [[ -f "$LOG" ]]; then
        echo
        echo "---- where the preprocessing ran ----"
        grep -a "Accelerated on this TaskManager\|Accelerator declined" "$LOG" | tail -1 \
            | sed 's/.* - /  /; s/: provider \([a-z]*\) claims.*/  (provider: \1)/' || true
        if [[ "${LLM_BENCH_SHOW_PHASES:-0}" == 1 ]]; then
            grep -a -A11 "GpuCalcOperator GpuCalcSpec" "$LOG" | tail -11 \
                | grep -aE "^(batches|gather|copy-in|kernel|drain|attributed|execute|compile)" | sed 's/^/  /' || true
        fi
    fi
fi
echo
echo "==> stopping the cluster"
