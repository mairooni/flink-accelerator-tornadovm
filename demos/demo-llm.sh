#!/usr/bin/env bash
#
# Demo 3 -- GPU-preprocessed Flink SQL feeding a language model in the same JVM.
#
#   ./demo-llm.sh            GPU preprocessing + jitllm, resident in the TaskManager
#   ./demo-llm.sh --cpu      CPU preprocessing + llama.cpp over HTTP
#   ./demo-llm.sh --warm     make the engine resident first, then run (see below)
#
# This demo has prerequisites the other two do not -- a jitllm checkout, a
# llama.cpp build and a GGUF model -- because it compares two inference
# engines, not two query plans. They are installed by the repository's own
# llm-bench-setup.sh, which this script checks for and points at.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh
[[ -f "$DEMO_ROOT/env.sh" ]] && source "$DEMO_ROOT/env.sh" && source ./common.sh

ARM=--gpu; EXTRA=()
while [[ $# -gt 0 ]]; do case "$1" in
    --cpu)  ARM=--cpu; shift ;;
    --gpu)  ARM=--gpu; shift ;;
    *)      EXTRA+=("$1"); shift ;;
esac; done

RUNNER="$PROVIDER_SRC/flink-accelerator-tornadovm/scripts/llm-bench-run.sh"
[[ -x "$RUNNER" ]] || demo_die "missing $RUNNER"
ENVFILE="${LLM_BENCH_WORK:-$HOME/gpu-bench-data/flink-llm}/llm-bench.env"
if [[ ! -f "$ENVFILE" ]]; then
    cat >&2 <<TXT

This demo is not set up yet. It needs jitllm, llama.cpp and a GGUF model, which
the repository's own setup script installs:

  $PROVIDER_SRC/flink-accelerator-tornadovm/scripts/llm-bench-setup.sh

It expects these in the environment (defaults in brackets):
  JITLLM_SRC    [~/Projects/GPULlama3-Beehive/GPULlama3.java]
  LLAMACPP_SRC  [~/Projects/llama.cpp]
  MODEL         [\$JITLLM_SRC/Qwen3-0.6B-f16.gguf]

TXT
    exit 2
fi

demo_banner "telemetry triage -- $( [[ $ARM == --gpu ]] && echo 'GPU preprocessing + jitllm' || echo 'CPU preprocessing + llama.cpp' )"
cat <<'TXT'
  The same SQL text in both arms: screen eight million telemetry readings for
  anomalies, roll the survivors up per machine, hand the digest to a language
  model for the maintenance note.

  What differs is where each half runs. The accelerated arm offloads the
  preprocessing and keeps the model resident in the TaskManager's own JVM; the
  other preprocesses on the cores and calls llama.cpp over HTTP.

  Watch for `resident=` in the output. Cold, the two arms are close -- the
  accelerated one spends its preprocessing advantage loading 1.4 GiB of
  weights. The result this demo is about is the warm one.
TXT
exec "$RUNNER" "$ARM" "${EXTRA[@]}"

if [[ ${KEEP:-0} == 1 ]]; then
    echo
    echo "  cluster left running: http://localhost:8081  (stop it with $FLINK_HOME/bin/stop-cluster.sh)"
fi
