#!/usr/bin/env bash
#
# Demo 3 -- GPU-preprocessed Flink SQL feeding a language model in the same JVM.
#
#   ./demo-llm.sh            GPU preprocessing + jitllm, resident in the TaskManager
#   ./demo-llm.sh --cpu      CPU preprocessing + llama.cpp over HTTP
#   ./demo-llm.sh --warm     make the engine resident first, then run (see below)
#   ./demo-llm.sh --keep-cluster   reuse a cluster that is already up, and leave
#                                  it up -- for running straight after demo 1
#
# This demo has prerequisites the other two do not -- a jitllm checkout, a
# llama.cpp build and a GGUF model -- because it compares two inference
# engines, not two query plans. They are installed by the repository's own
# llm-bench-setup.sh, which this script checks for and points at.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh
[[ -f "$DEMO_ROOT/env.sh" ]] && source "$DEMO_ROOT/env.sh" && source ./common.sh

ARM=--gpu; EXTRA=(); KEEP=0
# TelemetryTriage defaults to 256, which stops the note mid-sentence -- the run
# reports generated_tokens=255, the cap, every time. Raised here so the demo
# needs no flags. Passing --max-new-tokens explicitly still wins.
MAX_NEW_TOKENS=512
while [[ $# -gt 0 ]]; do case "$1" in
    --cpu)  ARM=--cpu; shift ;;
    --gpu)  ARM=--gpu; shift ;;
    --keep-cluster|--reuse-cluster) KEEP=1; shift ;;
    *)      EXTRA+=("$1"); shift ;;
esac; done

RUNNER="$PROVIDER_SRC/flink-accelerator-tornadovm/scripts/llm-bench-run.sh"
[[ -x "$RUNNER" ]] || demo_die "missing $RUNNER"
# The triage readings belong with the other corpora. An install made before
# that was true keeps working: the old location is still searched.
: "${LLM_BENCH_WORK:=$DATA_ROOT/flink-llm}"
ENVFILE="$LLM_BENCH_WORK/llm-bench.env"
[[ -f "$ENVFILE" ]] || { LLM_BENCH_WORK="$HOME/gpu-bench-data/flink-llm"; ENVFILE="$LLM_BENCH_WORK/llm-bench.env"; }
export LLM_BENCH_WORK
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
if [[ $ARM == --gpu ]]; then
cat <<'TXT'
  Eight million telemetry readings screened for anomalies and rolled up per
  machine on the GPU, then the digest handed to a language model in the same
  JVM for the maintenance note. One SQL statement; no process boundary between
  the two halves.
TXT
else
cat <<'TXT'
  The same SQL, with the preprocessing on the cores and the note from
  llama.cpp over HTTP -- the arrangement this is measured against.
TXT
fi

# Only supply the cap if the caller did not. TelemetryTriage defaults to 256,
# which stops the note mid-sentence.
case " ${EXTRA[*]-} " in
    *" --max-new-tokens "*) ;;
    *) EXTRA+=(--max-new-tokens "$MAX_NEW_TOKENS") ;;
esac

ensure_quiet_classloader
exec env LLM_BENCH_KEEP_CLUSTER="$KEEP" "$RUNNER" "$ARM" "${EXTRA[@]}"
