#!/usr/bin/env bash
#
# Step 2 of 3 -- generate the corpora the demos read.
#
#   ./2-generate-data.sh                 the log corpora (1M, 16M and 64M lines)
#   ./2-generate-data.sh --with-haversine  also materialise the 8M and 32M point CSVs
#   ./2-generate-data.sh --skip-model      do not fetch demo 3's GGUF or set it up
#
# Demo 3's setup runs from here too, because llm-bench-setup.sh refuses to start
# without the model and the model is fetched here. It builds jitllm and
# llama.cpp, so the first run adds roughly fifteen minutes.
#   ./2-generate-data.sh --rows 16000000,64000000   pick the log sizes
#
# The 1M corpus is two Parquet files rather than 32, so --print-bytecodes on the
# regex demo produces about 130 lines instead of 2,000. It costs 31 MB.
#
# Idempotent: a corpus that is already there is left alone. The log corpus is
# about 2.4 GB of Parquet and takes a few minutes; keep DATA_ROOT on a real
# disk, not on /tmp, which is tmpfs on most distributions and will eat RAM.
# Total is about 2.4 GB.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

LOGS="1000000,16000000,64000000"; WITH_HAVERSINE=0; WITH_MODEL=1
while [[ $# -gt 0 ]]; do case "$1" in
    --rows) LOGS="$2"; shift 2 ;;
    --with-haversine) WITH_HAVERSINE=1; shift ;;
    --skip-model) WITH_MODEL=0; shift ;;
    *) demo_die "unknown argument $1" ;;
esac; done

[[ -n "${JAVA_HOME:-}" ]]    || demo_die "no JDK 21 found. Run ./1-fetch.sh first, or set JAVA_HOME."
[[ -d "${FLINK_HOME:-}" ]]   || demo_die "no Flink distribution at ${FLINK_HOME:-<unset>}. Run ./1-fetch.sh first."
mkdir -p "$DATA_ROOT"

demo_banner "log corpus -> $DATA_ROOT"
export HADOOP_CLASSPATH="$(hadoop_classpath)"
EX="$PROVIDER_SRC/flink-accelerator-tornadovm-examples/target"
# The examples module shades one jar per main class, so the generator is in its
# own, not in GrokSQLExample's.
JAR="$EX/flink-accelerator-tornadovm-examples-0.1.0-SNAPSHOT-LogCorpusGenerator.jar"
[[ -f "$JAR" ]] || demo_die "missing $(basename "$JAR") in $EX -- run ./1-fetch.sh first"
unzip -l "$JAR" 2>/dev/null | grep -q 'LogCorpusGenerator\.class' \
    || demo_die "$(basename "$JAR") does not contain LogCorpusGenerator -- rebuild with ./1-fetch.sh"
CP="$(ls "$FLINK_HOME"/lib/*.jar | tr '\n' ':')"

TODO=()
for n in ${LOGS//,/ }; do
    tag="logs$(( n / 1000000 ))"
    if compgen -G "$DATA_ROOT/$tag/*.parquet" > /dev/null; then
        echo "  $tag already at $DATA_ROOT/$tag"
    else
        TODO+=("$n")
    fi
done
if [[ ${#TODO[@]} -gt 0 ]]; then
    echo "  generating: ${TODO[*]}"
    "$JAVA_HOME/bin/java" -Xmx12g -cp "$CP$JAR:$HADOOP_CLASSPATH" \
        org.apache.flink.table.examples.java.gpu.LogCorpusGenerator "$DATA_ROOT" "${TODO[@]}"
fi

# The haversine example writes its own CSV on first run, so this is only worth
# doing up front to keep a live demo from pausing to generate 1.4 GB.
if [[ $WITH_HAVERSINE == 1 ]]; then
    demo_banner "haversine corpora -> $DATA_ROOT"
    for n in 8000000 32000000; do
        if compgen -G "$DATA_ROOT/haversine-$n/*.csv" > /dev/null; then
            echo "  haversine-$n already present"
        else
            echo "  generating haversine-$n (runs the demo once)"
            ./demo-haversine.sh --rows "$n" > /dev/null 2>&1 || demo_die "haversine generation failed for $n rows"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Demo 3's weights. Data, not a repository, which is why it is here rather than
# in 1-fetch.sh. *.gguf is gitignored in the jitllm checkout, so cloning that
# brings nothing; the file comes from Hugging Face.
# ---------------------------------------------------------------------------
if [[ $WITH_MODEL == 1 ]]; then
    : "${JITLLM_SRC:=$DEMO_ROOT/jitllm}"
    : "${LLAMACPP_SRC:=$DEMO_ROOT/llama.cpp}"
    : "${MODEL:=$JITLLM_SRC/Qwen3-0.6B-f16.gguf}"
    : "${MODEL_URL:=https://huggingface.co/gvij/qwen3-0.6b-gguf/resolve/main/qwen3-0.6b-fp16.gguf}"
    demo_banner "model -> $MODEL"
    if [[ -f "$MODEL" ]]; then
        echo "  already there ($(du -h "$MODEL" | cut -f1))"
    elif [[ -d "$(dirname "$MODEL")" ]] || mkdir -p "$(dirname "$MODEL")"; then
        echo "  downloading Qwen3-0.6B fp16 (1.44 GiB) from $MODEL_URL"
        curl -fL --progress-bar -o "$MODEL.part" "$MODEL_URL" && mv -f "$MODEL.part" "$MODEL" \
            || { rm -f "$MODEL.part"; demo_die "download failed -- set MODEL to a GGUF you have, or MODEL_URL to another link"; }
    fi
    # A truncated GGUF fails deep inside the engine's loader rather than at
    # startup, so it is checked here where the message can still be useful.
    if [[ -f "$MODEL" ]]; then
        head -c 4 "$MODEL" | grep -q GGUF \
            || demo_die "$MODEL does not start with the GGUF magic -- delete it and re-run"
    else
        echo "  no model at $MODEL -- demo 3 will not run; demos 1 and 2 are unaffected"
    fi
fi

# ---------------------------------------------------------------------------
# Finish demo 3. llm-bench-setup.sh checks for the model before it does
# anything, so it cannot run from 1-fetch.sh -- the model only exists once the
# step above has run. TornadoVM and Flink are already built, so those phases are
# skipped and what is left is the two engines, the deploy and the triage data.
# ---------------------------------------------------------------------------
if [[ $WITH_MODEL == 1 && -f "$MODEL" ]]; then
    LLM_SETUP="$PROVIDER_SRC/flink-accelerator-tornadovm/scripts/llm-bench-setup.sh"
    if [[ -x "$LLM_SETUP" ]]; then
        demo_banner "demo 3: jitllm, llama.cpp and the triage dataset"
        echo "  building the two engines -- about fifteen minutes the first time"
        # Keep the triage readings with the other corpora rather than in the
        # script's own default under $HOME.
        : "${LLM_BENCH_WORK:=$DATA_ROOT/flink-llm}"
        # Every root this install resolved, passed explicitly. llm-bench-setup.sh
        # is also usable on its own and falls back to its siblings, and those
        # fallbacks are right for a 1-fetch tree -- but not for a machine where
        # the checkouts are somewhere else, and silently building jitllm against
        # a different TornadoVM than the cluster runs is not a failure you see.
        if JITLLM_SRC="$JITLLM_SRC" LLAMACPP_SRC="$LLAMACPP_SRC" MODEL="$MODEL" \
           WORK="$LLM_BENCH_WORK" JDK21="$JAVA_HOME" \
           TORNADOVM_SRC="$TORNADOVM_SRC" FLINK_SRC="$FLINK_SRC" \
           FLINK_HOME="$FLINK_HOME" \
           "$LLM_SETUP" --skip-tornadovm --skip-flink; then
            echo "  demo 3 ready"
        else
            echo
            echo "  demo 3 setup did not finish. Demos 1 and 2 are unaffected; retry with:"
            echo "    JITLLM_SRC=$JITLLM_SRC LLAMACPP_SRC=$LLAMACPP_SRC MODEL=$MODEL \\"
            echo "      WORK=$LLM_BENCH_WORK JDK21=$JAVA_HOME TORNADOVM_SRC=$TORNADOVM_SRC \\"
            echo "      FLINK_SRC=$FLINK_SRC FLINK_HOME=$FLINK_HOME \\"
            echo "      $LLM_SETUP --skip-tornadovm --skip-flink"
        fi
    fi
fi

demo_banner "data ready"
du -sh "$DATA_ROOT"/* 2>/dev/null | sed 's/^/  /' || true
echo
echo "  next:  source ./3-env.sh"
