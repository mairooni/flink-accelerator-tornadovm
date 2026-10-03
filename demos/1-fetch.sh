#!/usr/bin/env bash
#
# Step 1 of 3 -- fetch the three repositories at the right branches and build
# them, then deploy the provider into the Flink distribution.
#
#   ./1-fetch.sh                    everything
#   ./1-fetch.sh --skip-flink       when Flink is already built
#   ./1-fetch.sh --skip-tornadovm   when the TornadoVM SDK is already built
#   ./1-fetch.sh --with-llm         also fetch jitllm and llama.cpp for demo 3
#
# Idempotent: re-running skips what is already done, so a failed step can be
# fixed and the script re-run. Budget 40-70 minutes on a cold machine; Flink's
# own build is most of it.
#
# Requires: an NVIDIA GPU with a CUDA toolkit, JDK 21, Maven, CMake, Python 3,
# git, and a C++20 compiler. Everything else it fetches or builds.
#
# Then:  ./2-generate-data.sh   and   source ./3-env.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

SKIP_FLINK=0; SKIP_TORNADO=0; WITH_LLM=0
for a in "$@"; do case "$a" in
    --skip-flink) SKIP_FLINK=1 ;;
    --skip-tornadovm) SKIP_TORNADO=1 ;;
    --with-llm) WITH_LLM=1 ;;
    *) demo_die "unknown argument $a" ;;
esac; done

TORNADOVM_REPO="${TORNADOVM_REPO:-https://github.com/mairooni/TornadoVM.git}"
TORNADOVM_BRANCH="${TORNADOVM_BRANCH:-demo-integration}"
FLINK_REPO="${FLINK_REPO:-https://github.com/mairooni/flink.git}"
FLINK_BRANCH="${FLINK_BRANCH:-gpu-offload}"


# Clone a repository if it is not there; if it is, leave it alone unless it is
# already on the branch we want and clean. A demo setup script has no business
# switching someone's branch or discarding their work, and a tree that is
# deliberately on something else is usually deliberate.
ensure_repo() {
    local dir="$1" url="$2" branch="$3" name="$4"
    # -d "$dir/.git" is not enough: in a git worktree .git is a file.
    if ! git -C "$dir" rev-parse --git-dir >/dev/null 2>&1; then
        echo "  cloning $name -> $dir ($branch)"
        git clone --branch "$branch" "$url" "$dir"
        return
    fi
    local cur; cur="$(git -C "$dir" rev-parse --abbrev-ref HEAD)"
    if [[ -n "$(git -C "$dir" status --porcelain)" ]]; then
        echo "  $name: $dir is on '$cur' with local changes -- left untouched"
    elif [[ "$cur" != "$branch" ]]; then
        echo "  $name: $dir is on '$cur', not '$branch' -- left untouched"
        echo "         (check it out yourself, or point ${name^^}_SRC elsewhere)"
    else
        if git -C "$dir" fetch --quiet origin "$branch" 2>/dev/null \
           && git -C "$dir" merge --quiet --ff-only "origin/$branch" 2>/dev/null; then
            echo "  $name: $dir on '$branch' -- up to date with origin"
        else
            # A fork checked out under a different remote name, or a branch that
            # exists only locally. Nothing to do, and nothing worth failing over.
            echo "  $name: $dir on '$branch' -- kept as is (no fast-forward from origin)"
        fi
    fi
}

# ---------------------------------------------------------------------------
demo_banner "1/6  preflight"
# ---------------------------------------------------------------------------
command -v nvidia-smi >/dev/null || demo_die "no nvidia-smi: this needs an NVIDIA GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[[ -n "${JAVA_HOME:-}" ]] || demo_die "no JDK 21 found. Set JAVA_HOME to one."
"$JAVA_HOME/bin/javac" -version 2>&1 | grep -q " 21" || demo_die "JAVA_HOME is not a JDK 21: $JAVA_HOME"
echo "JDK      $JAVA_HOME"
for t in mvn cmake python3 git g++; do command -v $t >/dev/null || demo_die "missing: $t"; done

# A CUDA toolkit that has cudnn.h lets the cuDNN and CUTLASS bindings build,
# which the LLM demo wants for its fused attention. Without one the other two
# demos are unaffected and the LLM demo still runs, on a slower prefill path.
if [[ -z "${CUDA_PATH:-}" ]]; then
    for c in /usr/local/cuda-12.5 /usr/local/cuda-12 /usr/local/cuda; do
        [[ -f "$c/include/cuda_runtime.h" ]] && { export CUDA_PATH="$c"; break; }
    done
fi
[[ -n "${CUDA_PATH:-}" ]] || demo_die "no CUDA toolkit found. Set CUDA_PATH."
echo "CUDA     $CUDA_PATH"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--allow-unsupported-compiler}"

# The precondition that actually matters is that nvcc accepts this host's C++
# standard library -- not that any library is installed. cudnn-jni and
# cutlass-jni compile C++ through nvcc, and a toolkit older than the host GCC
# fails at cmake's compiler-identification step with errors inside
# <type_traits> ("identifier \"char8_t\" is undefined" and similar). That reads
# like a missing dependency and is not one: CUTLASS is fetched by cutlass-jni
# itself, as a sparse checkout of NVIDIA/cutlass.
_probe="$(mktemp -d)"; printf '#include <type_traits>\nint main(){return 0;}\n' > "$_probe/p.cu"
if ! "$CUDA_PATH/bin/nvcc" -std=c++17 -c "$_probe/p.cu" -o "$_probe/p.o" >"$_probe/err" 2>&1; then
    rm -rf "$_probe"
    demo_die "nvcc at $CUDA_PATH cannot compile against this host's libstdc++ ($(g++ -dumpversion 2>/dev/null)).
  cudnn-jni and cutlass-jni will fail at cmake compiler identification, which
  looks like a missing library and is not one.
  Use a newer CUDA toolkit, or put a wrapper nvcc that passes
  -allow-unsupported-compiler first on PATH, and set CUDA_PATH to it."
fi
rm -rf "$_probe"
mkdir -p "$DEMO_ROOT" "$DATA_ROOT"

# ---------------------------------------------------------------------------
demo_banner "2/6  RAPIDS libcudf"
# ---------------------------------------------------------------------------
# The cuDF shim links against RAPIDS. It is a binary distribution, so it is
# fetched rather than built; the pip wheels are the only packaging that does
# not require conda.
if [[ -d "$RAPIDS_HOME/libcudf/lib64" ]]; then
    echo "already at $RAPIDS_HOME"
else
    echo "fetching RAPIDS wheels into $RAPIDS_HOME (about 1.3 GB)"
    tmp="$(mktemp -d)"
    python3 -m pip install --quiet --target "$tmp" \
        libcudf-cu12 librmm-cu12 libkvikio-cu12 rapids-logger nvidia-nvcomp-cu12 \
        || demo_die "pip could not fetch the RAPIDS wheels"
    mkdir -p "$RAPIDS_HOME"
    cp -r "$tmp"/* "$RAPIDS_HOME/"
    rm -rf "$tmp"
fi
for d in libcudf/lib64 librmm/lib64 libkvikio/lib64 rapids_logger/lib64 \
         nvidia/libnvcomp/lib64 libkvikio_cu12.libs; do
    [[ -d "$RAPIDS_HOME/$d" ]] || echo "  warning: $RAPIDS_HOME/$d is missing; cuDF may report itself unavailable"
done
source ./common.sh   # re-resolve now that RAPIDS exists

# ---------------------------------------------------------------------------
demo_banner "3/6  TornadoVM"
# ---------------------------------------------------------------------------
if [[ $SKIP_TORNADO == 0 ]]; then
    ensure_repo "$TORNADOVM_SRC" "$TORNADOVM_REPO" "$TORNADOVM_BRANCH" tornadovm
    (
      cd "$TORNADOVM_SRC"
      # bin/compile is the canonical build: it also writes etc/tornado.backend,
      # share/java/graalJars and the argfile, none of which a plain `mvn
      # install` produces -- without the argfile every demo reports "no
      # TornadoVM CUDA SDK".
      # cuda-backend already builds cuda, cudnn-jni, cudf-jni and cutlass-jni,
      # and cutlass-jni fetches CUTLASS itself (a sparse checkout of
      # NVIDIA/cutlass); none of the three needs a separate step or a library
      # installed up front.
      python3 bin/compile --jdk jdk21 --backend cuda
    )
    export TORNADOVM_HOME="$(ls -d "$TORNADOVM_SRC"/dist/*/*/ | head -1)"; TORNADOVM_HOME="${TORNADOVM_HOME%/}"
    "$TORNADOVM_HOME/bin/tornado" --devices | sed -n '1,8p'
    # A missing cuDF shim is the most expensive silent failure here: the SDK
    # loads, reports CUDADriver, lists tornado.cudf in --add-modules, and every
    # cuDF region then declines and returns the right answer on the CPU. The
    # only symptom is that demo 2 takes 35 s instead of 5 s.
    if [[ -z "$(find "$TORNADOVM_HOME" -name 'libtornado-cudf.so' -print -quit)" ]]; then
        demo_die "TornadoVM built without the cuDF shim (no libtornado-cudf.so).
  CUDF_HOME=${CUDF_HOME:-<unset>} RMM_HOME=${RMM_HOME:-<unset>}
  Both must be set when bin/compile runs; its cmake step produces nothing
  without them and still reports success."
    fi
fi
export TORNADOVM_HOME="${TORNADOVM_HOME:-$(ls -d "$TORNADOVM_SRC"/dist/*/*/ | head -1)}"
TORNADOVM_HOME="${TORNADOVM_HOME%/}"

# ---------------------------------------------------------------------------
demo_banner "4/6  Flink (the long one -- 20-40 minutes the first time)"
# ---------------------------------------------------------------------------
if [[ $SKIP_FLINK == 0 ]]; then
    ensure_repo "$FLINK_SRC" "$FLINK_REPO" "$FLINK_BRANCH" flink
    (cd "$FLINK_SRC" && mvn -q -T1C install -DskipTests -Dcheckstyle.skip -Dspotless.check.skip=true \
                            -Drat.skip=true -Dmaven.javadoc.skip=true -Denforcer.skip=true)
fi
[[ -d "$FLINK_HOME" ]] || demo_die "no Flink distribution at $FLINK_HOME"

# ---------------------------------------------------------------------------
demo_banner "5/6  the accelerator provider and the example jars"
# ---------------------------------------------------------------------------
# This script ships inside the provider repository, so the provider is already
# here -- it is the clone the person is standing in. Nothing is fetched or
# checked out for it: doing so would switch the branch under someone who is
# working on one, or fail outright on a dirty tree. It is only built.
echo "provider: $PROVIDER_SRC ($(git -C "$PROVIDER_SRC" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'not a git checkout'))"
TVM_VERSION="$(basename "$(dirname "$TORNADOVM_HOME")" | sed -E 's/^tornadovm-(.*)-cuda-linux-amd64$/\1/')"
echo "building against TornadoVM $TVM_VERSION"
(cd "$PROVIDER_SRC" && mvn -q -DskipTests -Dcheckstyle.skip -Dspotless.check.skip=true -Drat.skip=true \
                          -Dtornado.version="$TVM_VERSION" install)

# ---------------------------------------------------------------------------
demo_banner "6/6  deploying into the Flink distribution"
# ---------------------------------------------------------------------------
TORNADOVM_HOME="$TORNADOVM_HOME" FLINK_SRC="$FLINK_SRC" \
    PROVIDER_JAR="$(ls "$PROVIDER_SRC"/flink-accelerator-tornadovm/target/flink-accelerator-tornadovm-*.jar | grep -v sources | head -1)" \
    "$PROVIDER_SRC/flink-accelerator-tornadovm/scripts/gpu-cluster-setup.sh" "$FLINK_HOME"

# The demos want more than the default slot and a bigger off-heap budget:
# TornadoVM stages through java.lang.foreign, which counts against the JVM
# direct-memory limit Flink sizes from taskmanager.memory.task.off-heap.size.
python3 - "$FLINK_HOME/conf/config.yaml" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
def setkey(s, key, val, indent):
    pat = re.compile(rf"^{indent}{re.escape(key)}:.*$", re.M)
    return pat.sub(f"{indent}{key}: {val}", s) if pat.search(s) else s
s = setkey(s, "numberOfTaskSlots", 4, "  ")
s = setkey(s, "size", "4g", "        ")   # taskmanager.memory.task.off-heap.size

# Hadoop's ShutdownHookManager runs after Flink has closed the job classloader
# and trips the leak check, so every Parquet demo ends in a stack trace that
# looks like a failure and is not -- the result and the region report are
# already printed by then. Off, because a demo that prints a scary-looking
# IllegalStateException on success is worse than the check is worth here.
if re.search(r"^classloader\.check-leaked-classloader:", s, re.M):
    s = re.sub(r"^classloader\.check-leaked-classloader:.*$",
               "classloader.check-leaked-classloader: false", s, flags=re.M)
else:
    s = s.rstrip() + "\n\nclassloader.check-leaked-classloader: false\n"
open(p, "w").write(s)
print("  config.yaml: 4 task slots, 4g task off-heap, leaked-classloader check off")
PY


# ---------------------------------------------------------------------------
# Demo 3 only. Separate because it is the one demo with prerequisites outside
# these three repositories, and because the model is a 1.4 GiB download.
# ---------------------------------------------------------------------------
if [[ $WITH_LLM == 1 ]]; then
    demo_banner "jitllm, llama.cpp and the model"
    : "${JITLLM_SRC:=$DEMO_ROOT/jitllm}"
    : "${LLAMACPP_SRC:=$DEMO_ROOT/llama.cpp}"
    : "${MODEL:=$JITLLM_SRC/Qwen3-0.6B-f16.gguf}"

    ensure_repo "$JITLLM_SRC"   "https://github.com/beehive-lab/jitllm.git"  main   jitllm
    ensure_repo "$LLAMACPP_SRC" "https://github.com/ggml-org/llama.cpp.git"         master llamacpp

    # *.gguf is gitignored in the jitllm repository, so cloning it gets no model.
    # Qwen3-0.6B in fp16 GGUF, 1.44 GiB. Verified to start with the GGUF magic and
    # to match the file the reported numbers were taken with to within 192 bytes.
    : "${MODEL_URL:=https://huggingface.co/gvij/qwen3-0.6b-gguf/resolve/main/qwen3-0.6b-fp16.gguf}"
    if [[ -f "$MODEL" ]]; then
        echo "  model: $MODEL"
    elif [[ -n "${MODEL_URL:-}" ]]; then
        echo "  downloading $(basename "$MODEL") from $MODEL_URL"
        mkdir -p "$(dirname "$MODEL")"
        curl -fL --progress-bar -o "$MODEL" "$MODEL_URL"
    fi
    if [[ -f "$MODEL" ]]; then
        # A truncated download is worse than none: the engine fails deep inside
        # a loader rather than at startup.
        head -c 4 "$MODEL" | grep -q GGUF \
            || demo_die "$MODEL does not start with the GGUF magic -- delete it and re-run"
    else
        echo "  no model at $MODEL, and the download did not produce one."
        echo "  Point MODEL at a Qwen3-0.6B GGUF you already have, or set MODEL_URL."
        echo "  Demos 1 and 2 are unaffected."
    fi

    echo "  then finish the LLM setup with:"
    echo "    JITLLM_SRC=$JITLLM_SRC LLAMACPP_SRC=$LLAMACPP_SRC MODEL=$MODEL \\"
    echo "      $PROVIDER_SRC/flink-accelerator-tornadovm/scripts/llm-bench-setup.sh"
fi

# ---------------------------------------------------------------------------
cat > "$DEMO_ROOT/env.sh" <<ENV
# Written by 1-fetch.sh on $(date -Is). Read by 3-env.sh.
export DEMO_ROOT="$DEMO_ROOT"
export TORNADOVM_SRC="$TORNADOVM_SRC"
export TORNADOVM_HOME="$TORNADOVM_HOME"
export FLINK_SRC="$FLINK_SRC"
export FLINK_HOME="$FLINK_HOME"
export PROVIDER_SRC="$PROVIDER_SRC"
export DATA_ROOT="$DATA_ROOT"
export RAPIDS_HOME="$RAPIDS_HOME"
export JAVA_HOME="$JAVA_HOME"
export CUDA_PATH="$CUDA_PATH"
${JITLLM_SRC:+export JITLLM_SRC="$JITLLM_SRC"}
${LLAMACPP_SRC:+export LLAMACPP_SRC="$LLAMACPP_SRC"}
${MODEL:+export MODEL="$MODEL"}
ENV
demo_banner "built"
echo "  env written to $DEMO_ROOT/env.sh"
echo "  next:  ./2-generate-data.sh   then   source ./3-env.sh"
