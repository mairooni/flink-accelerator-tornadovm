#!/usr/bin/env bash
#
# Sets a clean laptop up to run the three demos. Idempotent: re-running skips
# what is already done, so a failed step can be fixed and the script re-run.
#
#   ./setup.sh              everything
#   ./setup.sh --skip-flink only the parts that are quick to redo
#
# Requires: an NVIDIA GPU with a CUDA toolkit, JDK 21, Maven, CMake, Python 3,
# git, and a C++20 compiler. Everything else it fetches or builds.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

SKIP_FLINK=0; SKIP_TORNADO=0; SKIP_DATA=0
for a in "$@"; do case "$a" in
    --skip-flink) SKIP_FLINK=1 ;;
    --skip-tornadovm) SKIP_TORNADO=1 ;;
    --skip-data) SKIP_DATA=1 ;;
    *) demo_die "unknown argument $a" ;;
esac; done

TORNADOVM_REPO="${TORNADOVM_REPO:-https://github.com/mairooni/TornadoVM.git}"
TORNADOVM_BRANCH="${TORNADOVM_BRANCH:-demo-integration}"
FLINK_REPO="${FLINK_REPO:-https://github.com/mairooni/flink.git}"
FLINK_BRANCH="${FLINK_BRANCH:-gpu-offload}"
PROVIDER_REPO="${PROVIDER_REPO:-https://github.com/mairooni/flink-accelerator-tornadovm.git}"
PROVIDER_BRANCH="${PROVIDER_BRANCH:-feat/llm-pipeline-benchmark}"

# ---------------------------------------------------------------------------
demo_banner "1/7  preflight"
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
HAVE_CUDNN=0
[[ -f "$CUDA_PATH/include/cudnn.h" || -f "$CUDA_PATH/targets/x86_64-linux/include/cudnn.h" ]] && HAVE_CUDNN=1
echo "CUDA     $CUDA_PATH  (cudnn: $([[ $HAVE_CUDNN == 1 ]] && echo yes || echo 'no -- LLM prefill will use the JIT path'))"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--allow-unsupported-compiler}"
mkdir -p "$DEMO_ROOT" "$DATA_ROOT"

# ---------------------------------------------------------------------------
demo_banner "2/7  RAPIDS libcudf"
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
demo_banner "3/7  TornadoVM"
# ---------------------------------------------------------------------------
if [[ $SKIP_TORNADO == 0 ]]; then
    [[ -d "$TORNADOVM_SRC/.git" ]] || git clone --branch "$TORNADOVM_BRANCH" "$TORNADOVM_REPO" "$TORNADOVM_SRC"
    (cd "$TORNADOVM_SRC" && git fetch --quiet origin "$TORNADOVM_BRANCH" && git checkout --quiet "$TORNADOVM_BRANCH" && git pull --quiet --ff-only)
    (
      cd "$TORNADOVM_SRC"
      # bin/compile is the canonical build: it also writes etc/tornado.backend,
      # share/java/graalJars and the argfile, none of which a plain `mvn
      # install` produces -- without the argfile every demo reports "no
      # TornadoVM CUDA SDK".
      python3 bin/compile --jdk jdk21 --backend cuda
      # The cuDNN and CUTLASS bindings are their own profile, so that a host
      # without them still gets the core backend. Build them when we can.
      if [[ $HAVE_CUDNN == 1 ]]; then
          ./mvnw -q -Pjdk21,cuda-backend,cuda-libs -Dtornado.backend=cuda -DskipTests \
                 -pl tornado-drivers/cudnn-jni,tornado-drivers/cutlass-jni install || true
      fi
    )
    export TORNADOVM_HOME="$(ls -d "$TORNADOVM_SRC"/dist/*/*/ | head -1)"; TORNADOVM_HOME="${TORNADOVM_HOME%/}"
    # Place the optional native libraries beside the core ones.
    for m in cudnn cutlass; do
        so="$(find "$TORNADOVM_SRC/tornado-drivers/$m-jni/target" -name "libtornado-$m.so" 2>/dev/null | head -1)"
        [[ -n "$so" ]] && cp "$so" "$TORNADOVM_HOME/lib/" || true
    done
    "$TORNADOVM_HOME/bin/tornado" --devices | sed -n '1,8p'
fi
export TORNADOVM_HOME="${TORNADOVM_HOME:-$(ls -d "$TORNADOVM_SRC"/dist/*/*/ | head -1)}"
TORNADOVM_HOME="${TORNADOVM_HOME%/}"

# ---------------------------------------------------------------------------
demo_banner "4/7  Flink (the long one -- 20-40 minutes the first time)"
# ---------------------------------------------------------------------------
if [[ $SKIP_FLINK == 0 ]]; then
    [[ -d "$FLINK_SRC/.git" ]] || git clone --branch "$FLINK_BRANCH" "$FLINK_REPO" "$FLINK_SRC"
    (cd "$FLINK_SRC" && git fetch --quiet origin "$FLINK_BRANCH" && git checkout --quiet "$FLINK_BRANCH" && git pull --quiet --ff-only)
    (cd "$FLINK_SRC" && mvn -q -T1C install -DskipTests -Dcheckstyle.skip -Dspotless.check.skip=true \
                            -Drat.skip=true -Dmaven.javadoc.skip=true -Denforcer.skip=true)
fi
[[ -d "$FLINK_HOME" ]] || demo_die "no Flink distribution at $FLINK_HOME"

# ---------------------------------------------------------------------------
demo_banner "5/7  the accelerator provider and the example jars"
# ---------------------------------------------------------------------------
[[ -d "$PROVIDER_SRC/.git" ]] || git clone --branch "$PROVIDER_BRANCH" "$PROVIDER_REPO" "$PROVIDER_SRC"
(cd "$PROVIDER_SRC" && git fetch --quiet origin "$PROVIDER_BRANCH" && git checkout --quiet "$PROVIDER_BRANCH" && git pull --quiet --ff-only)
TVM_VERSION="$(basename "$(dirname "$TORNADOVM_HOME")" | sed -E 's/^tornadovm-(.*)-cuda-linux-amd64$/\1/')"
echo "building against TornadoVM $TVM_VERSION"
(cd "$PROVIDER_SRC" && mvn -q -DskipTests -Dcheckstyle.skip -Dspotless.check.skip=true -Drat.skip=true \
                          -Dtornado.version="$TVM_VERSION" install)

# ---------------------------------------------------------------------------
demo_banner "6/7  deploying into the Flink distribution"
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
open(p, "w").write(s)
print("  config.yaml: 4 task slots, 4g task off-heap")
PY

# ---------------------------------------------------------------------------
demo_banner "7/7  datasets"
# ---------------------------------------------------------------------------
if [[ $SKIP_DATA == 0 ]]; then
    export HADOOP_CLASSPATH="$(hadoop_classpath)"
    EX="$PROVIDER_SRC/flink-accelerator-tornadovm-examples/target"
    CP="$(ls "$FLINK_HOME"/lib/*.jar | tr '\n' ':')"
    # Haversine reads CSV it writes itself, so only the log corpus is built here.
    if [[ ! -d "$DATA_ROOT/logs16/part-00000.parquet" && ! -f "$DATA_ROOT/logs16/part-00000.parquet" ]]; then
        echo "generating the log corpus (16M and 64M lines, about 2.4 GB of Parquet)"
        "$JAVA_HOME/bin/java" -Xmx12g -cp "$CP$EX/flink-accelerator-tornadovm-examples-0.1.0-SNAPSHOT-GrokSQLExample.jar:$HADOOP_CLASSPATH" \
            org.apache.flink.table.examples.java.gpu.LogCorpusGenerator "$DATA_ROOT" 16000000 64000000
    else
        echo "log corpus already at $DATA_ROOT"
    fi
fi

# ---------------------------------------------------------------------------
cat > "$DEMO_ROOT/env.sh" <<ENV
# Written by setup.sh on $(date -Is). Sourced by the demo scripts.
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
ENV
demo_banner "ready"
echo "  env written to $DEMO_ROOT/env.sh"
echo "  run a demo:  ./demo-haversine.sh | ./demo-regex.sh | ./demo-llm.sh"
