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
# Prepares everything the two LLM pipeline experiments need, end to end, and writes an
# environment file that llm-bench-run.sh sources. Run it once.
#
#   ./llm-bench-setup.sh [--skip-tornadovm] [--skip-jitllm] [--skip-llamacpp]
#                        [--skip-flink] [--skip-deploy] [--skip-data] [--rows N]
#
# Every step is skippable because they take very different amounts of time and only some of
# them change when you are iterating. The order below is not a preference: each step installs
# the artifact the next one compiles against, and out of order the build succeeds against a
# stale jar and fails at run time instead.
#
# Paths are taken from the environment when set, so a checkout somewhere else needs no edit:
#
#   FLINK_SRC       the flink checkout on the gpu-offload branch
#   FLINK_HOME      the built distribution inside it
#   TORNADOVM_SRC   the TornadoVM checkout
#   JITLLM_SRC      the GPULlama3.java / jitllm checkout
#   LLAMACPP_SRC    the llama.cpp checkout
#   MODEL           the .gguf both engines load
#   WORK            scratch space for the dataset, the CUDA shim and the env file
#   JDK21           a JDK 21 (TornadoVM's SDK is built with it and the cluster must match)

set -euo pipefail

FLINK_SRC="${FLINK_SRC:-$HOME/Projects/flink}"
FLINK_HOME="${FLINK_HOME:-$FLINK_SRC/flink-dist/target/flink-2.3.0-bin/flink-2.3.0}"
TORNADOVM_SRC="${TORNADOVM_SRC:-$HOME/Projects/TornadoVM}"
JITLLM_SRC="${JITLLM_SRC:-$HOME/Projects/GPULlama3-Beehive/GPULlama3.java}"
LLAMACPP_SRC="${LLAMACPP_SRC:-$HOME/Projects/llama.cpp}"
MODEL="${MODEL:-$JITLLM_SRC/Qwen3-0.6B-f16.gguf}"
WORK="${WORK:-$HOME/gpu-bench-data/flink-llm}"
JDK21="${JDK21:-$HOME/Projects/JDKs/jdk-21.0.3}"
PROVIDER_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ROWS=8000000
MACHINES=48
do_tornadovm=1 do_jitllm=1 do_llamacpp=1 do_flink=1 do_deploy=1 do_data=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-tornadovm) do_tornadovm=0 ;;
        --skip-jitllm)    do_jitllm=0 ;;
        --skip-llamacpp)  do_llamacpp=0 ;;
        --skip-flink)     do_flink=0 ;;
        --skip-deploy)    do_deploy=0 ;;
        --skip-data)      do_data=0 ;;
        --rows)           ROWS="$2"; shift ;;
        *) echo "unknown flag $1" >&2; exit 2 ;;
    esac
    shift
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------------------
# 0. Prerequisites, checked before anything is built rather than after
# ---------------------------------------------------------------------------------------
say "checking prerequisites"
[[ -x "$JDK21/bin/java" ]]   || die "no JDK 21 at $JDK21 (set JDK21)"
[[ -d "$FLINK_SRC" ]]        || die "no flink checkout at $FLINK_SRC (set FLINK_SRC)"
[[ -d "$TORNADOVM_SRC" ]]    || die "no TornadoVM checkout at $TORNADOVM_SRC (set TORNADOVM_SRC)"
[[ -d "$JITLLM_SRC" ]]       || die "no jitllm checkout at $JITLLM_SRC (set JITLLM_SRC)"
[[ -f "$MODEL" ]]            || die "no model at $MODEL (set MODEL)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found; this needs an NVIDIA GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    GPU: /'

export JAVA_HOME="$JDK21"
mkdir -p "$WORK"

# The CUDA wrapper. CUDA 13.3 refuses GCC 16, and CUTLASS needs two more flags on top; the
# wrapper supplies both and points find_library at a toolkit that has cuBLAS and cuDNN. Without
# it TornadoVM's cudnn-jni and cutlass-jni fail at cmake's compiler-identification step, which
# reads like "cuDNN is not installed" and is not.
CUDA_WRAPPER="${CUDA_WRAPPER:-$HOME/.local/share/cuda-wrapper}"
if [[ -x "$CUDA_WRAPPER/bin/nvcc" ]]; then
    export PATH="$CUDA_WRAPPER/bin:$PATH" CUDA_PATH="$CUDA_WRAPPER"
    echo "    nvcc: $CUDA_WRAPPER/bin/nvcc (wrapper)"
else
    echo "    nvcc: $(command -v nvcc || echo 'not found') -- no wrapper at $CUDA_WRAPPER."
    echo "    If the TornadoVM build fails identifying the CUDA compiler, that is why."
fi

# ---------------------------------------------------------------------------------------
# 1. TornadoVM. Installs tornado-* into ~/.m2 and builds the SDK the cluster runs on.
# ---------------------------------------------------------------------------------------
if (( do_tornadovm )); then
    # `make BACKEND=cuda` runs `mvn clean install`: it deletes dist/ before it builds,
    # so a build that fails leaves no SDK at all and every demo stops working. On a
    # machine whose toolchain cannot complete that build -- gcc newer than nvcc
    # supports, a CUDA install without nvrtc.h -- this turns a working setup into a
    # broken one, and it has. So a usable SDK is never overwritten without being asked.
    existing="$(ls -d "$TORNADOVM_SRC"/dist/*/*/ 2>/dev/null | head -1)"
    if [[ -n "$existing" && -f "${existing}tornado-argfile" && -f "${existing}lib/libtornado-cudf.so" && -z "${FORCE_TORNADOVM:-}" ]]; then
        say "a usable TornadoVM SDK is already built -- keeping it"
        say "  rebuild deliberately with FORCE_TORNADOVM=1, or run $TORNADOVM_SRC/rebuild-for-demos.sh"
    else
        say "building TornadoVM (CUDA backend) -- several minutes"
        if [[ -x "$TORNADOVM_SRC/rebuild-for-demos.sh" ]]; then
            # Knows which modules this machine cannot build, and writes the argfile a
            # plain maven build leaves out.
            ( cd "$TORNADOVM_SRC" && ./rebuild-for-demos.sh )
        else
            ( cd "$TORNADOVM_SRC" && make BACKEND=cuda )
        fi
    fi
fi
TORNADOVM_HOME="$(ls -d "$TORNADOVM_SRC"/dist/*/*/ 2>/dev/null | head -1)"
[[ -n "$TORNADOVM_HOME" && -f "$TORNADOVM_HOME/tornado-argfile" ]] \
    || die "no TornadoVM SDK under $TORNADOVM_SRC/dist -- run without --skip-tornadovm"
TORNADOVM_HOME="${TORNADOVM_HOME%/}"
export TORNADOVM_HOME
echo "    SDK: $TORNADOVM_HOME"
TORNADO_VERSION="$(basename "$TORNADOVM_HOME" | sed 's/^tornadovm-//; s/-cuda$//')"
echo "    version: $TORNADO_VERSION"

# The backend must be CUDA. An OpenCL SDK on the PATH will run every test and every benchmark
# and report nothing wrong, on the wrong device.
"$TORNADOVM_HOME/bin/tornado" --devices 2>/dev/null | grep -q CUDADriver \
    || die "the TornadoVM SDK at $TORNADOVM_HOME is not reporting a CUDADriver"

# ---------------------------------------------------------------------------------------
# 2. jitllm, against that exact TornadoVM
# ---------------------------------------------------------------------------------------
if (( do_jitllm )); then
    say "building jitllm against TornadoVM ${TORNADO_VERSION%%-*}"
    ( cd "$JITLLM_SRC" && ./mvnw -q -Dtornadovm.base.version="${TORNADO_VERSION%%-*}" \
        -DskipTests -Dspotless.check.skip=true install )
fi
JITLLM_JAR="$(ls "$JITLLM_SRC"/target/jitllm-*.jar 2>/dev/null | grep -v original | head -1)"
[[ -n "$JITLLM_JAR" ]] || die "no jitllm jar in $JITLLM_SRC/target -- run without --skip-jitllm"
echo "    jar: $JITLLM_JAR"

# ---------------------------------------------------------------------------------------
# 3. llama.cpp with CUDA.
#
# CUDA 13 split cuBLAS out of the toolkit, so a 13.x install may have nvcc and no cublas_v2.h.
# When that is the case a shim prefix is assembled from NVIDIA's own wheels: the real toolkit
# symlinked in, plus cuBLAS, cuDNN and nvrtc beside it. The binary keeps an RPATH into the
# shim, so it needs no LD_LIBRARY_PATH afterwards -- but the shim must stay on disk.
# ---------------------------------------------------------------------------------------
if (( do_llamacpp )); then
    [[ -d "$LLAMACPP_SRC" ]] || die "no llama.cpp checkout at $LLAMACPP_SRC (set LLAMACPP_SRC)"
    say "building llama.cpp (CUDA)"
    CUDA_REAL="${CUDA_REAL:-/usr/local/cuda}"
    CMAKE_CUDA_ARGS=()
    if [[ -f "$CUDA_REAL/include/cublas_v2.h" || -f "$CUDA_REAL/targets/x86_64-linux/include/cublas_v2.h" ]]; then
        echo "    toolkit at $CUDA_REAL has cuBLAS; using it directly"
    else
        SHIM="$WORK/cuda-shim"
        echo "    $CUDA_REAL has no cuBLAS headers; assembling a shim at $SHIM"
        CUDA_MAJOR="$("$CUDA_REAL/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
        mkdir -p "$SHIM/targets/x86_64-linux/include" "$SHIM/targets/x86_64-linux/lib" "$SHIM/bin" "$WORK/whl"
        for f in "$CUDA_REAL"/bin/*;                                   do ln -sfn "$f" "$SHIM/bin/"; done
        for f in "$CUDA_REAL"/targets/x86_64-linux/include/*;          do ln -sfn "$f" "$SHIM/targets/x86_64-linux/include/"; done
        for f in "$CUDA_REAL"/targets/x86_64-linux/lib/*;              do ln -sfn "$f" "$SHIM/targets/x86_64-linux/lib/"; done
        ln -sfn "$CUDA_REAL/nvvm" "$SHIM/nvvm"          # nvcc finds cicc relative to itself
        ln -sfn targets/x86_64-linux/include "$SHIM/include"
        ln -sfn targets/x86_64-linux/lib     "$SHIM/lib64"
        for pkg in "nvidia-cublas" "nvidia-cuda-nvrtc"; do
            python3 -m pip download "$pkg" -d "$WORK/whl" --no-deps -q \
                || die "could not fetch $pkg; a complete CUDA toolkit would avoid this"
        done
        rm -rf "$WORK/wheels" && mkdir -p "$WORK/wheels"
        for w in "$WORK"/whl/nvidia_cublas-*.whl "$WORK"/whl/nvidia_cuda_nvrtc-*.whl; do
            [[ -f "$w" ]] && unzip -qo "$w" -d "$WORK/wheels"
        done
        for d in "$WORK"/wheels/nvidia/*/include; do
            [[ -d "$d" ]] && for f in "$d"/*; do ln -sfn "$f" "$SHIM/targets/x86_64-linux/include/"; done
        done
        for d in "$WORK"/wheels/nvidia/*/lib; do
            [[ -d "$d" ]] && for f in "$d"/*; do ln -sfn "$f" "$SHIM/targets/x86_64-linux/lib/"; done
        done
        # The wheels ship only the versioned sonames; the linker wants the bare .so beside them.
        for so in "$SHIM"/targets/x86_64-linux/lib/lib*.so."$CUDA_MAJOR"; do
            [[ -e "$so" ]] && ln -sfn "$so" "${so%.so.$CUDA_MAJOR}.so"
        done
        CMAKE_CUDA_ARGS=(-DCUDAToolkit_ROOT="$SHIM"
                         -DCMAKE_CUDA_COMPILER="$CUDA_REAL/bin/nvcc"
                         -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -I$SHIM/targets/x86_64-linux/include")
    fi
    ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}"
    ( cd "$LLAMACPP_SRC" \
      && cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
               -DLLAMA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES="$ARCH" "${CMAKE_CUDA_ARGS[@]}" \
      && cmake --build build --target llama-server llama-bench -j "$(nproc)" )
fi
LLAMA_SERVER="$LLAMACPP_SRC/build/bin/llama-server"
[[ -x "$LLAMA_SERVER" ]] || die "no llama-server at $LLAMA_SERVER -- run without --skip-llamacpp"

# ---------------------------------------------------------------------------------------
# 4. The provider and its example jars, against the Flink SPI already installed from FLINK_SRC
# ---------------------------------------------------------------------------------------
if (( do_flink )); then
    say "building the accelerator provider and the example jars"
    ( cd "$PROVIDER_SRC" && mvn -q -DskipTests -Dspotless.check.skip=true \
        -Drat.skip=true -Dcheckstyle.skip=true install )
fi
EXAMPLES="$(ls "$PROVIDER_SRC"/flink-accelerator-tornadovm-examples/target/*-GpuTriagePipeline.jar 2>/dev/null | head -1)"
[[ -n "$EXAMPLES" ]] || die "no example jars built -- run without --skip-flink"
JAR_PREFIX="${EXAMPLES%-GpuTriagePipeline.jar}"

# ---------------------------------------------------------------------------------------
# 5. Deploy into the distribution
# ---------------------------------------------------------------------------------------
[[ -d "$FLINK_HOME/bin" ]] || die "no Flink distribution at $FLINK_HOME (set FLINK_HOME)"
if (( do_deploy )); then
    say "deploying into $FLINK_HOME"
    "$PROVIDER_SRC/flink-accelerator-tornadovm/scripts/gpu-cluster-setup.sh"  "$FLINK_HOME"
    "$PROVIDER_SRC/flink-accelerator-tornadovm/scripts/llm-pipeline-setup.sh" "$FLINK_HOME" "$JITLLM_JAR"
    # The engine's device workspaces are direct buffers and Flink charges them to a budget it
    # sizes from its own memory model, which knows nothing about a model in the same JVM.
    python3 - "$FLINK_HOME/conf/config.yaml" <<'PY'
import pathlib, sys, re
p = pathlib.Path(sys.argv[1]); t = p.read_text()
t2 = re.sub(r"(task:\s*\n\s*off-heap:\s*\n\s*size: )\S+", r"\g<1>4g", t, count=1)
p.write_text(t2)
print("    taskmanager task off-heap set to 4g" if t2 != t else "    task off-heap already set")
PY
fi

# Hadoop on the classpath: the input is Parquet, and flink-sql-parquet bundles
# org.apache.parquet.* but none of org.apache.hadoop.*.
HADOOP_CP="${HADOOP_CLASSPATH:-}"
if [[ -z "$HADOOP_CP" ]] && command -v hadoop >/dev/null; then HADOOP_CP="$(hadoop classpath)"; fi
if [[ -z "$HADOOP_CP" ]]; then
    HADOOP_CP="$(find "$HOME/.m2/repository/org/apache/hadoop" "$HOME/.m2/repository/commons-"* \
                      "$HOME/.m2/repository/com/fasterxml/woodstox" "$HOME/.m2/repository/org/codehaus/woodstox" \
                      -name '*.jar' 2>/dev/null | paste -sd: || true)"
fi
[[ -n "$HADOOP_CP" ]] || die "no Hadoop jars found; set HADOOP_CLASSPATH before running this"

# ---------------------------------------------------------------------------------------
# 6. The environment file the run script sources
# ---------------------------------------------------------------------------------------
ENVFILE="$WORK/llm-bench.env"
cat > "$ENVFILE" <<ENV
# Written by llm-bench-setup.sh on $(date -Is). Sourced by llm-bench-run.sh.
export JAVA_HOME="$JDK21"
export TORNADOVM_HOME="$TORNADOVM_HOME"
export FLINK_HOME="$FLINK_HOME"
export HADOOP_CLASSPATH="$HADOOP_CP"
export LLM_BENCH_JAR_PREFIX="$JAR_PREFIX"
export LLM_BENCH_MODEL="$MODEL"
export LLM_BENCH_BINARY="$LLAMA_SERVER"
export LLM_BENCH_DATA="$WORK/readings"
export LLM_BENCH_WORK="$WORK"
ENV
say "wrote $ENVFILE"

# ---------------------------------------------------------------------------------------
# 7. The dataset, written once and read by both arms
# ---------------------------------------------------------------------------------------
if (( do_data )); then
    say "generating $ROWS rows into $WORK/readings"
    # shellcheck source=/dev/null
    source "$ENVFILE"
    "$FLINK_HOME/bin/stop-cluster.sh" >/dev/null 2>&1 || true
    sleep 2
    "$FLINK_HOME/bin/start-cluster.sh" >/dev/null
    until curl -s localhost:8081/overview 2>/dev/null | grep -q '"slots-available":1'; do sleep 2; done
    rm -rf "$WORK/readings"
    "$FLINK_HOME/bin/flink" run -c org.apache.flink.table.examples.java.gpu.llm.GpuTriagePipeline \
        "$JAR_PREFIX-GpuTriagePipeline.jar" --generate --rows "$ROWS" --machines "$MACHINES" \
        --data "$WORK/readings" 2>&1 | grep -E "writing|done in" || true
    "$FLINK_HOME/bin/stop-cluster.sh" >/dev/null
fi

say "setup complete"
cat <<DONE

    Run one experiment with:

        $PROVIDER_SRC/flink-accelerator-tornadovm/scripts/llm-bench-run.sh --gpu
        $PROVIDER_SRC/flink-accelerator-tornadovm/scripts/llm-bench-run.sh --cpu

DONE
