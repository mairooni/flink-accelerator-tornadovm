#!/usr/bin/env bash
#
# Step 3 of 3 -- set every path the demos need. SOURCE this, do not run it:
#
#     source ./3-env.sh
#
# On a machine that ran 1-fetch.sh this needs no arguments. On a machine that
# already has the three repositories somewhere else, put the overrides in
# demos/env.local.sh (uncommitted) and they win over everything below.
#
# Safe to source repeatedly.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "3-env.sh sets variables in the current shell, so it has to be sourced:" >&2
    echo "    source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

_demos_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- the four roots everything else is derived from ------------------------
: "${DEMO_ROOT:=$HOME/flink-tornadovm-demos}"
: "${TORNADOVM_SRC:=$DEMO_ROOT/TornadoVM}"
: "${FLINK_SRC:=$DEMO_ROOT/flink}"
: "${DATA_ROOT:=$DEMO_ROOT/data}"
: "${RAPIDS_HOME:=$HOME/.local/share/rapids-libcudf}"

# These scripts ship inside the provider repository, so when they are run from a
# clone they already know where it is.
if [[ -z "${PROVIDER_SRC:-}" && -d "$(dirname "$_demos_dir")/flink-accelerator-tornadovm/scripts" ]]; then
    PROVIDER_SRC="$(dirname "$_demos_dir")"
fi
: "${PROVIDER_SRC:=$DEMO_ROOT/flink-accelerator-tornadovm}"

# --- machine-specific overrides -------------------------------------------
# $DEMO_ROOT/env.sh is written by 1-fetch.sh; env.local.sh is hand-written on a
# machine whose checkouts live elsewhere. Both may reset any root above.
for _envf in "$DEMO_ROOT/env.sh" "$_demos_dir/env.local.sh"; do
    if [[ -f "$_envf" ]]; then source "$_envf"; fi
done

# --- the roots must be absolute -------------------------------------------
# A relative DEMO_ROOT (a missing leading slash is the usual way) resolves
# against whatever directory a script happens to cd into, so the repositories
# and the data end up somewhere nobody intended.
for _v in DEMO_ROOT TORNADOVM_SRC FLINK_SRC PROVIDER_SRC DATA_ROOT RAPIDS_HOME; do
    _p="${!_v}"
    if [[ -n "$_p" && "$_p" != /* ]]; then
        printf '\n\033[1;31mERROR: %s is a relative path: %s\033[0m\n' "$_v" "$_p" >&2
        printf '  Paths must be absolute -- a missing leading slash is the usual cause.\n' >&2
        printf '  Try:  export %s=%s/%s\n\n' "$_v" "$PWD" "$_p" >&2
        unset _v _p
        return 1 2>/dev/null || exit 1
    fi
done
unset _v _p

# --- JDK 21 ---------------------------------------------------------------
# TornadoVM's off-heap arrays are java.lang.foreign, preview on 21, so the
# cluster must run the JDK the SDK was built against -- not merely "a JDK 21".
if [[ -z "${JAVA_HOME:-}" || ! -x "${JAVA_HOME}/bin/javac" ]] \
   || ! "${JAVA_HOME}/bin/javac" -version 2>&1 | grep -q " 21"; then
    for _c in "$HOME/Projects/JDKs/jdk-21.0.3" /usr/lib/jvm/java-21-openjdk /usr/lib/jvm/jdk-21 \
              "$(dirname "$(dirname "$(readlink -f "$(command -v javac 2>/dev/null)")")")"; do
        if [[ -x "$_c/bin/javac" ]] && "$_c/bin/javac" -version 2>&1 | grep -q " 21"; then
            export JAVA_HOME="$_c"; break
        fi
    done
fi

# --- the built TornadoVM SDK ----------------------------------------------
# Resolved from TORNADOVM_SRC rather than inherited: a machine with SDKMAN on
# the PATH already exports TORNADOVM_HOME, and that install loads, reports no
# CUDA device, and sends every demo quietly down the CPU path.
# A plain `ls ... | head` here is fatal to a caller running set -euo pipefail
# when dist/ does not exist yet, which is exactly the state before 1-fetch.sh
# has built anything. Glob instead, and never return non-zero.
_sdk=""
for _d in "$TORNADOVM_SRC"/dist/*/*/; do
    if [[ -d "$_d" ]]; then _sdk="${_d%/}"; break; fi
done
if [[ -n "$_sdk" ]]; then export TORNADOVM_HOME="$_sdk"; fi

export FLINK_HOME="${FLINK_HOME:-$FLINK_SRC/flink-dist/target/flink-2.3.0-bin/flink-2.3.0}"
export DEMO_ROOT TORNADOVM_SRC FLINK_SRC PROVIDER_SRC DATA_ROOT RAPIDS_HOME

# --- RAPIDS ---------------------------------------------------------------
# Six directories, not three. The last is auditwheel's side-car and holds the
# only copy of libzstd; omit any one and cuDF reports itself unavailable with no
# diagnostic, which reads exactly like a shim that was never built.
if [[ -d "$RAPIDS_HOME" && ":${LD_LIBRARY_PATH:-}:" != *":$RAPIDS_HOME/libcudf/lib64:"* ]]; then
    export LD_LIBRARY_PATH="$RAPIDS_HOME/libcudf/lib64:$RAPIDS_HOME/librmm/lib64:\
$RAPIDS_HOME/libkvikio/lib64:$RAPIDS_HOME/rapids_logger/lib64:\
$RAPIDS_HOME/nvidia/libnvcomp/lib64:$RAPIDS_HOME/libkvikio_cu12.libs:${LD_LIBRARY_PATH:-}"
fi
export CUDF_HOME="${CUDF_HOME:-$RAPIDS_HOME/libcudf}"
export RMM_HOME="${RMM_HOME:-$RAPIDS_HOME/librmm}"

# --- report ---------------------------------------------------------------
_env_ok=1
_env_row() { local n="$1" p="$2"; if [[ -n "$p" && -e "$p" ]]; then printf '  %-14s %s\n' "$n" "$p"; else printf '  %-14s %s   <-- MISSING\n' "$n" "${p:-<unset>}"; _env_ok=0; fi; }
printf '\n\033[1;36m==> demo environment\033[0m\n'
_env_row JAVA_HOME "${JAVA_HOME:-}"
_env_row TORNADOVM "${TORNADOVM_HOME:-}"
_env_row FLINK_HOME "${FLINK_HOME:-}"
_env_row PROVIDER "${PROVIDER_SRC:-}"
_env_row DATA_ROOT "${DATA_ROOT:-}"
_env_row RAPIDS "${RAPIDS_HOME:-}"
if [[ -n "${TORNADOVM_HOME:-}" && -z "$(find "$TORNADOVM_HOME" -name 'libtornado-cudf.so' -print -quit 2>/dev/null)" ]]; then
    printf '  %-14s %s\n' "cuDF shim" "MISSING from the SDK -- demos 2 and 3 will run on the CPU"
    _env_ok=0
fi
if [[ $_env_ok == 1 ]]; then
    printf '\n  ready. run:  ./demo-haversine.sh --print-kernel | ./demo-regex.sh --patterns 8 | ./demo-llm.sh\n\n'
else
    printf '\n  something is missing above. On a fresh machine run ./1-fetch.sh then ./2-generate-data.sh;\n'
    printf '  on a machine with the repos elsewhere, set them in demos/env.local.sh\n\n'
fi
unset _demos_dir _envf _c _sdk _env_ok

# Never hand a non-zero status back: this file is sourced by scripts that run
# under set -e, and the last test above may legitimately be false.
return 0 2>/dev/null || true
