#!/usr/bin/env bash
#
# Shared environment for the three demos. Sourced by setup.sh and by each demo
# script; never run directly.
#
# Everything a demo needs is resolved here from four roots, so a different
# laptop only has to set those (or accept the defaults and let setup.sh clone
# into them).

: "${DEMO_ROOT:=$HOME/flink-tornadovm-demos}"     # checkouts and datasets live here
: "${TORNADOVM_SRC:=$DEMO_ROOT/TornadoVM}"
: "${FLINK_SRC:=$DEMO_ROOT/flink}"
: "${PROVIDER_SRC:=$DEMO_ROOT/flink-accelerator-tornadovm}"
: "${DATA_ROOT:=$DEMO_ROOT/data}"
: "${RAPIDS_HOME:=$HOME/.local/share/rapids-libcudf}"

# --- JDK 21 ----------------------------------------------------------------
# TornadoVM's off-heap arrays are java.lang.foreign, preview on 21, so the
# cluster must run the JDK the SDK was built against. Not "a JDK 21" -- the
# same one.
if [[ -z "${JAVA_HOME:-}" || ! -x "${JAVA_HOME}/bin/javac" ]]; then
    for candidate in "$HOME/Projects/JDKs/jdk-21.0.3" /usr/lib/jvm/java-21-openjdk \
                     /usr/lib/jvm/jdk-21 "$(dirname "$(dirname "$(readlink -f "$(command -v javac 2>/dev/null)")")")"; do
        if [[ -x "$candidate/bin/javac" ]] && "$candidate/bin/javac" -version 2>&1 | grep -q " 21"; then
            export JAVA_HOME="$candidate"; break
        fi
    done
fi

# --- the built TornadoVM SDK ------------------------------------------------
if [[ -z "${TORNADOVM_HOME:-}" ]]; then
    TORNADOVM_HOME="$(ls -d "$TORNADOVM_SRC"/dist/*/*/ 2>/dev/null | head -1)"
    TORNADOVM_HOME="${TORNADOVM_HOME%/}"
fi
export TORNADOVM_HOME

export FLINK_HOME="${FLINK_HOME:-$FLINK_SRC/flink-dist/target/flink-2.3.0-bin/flink-2.3.0}"

# --- RAPIDS ----------------------------------------------------------------
# Six directories, not three. The last is auditwheel's side-car and holds the
# only copy of libzstd; omit any one and CudfLibraryProvider.isAvailable()
# returns false with no diagnostic, which reads exactly like a shim that was
# never built.
if [[ -d "$RAPIDS_HOME" ]]; then
    export LD_LIBRARY_PATH="$RAPIDS_HOME/libcudf/lib64:$RAPIDS_HOME/librmm/lib64:\
$RAPIDS_HOME/libkvikio/lib64:$RAPIDS_HOME/rapids_logger/lib64:\
$RAPIDS_HOME/nvidia/libnvcomp/lib64:$RAPIDS_HOME/libkvikio_cu12.libs:${LD_LIBRARY_PATH:-}"
    export CUDF_HOME="$RAPIDS_HOME/libcudf"
    export RMM_HOME="$RAPIDS_HOME/librmm"
fi

# --- Hadoop ----------------------------------------------------------------
# flink-sql-parquet bundles org.apache.parquet.* and none of org.apache.hadoop.*,
# so a Parquet table needs a real Hadoop client on the classpath -- one
# consistent version of it. A find over ~/.m2 picks up two Hadoop versions and
# three woodstox builds and fails at run time with a NoSuchMethodError, so the
# classpath is resolved by Maven and cached.
hadoop_classpath() {
    local cache="$DEMO_ROOT/.hadoop-classpath"
    if [[ -s "$cache" ]]; then cat "$cache"; return; fi
    local tmp; tmp="$(mktemp -d)"
    cat > "$tmp/pom.xml" <<'POM'
<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion>
  <groupId>local</groupId><artifactId>hadoopcp</artifactId><version>1</version>
  <dependencies><dependency><groupId>org.apache.hadoop</groupId>
    <artifactId>hadoop-client</artifactId><version>3.3.4</version></dependency></dependencies>
</project>
POM
    (cd "$tmp" && mvn -q dependency:build-classpath -Dmdep.outputFile=cp.txt >/dev/null 2>&1)
    mkdir -p "$DEMO_ROOT"; cp "$tmp/cp.txt" "$cache"; rm -rf "$tmp"; cat "$cache"
}

demo_banner() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
demo_die()    { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

cluster_up() {
    "$FLINK_HOME/bin/stop-cluster.sh" >/dev/null 2>&1 || true
    "$FLINK_HOME/bin/start-cluster.sh" >/dev/null 2>&1
    sleep 6
}
cluster_down() { "$FLINK_HOME/bin/stop-cluster.sh" >/dev/null 2>&1 || true; }
