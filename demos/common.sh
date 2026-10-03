#!/usr/bin/env bash
#
# Helpers shared by the demo scripts. Sourced, never run directly.
#
# The paths themselves live in 3-env.sh, which is also the thing a person
# sources by hand to get a shell the demos run in.

_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The report is noise inside a script, but swallowing it outright turns any
# failure in there into a script that exits silently, which is how a bug in it
# cost an afternoon. Keep it quiet when it works, show everything when it does
# not.
_env_out="$(source "$_common_dir/3-env.sh" 2>&1)" || {
    printf '%s\n' "$_env_out" >&2
    printf '\n\033[1;31mERROR: 3-env.sh failed -- see above\033[0m\n' >&2
    exit 1
}
# shellcheck source=/dev/null
source "$_common_dir/3-env.sh" > /dev/null
unset _common_dir _env_out

# --- Hadoop ----------------------------------------------------------------
# flink-sql-parquet bundles org.apache.parquet.* and none of org.apache.hadoop.*,
# so a Parquet table needs a real Hadoop client on the classpath -- one
# consistent version of it. A find over ~/.m2 picks up two Hadoop versions and
# three woodstox builds and fails at run time with a NoSuchMethodError, so the
# classpath is resolved by Maven once and cached.
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


# --- TaskManager JVM options ----------------------------------------------
# tornado.printKernel and tornado.print.bytecodes are read on the TASKMANAGER,
# into static finals at class init. Setting them in JVM_ARGS reaches only the
# client, which compiles nothing -- so they have to go into config.yaml before
# the cluster starts. Added for the life of one run and always restored.
_TM_CFG_BAK=""
tm_opts_add() {
    local flags="$1"
    local cfg="$FLINK_HOME/conf/config.yaml"
    [[ -n "$flags" ]] || return 0
    _TM_CFG_BAK="$(mktemp)"
    cp "$cfg" "$_TM_CFG_BAK"
    python3 - "$cfg" "$flags" <<'PYCFG'
import re, sys
p, flags = sys.argv[1], sys.argv[2]
s = open(p).read()
# config.yaml is nested YAML -- env: / java: / opts: / all: -- so the key is
# matched by indent, not as a flat env.java.opts.all:.
m = re.search(r'^(\s+)all:[ \t]*(.*)$', s, re.M)
if m is None:
    sys.exit("could not find env.java.opts.all in " + p)
add = " ".join(f for f in flags.split() if f not in m.group(2))
if add:
    s = s[:m.start()] + f"{m.group(1)}all: {m.group(2)} {add}" + s[m.end():]
    open(p, "w").write(s)
PYCFG
}
tm_opts_restore() {
    if [[ -n "${_TM_CFG_BAK:-}" && -f "$_TM_CFG_BAK" ]]; then
        mv -f "$_TM_CFG_BAK" "$FLINK_HOME/conf/config.yaml"
        _TM_CFG_BAK=""
    fi
}

# Where TornadoVM writes kernels and bytecodes: a TaskManager's stdout, which
# is its .out file, not the .log everyone opens first.
tm_out() { ls -t "$FLINK_HOME"/log/*taskexecutor*.out 2>/dev/null | head -1; }

demo_banner() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
demo_die()    { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

cluster_up() {
    "$FLINK_HOME/bin/stop-cluster.sh" >/dev/null 2>&1 || true
    "$FLINK_HOME/bin/start-cluster.sh" >/dev/null 2>&1
    sleep 6
}
cluster_down() { "$FLINK_HOME/bin/stop-cluster.sh" >/dev/null 2>&1 || true; }
