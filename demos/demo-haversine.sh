#!/usr/bin/env bash
#
# Demo 1 -- a Flink SQL expression compiled to a CUDA kernel.
#
#   ./demo-haversine.sh                 2,000,000 points
#   ./demo-haversine.sh --rows 8000000
#   ./demo-haversine.sh --print-kernel  show the CUDA that was generated
#   ./demo-haversine.sh --cpu           the same query with the accelerator off
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh
[[ -f "$DEMO_ROOT/env.sh" ]] && source "$DEMO_ROOT/env.sh" && source ./common.sh

ROWS=8000000; PRINT_KERNEL=0; ARM=device
while [[ $# -gt 0 ]]; do case "$1" in
    --rows) ROWS="$2"; shift 2 ;;
    --print-kernel) PRINT_KERNEL=1; shift ;;
    --cpu) ARM=cpu; shift ;;
    *) demo_die "unknown argument $1" ;;
esac; done

JAR="$PROVIDER_SRC/flink-accelerator-tornadovm-examples/target/HaversineSQLExample.jar"
[[ -f "$JAR" ]] || demo_die "no example jar at $JAR -- run ./setup.sh"
DATA="$DATA_ROOT/haversine-$ROWS"

export HADOOP_CLASSPATH="$(hadoop_classpath)"
OPTS=""
[[ $ARM == cpu ]] && OPTS="-Dtable.exec.accelerator.enabled=false"
# tornado.printKernel must be set on the TASKMANAGER: that is where the kernel
# is compiled, and TornadoVM reads the property into a static final at class
# init. Putting it in JVM_ARGS sets it on the client, which compiles nothing --
# so --print-kernel used to print nothing at all. config.yaml is edited for the
# life of this run and restored on every exit path.
restore_config() {
    if [[ -n "${_CFG_BAK:-}" && -f "$_CFG_BAK" ]]; then
        mv -f "$_CFG_BAK" "$FLINK_HOME/conf/config.yaml"
    fi
}
if [[ $PRINT_KERNEL == 1 ]]; then
    _CFG="$FLINK_HOME/conf/config.yaml"
    _CFG_BAK="$(mktemp)"
    cp "$_CFG" "$_CFG_BAK"
    python3 - "$_CFG" <<'PYCFG'
import re, sys
p = sys.argv[1]; s = open(p).read()
# config.yaml is nested YAML -- env: / java: / opts: / all: -- not the flat
# env.java.opts.all: form, so the key has to be matched by indent.
m = re.search(r'^(\s+)all:[ \t]*(.*)$', s, re.M)
if m is None:
    sys.exit("could not find env.java.opts.all in " + p)
if 'tornado.printKernel' not in m.group(2):
    s = s[:m.start()] + f"{m.group(1)}all: {m.group(2)} -Dtornado.printKernel=true" + s[m.end():]
    open(p, 'w').write(s)
PYCFG
fi

demo_banner "Haversine -- $ROWS points, ${ARM}"
cat <<'TXT'
  SELECT COUNT(*), MIN(nearest_km), MAX(nearest_km)
  FROM (SELECT LEAST(<20 haversine distances>) AS nearest_km FROM Points)

  Ordinary Flink SQL. No hint, no annotation, nothing that names a device.
  The projection -- about 360 weighted operations a row -- is lowered to the
  accelerator IR and compiled to a CUDA kernel at job start.
TXT
cluster_up
trap 'cluster_down; restore_config' EXIT
JVM_ARGS="$OPTS" "$FLINK_HOME/bin/flink" run -c org.apache.flink.table.examples.java.gpu.HaversineSQLExample \
    "$JAR" "$ROWS" "$DATA" 2>&1 | grep -vE "^SLF4J|^WARNING"

if [[ $PRINT_KERNEL == 1 ]]; then
    echo
    demo_banner "the generated CUDA"
    # TornadoVM writes the kernel to stdout, which for a TaskManager is its .out
    # file, not the .log everyone looks in first.
    _out="$(ls -t "$FLINK_HOME"/log/*taskexecutor*.out 2>/dev/null | head -1)"
    if [[ -n "$_out" ]] && grep -qE "__kernel|__global__|\.visible \.entry" "$_out"; then
        sed -n '/__kernel\|__global__\|\.visible \.entry/,$p' "$_out" | head -60
        echo "  ... full text in $_out"
    else
        echo "  nothing in ${_out:-the TaskManager .out} -- the kernel may have been served"
        echo "  from TornadoVM's on-disk code cache. Clear it and re-run:"
        echo "    rm -rf ~/.tornadovm/  (or \$TORNADO_SDK/var) and ./demo-haversine.sh --print-kernel"
    fi
fi

echo
demo_banner "where it ran"
grep -hE "Accelerated on this TaskManager|Accelerator declined" "$FLINK_HOME"/log/*taskexecutor*.log \
    | tail -1 | sed 's/^.*\] - //' || echo "  (no accelerator decision logged -- it ran on the CPU)"
