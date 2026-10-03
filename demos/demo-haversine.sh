#!/usr/bin/env bash
#
# Demo 1 -- a Flink SQL expression compiled to a CUDA kernel.
#
#   ./demo-haversine.sh                 2,000,000 points
#   ./demo-haversine.sh --rows 8000000
#   ./demo-haversine.sh --print-kernel  show the CUDA that was generated
#   ./demo-haversine.sh --cpu           the same query with the accelerator off
#   ./demo-haversine.sh --keep-cluster  leave the cluster up afterwards, so the
#                                       web UI at http://localhost:8081 still
#                                       shows the finished job
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh
[[ -f "$DEMO_ROOT/env.sh" ]] && source "$DEMO_ROOT/env.sh" && source ./common.sh

ROWS=8000000; PRINT_KERNEL=0; ARM=device; KEEP=0
while [[ $# -gt 0 ]]; do case "$1" in
    --rows) ROWS="$2"; shift 2 ;;
    --print-kernel) PRINT_KERNEL=1; shift ;;
    --cpu) ARM=cpu; shift ;;
    --keep-cluster) KEEP=1; shift ;;
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
trap '[[ $KEEP == 1 ]] || cluster_down; restore_config' EXIT
JVM_ARGS="$OPTS" "$FLINK_HOME/bin/flink" run -c org.apache.flink.table.examples.java.gpu.HaversineSQLExample \
    "$JAR" "$ROWS" "$DATA" 2>&1 | grep -vE "^SLF4J|^WARNING"

if [[ $PRINT_KERNEL == 1 ]]; then
    echo
    demo_banner "the generated CUDA"
    echo "  TornadoVM wrote it to the TaskManager's stdout:"
    echo "    $(ls -t "$FLINK_HOME"/log/*taskexecutor*.out 2>/dev/null | head -1)"
    echo "  Nothing in this repository contains that text -- it is compiled from the SQL"
    echo "  at job start. The constants are the depot coordinates in radians."
fi

echo
demo_banner "where it ran"
# The log line carries the provider's "claims NNx over CPU" estimate. That is a
# static model used to decide whether to offload at all -- it is not measured
# here and it is nothing like the end-to-end speedup, so it is cut rather than
# shown next to a result it does not describe.
grep -hE "Accelerated on this TaskManager|Accelerator declined" "$FLINK_HOME"/log/*taskexecutor*.log \
    | tail -1 | sed 's/^.*\] - //; s/: provider \([a-z]*\) claims.*/  (provider: \1)/' \
    || echo "  (no accelerator decision logged -- it ran on the CPU)"

if [[ $KEEP == 1 ]]; then
    echo
    echo "  cluster left running: http://localhost:8081  (stop it with $FLINK_HOME/bin/stop-cluster.sh)"
fi
