#!/usr/bin/env bash
#
# Demo 1 -- a Flink SQL expression compiled to a CUDA kernel.
#
#   ./demo-haversine.sh                 2,000,000 points
#   ./demo-haversine.sh --rows 8000000
#   ./demo-haversine.sh --print-kernel     the generated CUDA, in the TaskManager .out
#   ./demo-haversine.sh --print-bytecodes  the TornadoVM bytecodes, same place
#   ./demo-haversine.sh --cpu           the same query with the accelerator off
#   ./demo-haversine.sh --keep-cluster  leave the cluster up afterwards, so the
#                                       web UI at http://localhost:8081 still
#                                       shows the finished job
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh
[[ -f "$DEMO_ROOT/env.sh" ]] && source "$DEMO_ROOT/env.sh" && source ./common.sh

ROWS=8000000; PRINT_KERNEL=0; PRINT_BYTECODES=0; ARM=device; KEEP=0
while [[ $# -gt 0 ]]; do case "$1" in
    --rows) ROWS="$2"; shift 2 ;;
    --print-kernel) PRINT_KERNEL=1; shift ;;
    --print-bytecodes) PRINT_BYTECODES=1; shift ;;
    --cpu) ARM=cpu; shift ;;
    --keep-cluster) KEEP=1; shift ;;
    *) demo_die "unknown argument $1" ;;
esac; done

JAR="$PROVIDER_SRC/flink-accelerator-tornadovm-examples/target/HaversineSQLExample.jar"
[[ -f "$JAR" ]] || demo_die "no example jar at $JAR -- run ./1-fetch.sh"
DATA="$DATA_ROOT/haversine-$ROWS"

export HADOOP_CLASSPATH="$(hadoop_classpath)"
OPTS=""
[[ $ARM == cpu ]] && OPTS="-Dtable.exec.accelerator.enabled=false"
TM_FLAGS=""
[[ $PRINT_KERNEL == 1 ]]    && TM_FLAGS="$TM_FLAGS -Dtornado.printKernel=true"
[[ $PRINT_BYTECODES == 1 ]] && TM_FLAGS="$TM_FLAGS -Dtornado.print.bytecodes=true"
tm_opts_add "$TM_FLAGS"

demo_banner "Haversine -- $ROWS points, ${ARM}"
cat <<'TXT'
  SELECT COUNT(*), MIN(nearest_km), MAX(nearest_km)
  FROM (SELECT LEAST(<20 haversine distances>) AS nearest_km FROM Points)

  Ordinary Flink SQL. No hint, no annotation, nothing that names a device.
  The projection -- about 360 weighted operations a row -- is lowered to the
  accelerator IR and compiled to a CUDA kernel at job start.
TXT
cluster_up
trap '[[ $KEEP == 1 ]] || cluster_down; tm_opts_restore' EXIT
JVM_ARGS="$OPTS" "$FLINK_HOME/bin/flink" run -c org.apache.flink.table.examples.java.gpu.HaversineSQLExample \
    "$JAR" "$ROWS" "$DATA" 2>&1 | grep -vE "^SLF4J|^WARNING"

if [[ -n "$TM_FLAGS" ]]; then
    echo
    demo_banner "what TornadoVM printed"
    echo "  TaskManager stdout: $(tm_out)"
    [[ $PRINT_KERNEL == 1 ]]    && echo "    the generated CUDA -- nothing in this repository contains it;"
    [[ $PRINT_KERNEL == 1 ]]    && echo "    the constants are the depot coordinates in radians"
    [[ $PRINT_BYTECODES == 1 ]] && echo "    the TornadoVM bytecodes -- ALLOC / TRANSFER / LAUNCH per task"
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
