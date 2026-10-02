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

ROWS=2000000; PRINT_KERNEL=0; ARM=device
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
[[ $PRINT_KERNEL == 1 ]] && OPTS="$OPTS -Dtornado.printKernel=true"

demo_banner "Haversine -- $ROWS points, ${ARM}"
cat <<'TXT'
  SELECT COUNT(*), MIN(nearest_km), MAX(nearest_km)
  FROM (SELECT LEAST(<20 haversine distances>) AS nearest_km FROM Points)

  Ordinary Flink SQL. No hint, no annotation, nothing that names a device.
  The projection -- about 360 weighted operations a row -- is lowered to the
  accelerator IR and compiled to a CUDA kernel at job start.
TXT
cluster_up
trap cluster_down EXIT
JVM_ARGS="$OPTS" "$FLINK_HOME/bin/flink" run -c org.apache.flink.table.examples.java.gpu.HaversineSQLExample \
    "$JAR" "$ROWS" "$DATA" 2>&1 | grep -vE "^SLF4J|^WARNING"

echo
demo_banner "where it ran"
grep -hE "Accelerated on this TaskManager|Accelerator declined" "$FLINK_HOME"/log/*taskexecutor*.log \
    | tail -1 | sed 's/^.*\] - //' || echo "  (no accelerator decision logged -- it ran on the CPU)"
