#!/usr/bin/env bash
#
# Demo 2 -- a CUDA library serving a whole SQL subtree.
#
#   ./demo-regex.sh                    16M lines, 8 patterns
#   ./demo-regex.sh --rows 64 --patterns 8
#   ./demo-regex.sh --patterns 1       below the floor: the region declines
#   ./demo-regex.sh --cpu              the same query with the region off
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh
[[ -f "$DEMO_ROOT/env.sh" ]] && source "$DEMO_ROOT/env.sh" && source ./common.sh

ROWS=16; PATTERNS=8; ARM=device; PAR=1; KEEP=0; REUSE=0; PRINT_KERNEL=0; PRINT_BYTECODES=0
while [[ $# -gt 0 ]]; do case "$1" in
    --rows) ROWS="$2"; shift 2 ;;
    --patterns) PATTERNS="$2"; shift 2 ;;
    --parallelism) PAR="$2"; shift 2 ;;
    --selective) SELECTIVE=--selective; shift ;;
    --cpu) ARM=cpu; shift ;;
    --keep-cluster) KEEP=1; shift ;;
    --reuse-cluster) REUSE=1; KEEP=1; shift ;;
    --print-kernel) PRINT_KERNEL=1; shift ;;
    --print-bytecodes) PRINT_BYTECODES=1; shift ;;
    *) demo_die "unknown argument $1" ;;
esac; done

JAR="$PROVIDER_SRC/flink-accelerator-tornadovm-examples/target/flink-accelerator-tornadovm-examples-0.1.0-SNAPSHOT-GrokSQLExample.jar"
[[ -f "$JAR" ]] || demo_die "no example jar at $JAR -- run ./1-fetch.sh"
DATA="$DATA_ROOT/logs$ROWS"
[[ -d "$DATA" ]] || demo_die "no corpus at $DATA
  Generate it with:  ./2-generate-data.sh --rows ${ROWS}000000
  (the corpora already generated are: $(ls -d "$DATA_ROOT"/logs* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '))"

export HADOOP_CLASSPATH="$(hadoop_classpath)"
OPTS=""
[[ $ARM == cpu ]] && OPTS="-Dflink.accelerator.device-scan=off"

demo_banner "log screening -- ${ROWS}M lines, $PATTERNS patterns, arm=$ARM"
cat <<'TXT'
  SELECT COUNT(*) FROM Logs
  WHERE REGEXP(line,'p1') AND REGEXP(line,'p2') AND ... AND REGEXP(line,'pk')

  One device region: cuDF reads the string column into TornadoVM buffers, k
  regexes match the same device memory, and a generated kernel folds the masks.
  The pattern count is the lever -- at one pattern the region declines, because
  the matching is a rounding error beside the read.
TXT
TM_FLAGS=""
[[ $PRINT_KERNEL == 1 ]]    && TM_FLAGS="$TM_FLAGS -Dtornado.printKernel=true"
[[ $PRINT_BYTECODES == 1 ]] && TM_FLAGS="$TM_FLAGS -Dtornado.print.bytecodes=true"
if [[ -n "$TM_FLAGS" && "$REUSE" == 1 ]] && cluster_running; then
    demo_die "--reuse-cluster cannot be combined with the print flags: those are
  TaskManager JVM options, read only when the JVM starts. Drop --reuse-cluster,
  or run ./stop.sh first so a fresh cluster picks them up."
fi
tm_opts_add "$TM_FLAGS"

cluster_up
trap '[[ ${KEEP:-0} == 1 ]] || cluster_down; tm_opts_restore' EXIT
JVM_ARGS="$OPTS" "$FLINK_HOME/bin/flink" run -c org.apache.flink.table.examples.java.gpu.GrokSQLExample \
    "$JAR" --data "$DATA" --patterns "$PATTERNS" --parallelism "$PAR" ${SELECTIVE:-} 2>&1 \
    | grep -vE "^SLF4J|^WARNING"

echo
demo_banner "what the planner and the region decided"
grep -hE "Accelerator reads the strings|declining the grok|not a grok" "$FLINK_HOME"/log/*client*.log \
    | tail -1 | sed 's/^.*\] - //' || echo "  (the region was not selected -- the CPU plan ran)"
grep -hE "grok region:" "$FLINK_HOME"/log/*taskexecutor*.log | tail -1 | sed 's/^.*\] - //' || true

if [[ ${KEEP:-0} == 1 ]]; then
    echo
    echo "  cluster left running: http://localhost:8081  (stop it with $FLINK_HOME/bin/stop-cluster.sh)"
fi

if [[ -n "${TM_FLAGS:-}" ]]; then
    echo
    demo_banner "what TornadoVM printed"
    echo "  TaskManager stdout: $(tm_out)"
    if [[ $PRINT_BYTECODES == 1 ]]; then
        echo "    the bytecodes show both kinds of node in one task graph:"
        echo "      LAUNCH task - grok.read[readParquetStrings]   library task"
        echo "      LAUNCH task - grok.reN[containsRe]            library task, once per pattern"
        echo "      LAUNCH task grok.combine - combine            generated kernel"
        echo "    grep for them with:  grep -E 'LAUNCH|ALLOC|TRANSFER' \"\$(tm_out)\""
    fi
fi
