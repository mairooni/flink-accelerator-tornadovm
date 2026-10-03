#!/usr/bin/env bash
#
# Step 2 of 3 -- generate the corpora the demos read.
#
#   ./2-generate-data.sh                 the log corpus (16M and 64M lines)
#   ./2-generate-data.sh --with-haversine  also materialise the 8M and 32M point CSVs
#   ./2-generate-data.sh --rows 16000000,64000000   pick the log sizes
#
# Idempotent: a corpus that is already there is left alone. The log corpus is
# about 2.4 GB of Parquet and takes a few minutes; keep DATA_ROOT on a real
# disk, not on /tmp, which is tmpfs on most distributions and will eat RAM.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

LOGS="16000000,64000000"; WITH_HAVERSINE=0
while [[ $# -gt 0 ]]; do case "$1" in
    --rows) LOGS="$2"; shift 2 ;;
    --with-haversine) WITH_HAVERSINE=1; shift ;;
    *) demo_die "unknown argument $1" ;;
esac; done

[[ -n "${JAVA_HOME:-}" ]]    || demo_die "no JDK 21 found. Run ./1-fetch.sh first, or set JAVA_HOME."
[[ -d "${FLINK_HOME:-}" ]]   || demo_die "no Flink distribution at ${FLINK_HOME:-<unset>}. Run ./1-fetch.sh first."
mkdir -p "$DATA_ROOT"

demo_banner "log corpus -> $DATA_ROOT"
export HADOOP_CLASSPATH="$(hadoop_classpath)"
EX="$PROVIDER_SRC/flink-accelerator-tornadovm-examples/target"
# The examples module shades one jar per main class, so the generator is in its
# own, not in GrokSQLExample's.
JAR="$EX/flink-accelerator-tornadovm-examples-0.1.0-SNAPSHOT-LogCorpusGenerator.jar"
[[ -f "$JAR" ]] || demo_die "missing $(basename "$JAR") in $EX -- run ./1-fetch.sh first"
unzip -l "$JAR" 2>/dev/null | grep -q 'LogCorpusGenerator\.class' \
    || demo_die "$(basename "$JAR") does not contain LogCorpusGenerator -- rebuild with ./1-fetch.sh"
CP="$(ls "$FLINK_HOME"/lib/*.jar | tr '\n' ':')"

TODO=()
for n in ${LOGS//,/ }; do
    tag="logs$(( n / 1000000 ))"
    if compgen -G "$DATA_ROOT/$tag/*.parquet" > /dev/null; then
        echo "  $tag already at $DATA_ROOT/$tag"
    else
        TODO+=("$n")
    fi
done
if [[ ${#TODO[@]} -gt 0 ]]; then
    echo "  generating: ${TODO[*]}"
    "$JAVA_HOME/bin/java" -Xmx12g -cp "$CP$JAR:$HADOOP_CLASSPATH" \
        org.apache.flink.table.examples.java.gpu.LogCorpusGenerator "$DATA_ROOT" "${TODO[@]}"
fi

# The haversine example writes its own CSV on first run, so this is only worth
# doing up front to keep a live demo from pausing to generate 1.4 GB.
if [[ $WITH_HAVERSINE == 1 ]]; then
    demo_banner "haversine corpora -> $DATA_ROOT"
    for n in 8000000 32000000; do
        if compgen -G "$DATA_ROOT/haversine-$n/*.csv" > /dev/null; then
            echo "  haversine-$n already present"
        else
            echo "  generating haversine-$n (runs the demo once)"
            ./demo-haversine.sh --rows "$n" > /dev/null 2>&1 || demo_die "haversine generation failed for $n rows"
        fi
    done
fi

demo_banner "data ready"
du -sh "$DATA_ROOT"/* 2>/dev/null | sed 's/^/  /' || true
echo
echo "  next:  source ./3-env.sh"
