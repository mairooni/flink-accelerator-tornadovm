#!/usr/bin/env bash
#
# Stop a cluster left running by --keep-cluster.
#
#   ./stop.sh
#
# The demos stop the cluster themselves unless --keep-cluster was passed, so
# this is only needed after that.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

if curl -s -m 2 "http://localhost:8081/overview" >/dev/null 2>&1; then
    demo_banner "stopping the cluster"
    "$FLINK_HOME/bin/stop-cluster.sh"
else
    demo_banner "nothing to stop"
    echo "  no cluster answering on http://localhost:8081"
    # Stop anyway: the UI may be bound elsewhere, or mid-shutdown.
    "$FLINK_HOME/bin/stop-cluster.sh" >/dev/null 2>&1 || true
fi
