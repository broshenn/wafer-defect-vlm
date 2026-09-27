#!/usr/bin/env bash
# Exit (and so wake the caller) as soon as a named report appears and is non-empty.
set -uo pipefail
DIR=/root/autodl-fs/wafer-vlm/outputs/reports
REP="${1:?usage: watch_report.sh <report-basename>}"
for i in $(seq 1 480); do
  if [ -s "$DIR/$REP" ]; then
    echo "APPEARED after ~$((i-1)) min: $REP"
    exit 0
  fi
  sleep 60
done
echo "TIMEOUT after 8h: $REP never appeared"
exit 0
