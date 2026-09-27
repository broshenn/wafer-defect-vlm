#!/usr/bin/env bash
# 44 and the pass behind it rewrite the same documents that the third seed's landing
# rewrites: comparison.json, paired_significance.json, FINAL_REPORT.md, LIMITATIONS.md.
# The driver that was going to start them stopped at run 42's landing (2026-09-16 17:36)
# and deliberately did not restart them. Starting them now would put two writers on one
# document -- whichever finished second would win, and the loser's counts would be the
# ones a reader sees. So the chain starts when the queue script exits, which is after its
# landing, which is the same order the driver used.
set -uo pipefail
ROOT=/root/autodl-fs/wafer-vlm
LOG="$ROOT/logs/finish_chain.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
PID="${1:?usage: finish_chain.sh <pid of the seed-3 queue script>}"
log "waiting for pid $PID (train, eval, and the seed-3 landing) before starting 44"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
log "pid $PID has exited"
sleep 30
setsid nohup bash "$ROOT/projects/wafer-defect-vlm/scripts/44_final_consolidation.sh" \
  >> "$ROOT/logs/44_driver.log" 2>&1 < /dev/null &
sleep 5
setsid nohup bash "$ROOT/tools/post_consolidation.sh" \
  >> "$ROOT/logs/post_consolidation.log" 2>&1 < /dev/null &
sleep 5
log "the end-of-day chain is running: 44, then the pass that keys on its marker"
