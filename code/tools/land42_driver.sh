#!/usr/bin/env bash
# Run 42's landing, sequenced against the end-of-day chain.
#
# WHY THIS EXISTS AS A DRIVER. Three things happen when run 42's queue exits, and two of
# them edit `LIMITATIONS.md`: this landing (5.2.7, the seed-variance prose) and
# 44_final_consolidation.sh (sync_counts and the number audit, which read and rewrite the
# same sentences). 44's waiter treats "no training, no eval, no queue driver" as idle, so
# it starts the moment the queue is gone -- exactly when the landing starts. Two writers
# to one document is a race with no useful outcome, so 44 is stopped for the duration and
# restarted here only after the landing has finished.
#
# The order inside the landing is patch75, then patch118, then land_run:
#   * patch75 is the fuller fix for the two defects in the training wrapper (it also
#     writes `grad_accum_source`, so a record whose caller forgot to export GRAD_ACCUM
#     says so instead of quietly claiming 4). patch118 edits the same two lines and
#     stands down when patch75's field is present -- it exists for its other two jobs,
#     which are item 11's regenerated table and the guards that read it;
#   * land_run waits for the report's retrieval numbers itself, so this driver does not
#     need to know when the queue's step 3b finishes.
set -uo pipefail
ROOT=/root/autodl-fs/wafer-vlm
PY="$ROOT/venvs/wafer/bin/python"
LOG="$ROOT/logs/land42_driver.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "waiting for pid 3699 (run 42's queue) to exit"
while kill -0 3699 2>/dev/null; do sleep 20; done
log "pid 3699 has exited; letting its last writes settle"
sleep 30

log "step 1/3: patch75_grpo_record.py (the training wrapper's two defects)"
"$PY" "$ROOT/tools/patch75_grpo_record.py" >> "$LOG" 2>&1 && log "  ok" \
  || log "  FAILED -- see above; continuing, because the landing does not depend on it"

log "step 2/3: patch118_grpo_train_fixes.py (item 11's table, the checker item, the idle table's declaration)"
"$PY" "$ROOT/tools/patch118_grpo_train_fixes.py" >> "$LOG" 2>&1 && log "  ok" \
  || log "  FAILED -- see above; continuing to the landing"

# step 3/3. The two --pre tools both have to run before anything is written, and the
# comment in land_run.sh says why --pre exists: they consume a record, and the sentences
# written below quote it. kl_length_confound is the one that was missing a caller: its own
# docstring said "no caller; nothing regenerates it", and the checker has an item that
# fails when the record is behind the run set. It is regenerated here so that this landing
# is what put the record in the state the landing's own counts read.
log "step 3/3: land run 42 (no column; --pre kl_length_confound, seed_variance; --patch patch94_seed, patch101_kl_sentences)"
if bash "$ROOT/tools/land_run.sh" --tag qwen35_9b_grpo_lr1e5_seed3408 --no-column \
     --pre tools/kl_length_confound.py \
     --pre tools/seed_variance.py \
     --patch tools/patch94_seed.py \
     --patch tools/patch101_kl_sentences.py >> "$LOG" 2>&1; then
  log "landing ok"
else
  log "LANDING STOPPED -- the log above names the sentence that has to be rewritten."
  log "The end-of-day chain is deliberately NOT restarted: 44 rewrites the same document."
  exit 1
fi

log "restarting the end-of-day chain (44, then the pass that keys on its marker)"
setsid nohup bash "$ROOT/projects/wafer-defect-vlm/scripts/44_final_consolidation.sh" \
  >> "$ROOT/logs/44_driver.log" 2>&1 < /dev/null &
sleep 5
setsid nohup bash "$ROOT/tools/post_consolidation.sh" \
  >> "$ROOT/logs/post_consolidation.log" 2>&1 < /dev/null &
sleep 5
log "LANDING COMPLETE and the end-of-day chain is running again"
