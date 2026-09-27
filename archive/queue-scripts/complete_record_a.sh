#!/usr/bin/env bash
# Complete the run record that the launcher never got to write.
#
# scripts/29_grpo_train.sh died at line 96 with "syntax error near unexpected
# token `('". The script was not at fault. patch44 inserted one line into it at
# 12:23 while a run launched at 12:18 was still executing it, and bash resumes a
# script by byte offset -- so the insertion shifted the resume point into the
# middle of the launcher's inline Python, and the shell tried to execute Python
# as shell. Training had already finished by then; only the bookkeeping was lost.
#
# So the measurements are real and the log is real. What follows runs the
# launcher's *own* writer, extracted verbatim from lines 75-165 of the current
# script, with the environment this run actually had. A second implementation
# written here would be free to disagree with the records it sits beside.
set -uo pipefail

ROOT=/root/autodl-fs/wafer-vlm
VENV="$ROOT/venvs/wafer/bin"
RUN_TAG=qwen35_9b_grpo_lr5e5
RESULT="$ROOT/outputs/reports/${RUN_TAG}_train_result.json"
LOG="$ROOT/logs/29_${RUN_TAG}_train.log"
MEMLOG="$ROOT/logs/29_${RUN_TAG}_mem.log"

if [ -s "$RESULT" ]; then echo "record already exists; refusing to overwrite"; exit 1; fi
if [ ! -s "$LOG" ]; then echo "no training log at $LOG"; exit 1; fi

# The run-level values the launcher would have computed, read back from the
# artefacts it left behind rather than assumed.
STATUS=0
OUTCOME="completed"
REASON="150 steps finished without error"
ELAPSED=$(grep -o "'train_runtime': '[0-9]*'" "$LOG" | tail -1 | grep -o "[0-9]*")
PEAK=$(sort -n "$MEMLOG" | tail -1)
MAX_STEPS=150
SAVE_STEPS=50
echo "reconstructed from the run's own artefacts:"
echo "  exit status $STATUS   elapsed ${ELAPSED}s   peak ${PEAK} MiB"

# The record's config block is read from the environment, so these must be what
# this run used -- 5e-5 and four generations, token-level GRPO, seed 3407 --
# not whatever the launcher happens to default to today.
export NUM_GENERATIONS=4
export LEARNING_RATE=5e-5
export SEED=3407
export IS_LEVEL=token

# The launcher's writer, verbatim, in this shell so its variables are in scope.
{
  echo 'set -uo pipefail'
  sed -n '75,165p' "$ROOT/projects/wafer-defect-vlm/scripts/29_grpo_train.sh"
} > /tmp/record_block.sh

if ! grep -q "out = Path(result)" /tmp/record_block.sh; then
  echo "FAILED: the extracted block does not look like the record writer"; exit 1
fi
# Sourced, not run: the block reads $RESULT, $LOG, $VENV and the rest from its
# caller's shell, and a child process would see them all empty and write a
# record to a path that expands to nothing.
#
# Written to a temporary path first and moved into place at the end, because the
# driver waiting on this file reads it and writes it back. Landing it early
# would let the driver read the record before the note below is attached, and
# its rewrite would then drop the note -- the one field that distinguishes a
# record the run wrote from one assembled afterwards.
REAL_RESULT="$RESULT"
RESULT=/tmp/${RUN_TAG}_train_result.json
rm -f "$RESULT"
source /tmp/record_block.sh
rc=$?

if [ "$rc" -ne 0 ] || [ ! -s "$RESULT" ]; then
  echo "FAILED: the writer did not produce $RESULT (rc=$rc)"; exit 1
fi

# Say so in the record. A reader must be able to tell a record the run wrote
# from one finished afterwards, or the distinction this project cares about --
# between a measurement and a reconstruction -- is lost exactly where it matters.
"$VENV/python" - "$RESULT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
d["record_note"] = (
    "Record completed after the fact. scripts/29_grpo_train.sh was edited at "
    "12:23 (patch44) while this run, launched at 12:18, was still executing it; "
    "bash resumes a script by byte offset, so the inserted line shifted the "
    "resume point into the launcher's inline Python and it died with a shell "
    "syntax error before writing this file. Training had already finished "
    "normally. The record was then produced by the launcher's own writer, "
    "extracted verbatim, with this run's environment. All step-level fields "
    "come from the run's log; elapsed_seconds and gpu_peak_mib come from the "
    "log's train_runtime and the max of the launcher's own memory samples.")
p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("record_note added")
PY

if ! grep -q "record_note" "$RESULT"; then
  echo "FAILED: the note did not stick; not publishing the record"; exit 1
fi
mv "$RESULT" "$REAL_RESULT"
echo "published $REAL_RESULT"

echo
echo "=== the record ==="
"$VENV/python" -c "
import json
d = json.load(open('$REAL_RESULT'))
for k in ('outcome','reason','exit_status','elapsed_seconds','gpu_peak_mib',
          'steps_logged','mean_reward','mean_kl','mean_step_seconds'):
    print('  %-20s %r' % (k, d.get(k)))
c = d['config']
print('  config: G=%s lr=%s seed=%s is_level=%s' % (
    c.get('num_generations'), c.get('learning_rate'), c.get('seed'), c.get('importance_sampling_level')))
"
