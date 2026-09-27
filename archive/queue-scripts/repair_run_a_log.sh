#!/usr/bin/env bash
# Point run A at the log that actually holds its data, and measure the idle share.
#
# Two consequences of the same mid-execution edit, and this is the second one.
# 24_train_grpo.sh routes the run's output through `| tee "$LOG"` on its last
# line, 127. patch44 changed line 115 -- before it, and by a different number of
# bytes -- so when bash resumed reading at its remembered offset the tee was
# lost. Swift's output went to stdout instead, which the launcher was capturing
# into the 29_ log, so the run lost nothing: 24_grpo_qwen35_9b_grpo_lr5e5.log is
# 0 bytes while 29_qwen35_9b_grpo_lr5e5_train.log holds all 150 steps.
#
# The driver then did exactly what it was written to do and repointed `log` at
# that empty file, so the record now cites a log with nothing in it. That is the
# same shape as section 8 item 5 -- a record pointing somewhere the data is not
# -- arriving by a different route. Point it at the real log, and record the
# idle share the driver could not compute from the empty one.
set -uo pipefail

ROOT=/root/autodl-fs/wafer-vlm
VENV="$ROOT/venvs/wafer/bin"
REC="$ROOT/outputs/reports/qwen35_9b_grpo_lr5e5_train_result.json"
REAL_LOG="$ROOT/logs/29_qwen35_9b_grpo_lr5e5_train.log"
EMPTY_LOG="$ROOT/logs/24_grpo_qwen35_9b_grpo_lr5e5.log"

[ -s "$REC" ] || { echo "no record at $REC"; exit 1; }
[ -s "$REAL_LOG" ] || { echo "no log at $REAL_LOG"; exit 1; }

echo "empty log cited by the record: $(stat -c%s "$EMPTY_LOG" 2>/dev/null || echo -) bytes"
echo "log that holds the data:       $(stat -c%s "$REAL_LOG") bytes"

"$VENV/python" - "$REC" "$REAL_LOG" "$EMPTY_LOG" <<'PY'
import json, re, sys
from pathlib import Path

rec, real_log, empty_log = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
d = json.loads(rec.read_text(encoding="utf-8"))
before = d.get("log")

vals = [float(v) for v in re.findall(
    r"'frac_reward_zero_std': '([-0-9.eE]+)'",
    real_log.read_text(encoding="utf-8", errors="replace"))]
if not vals:
    sys.exit("no frac_reward_zero_std rows in the real log; nothing written")

d["log"] = str(real_log)
d["log_note"] = (
    "Points at the launcher's log, not the per-run 24_grpo_*.log the other "
    "records name. That file is 0 bytes: 24_train_grpo.sh routes output through "
    "`| tee \"$LOG\"` on line 127, and patch44 edited line 115 while this run was "
    "executing the script, so bash's byte offset into the file was stale when it "
    "resumed and the tee was dropped. The output went to stdout instead, which "
    "the launcher captured here. No data was lost; the path named by default "
    "just does not hold it. See LIMITATIONS section 8 item 8.")
d["mean_frac_reward_zero_std"] = sum(vals) / len(vals)
d["frac_reward_zero_std_note"] = (
    "Share of logged optimizer steps where every generation in the group scored "
    "identically, so the advantage was zero and no gradient was produced. "
    "Measured over all 150 steps of this run's log -- computed here rather than "
    "by the usual driver step, because the driver read the empty 24_ log above "
    "and found nothing to average.")

rec.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
idle = [v for v in vals if v >= 1.0]
print(f"  log: {before} -> {real_log.name}")
print(f"  idle steps: {d['mean_frac_reward_zero_std']:.2%} ({len(idle)}/{len(vals)})")
print(f"  record_note preserved: {'record_note' in d}")
PY
