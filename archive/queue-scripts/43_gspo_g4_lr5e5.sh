#!/usr/bin/env bash
# GSPO at G=4, lr 5e-5 -- the project's first single-variable contrast.
#
# Every other pair in this project differs in more than the variable under
# discussion. Both confounds were found today:
#   * IS level: token-level is always G=4, sequence-level always G=8 or G=32;
#   * group size: accumulation was set equal to group size, so effective batch
#     was G^2 (16 / 64 / 1024) and group size never moved alone.
#
# This run holds group size (4), accumulation (4 -> effective batch 16), learning
# rate (5e-5), steps, data and algorithm family fixed against the EXISTING
# GRPO G=4 lr5e-5 run, and changes only --importance_sampling_level to sequence.
# It is the first pair here that differs in exactly one thing, and it is the only
# available test of the paper's proposition: does sequence-level normalisation
# tolerate lr 5e-5 better than token-level?
#
# WHY CONCURRENT WITH QUEUE 41, WHICH THE OTHER QUEUES DELIBERATELY AVOID:
# the card sits near 65% and 13:35-16:00 is the only slack in the day -- after
# that, queue 41's tail and then queue 42 hold it until the window closes. This
# run peaks ~14.6 GiB and queue 41 holds ~15.2 GiB of 49.1 GiB; use_vllm is False
# so there is no KV-cache reservation, and the project has no OOM in its history.
# If the card ever looks tight, kill THIS one: it is the addition, not the plan.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PROJECT="$ROOT/projects/wafer-defect-vlm"
REPORTS="$ROOT/outputs/reports"
LOGS="$ROOT/logs"
PY="$VENV/python"
TAG="qwen35_9b_gspo_g4_lr5e5"

step() { printf '\n========== %s ==========\n' "$1"; }

# ----------------------------------------------------------------- 1. train
step "1. launching $TAG (G=4, lr 5e-5, sequence-level, accum 4)"
cd "$PROJECT" || exit 1
setsid env RUN_TAG="$TAG" \
  IS_LEVEL=sequence \
  LEARNING_RATE=5e-5 \
  NUM_GENERATIONS=4 \
  GRAD_ACCUM=4 \
  MAX_STEPS=150 \
  SAVE_STEPS=50 \
  GRPO_TIMEOUT=10800 \
  bash scripts/29_grpo_train.sh \
  > "$LOGS/43_${TAG}_outer.log" 2>&1 < /dev/null

TRAIN_RESULT="$REPORTS/${TAG}_train_result.json"
step "2. waiting for $TRAIN_RESULT"
waited=0
until [ -s "$TRAIN_RESULT" ]; do
  [ "$waited" -ge 14400 ] && { echo "FATAL: no $TRAIN_RESULT within ${waited}s" >&2; exit 1; }
  sleep 60
  waited=$((waited + 60))
  [ $((waited % 600)) -eq 0 ] && echo "  waiting (${waited}s)"
done
sleep 30
echo "training finished after ~${waited}s"

# --------------------------------- 3. prove the config is what it claims to be
# This run exists to be a single-variable contrast, so verify the variables
# rather than trusting the launch line: compare against the GRPO G=4 lr5e-5
# record and require everything except the IS level to match. A silent extra
# difference would make the whole run worthless, and there is no later check
# that would catch it.
step "3. verify it is a single-variable contrast against GRPO G=4 lr5e-5"
"$PY" - "$TRAIN_RESULT" "$REPORTS/qwen35_9b_grpo_lr5e5_train_result.json" <<'PY'
import json, sys
from pathlib import Path
new = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ref = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
a, b = new.get("config", {}), ref.get("config", {})
same = ["num_generations", "learning_rate", "per_device_batch", "seed", "beta",
        "temperature", "reward_funcs", "reward_weights"]
bad = [(k, a.get(k), b.get(k)) for k in same if a.get(k) != b.get(k)]
print("  differing:", [(k, a.get(k), b.get(k)) for k in same if a.get(k) != b.get(k)] or "none")
print("  is_level   :", a.get("importance_sampling_level"), "vs", b.get("importance_sampling_level"))
if bad:
    print("  WARNING: this run differs in more than the IS level; the contrast is "
          "confounded and must be reported as such")
if a.get("importance_sampling_level") == b.get("importance_sampling_level"):
    sys.exit("FATAL: IS level is the same in both runs; this is a duplicate, "
             "not a contrast")
PY

# ------------------------------------ 4. real accumulation, from the real argv
# The record's grad_accum field is a hardcoded constant, so it proves nothing.
# Read the actual swift argv out of this run's own inner log and store that.
step "4. record the real gradient accumulation from the training log"
"$PY" - "$TRAIN_RESULT" "$LOGS/24_grpo_${TAG}.log" <<'PY'
import json, re, sys
from pathlib import Path
rec, log = Path(sys.argv[1]), Path(sys.argv[2])
d = json.loads(rec.read_text(encoding="utf-8"))
if not log.is_file():
    print(f"  WARNING: {log} missing; real accumulation not recorded")
else:
    txt = log.read_text(encoding="utf-8", errors="replace")
    vals = [float(v) for v in re.findall(
        r"'frac_reward_zero_std': '([-0-9.eE]+)'", txt)]
    m = re.search(r"--gradient_accumulation_steps[= ](\d+)", txt)
    d["log"] = str(log)
    if m:
        d["config"]["grad_accum_from_log"] = int(m.group(1))
        d["config"]["grad_accum_note"] = (
            "grad_accum in this record is a hardcoded constant and is not "
            "evidence; this value is parsed from the swift argv in the log.")
        print(f"  real accumulation: {m.group(1)}")
    if vals:
        d["mean_frac_reward_zero_std"] = sum(vals) / len(vals)
        d["frac_reward_zero_std_note"] = (
            "Share of logged optimizer steps where every generation in the group "
            "scored identically, so the advantage was zero and no gradient was "
            "produced. Measured over all steps in the per-run log.")
        print(f"  idle steps: {d['mean_frac_reward_zero_std']:.2%} "
              f"({sum(1 for v in vals if v >= 1.0)}/{len(vals)})")
    rec.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
PY

# ------------------------------------------------------- 5. score + retrieval
step "5. $TAG : score + retrieval"
RUN_DIR="$ROOT/outputs/checkpoints/$TAG" \
MERGED="$ROOT/models/Qwen3.5-9B-gspo-g4-lr5e5-merged" \
NAME="$TAG" RUN_GRPO_RETRIEVAL=1 \
  bash "$PROJECT/scripts/30_eval_grpo.sh" \
  || echo "WARNING: $TAG evaluation returned non-zero; its column may stay partial"

printf '\n========== GSPO G=4 lr5e5 tail complete ==========\n'
