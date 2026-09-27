#!/usr/bin/env bash
# GSPO at G=4, lr 1e-5 -- the fourth cell of the 2x2 the paper's proposition needs.
#
# Run 43 gives sequence-vs-token at lr 5e-5 with group size (4) and effective batch
# (16) held fixed. But "sequence-level tolerates a higher lr" is a claim about how
# the learning-rate EFFECT differs between IS levels, and an lr effect needs two lrs
# measured at the same IS level and the same group size. At G=4 this record has:
#
#     token    lr 1e-5 : GRPO G=4 lr1e-5   (exists, acc 0.6230)
#     token    lr 5e-5 : GRPO G=4 lr5e-5   (exists, acc 0.5675)
#     sequence lr 5e-5 : run 43            (running)
#     sequence lr 1e-5 : THIS RUN
#
# With all four, the lr effect is computable separately for each IS level at one
# group size, which is the only form in which "tolerates" gets a measurement rather
# than an impression. Every group size in this project except 4 changes with the IS
# level, so this is the only group size where that 2x2 can ever be complete.
#
# WHY IT WAITS: run 43 is training when this is launched, and queue 41 holds a G=32
# job. The card has 49.1 GiB; two G=4 jobs plus the G=32 job would leave ~4 GiB of
# headroom, which is not worth the risk to a queue that has been running since 13:15.
# This script therefore starts only after 43's own script has exited (training AND
# its evaluation), so at most two jobs overlap.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PROJECT="$ROOT/projects/wafer-defect-vlm"
REPORTS="$ROOT/outputs/reports"
LOGS="$ROOT/logs"
PY="$VENV/python"
TAG="qwen35_9b_gspo_g4_lr1e5"
REF_TAG="qwen35_9b_gspo_g4_lr5e5"

step() { printf '\n========== %s ==========\n' "$1"; }

# ------------------------------------------------------ 1. wait for the card
# Gate on run 43 being FINISHED, not on its driver process. Matching a process by
# the script's filename would match the shell that launched it -- that shell's
# command line still contains the path -- so this would never clear. The two
# conditions below cannot both hold while run 43 is still working: the report is
# written at the end of its evaluation, and a live training process for that run
# carries its output_dir.
step "1. wait for run 43 to finish before taking the card"
R43_REPORT="$REPORTS/qwen35_9b_gspo_g4_lr5e5__report.json"

run43_busy() {
  local p c
  for p in $(pgrep -f 'qwen35_9b_gspo_g4_lr5e5' 2>/dev/null); do
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$c" in
      *--output_dir*qwen35_9b_gspo_g4_lr5e5*) return 0 ;;
    esac
  done
  return 1
}

waited=0
while [ ! -s "$R43_REPORT" ] || run43_busy; do
  [ "$waited" -ge 14400 ] && {
    echo "FATAL: run 43 not finished after ${waited}s (report:$([ -s "$R43_REPORT" ] && echo yes || echo no)); refusing to start a second job on an unproven-free card" >&2
    exit 1
  }
  [ $((waited % 600)) -eq 0 ] && echo "  still waiting on run 43 (${waited}s)"
  sleep 60
  waited=$((waited + 60))
done
echo "run 43 is finished after ~${waited}s of waiting"

# ----------------------------------------------------------------- 2. train
step "2. launching $TAG (G=4, lr 1e-5, sequence-level, accum 4)"
cd "$PROJECT" || exit 1
setsid env RUN_TAG="$TAG" \
  IS_LEVEL=sequence \
  LEARNING_RATE=1e-5 \
  NUM_GENERATIONS=4 \
  GRAD_ACCUM=4 \
  MAX_STEPS=150 \
  SAVE_STEPS=50 \
  GRPO_TIMEOUT=10800 \
  bash scripts/29_grpo_train.sh \
  > "$LOGS/45_${TAG}_outer.log" 2>&1 < /dev/null

TRAIN_RESULT="$REPORTS/${TAG}_train_result.json"
step "3. waiting for $TRAIN_RESULT"
waited=0
until [ -s "$TRAIN_RESULT" ]; do
  [ "$waited" -ge 14400 ] && { echo "FATAL: no $TRAIN_RESULT within ${waited}s" >&2; exit 1; }
  sleep 60
  waited=$((waited + 60))
  [ $((waited % 600)) -eq 0 ] && echo "  waiting (${waited}s)"
done
sleep 30
echo "training finished after ~${waited}s"

# --------------------------------- 4. prove the pair differs only in the learning rate
# The whole point of this run is that its comparison with run 43 changes one thing.
# Verify it against run 43's own record rather than trusting this launch line: a
# silent second difference would make the 2x2 unreadable and nothing later would
# catch it.
step "4. verify it is single-variable against $REF_TAG (only lr differs)"
"$PY" - "$TRAIN_RESULT" "$REPORTS/${REF_TAG}_train_result.json" <<'PY'
import json, sys
from pathlib import Path
new_p, ref_p = Path(sys.argv[1]), Path(sys.argv[2])
new = json.loads(new_p.read_text(encoding="utf-8"))
if not ref_p.is_file():
    print(f"  WARNING: reference {ref_p.name} missing; cannot verify the pair "
          f"differs in one thing. The 2x2 must not be read as single-variable.")
    sys.exit(0)
ref = json.loads(ref_p.read_text(encoding="utf-8"))
a, b = new.get("config", {}), ref.get("config", {})
same = ["num_generations", "per_device_batch", "seed", "beta", "temperature",
        "reward_funcs", "reward_weights", "importance_sampling_level"]
bad = [(k, a.get(k), b.get(k)) for k in same if a.get(k) != b.get(k)]
print("  must match, differing:", bad or "none")
print("  learning_rate :", a.get("learning_rate"), "vs", b.get("learning_rate"))
print("  is_level      :", a.get("importance_sampling_level"), "vs",
      b.get("importance_sampling_level"))
if bad:
    print("  WARNING: this run differs from run 43 in more than the learning "
          "rate; the 2x2 is confounded and must be reported as such")
if a.get("learning_rate") == b.get("learning_rate"):
    sys.exit("FATAL: same learning rate as run 43; this is a duplicate, not the "
             "fourth cell of the 2x2")
PY

# ------------------------------------ 5. real accumulation, from the real argv
# The record's grad_accum field is a hardcoded constant, so it proves nothing.
step "5. record the real gradient accumulation from the training log"
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

# ------------------------------------------------------- 6. score + retrieval
step "6. $TAG : score + retrieval"
RUN_DIR="$ROOT/outputs/checkpoints/$TAG" \
MERGED="$ROOT/models/Qwen3.5-9B-gspo-g4-lr1e5-merged" \
NAME="$TAG" RUN_GRPO_RETRIEVAL=1 \
  bash "$PROJECT/scripts/30_eval_grpo.sh" \
  || echo "WARNING: $TAG evaluation returned non-zero; its column may stay partial"

printf '\n========== GSPO G=4 lr1e5 (fourth cell) complete ==========\n'
