#!/usr/bin/env bash
# Second seed of GRPO at lr 1e-5 -- the run the main conclusion depends on.
#
# Every RL number in this project comes from a single run. The headline result is
# negative ("no RL configuration beats SFT") and is supported by confidence
# intervals computed inside those single runs, so the obvious question is
# whether the intervals describe the comparison or only sampling noise. GRPO at
# lr 1e-5 is the configuration that came closest to SFT (macro-F1 0.6197 against
# 0.6114), which makes it the one worth re-running: if a second seed lands below
# SFT the negative result is strengthened, and if it lands well above, the
# conclusion has to be revisited.
#
# Identical to the first run except for SEED, which the launcher now forwards to
# ms-swift and writes into the run record, so the two runs can be shown to
# differ rather than merely asserted to.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PROJECT="$ROOT/projects/wafer-defect-vlm"
REPORTS="$ROOT/outputs/reports"
LOGS="$ROOT/logs"
PY="$VENV/python"
TAG="qwen35_9b_grpo_lr1e5_seed3408"
SEED=3408
PREV_DRIVER="41_queue_gspo_g32_lr1e5.sh"
PREV_LOG="$LOGS/41_queue_outer.log"

step() { printf '\n========== %s ==========\n' "$1"; }

# ---------------------------------------------- 0. wait for the GPU to be free
step "0. waiting for $PREV_DRIVER to finish"
waited=0
while : ; do
  if ! pgrep -f "$PREV_DRIVER" >/dev/null 2>&1; then
    if grep -q "queue complete" "$PREV_LOG" 2>/dev/null; then
      echo "$PREV_DRIVER completed after ~${waited}s"
      break
    fi
    echo "FATAL: $PREV_DRIVER gone without its completion marker; refusing to start" >&2
    tail -20 "$PREV_LOG" 2>&1 >&2
    exit 1
  fi
  [ "$waited" -ge 25200 ] && { echo "FATAL: $PREV_DRIVER still running after ${waited}s" >&2; exit 1; }
  sleep 60
  waited=$((waited + 60))
done

# ----------------------------------------------------------------- 1. train
step "1. launching $TAG (GRPO G=4, lr 1e-5, seed $SEED)"
cd "$PROJECT" || exit 1
setsid env RUN_TAG="$TAG" \
  IS_LEVEL=token \
  LEARNING_RATE=1e-5 \
  NUM_GENERATIONS=4 \
  GRAD_ACCUM=4 \
  SEED="$SEED" \
  MAX_STEPS=150 \
  SAVE_STEPS=50 \
  GRPO_TIMEOUT=9000 \
  bash scripts/29_grpo_train.sh \
  > "$LOGS/42_${TAG}_outer.log" 2>&1 < /dev/null

TRAIN_RESULT="$REPORTS/${TAG}_train_result.json"
step "2. waiting for $TRAIN_RESULT"
waited=0
until [ -s "$TRAIN_RESULT" ]; do
  [ "$waited" -ge 10800 ] && { echo "FATAL: no train result within ${waited}s" >&2; exit 1; }
  sleep 60
  waited=$((waited + 60))
done
sleep 30
"$PY" -c "
import json
d = json.load(open('$TRAIN_RESULT'))
c = d.get('config', {})
print('  outcome:', d.get('outcome'), '|', d.get('reason'))
print('  steps:', d.get('steps_logged'), '| G:', c.get('num_generations'),
      '| lr:', c.get('learning_rate'), '| seed:', c.get('seed'))
print('  mean_reward:', d.get('mean_reward'), '| mean_kl:', d.get('mean_kl'))
"

# --------------------------------------- 3. prove the seed actually differed
step "3. seed check against the first run (must differ)"
"$PY" - "$REPORTS/qwen35_9b_grpo_train_result.json" "$TRAIN_RESULT" <<'PY'
import json, sys
from pathlib import Path
a = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("config", {})
b = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")).get("config", {})
sa, sb = a.get("seed"), b.get("seed")
print(f"  first run seed: {sa}   this run seed: {sb}")
if sa is None or sb is None:
    print("  WARNING: a record has no seed field; cannot prove the runs differ")
elif sa == sb:
    print("  ERROR: seeds are identical -- this is a duplicate, not a replication")
    sys.exit(3)
else:
    print("  seeds differ: this is a genuine replication")
PY

step "4. repoint log, measure idle-step share"
"$PY" - "$TRAIN_RESULT" "$LOGS/24_grpo_${TAG}.log" <<'PY'
import json, re, sys
from pathlib import Path
rec, log = Path(sys.argv[1]), Path(sys.argv[2])
d = json.loads(rec.read_text(encoding="utf-8"))
if not log.is_file():
    print(f"  WARNING: {log} missing; record left as written")
else:
    vals = [float(v) for v in re.findall(
        r"'frac_reward_zero_std': '([-0-9.eE]+)'",
        log.read_text(encoding="utf-8", errors="replace"))]
    d["log"] = str(log)
    if vals:
        d["mean_frac_reward_zero_std"] = sum(vals) / len(vals)
        d["frac_reward_zero_std_note"] = (
            "Share of logged optimizer steps where every generation in the group "
            "scored identically, so the advantage was zero and no gradient was "
            "produced. Measured over all steps in the per-run log.")
        print(f"  idle steps: {d['mean_frac_reward_zero_std']:.2%} "
              f"({sum(1 for v in vals if v >= 1.0)}/{len(vals)})")
    rec.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  log repointed to {log.name}")
PY

step "5. $TAG : score + retrieval"
RUN_DIR="$ROOT/outputs/checkpoints/$TAG" \
MERGED="$ROOT/models/Qwen3.5-9B-grpo-lr1e5-seed3408-merged" \
NAME="$TAG" RUN_GRPO_RETRIEVAL=1 \
  bash "$PROJECT/scripts/30_eval_grpo.sh" \
  || echo "WARNING: $TAG evaluation returned non-zero; its column may stay partial"

step "6. comparison table (all runs)"
RUN_ARGS=()
for entry in \
  "Base:$REPORTS/qwen35_9b_zero_shot__report.json" \
  "SFT:$REPORTS/qwen35_9b_adapter__report.json" \
  "GRPO_lr1e5:$REPORTS/qwen35_9b_grpo__report.json" \
  "GRPO_lr5e5:$REPORTS/qwen35_9b_grpo_lr5e5__report.json" \
  "GRPO_lr1e5_s2:$REPORTS/${TAG}__report.json" \
  "GSPO_lr5e5:$REPORTS/qwen35_9b_gspo_v1__report.json" \
  "GSPO_lr1e5:$REPORTS/gspo_lr1e5__report.json" \
  "GSPO_G32:$REPORTS/qwen35_9b_gspo_g32__report.json" \
  "GSPO_G32_lr1e5:$REPORTS/qwen35_9b_gspo_g32_lr1e5__report.json" ; do
  name="${entry%%:*}"; path="${entry#*:}"
  [ -s "$path" ] && RUN_ARGS+=(--run "$name=$path")
done
[ ${#RUN_ARGS[@]} -gt 0 ] && "$PY" "$ROOT/tools/make_report.py" "${RUN_ARGS[@]}" \
  --output "$REPORTS/comparison.json" --markdown "$REPORTS/comparison.md" \
  --note "Metrics shown as not run were never measured; they are not zeros."

step "7. FINAL_REPORT.md"
"$PY" "$ROOT/tools/final_report.py" --root "$ROOT" --output "$ROOT/FINAL_REPORT.md"

printf '\n========== GRPO seed-2 queue complete ==========\n'
