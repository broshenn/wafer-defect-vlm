#!/usr/bin/env bash
# Queue for GSPO at G=32, lr 1e-5 -- the last corner of the group-size x lr grid.
#
# It exists to answer one question the current results cannot: G=32 has the best
# retrieval of any run (mAP@10 0.4337 against SFT's 0.3873) and the best
# radial_zone, but it is also the run with the most degenerate policy (idle
# steps 42.00%, the worst of four). Holding the group size at 32 and moving the
# learning rate to 1e-5 separates those: if the retrieval advantage survives, it
# belongs to the group size; if it goes, it was a side effect of the collapsed
# 5e-5 policy and the project has no positive RL result at all.
#
# Waits for the previous tail driver to finish before touching the GPU, because
# both jobs need the whole card.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PROJECT="$ROOT/projects/wafer-defect-vlm"
REPORTS="$ROOT/outputs/reports"
LOGS="$ROOT/logs"
PY="$VENV/python"
TAG="qwen35_9b_gspo_g32_lr1e5"
PREV_DRIVER="40_after_grpo_lr5e5.sh"

step() { printf '\n========== %s ==========\n' "$1"; }

# ---------------------------------------------- 0. wait for the GPU to be free
step "0. waiting for $PREV_DRIVER to finish"
waited=0
while : ; do
  if ! pgrep -f "$PREV_DRIVER" >/dev/null 2>&1; then
    # Process gone. Confirm it finished rather than died, by looking for its
    # own completion marker; a crash would otherwise silently start this run on
    # top of a half-written comparison table.
    if grep -q "tail complete" "$LOGS/40_after_grpo_lr5e5_outer.log" 2>/dev/null; then
      echo "$PREV_DRIVER completed after ~${waited}s"
      break
    fi
    echo "FATAL: $PREV_DRIVER is gone but never reported completion; refusing to start" >&2
    tail -20 "$LOGS/40_after_grpo_lr5e5_outer.log" 2>&1 >&2
    exit 1
  fi
  [ "$waited" -ge 14400 ] && { echo "FATAL: $PREV_DRIVER still running after ${waited}s" >&2; exit 1; }
  sleep 60
  waited=$((waited + 60))
done

# ----------------------------------------------------------------- 1. train
step "1. launching $TAG (G=32, lr 1e-5, sequence-level)"
cd "$PROJECT" || exit 1
setsid env RUN_TAG="$TAG" \
  IS_LEVEL=sequence \
  LEARNING_RATE=1e-5 \
  NUM_GENERATIONS=32 \
  GRAD_ACCUM=32 \
  MAX_STEPS=150 \
  SAVE_STEPS=50 \
  GRPO_TIMEOUT=18000 \
  bash scripts/29_grpo_train.sh \
  > "$LOGS/41_${TAG}_outer.log" 2>&1 < /dev/null
echo "training launcher returned; verifying via the result record"

# ------------------------------------------- 2. record + idle-step share
TRAIN_RESULT="$REPORTS/${TAG}_train_result.json"
step "2. waiting for $TRAIN_RESULT"
waited=0
until [ -s "$TRAIN_RESULT" ]; do
  [ "$waited" -ge 18000 ] && { echo "FATAL: no train result within ${waited}s" >&2; exit 1; }
  sleep 60
  waited=$((waited + 60))
  [ $((waited % 900)) -eq 0 ] && echo "  waiting (${waited}s); $(tail -c 160 "$LOGS/29_${TAG}_train.log" 2>/dev/null | tr '\r' '\n' | tail -1)"
done
sleep 30
"$PY" -c "
import json
d = json.load(open('$TRAIN_RESULT'))
print('  outcome:', d.get('outcome'), '|', d.get('reason'))
print('  steps:', d.get('steps_logged'), '| G:', d.get('config', {}).get('num_generations'),
      '| lr:', d.get('config', {}).get('learning_rate'))
print('  mean_reward:', d.get('mean_reward'), '| mean_kl:', d.get('mean_kl'))
"

step "3. repoint log, measure idle-step share"
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

# ------------------------------------------------------- 4. score + retrieval
step "4. $TAG : score + retrieval"
RUN_DIR="$ROOT/outputs/checkpoints/$TAG" \
MERGED="$ROOT/models/Qwen3.5-9B-gspo-g32-lr1e5-merged" \
NAME="$TAG" RUN_GRPO_RETRIEVAL=1 \
  bash "$PROJECT/scripts/30_eval_grpo.sh" \
  || echo "WARNING: $TAG evaluation returned non-zero; its column may stay partial"

# ------------------------------------------------------------- 5. rebuild all
step "5. comparison table (all runs)"
RUN_ARGS=()
for entry in \
  "Base:$REPORTS/qwen35_9b_zero_shot__report.json" \
  "SFT:$REPORTS/qwen35_9b_adapter__report.json" \
  "GRPO_lr1e5:$REPORTS/qwen35_9b_grpo__report.json" \
  "GRPO_lr5e5:$REPORTS/qwen35_9b_grpo_lr5e5__report.json" \
  "GSPO_lr5e5:$REPORTS/qwen35_9b_gspo_v1__report.json" \
  "GSPO_lr1e5:$REPORTS/gspo_lr1e5__report.json" \
  "GSPO_G32:$REPORTS/qwen35_9b_gspo_g32__report.json" \
  "GSPO_G32_lr1e5:$REPORTS/${TAG}__report.json" ; do
  name="${entry%%:*}"; path="${entry#*:}"
  [ -s "$path" ] && RUN_ARGS+=(--run "$name=$path")
done
[ ${#RUN_ARGS[@]} -gt 0 ] && "$PY" "$ROOT/tools/make_report.py" "${RUN_ARGS[@]}" \
  --output "$REPORTS/comparison.json" --markdown "$REPORTS/comparison.md" \
  --note "Metrics shown as not run were never measured; they are not zeros."

step "6. FINAL_REPORT.md"
"$PY" "$ROOT/tools/final_report.py" --root "$ROOT" --output "$ROOT/FINAL_REPORT.md"

printf '\n========== GSPO G=32 lr1e5 queue complete ==========\n'
