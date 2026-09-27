#!/usr/bin/env bash
# Tail for the G=32 run, the paper's actual group size.
#
# Waits for it, records its log and idle-step share, scores it on the frozen
# benchmark, then fills the retrieval cells that would otherwise read "not run"
# for every GSPO row. Retrieval needs a bf16 merge per run because the ranking
# tool embeds with plain transformers, and Base/SFT/GRPO were all measured that
# way -- an adapter shortcut would not be comparable.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PROJECT="$ROOT/projects/wafer-defect-vlm"
REPORTS="$ROOT/outputs/reports"
PY="$VENV/python"
TAG="qwen35_9b_gspo_g32"

step() { printf '\n========== %s ==========\n' "$1"; }

# ------------------------------------------------------------- 1. wait for it
TRAIN_RESULT="$REPORTS/${TAG}_train_result.json"
WAIT_MAX="${WAIT_MAX:-14400}"
waited=0
step "1. waiting for $TAG training (max ${WAIT_MAX}s)"
until [ -s "$TRAIN_RESULT" ]; do
  if [ "$waited" -ge "$WAIT_MAX" ]; then
    echo "FATAL: no $TRAIN_RESULT within ${WAIT_MAX}s" >&2
    exit 1
  fi
  sleep 60
  waited=$((waited + 60))
  [ $((waited % 900)) -eq 0 ] && echo "  waiting (${waited}s); $(tail -c 120 "$ROOT/logs/29_${TAG}_train.log" 2>/dev/null | tr '\r' '\n' | tail -1)"
done
sleep 30   # let the record finish being written
echo "training finished after ~${waited}s"
"$PY" -c "
import json
d = json.load(open('$TRAIN_RESULT'))
print('  outcome:', d.get('outcome'), '|', d.get('reason'))
print('  steps:', d.get('steps_logged'), '| G:', d.get('config', {}).get('num_generations'),
      '| lr:', d.get('config', {}).get('learning_rate'))
print('  mean_reward:', d.get('mean_reward'), '| mean_kl:', d.get('mean_kl'))
"

# ------------------------------------------- 2. per-run log + idle-step share
# Same correction as the other two RL records: point `log` at this run's own
# inner log and measure the idle share from all steps, rather than leaving a
# number that only the first and last step could support.
step "2. repoint log, measure idle-step share"
"$PY" - "$TRAIN_RESULT" "$ROOT/logs/24_grpo_${TAG}.log" <<'PY'
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

# ------------------------------------------------------- 3. score + retrieval
for spec in \
  "$TAG|$ROOT/outputs/checkpoints/$TAG|$ROOT/models/Qwen3.5-9B-gspo-g32-merged" \
  "gspo_lr1e5|$ROOT/outputs/checkpoints/gspo_lr1e5|$ROOT/models/Qwen3.5-9B-gspo-lr1e5-merged" \
  "qwen35_9b_gspo_v1|$ROOT/outputs/checkpoints/qwen35_9b_gspo_v1|$ROOT/models/Qwen3.5-9B-gspo-lr5e5-merged" ; do
  IFS='|' read -r name rundir merged <<< "$spec"
  step "3. $name : score + retrieval"
  RUN_DIR="$rundir" MERGED="$merged" NAME="$name" RUN_GRPO_RETRIEVAL=1 \
    bash "$PROJECT/scripts/30_eval_grpo.sh" \
    || echo "WARNING: $name evaluation returned non-zero; its column may stay partial"
done

# ------------------------------------------------------------- 4. rebuild all
step "4. comparison table (all runs)"
RUN_ARGS=()
for entry in \
  "Base:$REPORTS/qwen35_9b_zero_shot__report.json" \
  "SFT:$REPORTS/qwen35_9b_adapter__report.json" \
  "GRPO:$REPORTS/qwen35_9b_grpo__report.json" \
  "GSPO_lr5e5:$REPORTS/qwen35_9b_gspo_v1__report.json" \
  "GSPO_lr1e5:$REPORTS/gspo_lr1e5__report.json" \
  "GSPO_G32:$REPORTS/${TAG}__report.json" ; do
  name="${entry%%:*}"; path="${entry#*:}"
  [ -s "$path" ] && RUN_ARGS+=(--run "$name=$path")
done
[ ${#RUN_ARGS[@]} -gt 0 ] && "$PY" "$ROOT/tools/make_report.py" "${RUN_ARGS[@]}" \
  --output "$REPORTS/comparison.json" --markdown "$REPORTS/comparison.md" \
  --note "Metrics shown as not run were never measured; they are not zeros."

step "5. FINAL_REPORT.md"
"$PY" "$ROOT/tools/final_report.py" --root "$ROOT" --output "$ROOT/FINAL_REPORT.md"

printf '\n========== G=32 tail complete ==========\n'
