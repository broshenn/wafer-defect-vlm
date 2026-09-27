#!/usr/bin/env bash
# Run the whole GSPO tail in one shot: wait for the second training run, score
# both GSPO adapters on the frozen benchmark, fill the retrieval column, then
# rebuild the comparison table and the final report.
#
# Why a driver rather than hand-run steps: the two runs' results must land in
# files named after their own run, and the first GSPO run already demonstrated
# what happens when a shared default path is reused -- it overwrote the GRPO
# record. Every path here is per-run, and `--tag` is required rather than
# defaulted, so a third run cannot silently collide.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PROJECT="$ROOT/projects/wafer-defect-vlm"
REPORTS="$ROOT/outputs/reports"
PY="$VENV/python"

step() { printf '\n========== %s ==========\n' "$1"; }

# ------------------------------------------------- 1. wait for the lr 1e-5 run
# The training wrapper writes this file last, so its presence is the completion
# signal. Polling a file beats polling for a process: the process can also end
# in a crash, and then the file never appears, which is what we want to detect.
TRAIN_RESULT="$REPORTS/gspo_lr1e5_train_result.json"
WAIT_MAX="${WAIT_MAX:-7200}"
waited=0
step "1. waiting for gspo_lr1e5 training (max ${WAIT_MAX}s)"
until [ -s "$TRAIN_RESULT" ]; do
  if [ "$waited" -ge "$WAIT_MAX" ]; then
    echo "FATAL: training did not produce $TRAIN_RESULT within ${WAIT_MAX}s" >&2
    exit 1
  fi
  sleep 60
  waited=$((waited + 60))
  if [ $((waited % 600)) -eq 0 ]; then
    echo "  still waiting (${waited}s); $(tail -c 120 "$ROOT/logs/29_gspo_lr1e5_train.log" 2>/dev/null | tr '\r' '\n' | tail -1)"
  fi
done
echo "training finished after ~${waited}s"
"$PY" -c "
import json,sys
d=json.load(open('$TRAIN_RESULT'))
print('  outcome:',d.get('outcome'),'|',d.get('reason'))
print('  steps:',d.get('steps_logged'),'| kl:',d.get('mean_kl'),'| reward:',d.get('mean_reward'))
"

# --------------------------------------------------------- 2. score both runs
# Retrieval is on for both: the comparator rows (Base/SFT/GRPO) all carry
# retrieval numbers, and leaving two blank cells invites reading them as zeros.
# Each run gets its own merged model directory and its own rankings file.
run_one () {
  local tag="$1" rundir="$2" merged="$3"
  step "2/3. $tag : score + retrieval"
  RUN_DIR="$rundir" MERGED="$merged" NAME="$tag" RUN_GRPO_RETRIEVAL=1 \
    bash "$PROJECT/scripts/30_eval_grpo.sh" \
    || echo "WARNING: $tag evaluation returned non-zero; its column may stay partial"
}

run_one "gspo_lr1e5" "$ROOT/outputs/checkpoints/gspo_lr1e5" \
        "$ROOT/models/Qwen3.5-9B-gspo-lr1e5-merged"
run_one "qwen35_9b_gspo_v1" "$ROOT/outputs/checkpoints/qwen35_9b_gspo_v1" \
        "$ROOT/models/Qwen3.5-9B-gspo-lr5e5-merged"

# --------------------------------------------- 4. comparison table, all columns
step "4. comparison table"
RUN_ARGS=()
for entry in \
  "Base:$REPORTS/qwen35_9b_zero_shot__report.json" \
  "SFT:$REPORTS/qwen35_9b_adapter__report.json" \
  "GRPO:$REPORTS/qwen35_9b_grpo__report.json" \
  "GSPO_lr5e5:$REPORTS/qwen35_9b_gspo_v1__report.json" \
  "GSPO_lr1e5:$REPORTS/gspo_lr1e5__report.json" ; do
  name="${entry%%:*}"; path="${entry#*:}"
  [ -s "$path" ] && RUN_ARGS+=(--run "$name=$path")
done
if [ ${#RUN_ARGS[@]} -gt 0 ]; then
  "$PY" "$ROOT/tools/make_report.py" "${RUN_ARGS[@]}" \
    --output "$REPORTS/comparison.json" --markdown "$REPORTS/comparison.md" \
    --note "Metrics shown as not run were never measured; they are not zeros."
else
  echo "WARNING: no run reports found"
fi

# ------------------------------------------------------------ 5. final report
step "5. regenerate FINAL_REPORT.md"
"$PY" "$ROOT/tools/final_report.py" --root "$ROOT" --output "$ROOT/FINAL_REPORT.md"

printf '\n========== GSPO tail complete ==========\n'
