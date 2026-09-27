#!/usr/bin/env bash
# Wait for the GRPO run to exit, then re-evaluate everything that GRPO changes:
# the benchmark column, the comparison table, the leak check (GRPO reuses the
# training split by design), acceptance, and the final report.
#
# Chained under nohup so it proceeds whether or not the ssh session survives.
# Every stage is skipped when its input is missing and says so, rather than
# writing a row of zeros for a run that never produced numbers.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PY="$VENV/python"
PROJECT="$ROOT/projects/wafer-defect-vlm"
BENCH="$ROOT/benchmarks/wafer_bench_v1"
REPORTS="$ROOT/outputs/reports"
DRIVER="$ROOT/logs/31_after_grpo.log"

mkdir -p "$REPORTS"
exec > >(tee -a "$DRIVER") 2>&1
step() { printf '\n========== %s ==========\n' "$1"; }

echo "=== $(date) driver started; waiting for GRPO training to exit ==="
while pgrep -f "[2]4_train_grpo" > /dev/null; do
  sleep 60
done
echo "=== $(date) GRPO training process is gone ==="

step "0. what the GRPO run actually did"
cat "$REPORTS/grpo_train_result.json" 2>/dev/null || echo "no grpo_train_result.json"

step "1. GRPO on the frozen benchmark"
if [ -s "$REPORTS/grpo_train_result.json" ] \
   && [ "$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["outcome"])' \
            "$REPORTS/grpo_train_result.json" 2>/dev/null)" = "completed" ]; then
  GRPO_RETRIEVAL="${GRPO_RETRIEVAL:-0}" bash "$PROJECT/scripts/30_eval_grpo.sh" \
    || echo "WARNING: GRPO evaluation returned non-zero; comparison will show 'not run'"
else
  echo "GRPO did not complete; no benchmark column will be produced for it."
  echo "The Base and SFT columns are unaffected and remain the reported result."
fi

step "2. comparison table (Base | SFT | GRPO)"
RUN_ARGS=()
[ -s "$REPORTS/qwen35_9b_zero_shot__report.json" ] && RUN_ARGS+=(--run "Base=$REPORTS/qwen35_9b_zero_shot__report.json")
[ -s "$REPORTS/qwen35_9b_adapter__report.json" ]   && RUN_ARGS+=(--run "SFT=$REPORTS/qwen35_9b_adapter__report.json")
[ -s "$REPORTS/qwen35_9b_grpo__report.json" ]      && RUN_ARGS+=(--run "GRPO=$REPORTS/qwen35_9b_grpo__report.json")
if [ ${#RUN_ARGS[@]} -gt 0 ]; then
  "$PY" "$ROOT/tools/make_report.py" "${RUN_ARGS[@]}" \
    --output "$REPORTS/comparison.json" --markdown "$REPORTS/comparison.md" \
    --note "Metrics shown as 'not run' were never measured; they are not zeros."
else
  echo "WARNING: no run reports found; comparison left untouched"
fi

step "3. leak check (GRPO reuses the train split, which is expected)"
"$PY" "$ROOT/tools/leak_check.py" --benchmark "$BENCH" \
  --curated "$ROOT/data/curated_v2" --manifest "$ROOT/data/prepared_v1/manifest.jsonl" \
  --grpo "$ROOT/data/grpo_v1/train.jsonl" --seed 3407 \
  --output "$REPORTS/leak_check.json" || echo "WARNING: leak check returned non-zero"

step "4. acceptance"
"$PY" "$ROOT/tools/final_acceptance.py" --root "$ROOT" --project "$PROJECT" \
  --output "$REPORTS/acceptance.json" || echo "WARNING: acceptance returned non-zero"

step "5. final report"
"$PY" "$ROOT/tools/final_report.py" --root "$ROOT" --output "$ROOT/FINAL_REPORT.md" \
  || echo "WARNING: final report returned non-zero"

echo "=== $(date) driver finished ==="
