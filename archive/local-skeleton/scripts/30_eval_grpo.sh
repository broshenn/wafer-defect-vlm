#!/usr/bin/env bash
# Score the GRPO adapter on the identical frozen benchmark the Base and SFT runs
# used, so the three columns of the comparison table come from the same prompts.
#
# Retrieval is opt-in (RUN_GRPO_RETRIEVAL=1) and expensive: the ranking tool
# embeds with a plain transformers model, so a LoRA-only adapter has to be
# merged into bf16 first. Base and SFT retrieval were measured that way, which
# is why it is the comparable path rather than an adapter shortcut.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PROJECT="$ROOT/projects/wafer-defect-vlm"
BENCH="$ROOT/benchmarks/wafer_bench_v1"
REPORTS="$ROOT/outputs/reports"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/checkpoints/qwen35_9b_grpo_v1}"
MERGED="${MERGED:-$ROOT/models/Qwen3.5-9B-grpo-v1-merged}"

export PATH="$VENV:$PATH"
mkdir -p "$REPORTS" "$ROOT/outputs/baselines"
step() { printf '\n========== %s ==========\n' "$1"; }

# ------------------------------------------------------------- 1. checkpoint
step "1. newest GRPO checkpoint that actually holds adapter weights"
if [ -z "${ADAPTER:-}" ]; then
  # `find` and not a shell loop: under `set -e` a loop whose last iteration
  # fails its test returns 1 and aborts the script silently. This exact bug hid
  # a GRPO launch behind an empty log and an exit status of 1 once already.
  ADAPTER=$(find "$RUN_DIR" -maxdepth 3 -name adapter_config.json -printf '%h\n' 2>/dev/null \
            | sort -V | tail -1)
fi
if [ -z "$ADAPTER" ] || [ ! -f "$ADAPTER/adapter_config.json" ]; then
  echo "FATAL: no GRPO adapter under $RUN_DIR; did the training run finish?" >&2
  exit 1
fi
echo "ADAPTER=$ADAPTER"

# ------------------------------------------------------------------ 2. infer
step "2. GRPO adapter on the full frozen benchmark"
GRPO_RESULTS="$ROOT/outputs/baselines/qwen35_9b_grpo.jsonl"
if [ -s "$GRPO_RESULTS" ]; then
  echo "skip: $GRPO_RESULTS exists"
else
  ADAPTER_DIR="$ADAPTER" RUN_NAME=qwen35_9b_grpo \
    RESULT_PATH="$GRPO_RESULTS" REQUESTS="$BENCH/all_requests.jsonl" \
    "$PROJECT/scripts/22_eval_adapter.sh" || {
      echo "FATAL: inference failed" >&2; exit 1; }
fi

step "2b. score it"
"$VENV/python" -m wafer_vlm.cli evaluate \
  --benchmark "$BENCH" --predictions "$GRPO_RESULTS" \
  --output "$REPORTS/qwen35_9b_grpo__report.json" || exit 1

# --------------------------------------------------------------- 3. retrieval
if [ "${RUN_GRPO_RETRIEVAL:-0}" = "1" ]; then
  step "3. merge GRPO to bf16, then rank (same path Base and SFT used)"
  if [ ! -d "$MERGED" ]; then
    "$VENV/swift" export --model "$ROOT/models/Qwen3.5-9B" --adapters "$ADAPTER" \
      --merge_lora true --safe_serialization true --exist_ok true \
      --output_dir "$MERGED" || echo "WARNING: GRPO merge failed; retrieval stays unmeasured"
  fi
  RANK="$ROOT/outputs/retrieval/grpo_rankings.jsonl"
  if [ -d "$MERGED" ] && [ ! -s "$RANK" ]; then
    "$VENV/python" "$ROOT/tools/retrieval_rank.py" --model "$MERGED" \
      --benchmark "$BENCH" --output "$RANK" --batch-size 4 --top-k 50 \
      || echo "WARNING: GRPO retrieval failed"
  fi
  if [ -s "$RANK" ]; then
    step "3b. re-score GRPO including retrieval"
    "$VENV/python" -m wafer_vlm.cli evaluate --benchmark "$BENCH" \
      --predictions "$GRPO_RESULTS" --rankings "$RANK" \
      --output "$REPORTS/qwen35_9b_grpo__report.json"
  else
    echo "note: no GRPO rankings; retrieval metrics stay 'not run' for this row"
  fi
fi

printf '\n========== GRPO evaluation complete ==========\n'
