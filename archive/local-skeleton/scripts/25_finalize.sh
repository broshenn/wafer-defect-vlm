#!/usr/bin/env bash
# Drive everything that has to happen after SFT finishes.
#
# Each stage is skipped when its output already exists, so the script can be
# re-run after an interruption without redoing an hour of inference. Delete the
# relevant output file to force a stage to run again.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PROJECT="$ROOT/projects/wafer-defect-vlm"
BENCH="$ROOT/benchmarks/wafer_bench_v1"
REPORTS="$ROOT/outputs/reports"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/checkpoints/qwen35_9b_qlora_v1}"
ADAPTER="${ADAPTER:-}"
MERGED="${MERGED:-$ROOT/models/Qwen3.5-9B-sft-lora-v1-merged}"

export PATH="$VENV:$PATH"
mkdir -p "$REPORTS"

step() { printf '\n========== %s ==========\n' "$1"; }

# ---------------------------------------------------------------- 1. checkpoint
step "1. choose the best checkpoint by validation loss"
if [ -z "$ADAPTER" ]; then
  BEST=$("$VENV/python" "$ROOT/tools/pick_checkpoint.py" --run-dir "$RUN_DIR" \
           --log "$ROOT/logs/08_train.log" --output "$REPORTS/checkpoint_choice.json" 2>/dev/null \
         | "$VENV/python" -c 'import json,sys; print(json.load(sys.stdin).get("best_checkpoint") or "")')
  if [ -z "$BEST" ]; then
    # No eval_loss recorded; the newest checkpoint is the only defensible choice,
    # and we say so rather than implying it was selected on merit.
    BEST=$(ls -d "$RUN_DIR"/*/checkpoint-* 2>/dev/null | sort -V | tail -1)
    echo "WARNING: no eval_loss found; falling back to the last checkpoint: $BEST"
  fi
  ADAPTER="$BEST"
fi
echo "ADAPTER=$ADAPTER"
if [ -z "$ADAPTER" ] || [ ! -d "$ADAPTER" ]; then
  echo "FATAL: no usable adapter checkpoint under $RUN_DIR" >&2
  exit 1
fi

# ------------------------------------------------------- 2. benchmark (adapter)
step "2. adapter on the full frozen benchmark"
ADAPTER_RESULTS="$ROOT/outputs/baselines/qwen35_9b_adapter.jsonl"
if [ -s "$ADAPTER_RESULTS" ]; then
  echo "skip: $ADAPTER_RESULTS exists"
else
  ADAPTER_DIR="$ADAPTER" "$PROJECT/scripts/22_eval_adapter.sh"
fi

step "2b. score the adapter"
if [ -s "$REPORTS/qwen35_9b_adapter__report.json" ]; then
  echo "skip: report exists"
else
  "$VENV/python" -m wafer_vlm.cli evaluate \
    --benchmark "$BENCH" --predictions "$ADAPTER_RESULTS" \
    --output "$REPORTS/qwen35_9b_adapter__report.json"
fi

# ---------------------------------------------------- 2c. training-format probe
step "2c. adapter on the training-format prompt (diagnostic, not the headline)"
AUX_RESULTS="$ROOT/outputs/baselines/qwen35_9b_adapter_trainfmt.jsonl"
if [ -s "$AUX_RESULTS" ]; then
  echo "skip: $AUX_RESULTS exists"
else
  "$VENV/swift" infer \
    --adapters "$ADAPTER" \
    --val_dataset "$REPORTS/aux_trainfmt_requests.jsonl" \
    --result_path "$AUX_RESULTS" \
    --infer_backend transformers --max_batch_size 8 \
    --temperature 0 --max_new_tokens 512 --stream false || echo "WARNING: training-format probe failed; continuing"
fi
if [ -s "$AUX_RESULTS" ]; then
  "$VENV/python" "$ROOT/tools/trainfmt_check.py" score \
    --benchmark "$BENCH" --predictions "$AUX_RESULTS" \
    --requests "$REPORTS/aux_trainfmt_requests.jsonl" \
    --output "$REPORTS/trainfmt_diagnostic.json" || true
fi

# --------------------------------------------------------------- 3. merge check
step "3. merge the adapter and prove Adapter ~= Merged"
if [ -s "$ROOT/outputs/merge_check/equivalence.json" ]; then
  echo "skip: equivalence.json exists"
else
  ADAPTER_DIR="$ADAPTER" MERGED_DIR="$MERGED" "$PROJECT/scripts/23_merge_and_check.sh"
fi

# ------------------------------------------------------------------ 4. retrieval
step "4. retrieval rankings (base, then merged SFT)"
for KIND in base merged; do
  if [ "$KIND" = "base" ]; then
    MODEL="$ROOT/models/Qwen3.5-9B"; OUT="$ROOT/outputs/retrieval/base_rankings.jsonl"
  else
    MODEL="$MERGED"; OUT="$ROOT/outputs/retrieval/merged_rankings.jsonl"
  fi
  if [ -s "$OUT" ]; then echo "skip: $OUT exists"; continue; fi
  if [ ! -d "$MODEL" ]; then echo "skip: $MODEL not present"; continue; fi
  "$VENV/python" "$ROOT/tools/retrieval_rank.py" \
    --model "$MODEL" --benchmark "$BENCH" --output "$OUT" --batch-size 4 --top-k 50 \
    || echo "WARNING: retrieval for $KIND failed"
done

step "4b. re-score Base and SFT including retrieval"
# The SFT retrieval vectors come from the merged model because the ranking tool
# embeds with a plain transformers model. Merged is the same weights the adapter
# produces, which is exactly what the equivalence check in step 3 verifies.
for KIND in base sft; do
  if [ "$KIND" = "base" ]; then
    RANKINGS="$ROOT/outputs/retrieval/base_rankings.jsonl"
    PRED="$ROOT/outputs/baselines/qwen35_9b_zero_shot.jsonl"
    OUT="$REPORTS/qwen35_9b_zero_shot__report.json"
  else
    RANKINGS="$ROOT/outputs/retrieval/merged_rankings.jsonl"
    PRED="$ADAPTER_RESULTS"
    OUT="$REPORTS/qwen35_9b_adapter__report.json"
  fi
  [ -s "$RANKINGS" ] || { echo "skip: no rankings for $KIND"; continue; }
  [ -s "$PRED" ] || { echo "skip: predictions missing for $KIND ($PRED)"; continue; }
  "$VENV/python" -m wafer_vlm.cli evaluate \
    --benchmark "$BENCH" --predictions "$PRED" --rankings "$RANKINGS" --output "$OUT"
done

# Optional and expensive (~40 min): the full benchmark through the merged
# weights. Step 3 already proves the two paths agree on 60 samples, so this is
# redundancy rather than evidence, and it only runs when asked for.
if [ "${RUN_MERGED_BENCH:-0}" = "1" ]; then
  step "5. optional: full benchmark through the merged model"
  MERGED_RESULTS="$ROOT/outputs/baselines/qwen35_9b_merged_sft.jsonl"
  if [ -s "$MERGED_RESULTS" ]; then
    echo "skip: $MERGED_RESULTS exists"
  else
    "$VENV/swift" infer --model "$MERGED" --val_dataset "$BENCH/all_requests.jsonl" \
      --result_path "$MERGED_RESULTS" --infer_backend transformers --max_batch_size 8 \
      --temperature 0 --max_new_tokens 512 --stream false
  fi
  "$VENV/python" -m wafer_vlm.cli evaluate --benchmark "$BENCH" --predictions "$MERGED_RESULTS" \
    --rankings "$ROOT/outputs/retrieval/merged_rankings.jsonl" \
    --output "$REPORTS/qwen35_9b_merged_sft__report.json"
fi

printf '\n========== finalize stages complete ==========\n'
