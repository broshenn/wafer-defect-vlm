#!/usr/bin/env bash
# Merge the SFT adapter into the base weights and prove Adapter ~= Merged.
#
# The equivalence check runs a fixed subset through both paths and compares the
# generations, so a merge bug cannot hide behind aggregate benchmark scores.
set -euo pipefail

ROOT_DIR="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
ADAPTER_DIR="${ADAPTER_DIR:?Set ADAPTER_DIR to a completed ms-swift checkpoint}"
MERGED_DIR="${MERGED_DIR:-$ROOT_DIR/models/Qwen3.5-9B-sft-lora-v1-merged}"
BENCH_DIR="${BENCH_DIR:-$ROOT_DIR/benchmarks/wafer_bench_v1}"
CHECK_DIR="$ROOT_DIR/outputs/merge_check"
N="${N:-60}"
source "$ROOT_DIR/venvs/wafer/bin/activate"

mkdir -p "$CHECK_DIR"
head -n "$N" "$BENCH_DIR/all_requests.jsonl" > "$CHECK_DIR/subset${N}.jsonl"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-256}"

echo "=== merging adapter into base weights ==="
# --exist_ok so a retry after an interrupted merge regenerates the weights
# instead of dying with FileExistsError, which would otherwise stall the whole
# post-training chain at its first re-run. The equivalence check below is what
# decides whether the merged weights are actually usable.
swift export \
  --model "$ROOT_DIR/models/Qwen3.5-9B" \
  --adapters "$ADAPTER_DIR" \
  --merge_lora true \
  --safe_serialization true \
  --exist_ok true \
  --output_dir "$MERGED_DIR"

echo "=== inferring the subset through both paths ==="
for KIND in adapter merged; do
  if [ "$KIND" = adapter ]; then
    MODEL_ARGS=(--adapters "$ADAPTER_DIR")
  else
    MODEL_ARGS=(--model "$MERGED_DIR")
  fi
  swift infer \
    "${MODEL_ARGS[@]}" \
    --val_dataset "$CHECK_DIR/subset${N}.jsonl" \
    --result_path "$CHECK_DIR/${KIND}.jsonl" \
    --infer_backend transformers \
    --max_batch_size 4 \
    --temperature 0 \
    --max_new_tokens 512 \
    --stream false
done

echo "=== comparing ==="
python3 "$ROOT_DIR/tools/compare_results.py" \
  --a "$CHECK_DIR/adapter.jsonl" --b "$CHECK_DIR/merged.jsonl" \
  --output "$CHECK_DIR/equivalence.json"
