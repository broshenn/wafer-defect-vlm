#!/usr/bin/env bash
# Score a trained LoRA adapter on the full frozen benchmark (core + robustness).
#
# This deliberately scores all_requests.jsonl, the same file the zero-shot
# baseline used, so Base and SFT numbers come from identical prompts.
set -euo pipefail

ROOT_DIR="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
ADAPTER_DIR="${ADAPTER_DIR:?Set ADAPTER_DIR to a completed ms-swift checkpoint}"
BENCH_DIR="${BENCH_DIR:-$ROOT_DIR/benchmarks/wafer_bench_v1}"
RUN_NAME="${RUN_NAME:-qwen35_9b_adapter}"
RESULT_PATH="${RESULT_PATH:-$ROOT_DIR/outputs/baselines/$RUN_NAME.jsonl}"
REQUESTS="${REQUESTS:-$BENCH_DIR/all_requests.jsonl}"
source "$ROOT_DIR/venvs/wafer/bin/activate"

test -f "$ADAPTER_DIR/args.json"
test -s "$REQUESTS"
mkdir -p "$(dirname "$RESULT_PATH")"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-256}"

swift infer \
  --adapters "$ADAPTER_DIR" \
  --val_dataset "$REQUESTS" \
  --result_path "$RESULT_PATH" \
  --infer_backend transformers \
  --max_batch_size "${MAX_BATCH_SIZE:-8}" \
  --temperature 0 \
  --max_new_tokens 512 \
  --stream false
