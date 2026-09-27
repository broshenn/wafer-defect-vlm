#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
PROJECT_DIR="$ROOT_DIR/projects/wafer-defect-vlm"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Qwen3.5-9B}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/curated_v1/splits}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/checkpoints/qwen35_9b_lora_bf16_v1}"
source "$ROOT_DIR/venvs/wafer/bin/activate"

test -f "$MODEL_DIR/config.json"
test -s "$DATA_DIR/train.jsonl"
test -s "$DATA_DIR/val.jsonl"
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-256}"
swift sft \
  --model "$MODEL_DIR" \
  --dataset "$DATA_DIR/train.jsonl" \
  --val_dataset "$DATA_DIR/val.jsonl" \
  --tuner_type lora \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype bfloat16 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --learning_rate 1e-4 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --max_length 2048 \
  --warmup_ratio 0.05 \
  --gradient_checkpointing true \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 3 \
  --logging_steps 5 \
  --dataset_num_proc 1 \
  --dataloader_num_workers 1 \
  --seed 3407 \
  --output_dir "$OUTPUT_DIR"
