#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
PROJECT_DIR="$ROOT_DIR/projects/wafer-defect-vlm"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Qwen3.5-9B}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/curated_v1/splits}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/checkpoints/qwen35_9b_vision_lora_v1}"
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
  --tuner_type lora_llm \
  --target_modules all-linear \
  --freeze_llm false \
  --freeze_vit false \
  --freeze_aligner false \
  --torch_dtype bfloat16 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --learning_rate 5e-5 \
  --vit_lr 1e-5 \
  --aligner_lr 1e-5 \
  --num_train_epochs 1 \
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
