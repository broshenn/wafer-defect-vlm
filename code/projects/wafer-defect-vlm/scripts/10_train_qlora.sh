#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
PROJECT_DIR="$ROOT_DIR/projects/wafer-defect-vlm"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Qwen3.5-9B}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/curated_v2/splits}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/checkpoints/qwen35_9b_qlora_v1}"
# Epochs and max_steps are knobs because the main run is time-boxed; the values
# actually used are recorded in the run manifest written next to the checkpoints.
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-300}"
# micro-batch 4 / accum 8 keeps the specified effective batch of 32 while cutting
# wall-clock ~3x versus micro-batch 1: measured 12.05 GiB vs 9.12 GiB peak on the
# 48GB card, so the larger micro-batch costs almost nothing in memory.
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
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
  --quant_method bnb \
  --quant_bits 4 \
  --bnb_4bit_compute_dtype bfloat16 \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_use_double_quant true \
  --torch_dtype bfloat16 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --learning_rate 1e-4 \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --max_length 2048 \
  --warmup_ratio 0.05 \
  --gradient_checkpointing true \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 8 \
  --logging_steps 5 \
  --dataset_num_proc 1 \
  --dataloader_num_workers 1 \
  --seed 3407 \
  --output_dir "$OUTPUT_DIR"
