#!/usr/bin/env bash
# 20-example end-to-end ms-swift SFT smoke. Proves the curated JSONL loads under
# the qwen3_5 template, that one forward+backward+save cycle fits in 48GB, and
# logs steady-state per-example cost so the main run can be sized to the clock.
# gradient_accumulation_steps=1 here on purpose: it makes s/step equal the cost
# of a single training example, which is what the main run is budgeted from.
set -uo pipefail
F=/root/autodl-fs/wafer-vlm
S=$F/data/curated_v2/splits
SM=$F/outputs/smoke/sft_smoke
mkdir -p "$SM"
head -n 20 "$S/train.jsonl" > "$SM/train20.jsonl"
head -n 4  "$S/val.jsonl"   > "$SM/val4.jsonl"
echo "train20=$(wc -l < "$SM/train20.jsonl") val4=$(wc -l < "$SM/val4.jsonl")"
cd "$F/projects/wafer-defect-vlm"
export IMAGE_MAX_TOKEN_NUM=256
$F/venvs/wafer/bin/swift sft \
  --model "$F/models/Qwen3.5-9B" \
  --dataset "$SM/train20.jsonl" --val_dataset "$SM/val4.jsonl" \
  --tuner_type lora --target_modules all-linear \
  --freeze_vit true --freeze_aligner true \
  --quant_method bnb --quant_bits 4 --bnb_4bit_compute_dtype bfloat16 \
  --bnb_4bit_quant_type nf4 --bnb_4bit_use_double_quant true \
  --torch_dtype bfloat16 --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --learning_rate 1e-4 --per_device_train_batch_size "${SMOKE_BS:-1}" \
  --gradient_accumulation_steps 1 \
  --max_length 2048 --gradient_checkpointing true \
  --logging_steps 1 --save_steps 2 --save_total_limit 1 \
  --report_to none --dataset_num_proc 1 --dataloader_num_workers 1 --seed 3407 \
  --max_steps 4 --output_dir "$SM/ckpt"
echo "SMOKE_EXIT=$?"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
