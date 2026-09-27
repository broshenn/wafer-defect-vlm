#!/usr/bin/env bash
# Launch the main QLoRA SFT run and record the exact parameters used.
set -uo pipefail
F=/root/autodl-fs/wafer-vlm
V=$F/venvs/wafer/bin/python
RUN=${RUN:-qwen35_9b_qlora_v1}
OUT=$F/outputs/checkpoints/$RUN
DATA=$F/data/curated_v2/splits
MODEL=$F/models/Qwen3.5-9B
EPOCHS=${NUM_TRAIN_EPOCHS:-1}
STEPS=${MAX_STEPS:-300}
BS=${PER_DEVICE_BATCH:-4}
GA=${GRAD_ACCUM:-8}
mkdir -p "$OUT"

if [ ! -s "$DATA/train.jsonl" ]; then echo "FATAL: $DATA/train.jsonl missing"; exit 1; fi
NTRAIN=$(wc -l < "$DATA/train.jsonl")
NVAL=$(wc -l < "$DATA/val.jsonl")
TRAIN_SHA=$(sha256sum "$DATA/train.jsonl" | cut -d" " -f1)
VAL_SHA=$(sha256sum "$DATA/val.jsonl" | cut -d" " -f1)
DRAWN=$(python3 -c "print($STEPS*$BS*$GA)" 2>/dev/null || echo 0)

cat > "$OUT/run_manifest.json" <<EOG
{
  "run": "$RUN",
  "started": "$(date -Iseconds)",
  "base_model": "$MODEL",
  "curation": {
    "dir": "$F/data/curated_v2",
    "arbitrate_disagreement": true,
    "geometry_gold_structure": true,
    "note": "geometry arbitrates passing-but-disagreeing teachers; the four geometry-scored structured fields are written from trusted label + features"
  },
  "train_jsonl": "$DATA/train.jsonl",
  "train_sha256": "$TRAIN_SHA",
  "train_examples_in_file": $NTRAIN,
  "val_jsonl": "$DATA/val.jsonl",
  "val_sha256": "$VAL_SHA",
  "val_examples": $NVAL,
  "tuner_type": "lora",
  "quant_method": "bnb",
  "quant_bits": 4,
  "bnb_4bit_quant_type": "nf4",
  "bnb_4bit_compute_dtype": "bfloat16",
  "bnb_4bit_use_double_quant": true,
  "torch_dtype": "bfloat16",
  "lora_rank": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": "all-linear",
  "freeze_vit": true,
  "freeze_aligner": true,
  "learning_rate": 1e-4,
  "num_train_epochs": $EPOCHS,
  "max_steps": $STEPS,
  "per_device_train_batch_size": $BS,
  "per_device_eval_batch_size": 4,
  "gradient_accumulation_steps": $GA,
  "effective_batch_size": $((BS*GA)),
  "eval_steps": 100,
  "save_steps": 100,
  "deviation_from_spec": "spec asked micro-batch 1 / accum 32 (effective 32). Effective batch kept at 32; micro-batch raised to $BS for throughput. Measured peak 12.05 GiB vs 9.12 GiB at micro-batch 1, so this is a time-budget change, not a memory one.",
  "max_length": 2048,
  "image_max_token_num": 256,
  "gradient_checkpointing": true,
  "warmup_ratio": 0.05,
  "seed": 3407,
  "output_dir": "$OUT"
}
EOG
echo "=== run_manifest.json ==="; cat "$OUT/run_manifest.json"
echo "=== launching ==="
cd "$F/projects/wafer-defect-vlm"
export IMAGE_MAX_TOKEN_NUM=256
export WAFER_ROOT=$F
NUM_TRAIN_EPOCHS=$EPOCHS MAX_STEPS=$STEPS PER_DEVICE_BATCH=$BS GRAD_ACCUM=$GA \
  bash scripts/10_train_qlora.sh
echo "TRAIN_EXIT=$? $(date -Iseconds)"
