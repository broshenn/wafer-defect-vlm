#!/usr/bin/env bash
# GRPO on the structured-output task, starting from the SFT adapter.
#
# vLLM is not installed in this environment, so rollouts go through ms-swift's
# TransformersEngine (`--use_vllm false`). That is a supported path but it is
# far slower than vLLM, which is why the smoke run comes first and the budget is
# capped by wall-clock rather than by steps.
#
# Rewards are deterministic: the trusted class label plus the circle-fit
# geometry. No judge model is involved.
set -euo pipefail

ROOT="${ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="${VENV:-/root/autodl-tmp/wafer-vlm/venvs/wafer}"
SWIFT_SRC="${SWIFT_SRC:-/root/autodl-tmp/wafer-vlm/src/ms-swift}"
export PYTHONPATH="$SWIFT_SRC:${PYTHONPATH:-}"

BASE_MODEL="${BASE_MODEL:-$ROOT/models/Qwen3.5-9B}"
SFT_CKPT="${SFT_CKPT:-$ROOT/outputs/checkpoints/qwen35_9b_qlora_v1}"
PLUGIN="${PLUGIN:-$ROOT/tools/wafer_grpo_plugin.py}"
DATASET="${DATASET:-$ROOT/data/grpo_v1/train.jsonl}"
RUN="${RUN:-qwen35_9b_grpo_v1}"

# SMOKE=1 trains on a 20-row slice for 2 steps: enough to prove the rollout,
# the reward wiring and the backward pass all work before spending real time.
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  DATASET="$ROOT/data/grpo_v1/smoke20.jsonl"
  head -n 20 "$ROOT/data/grpo_v1/train.jsonl" > "$DATASET"
  RUN="${RUN}_smoke"
fi

NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_STEPS="${MAX_STEPS:-2}"
MAX_COMPLETION="${MAX_COMPLETION:-256}"
SAVE_STEPS="${SAVE_STEPS:-50}"
OUT="$ROOT/outputs/checkpoints/$RUN"
LOG="$ROOT/logs/24_grpo_${RUN}.log"

mkdir -p "$OUT" "$ROOT/logs"

# ms-swift insists on an actual adapter directory: handing it the run directory
# fails with "is not an adapter". Resolve a run directory to its newest
# checkpoint that really holds adapter weights.
#
# The filter matters. Merging writes `checkpoint-604-merged` next to
# `checkpoint-604`, and version-sort puts it last, so a plain
# `ls -d checkpoint-* | sort -V | tail -1` hands ms-swift a full merged model
# and the run dies with "is not an adapter". Only directories containing
# adapter_config.json qualify.
#
# `find` rather than a shell loop, because under `set -e` the loop was an
# abort: its last iteration is `checkpoint-604-merged`, where the `[ -f ]` test
# fails, so the loop returns 1, `pipefail` propagates that, and the assignment
# kills the script before it can print anything — an exit status of 1, an empty
# log and no Python process, which is exactly how this failed the first time.
# find/sort/tail all exit 0 whether or not they match, so it cannot abort.
if [ ! -f "$SFT_CKPT/adapter_config.json" ] && [ ! -d "$SFT_CKPT/default" ]; then
  RESOLVED=$(find "$SFT_CKPT" -maxdepth 3 -name adapter_config.json -printf '%h\n' 2>/dev/null \
             | sort -V | tail -1)
  if [ -n "$RESOLVED" ]; then
    echo "resolved run directory $SFT_CKPT to checkpoint $RESOLVED"
    SFT_CKPT="$RESOLVED"
  else
    echo "SFT adapter not found under $SFT_CKPT; GRPO must start from a finished SFT run." >&2
    exit 1
  fi
fi

# The flags live in one array so the offline validator and the real run cannot
# disagree about what was requested.
ARGS=(
  --rlhf_type grpo
  --model "$BASE_MODEL"
  --adapters "$SFT_CKPT"
  --tuner_type lora
  --target_modules all-linear
  --freeze_vit true
  --freeze_aligner true
  --quant_method bnb
  --quant_bits 4
  --bnb_4bit_compute_dtype bfloat16
  --bnb_4bit_quant_type nf4
  --bnb_4bit_use_double_quant true
  --torch_dtype bfloat16
  --use_vllm false
  --dataset "$DATASET"
  --external_plugins "$PLUGIN"
  --reward_funcs wafer_class wafer_format wafer_radial wafer_clock
  --reward_weights 1.0 0.2 0.4 0.4
  --num_generations "$NUM_GENERATIONS"
  --per_device_train_batch_size "$PER_DEVICE_BATCH"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --learning_rate "${LEARNING_RATE:-1e-5}"
  --beta "${BETA:-0.04}"
  --temperature "${TEMPERATURE:-1.0}"
  --num_train_epochs 1
  --max_steps "$MAX_STEPS"
  --max_length 2048
  --max_completion_length "$MAX_COMPLETION"
  --gradient_checkpointing true
  --logging_steps 1
  --save_steps "$SAVE_STEPS"
  --save_total_limit 2
  --dataset_num_proc 1
  --dataloader_num_workers 1
  --seed 3407
  --report_to none
  --output_dir "$OUT"
)

# VALIDATE_ONLY=1 checks the flags against ms-swift's parser on the CPU and
# exits, so an unsupported option is caught without loading a model.
if [ "${VALIDATE_ONLY:-0}" = "1" ]; then
  SWIFT_SRC="$SWIFT_SRC" "$VENV/bin/python" "$ROOT/tools/validate_rlhf_args.py" "${ARGS[@]}"
  exit $?
fi

"$VENV/bin/swift" rlhf "${ARGS[@]}" 2>&1 | tee "$LOG"
