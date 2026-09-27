#!/usr/bin/env bash
# Wait for the SFT run to exit, then run the whole post-training chain.
#
# Launched with nohup so it keeps going if the ssh session drops. Every stage is
# idempotent: re-running this script picks up wherever it stopped instead of
# redoing completed inference.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PY="$VENV/python"
PROJECT="$ROOT/projects/wafer-defect-vlm"
BENCH="$ROOT/benchmarks/wafer_bench_v1"
REPORTS="$ROOT/outputs/reports"
DRIVER="$ROOT/logs/26_after_training.log"

mkdir -p "$REPORTS"
exec > >(tee -a "$DRIVER") 2>&1

echo "=== $(date) driver started; waiting for training to finish ==="
while pgrep -f "[1]0_train_qlora" > /dev/null; do
  sleep 60
done
echo "=== $(date) training process is gone ==="

tail -c 2000 "$ROOT/logs/08_train.log" || true

# A killed run leaves no final checkpoint; failing loudly beats producing a
# comparison table from half-trained weights without saying so.
LAST_CKPT=$(ls -d "$ROOT"/outputs/checkpoints/qwen35_9b_qlora_v1/*/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$LAST_CKPT" ]; then
  echo "FATAL: no checkpoint was written; nothing to evaluate." >&2
  exit 1
fi
echo "=== latest checkpoint: $LAST_CKPT ==="

# ------------------------------------------------------------------- post train
bash "$PROJECT/scripts/25_finalize.sh"

# ------------------------------------------------------------ trivial baselines
if [ ! -s "$REPORTS/trivial_baselines.json" ]; then
  echo "=== trivial baselines ==="
  "$PY" "$ROOT/tools/trivial_baselines.py" --benchmark "$BENCH" \
    --output "$REPORTS/trivial_baselines.json" --markdown "$REPORTS/trivial_baselines.md"
fi

# -------------------------------------------------------------------- provenance
echo "=== provenance ==="
"$PY" "$ROOT/tools/provenance.py" --root "$ROOT" --swift-src "$ROOT/src/ms-swift" \
  --artifact data/prepared_v1/manifest.jsonl \
  --artifact data/curated_v2/splits/train.jsonl \
  --artifact data/curated_v2/splits/val.jsonl \
  --artifact data/curated_v2/curation_report.json \
  --artifact benchmarks/wafer_bench_v1/all_requests.jsonl \
  --artifact benchmarks/wafer_bench_v1/SHA256SUMS \
  --deviation "GRPO/GSPO: GSPO is not offered by this ms-swift commit; GRPO is available but runs without vLLM because vLLM is not installed." \
  --deviation "Micro-batch 4 with gradient accumulation 8 replaces micro-batch 1 with accumulation 32; the effective batch of 32 is unchanged." \
  --deviation "deepseek-v4.1-flash free quota was exhausted; part of the teacher pass was served by deepseek-v3.2, recorded per row." \
  --deviation "venv and the ms-swift checkout live on the instance disk and are symlinked into autodl-fs, because FUSE file creation is ~850x slower." \
  --output "$REPORTS/provenance.json" > /dev/null

# --------------------------------------------------------------- comparison table
echo "=== comparison table ==="
RUN_ARGS=()
[ -s "$REPORTS/qwen35_9b_zero_shot__report.json" ] && RUN_ARGS+=(--run "Base=$REPORTS/qwen35_9b_zero_shot__report.json")
[ -s "$REPORTS/qwen35_9b_adapter__report.json" ] && RUN_ARGS+=(--run "SFT=$REPORTS/qwen35_9b_adapter__report.json")
[ -s "$REPORTS/qwen35_9b_merged_sft__report.json" ] && RUN_ARGS+=(--run "SFT-merged=$REPORTS/qwen35_9b_merged_sft__report.json")
if [ ${#RUN_ARGS[@]} -gt 0 ]; then
  "$PY" "$ROOT/tools/make_report.py" "${RUN_ARGS[@]}" \
    --output "$REPORTS/comparison.json" --markdown "$REPORTS/comparison.md" \
    --note "Metrics shown as 'not run' were never measured; they are not zeros."
else
  echo "WARNING: no run reports found; skipping comparison"
fi

# -------------------------------------------------------------------- acceptance
echo "=== acceptance ==="
"$PY" "$ROOT/tools/final_acceptance.py" --root "$ROOT" --project "$PROJECT" \
  --output "$REPORTS/acceptance.json"

# ------------------------------------------------------------------------- RL
# Runs last and on purpose: it is the most likely thing to fail, and by this
# point every required result is already on disk, so a hang or an OOM here
# cannot cost us the deliverables.
if [ "${RUN_GRPO:-1}" = "1" ]; then
  echo "=== GRPO smoke (bounded, failure recorded honestly) ==="
  bash "$PROJECT/scripts/27_grpo_smoke.sh" || echo "WARNING: GRPO smoke stage returned non-zero"
fi

# ------------------------------------------------------------------ final report
echo "=== final report ==="
"$PY" "$ROOT/tools/final_report.py" --root "$ROOT" --output "$ROOT/FINAL_REPORT.md"

echo "=== $(date) driver finished ==="
