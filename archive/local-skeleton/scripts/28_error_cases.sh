#!/usr/bin/env bash
# Collect success / failure / high-confidence-error cases with real confidences.
#
# The classification split is re-run with --logprobs so each answer carries the
# model's own token probabilities. Without that the error-case report can still
# say which cases were wrong, but it cannot honestly call any of them
# confident, and "wrong" and "wrong while certain" are different findings.
#
# This is a second pass over 252 prompts for each model, so it runs after the
# main chain has released the GPU rather than competing with it.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
BENCH="$ROOT/benchmarks/wafer_bench_v1"
REPORTS="$ROOT/outputs/reports"
BASELINES="$ROOT/outputs/baselines"
ADAPTER="${ADAPTER:-}"
TOP_LOGPROBS="${TOP_LOGPROBS:-5}"

mkdir -p "$REPORTS" "$BASELINES"

# A classification-only request file, keeping the order of all_requests.jsonl.
# Alignment in the scorer is positional, so the results come back in the order
# of whatever file was passed here, and the scorer is told to align against this
# same file rather than the full benchmark.
CLS_REQUESTS="$REPORTS/aux_classification_requests.jsonl"
if [ ! -s "$CLS_REQUESTS" ]; then
  "$VENV/python" - "$BENCH/all_requests.jsonl" "$CLS_REQUESTS" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as fh, open(dst, "w", encoding="utf-8", newline="\n") as out:
    for line in fh:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task") == "classification":
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
fi
echo "classification requests: $(wc -l < "$CLS_REQUESTS")"

run_one() {
  local name="$1"; shift
  local preds="$BASELINES/${name}_cls_logprobs.jsonl"
  if [ ! -s "$preds" ]; then
    echo "=== logprob inference: $name ==="
    "$VENV/swift" infer "$@" \
      --val_dataset "$CLS_REQUESTS" \
      --result_path "$preds" \
      --infer_backend transformers \
      --max_batch_size 8 \
      --temperature 0 \
      --max_new_tokens 512 \
      --logprobs true \
      --top_logprobs "$TOP_LOGPROBS" \
      --stream false || { echo "WARNING: inference failed for $name"; return 1; }
  else
    echo "skip: $preds exists"
  fi
  "$VENV/python" "$ROOT/tools/error_cases.py" \
    --benchmark "$BENCH" --predictions "$preds" --requests "$CLS_REQUESTS" \
    --output "$REPORTS/error_cases_${name}.json" \
    --markdown "$REPORTS/error_cases_${name}.md" || echo "WARNING: scoring failed for $name"
}

export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-256}"

run_one base --model "$ROOT/models/Qwen3.5-9B"

if [ -n "$ADAPTER" ] && [ -d "$ADAPTER" ]; then
  run_one sft --adapters "$ADAPTER"
else
  echo "ADAPTER not set or missing; skipping the SFT error-case pass"
fi

echo "=== error-case reports ==="
ls -l "$REPORTS"/error_cases_*.json 2>/dev/null
