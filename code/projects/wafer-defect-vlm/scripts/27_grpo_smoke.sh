#!/usr/bin/env bash
# Run the GRPO smoke test and record what actually happened.
#
# GRPO is the part of this project most likely to fail on a 48 GB card without
# vLLM, so the outcome is written to a JSON result either way. A failure is
# recorded as a failure with its reason, never smoothed over: "we could not run
# it" and "it worked" are different findings and the report needs both.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PROJECT="$ROOT/projects/wafer-defect-vlm"
RESULT="$ROOT/outputs/reports/grpo_result.json"
LOG="$ROOT/logs/27_grpo_smoke.log"
TIMEOUT="${GRPO_TIMEOUT:-2700}"   # 45 min: enough to prove or disprove viability

mkdir -p "$ROOT/outputs/reports"
START=$(date +%s)

# Memory is watched because the honest question is not just "did it run" but
# "how close to the card's limit did it get".
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$ROOT/logs/27_grpo_mem.log" 2>/dev/null
    sleep 10
  done ) &
WATCHER=$!
trap 'kill $WATCHER 2>/dev/null' EXIT

set +e
SMOKE=1 timeout "$TIMEOUT" bash "$PROJECT/scripts/24_train_grpo.sh" > "$LOG" 2>&1
STATUS=$?
set -e
kill $WATCHER 2>/dev/null
PEAK=$(sort -n "$ROOT/logs/27_grpo_mem.log" 2>/dev/null | tail -1)
ELAPSED=$(( $(date +%s) - START ))

OUTCOME="failed"
REASON="see log"
if [ "$STATUS" -eq 0 ]; then
  OUTCOME="smoke_completed"
  REASON="2 steps finished without error"
elif [ "$STATUS" -eq 124 ]; then
  REASON="exceeded the ${TIMEOUT}s budget and was killed"
fi
grep -qi "out of memory" "$LOG" && REASON="CUDA out of memory"
grep -qi "ModuleNotFoundError\|ImportError" "$LOG" && REASON="missing dependency: $(grep -i 'ModuleNotFoundError\|ImportError' "$LOG" | head -1)"

"$VENV/python" - "$RESULT" "$OUTCOME" "$REASON" "$STATUS" "$ELAPSED" "${PEAK:-0}" "$LOG" <<'PY'
import json, sys
result, outcome, reason, status, elapsed, peak, log = sys.argv[1:8]
payload = {
    "outcome": outcome,
    "reason": reason,
    "exit_status": int(status),
    "elapsed_seconds": int(elapsed),
    "gpu_peak_mib": int(peak or 0),
    "log": log,
    "config": {
        "rlhf_type": "grpo",
        "use_vllm": False,
        "note": "vLLM is not installed, so rollouts use ms-swift's TransformersEngine",
        "reward_funcs": ["wafer_class", "wafer_format", "wafer_radial", "wafer_clock"],
        "rewards_are_deterministic": True,
        "gspo": "not offered by this ms-swift commit; rlhf_type choices exclude gspo",
    },
}
open(result, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=2))
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
