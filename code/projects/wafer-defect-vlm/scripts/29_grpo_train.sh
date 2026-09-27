#!/usr/bin/env bash
# The real GRPO run, after the smoke proved the path works.
#
# Same honest-result contract as the smoke: whatever happens is written to
# outputs/reports/grpo_train_result.json, including failure. The memory watcher
# runs because the interesting question on a 48 GB card without vLLM is not only
# "did it fit" but "how close did it get".
#
# Rollouts go through ms-swift's TransformersEngine (no vLLM in this
# environment), so a step costs ~15-20 s and the run is budgeted by wall clock
# rather than by epochs. Checkpoints land every SAVE_STEPS so an interruption
# leaves a resumable adapter behind.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PROJECT="$ROOT/projects/wafer-defect-vlm"
RUN_TAG="${RUN_TAG:?RUN_TAG must be set: it names the result file and both logs, and a fixed default lets a later run overwrite the artefacts of an earlier one instead of failing}"
RESULT="$ROOT/outputs/reports/${RUN_TAG}_train_result.json"
LOG="$ROOT/logs/29_${RUN_TAG}_train.log"
MEMLOG="$ROOT/logs/29_${RUN_TAG}_mem.log"

# 150 steps at ~20 s/step is roughly 50 minutes, which fits the window and
# leaves a checkpoint-100 behind if the run has to be cut short.
MAX_STEPS="${MAX_STEPS:-150}"
# Algorithm flags. `sequence` importance sampling is GSPO; the ms-swift
# default is `token`, which is plain GRPO. Kept as variables so the two
# runs differ by a documented flag rather than by an edited script.
IS_LEVEL="${IS_LEVEL:-token}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SAVE_STEPS="${SAVE_STEPS:-50}"
TIMEOUT="${GRPO_TIMEOUT:-5400}"

mkdir -p "$ROOT/outputs/reports"
: > "$MEMLOG"
START=$(date +%s)

( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$MEMLOG" 2>/dev/null
    sleep 10
  done ) &
WATCHER=$!
trap 'kill $WATCHER 2>/dev/null' EXIT

set +e
SMOKE=0 MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
  RUN="$RUN_TAG" IS_LEVEL="$IS_LEVEL" LEARNING_RATE="$LEARNING_RATE" \
  NUM_GENERATIONS="$NUM_GENERATIONS" TEMPERATURE="$TEMPERATURE" \
  SEED="${SEED:-3407}" \
  timeout "$TIMEOUT" bash "$PROJECT/scripts/24_train_grpo.sh" > "$LOG" 2>&1
STATUS=$?
set -e
kill $WATCHER 2>/dev/null
PEAK=$(sort -n "$MEMLOG" 2>/dev/null | tail -1)
ELAPSED=$(( $(date +%s) - START ))

OUTCOME="failed"
REASON="see log"
if [ "$STATUS" -eq 0 ]; then
  OUTCOME="completed"
  REASON="${MAX_STEPS} steps finished without error"
elif [ "$STATUS" -eq 124 ]; then
  REASON="exceeded the ${TIMEOUT}s budget and was killed"
fi
grep -qi "out of memory" "$LOG" && REASON="CUDA out of memory"
grep -qi "ModuleNotFoundError\|ImportError" "$LOG" \
  && REASON="missing dependency: $(grep -i 'ModuleNotFoundError\|ImportError' "$LOG" | head -1)"

# The reward breakdown is the part that decides whether the run meant anything.
# With four generations per prompt, a reward whose per-group std is always zero
# contributes no advantage signal at all, so the run's real learning signal is
# whichever reward still varies.
"$VENV/python" - "$RESULT" "$OUTCOME" "$REASON" "$STATUS" "$ELAPSED" "${PEAK:-0}" \
                "$LOG" "$MAX_STEPS" "$SAVE_STEPS" "$ROOT" <<'PY'
import json, os, re, sys
from pathlib import Path

(result, outcome, reason, status, elapsed, peak, logpath,
 max_steps, save_steps, root) = sys.argv[1:11]

text = Path(logpath).read_text(encoding="utf-8", errors="replace") if Path(logpath).is_file() else ""
rows = re.findall(r"\{'loss'.*?\}", text)

def series(name, row):
    m = re.search(r"'%s': '([-0-9.eE]+)'" % re.escape(name), row)
    return float(m.group(1)) if m else None

steps = []
for row in rows:
    entry = {"loss": series("loss", row), "reward": series("reward", row),
             "reward_std": series("reward_std", row),
             "frac_reward_zero_std": series("frac_reward_zero_std", row),
             "kl": series("kl", row), "step_time": series("step_time", row),
             "memory_GiB": series("memory(GiB)", row)}
    for fn in ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock"):
        entry[fn] = series("rewards/%s/mean" % fn, row)
        entry[fn + "_std"] = series("rewards/%s/std" % fn, row)
    steps.append(entry)

# A reward that never varies inside a group cannot teach the policy anything;
# it is worth naming explicitly rather than leaving the reader to diff the log.
signal = {}
for fn in ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock"):
    stds = [s[fn + "_std"] for s in steps if s.get(fn + "_std") is not None]
    signal[fn] = {
        "mean_std_across_steps": (sum(stds) / len(stds)) if stds else None,
        "carries_group_signal": any(v and v > 0 for v in stds),
    }

def mean(key):
    vals = [s[key] for s in steps if s.get(key) is not None]
    return (sum(vals) / len(vals)) if vals else None

payload = {
    "outcome": outcome,
    "reason": reason,
    "exit_status": int(status),
    "elapsed_seconds": int(elapsed),
    "gpu_peak_mib": int(peak or 0),
    "log": logpath,
    "max_steps_requested": int(max_steps),
    "save_steps": int(save_steps),
    "steps_logged": len(steps),
    "first_step": steps[0] if steps else None,
    "last_step": steps[-1] if steps else None,
    "mean_reward": mean("reward"),
    "mean_kl": mean("kl"),
    "mean_memory_GiB": mean("memory_GiB"),
    "mean_step_seconds": mean("step_time"),
    "reward_signal": signal,
    "config": {
        "rlhf_type": "grpo",
        "use_vllm": False,
        "note": "vLLM is not installed, so rollouts use ms-swift's TransformersEngine",
        "num_generations": int(os.environ.get("NUM_GENERATIONS", 4)),
        "per_device_batch": 1,
        "grad_accum": int(os.environ.get("GRAD_ACCUM", 4)),
        "grad_accum_source": (
            "GRAD_ACCUM environment variable"
            if "GRAD_ACCUM" in os.environ else
            "DEFAULT 4 -- GRAD_ACCUM was not exported by the caller, so this "
            "value is not evidence about how the run was actually configured"),
        "learning_rate": float(os.environ.get("LEARNING_RATE", 1e-5)),
        "beta": 0.04,
        "temperature": 1.0,
        "seed": int(os.environ.get("SEED", 3407)),
        "reward_funcs": ["wafer_class", "wafer_format", "wafer_radial", "wafer_clock"],
        "reward_weights": [1.0, 0.2, 0.4, 0.4],
        "rewards_are_deterministic": True,
        "importance_sampling_level": os.environ.get("IS_LEVEL", "token"),
        "algorithm": ("GSPO (sequence-level importance sampling)"
                      if os.environ.get("IS_LEVEL") == "sequence" else "GRPO"),
        "gspo_note": ("GSPO is not a separate rlhf_type in ms-swift; it is GRPO with "
                      "--importance_sampling_level sequence. An earlier revision of "
                      "this file claimed GSPO was unavailable because the rlhf_type "
                      "choices exclude the name gspo -- that was wrong."),
    },
}
# argv[1], not a literal: the launcher computes a per-run path and passes it
# in, so a hardcoded write here silently collides between runs. It did --
# twice, both times overwriting the GRPO record.
out = Path(result)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: payload[k] for k in
                  ("outcome", "reason", "elapsed_seconds", "gpu_peak_mib",
                   "steps_logged", "mean_reward", "mean_kl", "mean_step_seconds")}, indent=2))
print("reward_signal:", json.dumps(signal, indent=2))
PY
