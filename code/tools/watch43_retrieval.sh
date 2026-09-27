#!/bin/bash
# Run 43's eval writes its report at step 2b, before the retrieval path exists, and
# re-scores it at step 3b with retrieval metrics merged in. If 3a (merge + ranking)
# or 3b (re-score) fails -- plausible now, because two trainings are running and the
# card is at 47.7/49.1 GiB -- the report keeps `retrieval: null` forever. The 5.2.2
# table has a `retrieval mAP@10` row, and a null there renders as a blank cell,
# which is indistinguishable from "measured, and it was zero". That is the failure
# this watcher exists to prevent.
#
# It never invents a value: either the field gets filled from a real re-score, or the
# log ends with UNMEASURED so the cell can say "not measured" instead of nothing.
#
# Resumable: it skips any stage whose artefact already exists (merged model,
# rankings), and it re-checks the report rather than trusting that a stage ran.
ROOT=/root/autodl-fs/wafer-vlm
PY="$ROOT/venvs/wafer/bin/python"
LOG="$ROOT/logs/43_retrieval_watch.log"
REP="$ROOT/outputs/reports/qwen35_9b_gspo_g4_lr5e5__report.json"
RANK="$ROOT/outputs/retrieval/qwen35_9b_gspo_g4_lr5e5_rankings.jsonl"
MERGED="$ROOT/models/Qwen3.5-9B-gspo-g4-lr5e5-merged"
BENCH="$ROOT/benchmarks/wafer_bench_v1"
PRED="$ROOT/outputs/baselines/qwen35_9b_gspo_g4_lr5e5.jsonl"
PID=181567

exec >> "$LOG" 2>&1
say() { echo "[$(date +%H:%M:%S)] $*"; }

# Prints the retrieval state of the report as one word, so success is decided by
# reading the artefact and not by a stage's exit code.
state() {
  "$PY" - "$REP" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception as e:
    print("UNREADABLE(%s)" % type(e).__name__); raise SystemExit(0)
r = d.get("retrieval")
if r is None:
    print("NULL")
elif not r:
    print("EMPTY")
else:
    print("OK mAP@10=%.6f" % r.get("mAP@10"))
PYEOF
}

merged_ok() { ls "$MERGED"/*.safetensors >/dev/null 2>&1; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits; }

say "start: report state = $(state); waiting for pid $PID (run 43 eval) to exit"
while [ -d "/proc/$PID" ]; do sleep 30; done
say "pid $PID gone; report state = $(state)"

for attempt in $(seq 1 30); do
  S=$(state)
  case "$S" in OK*) say "attempt $attempt: retrieval already populated ($S); done"; exit 0;; esac
  # 19 GiB is roughly what loading the merged 9B bf16 model costs; below that the
  # stage would OOM and, worse, might take a training process down with it.
  while [ "$(free_mib)" -lt 21000 ]; do
    say "attempt $attempt: waiting for GPU room ($(free_mib) MiB free, need 21000)"
    sleep 120
  done
  if ! merged_ok; then
    say "attempt $attempt: merged model incomplete; re-merging"
    rm -rf "$MERGED"
    "$ROOT/venvs/wafer/bin/swift" export --model "$ROOT/models/Qwen3.5-9B" \
      --adapters "$ROOT/outputs/checkpoints/qwen35_9b_gspo_g4_lr5e5" \
      --merge_lora true --safe_serialization true --exist_ok true \
      --output_dir "$MERGED" || { say "  merge FAILED"; sleep 60; continue; }
  fi
  if [ ! -s "$RANK" ]; then
    say "attempt $attempt: ranking ($(free_mib) MiB free)"
    "$PY" "$ROOT/tools/retrieval_rank.py" --model "$MERGED" --benchmark "$BENCH" \
      --output "$RANK" --batch-size 4 --top-k 50 || say "  rank FAILED"
  fi
  if [ -s "$RANK" ]; then
    say "attempt $attempt: re-scoring with retrieval"
    "$PY" -m wafer_vlm.cli evaluate --benchmark "$BENCH" --predictions "$PRED" \
      --rankings "$RANK" --output "$REP" || say "  re-score FAILED"
  else
    say "attempt $attempt: no rankings; cannot measure retrieval"
  fi
  sleep 30
done

say "FINAL: report state = $(state)"
case "$(state)" in OK*) say "retrieval measured; 5.2.2 retrieval row may quote it";;
  *) say "UNMEASURED: run 43 has no retrieval metrics. The table must say so in words, not leave a blank.";; esac
