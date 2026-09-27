#!/bin/bash
# Watch every report whose eval can leave `retrieval: null` on disk.
#
# scripts/30_eval_grpo.sh writes <tag>__report.json at step 2b -- before the merge,
# the ranking and the re-score of step 3b -- and then rewrites the same file with
# the retrieval metrics merged in. Between those points the file on disk is a
# complete-looking report with a null retrieval field, and a null in 5.2.2's
# `retrieval mAP@10` row renders as a blank cell, which is indistinguishable from
# "measured, and it was zero". That is the failure this watcher exists to prevent,
# and with three runs still to land it is now a loop rather than a one-off.
#
# It never invents a value: either the field is filled from a real re-score, or the
# log ends with UNMEASURED so the table can say "not measured" in words.
#
# Resumable: every stage is skipped if its artefact already exists (merged model,
# rankings), and the report is re-read rather than trusting that a stage ran. The
# merge/rank/scoring path also refuses to start below 21 GiB free, because the
# merged 9B needs ~19 GiB and an OOM there could take a training process with it.
#
# usage: watch_retrieval.sh <tag>:<merged-dir-name> [<tag>:<merged-dir-name> ...]
set -u
ROOT=/root/autodl-fs/wafer-vlm
PY="$ROOT/venvs/wafer/bin/python"
BENCH="$ROOT/benchmarks/wafer_bench_v1"
say() { echo "[$(date +%H:%M:%S)] $*"; }

state() {
  "$PY" - "$1" <<'PYEOF'
import json, os, sys
p = sys.argv[1]
if not os.path.isfile(p):
    print("ABSENT"); raise SystemExit(0)
try:
    d = json.load(open(p, encoding="utf-8"))
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

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits; }

# Is work in progress that would write the artefacts this watcher writes?
#
# The eval script does its own merge -> rank -> re-score as step 3b, so the watcher's
# real job is the *fallback*: a report that stayed null because that step failed or
# was skipped. Merging into a directory while 3b is merging into the same one, or
# ranking while 3b is ranking, would corrupt the eval's output rather than repair it,
# so no stage here starts while anything that could be doing that stage is alive.
#
# The match is by argument signature and by the tag, not by script filename: scripts
# here are pushed as a base64 blob, so the launching shell's command line contains the
# path of whatever it launched and a filename match would never clear. Every process
# in this ladder carries its run's identity in its arguments -- swift rlhf carries
# --output_dir <tag>, swift export carries --adapters <tag>, retrieval_rank.py and the
# scorer carry <tag> in --output/--rankings/--predictions -- so one substring test
# covers training, merging, ranking and scoring for that tag.
tag_busy() {
  local tag="$1" p c
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    [ "$p" = "$$" ] && continue
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    [ -z "$c" ] && continue
    case "$c" in
      *"$tag"*) return 0 ;;
    esac
    if [ "$p" != "$PPID" ]; then
      case "$c" in
        *30_eval_grpo.sh*|*swift/cli/rlhf.py*|*24_train_grpo.sh*) return 0 ;;
      esac
    fi
  done
  return 1
}

for SPEC in "$@"; do
  TAG="${SPEC%%:*}"
  MERGED="$ROOT/models/${SPEC#*:}"
  REP="$ROOT/outputs/reports/${TAG}__report.json"
  RANK="$ROOT/outputs/retrieval/${TAG}_rankings.jsonl"
  PRED="$ROOT/outputs/baselines/${TAG}.jsonl"
  # Not the checkpoint root: training writes <root>/<run-id>/checkpoint-N/adapter_config.json,
  # and `swift export` rejects the root with "is not an adapter". Same resolution as
  # 30_eval_grpo.sh, for the reason recorded there -- a `find`, not a shell loop, because
  # under `set -e` a loop whose last iteration fails its test aborts the script silently.
  ADAPTER=$(find "$ROOT/outputs/checkpoints/${TAG}" -maxdepth 3 -name adapter_config.json \
            -printf '%h\n' 2>/dev/null | sort -V | tail -1)
  [ -n "$ADAPTER" ] || say "$TAG: WARNING: no adapter_config.json under \
$ROOT/outputs/checkpoints/${TAG}; the merge cannot run for this run"

  say "=== $TAG: waiting for $REP to appear (state = $(state "$REP"))"
  for _ in $(seq 1 480); do               # up to 8h at 60s
    S=$(state "$REP")
    [ "$S" != "ABSENT" ] && break
    sleep 60
  done
  say "$TAG: report state = $(state "$REP")"

  # The eval's own 3b step normally fills retrieval; only a report that stays null
  # is a problem. 30 attempts at 60s is the same budget the run-43 watcher used.
  for attempt in $(seq 1 30); do
    S=$(state "$REP")
    case "$S" in
      OK*)     say "$TAG: attempt $attempt: retrieval present ($S); done"; break;;
      ABSENT)  say "$TAG: attempt $attempt: report still absent"; sleep 60; continue;;
    esac
    # A null report while the run's own evaluation is still working means 3b has not
    # finished, not that it failed: waiting is correct, and merging now would fight
    # the eval for the card and for the same output directory.
    if tag_busy "$TAG"; then
      say "$TAG: attempt $attempt: null, but this run still has work in flight; waiting"
      sleep 120; continue
    fi
    while [ "$(free_mib)" -lt 21000 ]; do
      say "$TAG: attempt $attempt: waiting for GPU room ($(free_mib) MiB free, need 21000)"
      sleep 120
    done
    # An empty adapter is not something to retry: without it there is nothing to merge,
    # and the loop below would call `swift export --adapters ""` once a minute for half an
    # hour before saying UNMEASURED. Say it now, and say the honest outcome.
    if [ ! -s "$RANK" ] && [ -z "$ADAPTER" ]; then
      say "$TAG: attempt $attempt: nothing to merge; retrieval stays unmeasured"
      break
    fi
    if [ ! -s "$RANK" ]; then
      if ! ls "$MERGED"/*.safetensors >/dev/null 2>&1; then
        say "$TAG: attempt $attempt: merging into $MERGED"
        rm -rf "$MERGED"
        "$ROOT/venvs/wafer/bin/swift" export --model "$ROOT/models/Qwen3.5-9B" \
          --adapters "$ADAPTER" --merge_lora true --safe_serialization true \
          --exist_ok true --output_dir "$MERGED" \
          || { say "$TAG:   merge FAILED"; sleep 60; continue; }
      fi
      say "$TAG: attempt $attempt: ranking ($(free_mib) MiB free)"
      "$PY" "$ROOT/tools/retrieval_rank.py" --model "$MERGED" --benchmark "$BENCH" \
        --output "$RANK" --batch-size 4 --top-k 50 || say "$TAG:   rank FAILED"
    fi
    if [ -s "$RANK" ] && [ -s "$PRED" ]; then
      say "$TAG: attempt $attempt: re-scoring with retrieval"
      "$PY" -m wafer_vlm.cli evaluate --benchmark "$BENCH" --predictions "$PRED" \
        --rankings "$RANK" --output "$REP" || say "$TAG:   re-score FAILED"
    else
      say "$TAG: attempt $attempt: no rankings or no predictions yet" \
          "(rankings: $([ -s "$RANK" ] && echo yes || echo no)," \
          "predictions: $([ -s "$PRED" ] && echo yes || echo no))"
    fi
    sleep 30
  done
  case "$(state "$REP")" in
    OK*) say "$TAG: FINAL: retrieval measured ($(state "$REP"))";;
    *)   say "$TAG: FINAL: UNMEASURED -- the table must say 'not measured' in words, not leave a blank cell";;
  esac
done
say "watcher done for: $*"
