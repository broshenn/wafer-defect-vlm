#!/usr/bin/env bash
# Align ms-swift results to benchmark keys and score them.
set -euo pipefail
F=/root/autodl-fs/wafer-vlm
V=/root/autodl-tmp/wafer-vlm/venvs/wafer/bin/python
B=$F/benchmarks/wafer_bench_v1
RUN=${1:?usage: run_eval.sh <run_name> [results.jsonl] [requests.jsonl] [rankings.jsonl]}
RESULTS=${2:-$F/outputs/baselines/$RUN.jsonl}
REQUESTS=${3:-$B/all_requests.jsonl}
RANKINGS=${4:-}
ALIGNED=$F/outputs/baselines/${RUN}__aligned.jsonl
REPORT=$F/outputs/reports/${RUN}__report.json
mkdir -p "$F/outputs/reports" "$F/outputs/baselines"

$V "$F/tools/attach_ids.py" --requests "$REQUESTS" --results "$RESULTS" --output "$ALIGNED"

ARGS=(--benchmark "$B" --predictions "$ALIGNED" --output "$REPORT")
if [ -n "$RANKINGS" ] && [ -s "$RANKINGS" ]; then ARGS+=(--rankings "$RANKINGS"); fi
$V -m wafer_vlm.evaluate "${ARGS[@]}"
echo "report: $REPORT"
