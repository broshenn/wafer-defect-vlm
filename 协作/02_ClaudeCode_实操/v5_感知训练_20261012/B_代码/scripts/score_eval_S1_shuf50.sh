#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python score_v5.py --raw /WS/v5/eval/S1/shuf50/S1_raw.jsonl --ref /WS/v5/data/shuf50_main.jsonl --kind main --out /WS/v5/eval/S1/shuf50/metrics.json --per-sample /WS/v5/eval/S1/shuf50/per_sample.jsonl
