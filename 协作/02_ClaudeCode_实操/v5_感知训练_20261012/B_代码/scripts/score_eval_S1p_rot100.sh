#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python score_v5.py --raw /WS/v5/eval/S1p/rot100/S1p_raw.jsonl --ref /WS/v5/data/rot100_main.jsonl --kind main --out /WS/v5/eval/S1p/rot100/metrics.json --per-sample /WS/v5/eval/S1p/rot100/per_sample.jsonl
