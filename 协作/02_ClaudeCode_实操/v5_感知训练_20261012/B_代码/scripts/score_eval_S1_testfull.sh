#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python score_v5.py --raw /WS/v5/eval/S1/testfull/S1_raw.jsonl --ref /WS/v5/data/test_main.jsonl --kind main --out /WS/v5/eval/S1/testfull/metrics.json --per-sample /WS/v5/eval/S1/testfull/per_sample.jsonl
