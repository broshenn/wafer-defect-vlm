#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python score_v5.py --raw /WS/v5/eval/BASE/testfull/BASE_raw.jsonl --ref /WS/v5/data/test_main.jsonl --kind main --out /WS/v5/eval/BASE/testfull/metrics.json --per-sample /WS/v5/eval/BASE/testfull/per_sample.jsonl
