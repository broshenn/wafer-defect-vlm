#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python eval_v5.py --model /WS/models/Qwen3.5-9B --adapter-from-line S0 --runs-root /WS/v5/runs --tag S0 --dataset /WS/v5/data/shuf50_main.jsonl --out /WS/v5/eval/S0/shuf50 --max-tokens 384 --batch-size 8
