#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python eval_v5.py --model /WS/models/Qwen3.5-9B --adapter-from-line S4 --runs-root /WS/v5/runs --tag S4 --dataset /WS/v5/data/select200_main.jsonl --out /WS/v5/eval/S4/select200 --max-tokens 384 --batch-size 8
