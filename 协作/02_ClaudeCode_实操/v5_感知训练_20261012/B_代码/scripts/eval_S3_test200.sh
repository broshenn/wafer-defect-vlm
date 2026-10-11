#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python eval_v5.py --model /WS/models/Qwen3.5-9B --adapter-from-line S3 --runs-root /WS/v5/runs --tag S3 --dataset /WS/v5/data/test200_main_x2.jsonl --out /WS/v5/eval/S3/test200 --max-tokens 384 --batch-size 8
