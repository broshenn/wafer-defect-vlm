#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python eval_v5.py --model /WS/models/Qwen3.5-9B --tag BASE --dataset /WS/v5/data/test200_main.jsonl --out /WS/v5/eval/BASE/test200 --max-tokens 384 --batch-size 8
