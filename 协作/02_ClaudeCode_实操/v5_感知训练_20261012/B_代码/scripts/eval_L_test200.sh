#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python eval_v5.py --model /WS/models/Qwen3.5-9B --adapter /WS/night/L-N3072-3407.ckpt --tag L --dataset /WS/v5/data/test_L.jsonl --out /WS/v5/eval/L/test200 --max-tokens 64 --batch-size 8
