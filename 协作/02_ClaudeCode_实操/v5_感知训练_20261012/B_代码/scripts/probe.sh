#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python probe_v5.py --model /WS/models/Qwen3.5-9B --train-jsonl /WS/v5/data/sft_meta_train.jsonl --test-jsonl /WS/v5/data/test_main.jsonl --out /WS/v5/eval/PROBE --seed 3407
