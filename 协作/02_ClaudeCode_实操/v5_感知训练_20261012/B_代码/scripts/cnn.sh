#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python cnn_v5.py --train-jsonl /WS/v5/data/sft_meta_train.jsonl --test-jsonl /WS/v5/data/test_main.jsonl --out /WS/v5/eval/CNN --seed 3407
