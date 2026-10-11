#!/bin/bash
set -u
cd /WS/v5/code || exit 9
WAFER_REWARD_MODE=full /WS/envs/wafer/bin/python screen_rl_pool.py --shard 1 --nshards 6 --winner /WS/v5/night/winner.json --model /WS/models/Qwen3.5-9B --candidates /WS/v5/data/rl_candidates800.jsonl --out-dir /WS/v5/night/screen --batch-size 8
