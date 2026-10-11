#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python screen_rl_pool.py --merge --out-dir /WS/v5/night/screen --candidates /WS/v5/data/rl_candidates800.jsonl --template /WS/v5/data/rsft_rows_template.jsonl --out-pool /WS/v5/data/rl_pool320.jsonl --out-rsft /WS/v5/data/rsft_pool320.jsonl --passk /WS/v5/night/passk.json
