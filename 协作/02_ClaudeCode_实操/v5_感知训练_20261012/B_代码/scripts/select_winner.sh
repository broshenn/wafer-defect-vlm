#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python select_winner.py --runs-root /WS/v5/runs --eval-root /WS/v5/eval --l-metrics /WS/v5/eval/L/select200/metrics.json --out /WS/v5/night/winner.json
