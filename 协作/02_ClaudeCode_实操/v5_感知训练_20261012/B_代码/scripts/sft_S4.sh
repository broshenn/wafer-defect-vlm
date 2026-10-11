#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python train_v5.py --config configs/S4.json --out /WS/v5/runs/S4
