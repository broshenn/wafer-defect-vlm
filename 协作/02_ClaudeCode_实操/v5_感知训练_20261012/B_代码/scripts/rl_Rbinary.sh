#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python train_v5.py --config configs/Rbinary.json --out /WS/v5/runs/Rbinary
