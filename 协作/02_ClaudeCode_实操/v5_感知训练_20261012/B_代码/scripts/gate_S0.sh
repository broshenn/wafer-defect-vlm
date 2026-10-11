#!/bin/bash
set -u
cd /WS/v5/code || exit 9
/WS/envs/wafer/bin/python train_v5.py --config configs/S0.json --out /WS/v5/runs/S0_probe --probe-steps 2 && CKPT=$(find /WS/v5/runs/S0_probe -name checkpoint-2 -type d | head -1) && test -n "$CKPT" && /WS/envs/wafer/bin/python gate_sft.py --config configs/S0.json --ckpt "$CKPT" --out /WS/v5/runs/S0_probe/gate_sft.json
