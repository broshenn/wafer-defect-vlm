#!/usr/bin/env bash
set -x
F=/root/autodl-fs/wafer-vlm
T=/root/autodl-tmp/wafer-vlm
echo "START $(date)"
cp -a $T/models/Qwen3.5-9B $F/models/ 2>&1
echo "model copied $(date)"
cp -a $T/data/raw/wm811k $F/data/raw/ 2>&1
echo "data copied $(date)"
echo "COPY_DONE $(date)"
