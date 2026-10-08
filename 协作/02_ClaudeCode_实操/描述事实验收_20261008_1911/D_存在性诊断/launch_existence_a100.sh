#!/bin/bash
# A100 侧的 D 段启动器（与 5090 版差别只在 adapter 存放位置）。
# 用法: launch_existence_a100.sh <tag> <GPU序号> <模型spec>
set -u
TAG="$1"; I="$2"; SPEC="$3"
R="${WS_ROOT:-/WS}"
cd "$R" || exit 9
L=$R/logs/existence_a100_${TAG}_gpu${I}.log
mkdir -p "$R/existence/out"

# 现场核卡：UUID 查不到 = 故障卡，跳过
UUID=$(nvidia-smi -i "$I" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
case "$UUID" in
  GPU-*) : ;;
  *) echo "**GPU$I UUID 查询失败（'$UUID'）—— 跳过**" >> "$L"; exit 5 ;;
esac
export CUDA_VISIBLE_DEVICES="$UUID"
export IMAGE_MAX_TOKEN_NUM=256 PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8

echo "=== ${TAG} 存在性诊断 (A100) GPU$I ($UUID) $(date '+%F %T') ===" > "$L"
echo "环境: $(hostname)  torch 见下" >> "$L"
"$R/envs/wafer/bin/python" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)" >> "$L" 2>&1

"$R/envs/wafer/bin/python" "$R/existence/eval_existence.py" \
    --model "${TAG}=${SPEC}" \
    --outdir "$R/existence/out" \
    --assets "$R/existence/派生图" >> "$L" 2>&1
echo "返回码 $?" >> "$L"
echo "=== ${TAG} 完成 $(date '+%F %T') ===" >> "$L"
