#!/bin/bash
# 用法: launch_existence.sh <tag> <GPU序号> <模型spec：base 或 base::adapter>
set -u
TAG="$1"; I="$2"; SPEC="$3"
# 根目录从环境变量取，便于在仓库里用别名、在服务器上用真实路径；
# 同**一份**文件两处都能跑，不产生「仓库版 / 部署版」漂移。
R="${WS_ROOT:-/WS}"
cd "$R" || exit 9
L=$R/logs/existence_${TAG}_gpu${I}.log
mkdir -p "$R/existence/out"

# 现场核卡：故障卡（UUID 查询失败）必须跳过 —— 2026-10-08 遇到的 GPU0 ERR! 状态
UUID=$(nvidia-smi -i "$I" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
case "$UUID" in
  GPU-*) : ;;
  *) echo "**GPU$I UUID 查询失败（'$UUID'）—— 疑似故障卡，跳过**" >> "$L"; exit 5 ;;
esac
export CUDA_VISIBLE_DEVICES="$UUID"
export IMAGE_MAX_TOKEN_NUM=256 PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8

echo "=== ${TAG} 存在性诊断 GPU$I ($UUID) $(date '+%F %T') ===" > "$L"
"$R/envs/wafer/bin/python" "$R/existence/eval_existence.py" \
    --model "${TAG}=${SPEC}" \
    --outdir "$R/existence/out" \
    --assets "$R/existence/派生图" >> "$L" 2>&1
echo "返回码 $?" >> "$L"
echo "=== ${TAG} 完成 $(date '+%F %T') ===" >> "$L"
