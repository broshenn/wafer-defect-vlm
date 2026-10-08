#!/bin/bash
# D 段 v2：在**一张已认领的卡**上**串行**跑三个模型，共用**同一个累计计时器**。
#
# v1 的四个缺口（Codex 独立审计指出，本版逐条修）：
#   1. `launch_existence.sh` 在 Python 失败后仍继续 echo「完成」、最终返回 0
#      → 本版**透传子进程返回码**，并把每段耗时/返回码写进账本。
#   2. 队列没透传失败码 → 本版队列读账本，非零即如实上报。
#   3. 没有预算控制 → 本版**共享同一累计计时器**，耗尽即停，写**部分完成**。
#   4. 计时起点含糊 → 本版从**子进程启动**（含 CUDA 上下文与模型加载）
#      到**进程退出释放**为止，**失败与已跑部分全部计入**。
#
# 用法: 串行跑三模型_v2.sh <GPU序号> <预算秒> <账本文件> <输出目录>
set -u
I="$1"; BUDGET="$2"; LEDGER="$3"; OUT="$4"
R="${WS_ROOT:-/WS}"
NVSMI="${NVSMI:-nvidia-smi}"          # 可注入，供离线模拟替换
PY="$R/envs/wafer/bin/python"
mkdir -p "$OUT" "$(dirname "$LEDGER")"

UUID=$(timeout 15 "$NVSMI" -i "$I" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
case "${UUID:-}" in
  GPU-*) : ;;
  *) echo "**GPU$I UUID 无效（'${UUID:-}'）—— 拒绝启动**" >> "$LEDGER"; exit 5 ;;
esac
export CUDA_VISIBLE_DEVICES="$UUID"
export IMAGE_MAX_TOKEN_NUM=256 PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8

echo "# 串行开始 GPU$I UUID=$UUID 预算=${BUDGET}s $(date '+%F %T')" >> "$LEDGER"
USED=0; FAILED=0; RAN=0; NOTRUN=0
for spec in ${SPECS:-}; do
  TAG=${spec%%=*}; SPEC=${spec#*=}
  if [ "$USED" -ge "$BUDGET" ]; then
    echo "$TAG	NOT_RUN	0	预算耗尽（已用 ${USED}s / ${BUDGET}s）" >> "$LEDGER"
    echo "**预算耗尽，未启动 $TAG**"
    NOTRUN=$((NOTRUN+1)); continue
  fi
  L="$R/logs/existence_v2_${TAG}_gpu${I}.log"
  T0=$(date +%s)
  "$PY" "$R/existence/eval_existence.py" --model "${TAG}=${SPEC}" \
        --outdir "$OUT" --assets "$R/existence/派生图" >> "$L" 2>&1
  RC=$?
  T1=$(date +%s); D=$((T1-T0)); USED=$((USED+D)); RAN=$((RAN+1))
  if [ "$RC" = "0" ]; then
    echo "$TAG	OK	$D	" >> "$LEDGER"; echo "$TAG 完成 ${D}s"
  else
    FAILED=$((FAILED+1))
    echo "$TAG	FAILED	$D	rc=$RC（已计入）" >> "$LEDGER"
    echo "**$TAG 失败 rc=$RC，耗时 ${D}s 已计入预算**"
  fi
done
echo "# 串行结束 已用 ${USED}s / ${BUDGET}s，跑了 $RAN 个，失败 $FAILED 个，未启动 $NOTRUN 个 $(date '+%F %T')" >> "$LEDGER"

# 返回码要能区分四种结局 —— **部分完成绝不能返回 0**（否则等于谎报完全成功）：
#   0 = 三个都跑成功；1 = 有失败；7 = 预算耗尽、有未启动（部分完成）；4 = 一个都没跑
if [ "$FAILED" -gt 0 ]; then exit 1; fi
if [ "$RAN" -eq 0 ]; then exit 4; fi
if [ "$NOTRUN" -gt 0 ]; then exit 7; fi
exit 0
