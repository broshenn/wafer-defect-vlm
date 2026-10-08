#!/bin/bash
# D 段存在性诊断：等**任意一张完全空闲**的卡，原子认领后启动一个模型。
#
# 判据只认 n == 0（完全没有进程）。
# **不要**沿用 chain_one.sh 里那句「进程属主等于自己也算空闲」的判断——
# 那等于「卡上只有我自己的进程也算空闲」，是 2026-10-08 三条链同时 OOM 的直接原因。
#
# 认领用 mkdir 原子锁，避免三个队列抢同一张卡。
# 用法: queue_existence.sh <tag> <模型spec>
set -u
TAG="$1"; SPEC="$2"
R="${WS_ROOT:-/WS}"
L=$R/logs/_queue_ex_${TAG}.log
mkdir -p "$R/existence/locks" "$R/existence/logs"
echo "=== $TAG 等任意卡**完全**空出 $(date '+%F %T') ===" >> "$L"

CLAIM=""
for i in $(seq 1 720); do          # 720 × 30s = 6 小时
  for I in 0 1 2 3 4 5 6 7; do
    # 先确认 nvidia-smi 本身**成功**：它失败时输出为空，`grep -c` 会得 0，
    # 看着像「卡是空的」—— 那是会直接导致 OOM 的误判。
    if ! nvidia-smi -i "$I" --query-gpu=uuid --format=csv,noheader >/dev/null 2>&1; then
      continue
    fi
    n=$(nvidia-smi -i "$I" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . )
    [ "$n" = "0" ] || continue
    # 故障卡（UUID 查不到，如 2026-10-08 的 ERR!/ECC 状态）跳过
    UUID=$(nvidia-smi -i "$I" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
    case "$UUID" in GPU-*) : ;; *) continue ;; esac
    if mkdir "$R/existence/locks/gpu$I" 2>/dev/null; then
      CLAIM="$I"
      echo "  认领 GPU$I（$UUID）$(date '+%F %T')" >> "$L"
      break 2
    fi
  done
  sleep 30
done

if [ -z "$CLAIM" ]; then
  echo "  **等待超时，未启动 $TAG** $(date '+%F %T')" >> "$L"
  exit 4
fi

cd "$R"
nohup bash "$R/existence/launch_existence.sh" "$TAG" "$CLAIM" "$SPEC" >> "$L" 2>&1
rc=$?
rmdir "$R/existence/locks/gpu$CLAIM" 2>/dev/null
echo "=== $TAG 启动器退出 rc=$rc $(date '+%F %T') ===" >> "$L"
