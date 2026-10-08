#!/bin/bash
# A100 侧：等 GPU 子系统**恢复**、再等**任意一张卡完全空闲**，然后抢下来跑 D 段。
#
# ⚠️ 这台机器 2026-10-08 出现过**驱动整体卡死**：任何碰 GPU 的进程都会进入
#    D 状态（不可中断睡眠），连 `kill -9` 都杀不掉。当时全机 52 个 D 进程，
#    `gq` 的 python 卡了一个多小时、`lzj` 的 nvidia-smi 也卡住。
#    所以本脚本的**检测本身绝不能反过来制造新的卡死进程**。
#
# 恢复判据（按风险从低到高，前两条完全不碰 GPU）：
#   1. **重启检测**：uptime 变小 = 机器重启过 → 驱动必然是新的，零风险
#   2. **D 进程清空**：当前 nvidia 相关 D 状态进程数掉到 ≤1 → 很可能已恢复
#   3. 满足 1 或 2 后，才做**一次**真实探测（`timeout 25 nvidia-smi -L`）；
#      若这次反而挂住，它会自己变成 D 进程，所以**之后至少歇 40 分钟不再探测**。
#
# 抢卡：`mkdir` 原子锁（防多个队列抢同一张卡）；判据**只认 n == 0**。
#
# ⚠️ **预算互斥**：D 段总上限 40 设备分钟，两台机器各跑一遍就是 ~42，**超了**。
#    而任务书把 D 段冻结在「同 5090 环境」，A100 的 torch/CUDA 组合**并不相同**。
#    所以 A100 只作**后备**：**延迟 60 分钟**才允许认领，把优先权让给 5090；
#    真在 A100 上跑出来的结果必须**标注环境差异**，不能与 5090 数字并列。
#    用法: A100_监视与抢卡.sh <tag> <模型spec>
set -u
TAG="$1"; SPEC="$2"
R="${WS_ROOT:-/WS}"
L=$R/logs/_a100_watch_${TAG}.log
mkdir -p "$R/existence/locks" "$R/existence/logs" "$(dirname "$L")"

UP0=$(cut -d. -f1 /proc/uptime)
START=$(date +%s)
GRACE=3600                          # 让 5090 先跑：60 分钟内不认领
DPROBE=0
echo "=== $TAG 监视启动，基线 uptime=${UP0}s，让位窗口 ${GRACE}s $(date '+%F %T') ===" >> "$L"

dprocs() {   # 只数「nvidia 相关」的 D 状态进程
  ps -eo stat,comm 2>/dev/null | awk '$1 ~ /D/ && $2 ~ /nvidia/ {n++} END {print n+0}'
}

for i in $(seq 1 2880); do          # 2880 × 60s = 48 小时
  UP=$(cut -d. -f1 /proc/uptime)
  D=$(dprocs)
  REBOOTED=0
  [ "$UP" -lt "$UP0" ] && REBOOTED=1

  if [ "$REBOOTED" = "1" ] || [ "$D" -le 1 ]; then
    NOW=$(date +%s)
    if [ $((NOW - START)) -lt "$GRACE" ]; then
      sleep 60; continue            # 让位窗口内不探测、不认领
    fi
    if [ $((NOW - DPROBE)) -ge 2400 ]; then     # 探测间隔 ≥40 分钟
      DPROBE=$NOW
      echo "  疑似已恢复（uptime=${UP}s 重启=$REBOOTED nvidia-D进程=$D），做一次探测 $(date '+%F %T')" >> "$L"
      if timeout 25 nvidia-smi -L >/dev/null 2>&1; then
        echo "  **nvidia-smi 已恢复** $(date '+%F %T')" >> "$L"
        # 找一张完全空闲的卡
        for I in 0 1 2 3 4 5; do
          n=$(nvidia-smi -i "$I" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
          [ "$n" = "0" ] || continue
          UUID=$(nvidia-smi -i "$I" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
          case "$UUID" in GPU-*) : ;; *) continue ;; esac
          if mkdir "$R/existence/locks/gpu$I" 2>/dev/null; then
            echo "  认领 GPU$I（$UUID）$(date '+%F %T')" >> "$L"
            cd "$R" || exit 9
            nohup bash "$R/existence/launch_existence_a100.sh" "$TAG" "$I" "$SPEC" >> "$L" 2>&1
            rc=$?
            rmdir "$R/existence/locks/gpu$I" 2>/dev/null
            echo "=== $TAG 启动器退出 rc=$rc $(date '+%F %T') ===" >> "$L"
            exit "$rc"
          fi
        done
        echo "  驱动已恢复但没有完全空闲的卡，继续等 $(date '+%F %T')" >> "$L"
      else
        echo "  探测超时 —— 仍在挂死，40 分钟内不再探测 $(date '+%F %T')" >> "$L"
      fi
    fi
  fi
  sleep 60
done
echo "=== $TAG 监视超时退出（48 小时）$(date '+%F %T') ===" >> "$L"
exit 4
