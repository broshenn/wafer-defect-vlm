#!/bin/bash
# D 段 v2 队列：等**一张完全空闲**的卡 → 原子锁 → **锁后复查** → 串行跑三个模型。
#
# v1 的缺口（Codex 独立审计指出，逐条修）：
#   · 「720×30 秒 = 6 小时」不是承诺的 40 分钟 → 本版改成**绝对截止时间戳**，
#     由调用方传入，脚本自己不再暗含任何时长。
#   · 「UUID 查询成功 ≠ 进程查询成功」；进程查询失败产出空输出，
#     `grep -c` 得 0，会被当成空卡 → 本版**每次查询都判返回码**，
#     失败与「确实是 0 个进程」严格区分开。
#   · 「原子锁后未复查占用」→ 本版**锁后再查一次**，被占就释放锁走人。
#   · 「GPU 查询整体异常时反复堆探针」→ 本版连续失败超过阈值就**停止派发**。
#
# 用法: 队列_v2.sh <截止epoch> <预算秒> <账本文件> <输出目录>
set -u
DEADLINE="$1"; BUDGET="$2"; LEDGER="$3"; OUT="$4"
R="${WS_ROOT:-/WS}"
NVSMI="${NVSMI:-nvidia-smi}"          # 可注入，供离线模拟替换
QCMD=(timeout 15 "$NVSMI")
MAXQFAIL="${MAXQFAIL:-10}"            # 连续查询失败达到此数 → 停止派发
QFAIL=0; LAUNCHED=0
SLEEP_SECS="${SLEEP_SECS:-30}"        # 可注入，供离线模拟把等待压到 0

L=$R/logs/_queue_v2.log
mkdir -p "$R/existence/locks" "$(dirname "$L")" "$OUT"
echo "=== v2 队列启动 截止=$(date -d @"$DEADLINE" '+%F %T') 预算=${BUDGET}s $(date '+%F %T') ===" >> "$L"

qfail() { QFAIL=$((QFAIL+1)); echo "  查询失败 ×$QFAIL $(date '+%F %T')" >> "$L"
          [ "$QFAIL" -ge "$MAXQFAIL" ] && { echo "  **连续查询失败 $QFAIL 次 —— 停止派发**" >> "$L"; return 1; }
          return 0; }

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  for I in 0 1 2 3 4 5 6 7; do
    # ---- 查询 1：进程列表。必须**成功返回**才继续 ----
    if ! OUT1=$("${QCMD[@]}" -i "$I" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
      qfail || exit 6; continue
    fi
    n=$(printf '%s' "$OUT1" | grep -c .)
    [ "$n" = "0" ] || { QFAIL=0; continue; }        # 有进程 → 这张卡跳过（不算查询失败）

    # ---- 查询 2：UUID。失败或非法 → 跳过（故障卡）----
    if ! U=$("${QCMD[@]}" -i "$I" --query-gpu=uuid --format=csv,noheader 2>/dev/null); then
      qfail || exit 6; continue
    fi
    U=$(printf '%s' "$U" | tr -d ' \r\n')
    case "$U" in GPU-*) : ;; *) continue ;; esac
    QFAIL=0

    # ---- 原子认领 ----
    if ! mkdir "$R/existence/locks/gpu$I" 2>/dev/null; then continue; fi

    # ---- **锁后复查**：这一小段时间里可能有人占了 ----
    if ! OUT2=$("${QCMD[@]}" -i "$I" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
      rmdir "$R/existence/locks/gpu$I" 2>/dev/null; qfail || exit 6; continue
    fi
    m=$(printf '%s' "$OUT2" | grep -c .)
    if [ "$m" != "0" ]; then
      echo "  锁后复查发现 GPU$I 已被占（$m 个进程）—— 放弃 $(date '+%F %T')" >> "$L"
      rmdir "$R/existence/locks/gpu$I" 2>/dev/null; continue
    fi

    # ---- 启动：串行跑三个模型，共用同一计时器 ----
    echo "  认领 GPU$I（$U），开始串行 $(date '+%F %T')" >> "$L"
    REMAIN=$(( BUDGET ))
    cd "$R" || exit 9
    bash "$R/existence/串行跑三模型_v2.sh" "$I" "$REMAIN" "$LEDGER" "$OUT" >> "$L" 2>&1
    RC=$?
    rmdir "$R/existence/locks/gpu$I" 2>/dev/null
    echo "  串行返回码 $RC $(date '+%F %T')" >> "$L"
    LAUNCHED=1
    # 无论成败都退出：预算已由串行脚本内部核算，不重复跑
    exit "$RC"
  done
  sleep "$SLEEP_SECS"
done

echo "=== v2 队列截止，未启动（launched=$LAUNCHED）$(date '+%F %T') ===" >> "$L"
exit 4
