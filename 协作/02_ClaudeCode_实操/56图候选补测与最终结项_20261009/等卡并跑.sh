#!/bin/bash
# 注意：本脚本是**学校 5090 的等卡脚本**，本轮因八卡全被他人占用而**从未锁到卡**（未执行任何模型）。
# 真实根路径按项目约定用 WS_ROOT 别名注入，仓库内不写实路径。
# 在获准的学校 5090 上等待一张**真正空闲**的卡，然后顺序跑 Base / L / D。
#
# 纪律：
#  · 不抢卡：同一张卡**连续两次检查（间隔 45 秒）都无计算进程**才锁；
#  · 不共用：要求 0 个计算进程、且空闲显存 > 24 GB；
#  · 不占位：等待期间不占任何显存，只轮询；
#  · 不杀别人：绝不 kill 任何进程；
#  · 记设备分钟：锁卡时刻 → 三模型跑完时刻，逐卡区间。
#
# 用法：setsid nohup bash 等卡并跑.sh > 等卡并跑.log 2>&1 < /dev/null &

R=${WS_ROOT:?请先设 WS_ROOT}/probe56
PY=$R/envs/wafer/bin/python
BASE=$R/models/Qwen3.5-9B
L_AD=$(cat $R/night/L-N3072-3407.ckpt)
D_AD=$(cat $R/night/D-N3072-3407.ckpt)
OUT=$R/out
MIN_FREE=24000          # MiB，9B bf16 实测峰值约 18 GiB
MAX_ROUND=200           # 200 × 30s ≈ 100 分钟；超时就退出并留日志

mkdir -p "$OUT" "$R/logs"
say() { echo "[$(date '+%F %T')] $*"; }

say "等待空闲卡：需连续两次检查均 0 计算进程且 free>${MIN_FREE}MiB"
prev=""
for round in $(seq 1 $MAX_ROUND); do
  for i in 0 1 2 3 4 5 6 7; do
    uuid=$(nvidia-smi -i $i --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -z "$uuid" ] && continue                       # 坏卡（UUID 为空）跳过
    procs=$(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    free=$(nvidia-smi -i $i --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ "$procs" = "0" ] && [ -n "$free" ] && [ "$free" -gt "$MIN_FREE" ]; then
      if [ "$prev" = "$i" ]; then
        say "GPU$i 连续两次空闲（free=${free}MiB）→ 锁卡跑"
        export CUDA_VISIBLE_DEVICES="$uuid"
        START=$(date +%s)
        echo "{\"锁卡时刻\":\"$(date '+%F %T')\",\"gpu_index\":$i,\"uuid\":\"$uuid\",\"free_mib\":$free}" > "$OUT/锁卡记录.json"
        RC=0
        for spec in "Base::" "L-N3072-3407:$L_AD:" "D-N3072-3407:$D_AD:"; do
          tag=${spec%%:*}; rest=${spec#*:}; ad=${rest%%:*}
          say "=== 跑 $tag ==="
          if [ -n "$ad" ]; then
            "$PY" "$R/跑56图.py" --model "$BASE" --adapter "$ad" --tag "$tag" \
              --prompt "$R/prompt_7f.txt" --checker "$R/schema_check.py" \
              --images "$R/图" --frozen "$R/冻结清单_无gold.json" --out "$OUT" || RC=$?
          else
            "$PY" "$R/跑56图.py" --model "$BASE" --tag "$tag" \
              --prompt "$R/prompt_7f.txt" --checker "$R/schema_check.py" \
              --images "$R/图" --frozen "$R/冻结清单_无gold.json" --out "$OUT" || RC=$?
          fi
          [ "$RC" != "0" ] && { say "**$tag 返回码 $RC，停（系统错误或已达预算）**"; break; }
        done
        END=$(date +%s)
        say "释放卡。本次占用 $(( (END-START)/60 )) 分 $(( (END-START)%60 )) 秒（GPU$i 设备区间）"
        echo "{\"锁卡\":\"$(date -d @$START '+%F %T')\",\"释放\":\"$(date -d @$END '+%F %T')\",\"gpu_index\":$i,\"uuid\":\"$uuid\",\"设备秒\":$((END-START))}" > "$OUT/释放记录.json"
        exit 0
      else
        prev="$i"; say "GPU$i 首次空闲（free=${free}MiB），45 秒后复查"
        sleep 45; continue 2
      fi
    fi
  done
  prev=""
  sleep 30
done
say "**超时未等到空闲卡（约 100 分钟），退出，未跑任何模型**"
exit 9
