#!/bin/bash
# 占位锁卡发射器（服务器端执行；仓库副本不含真实根路径）。
# 用法：WS_ROOT=<真实根> [MAX_HOLD=7] [DURATION=12600] bash launch_holds.sh
# 规则：只占「memory.used<500MiB 且无 compute 进程」且 UUID 可查的卡；每卡一个占位进程；
#       记录到 $WS/v5/night/holds/<uuid>.json；到点自动过期；绝不触碰他人进程。
set -u
WS=${WS_ROOT:?必须提供 WS_ROOT}
MAX_HOLD=${MAX_HOLD:-7}
DURATION=${DURATION:-12600}
NIGHT=$WS/v5/night
PY=$WS/envs/wafer/bin/python
mkdir -p "$NIGHT/holds" "$NIGHT/logs"
n=0
for i in 0 1 2 3 4 5 6 7; do
  [ "$n" -ge "$MAX_HOLD" ] && break
  used=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  uuid=$(nvidia-smi -i $i --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
  procs=$(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ -z "$uuid" ]; then echo "GPU$i: 无 UUID（坏卡），跳过"; continue; fi
  if [ -z "$used" ]; then echo "GPU$i: 状态查询失败，跳过"; continue; fi
  if [ "$used" -ge 500 ] || [ "$procs" -gt 0 ]; then
    echo "GPU$i: 有占用（used=${used}MiB procs=$procs），跳过"; continue
  fi
  stop=$NIGHT/holds/${uuid}.stop
  rm -f "$stop"
  CUDA_VISIBLE_DEVICES=$uuid setsid nohup "$PY" "$WS/v5/code/hold_gpu.py" \
      --outdir "$NIGHT/holds" --stop-file "$stop" --duration-s "$DURATION" \
      > "$NIGHT/logs/hold_${uuid}.log" 2>&1 < /dev/null &
  echo "GPU$i: 已发起占位 uuid=$uuid"
  n=$((n+1))
done
echo "共发起 $n 张占位；等待 25s 后核验"
sleep 25
ls -la "$NIGHT/holds/" 2>/dev/null
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
