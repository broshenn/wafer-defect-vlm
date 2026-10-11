#!/bin/bash
# 释放占位（服务器端执行；仓库副本不含真实根路径）。
# 用法：WS_ROOT=<真实根> bash release_holds.sh [uuid]   # 不带 uuid 释放全部
# 只 kill 记录在 holds/*.json 且命令行确认为 hold_gpu.py 的进程；先写停止文件再 kill。
set -u
WS=${WS_ROOT:?必须提供 WS_ROOT}
HOLDS=$WS/v5/night/holds
PY=$WS/envs/wafer/bin/python
target="${1:-}"
released=0
for f in "$HOLDS"/*.json; do
  [ -e "$f" ] || continue
  fu=$(basename "$f" .json)
  if [ -n "$target" ] && [ "$target" != "$fu" ]; then continue; fi
  touch "$HOLDS/$fu.stop"
  pid=$("$PY" -c "import json;print(json.load(open('$f'))['pid'])" 2>/dev/null)
  if [ -n "$pid" ] && ps -p "$pid" -o args= 2>/dev/null | grep -q hold_gpu.py; then
    kill "$pid" 2>/dev/null && echo "已释放 $fu pid=$pid"
    released=$((released+1))
  else
    echo "$fu: PID 非本进程或已退出（pid=${pid:-?}），仅删记录"
  fi
  rm -f "$f"
done
echo "共释放 $released 张"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
