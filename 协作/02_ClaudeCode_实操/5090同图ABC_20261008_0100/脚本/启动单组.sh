#!/bin/bash
# 用法: launch_one.sh <组 A|B|C|S0> <物理GPU序号>
# 一个进程 = 一张卡 = 训练(+该组评测)；S0 只做基座评测
set -u
G="${1:?组}"; I="${2:?GPU序号}"
R=/WS
cd $R
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$I" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
export IMAGE_MAX_TOKEN_NUM=256
export PYTHONNOUSERSITE=1
export PYTHONIOENCODING=utf-8
BASE=$R/models/Qwen3.5-9B
L=$R/logs/run_${G}_gpu${I}.log
mkdir -p $R/logs $R/eval
echo "=== $G on GPU$I uuid=$CUDA_VISIBLE_DEVICES  $(date '+%F %T') ===" > $L

if [ "$G" != "S0" ]; then
  OUT=$R/checkpoints/abc_${G}_$(date +%Y%m%d_%H%M%S)
  python_bin=$R/envs/wafer/bin/python
  $python_bin $R/train_abc.py --data $R/datasets/abc_${G}.server.jsonl --out "$OUT" >> $L 2>&1
  RC=$?
  echo "训练返回码 $RC" >> $L
  if [ $RC -ne 0 ]; then echo "**训练失败，跳过评测**" >> $L; exit $RC; fi
  # 门槛必须执行且通过
  $python_bin - "$OUT" <<'PYEOF' >> $L 2>&1
import io, json, sys
d = json.load(io.open(sys.argv[1] + "/gate_report.json", encoding="utf-8"))
ok = d.get("门槛已执行") is True and not d.get("门槛未通过项")
print("门槛已执行", d.get("门槛已执行"), "未通过项", d.get("门槛未通过项"), "=>", "通过" if ok else "**不成立**")
sys.exit(0 if ok else 9)
PYEOF
  [ $? -ne 0 ] && { echo "**门槛不成立，终止**" >> $L; exit 9; }
  CKPT=$(ls -d $OUT/*/checkpoint-89 2>/dev/null | head -1)
  echo "末步 checkpoint: $CKPT" >> $L
  SPEC="${G}=${BASE}::${CKPT}"
else
  SPEC="S0=${BASE}"
fi

$R/envs/wafer/bin/python $R/eval_abc.py --models "$SPEC" --outdir $R/eval >> $L 2>&1
echo "评测返回码 $?" >> $L
echo "=== $G 完成 $(date '+%F %T') ===" >> $L
