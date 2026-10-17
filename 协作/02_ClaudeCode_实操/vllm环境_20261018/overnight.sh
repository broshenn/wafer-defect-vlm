#!/bin/bash
# 断线后继续跑：等 torch cu129 装完 → CUDA 自检 → 装 vllm cu129 → Qwen3.5-9B 实跑自检
LOG=/data/renjinyao/vllm_test/overnight_$(date +%Y%m%d_%H%M%S).log
exec >> "$LOG" 2>&1
set -x
P=/data/renjinyao/envs/vllm026/bin/python
export PIP_CACHE_DIR=/data/renjinyao/.cache/pip TMPDIR=/data/renjinyao/tmp HF_HUB_OFFLINE=1
echo "=== START $(date)"

# 1) 等正在跑的 torch 安装结束（最多 60 分钟）
for i in $(seq 1 720); do
  pgrep -u renjinyao -f "pip install --force-reinstall torch" >/dev/null || break
  sleep 5
done
echo "=== torch 等待结束 $(date)"
$P -m pip list 2>/dev/null | grep -iE "^torch "

# 2) CUDA 自检（GPU4 空卡）
CUDA_VISIBLE_DEVICES=4 $P -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(10)
x = torch.randn(2048, 2048, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('matmul OK on', torch.cuda.get_device_name(0))
"
rc=$?
if [ $rc -ne 0 ]; then echo "=== CU129_TORCH_FAILED rc=$rc"; exit 10; fi

# 3) 装 vllm 0.26.0 的 cu129 构建（同版本替换掉 PyPI 的 cu13 构建）
$P -m pip install --force-reinstall --no-deps \
  "https://wheels.vllm.ai/568afb3a13806beb53bb2e6bd518269357b237c0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
rc=$?
if [ $rc -ne 0 ]; then echo "=== VLLM_WHEEL_INSTALL_FAILED rc=$rc"; exit 11; fi

# 4) import 自检
CUDA_VISIBLE_DEVICES=4 $P -c "import vllm; print('vllm', vllm.__version__)"
rc=$?
if [ $rc -ne 0 ]; then echo "=== VLLM_IMPORT_FAILED rc=$rc"; exit 12; fi

# 5) 真实加载 + 纯文本 + 带图
cd /data/renjinyao/vllm_test
CUDA_VISIBLE_DEVICES=4 timeout 1800 $P test_qwen35.py \
  "/data/renjinyao/probe56/图/wafer_00044911_011.png" \
  /data/renjinyao/vllm_test/prompt_v5.txt \
  > result_cu129.json 2> err_cu129.log
echo "=== TEST exit=$? $(date)"
echo "--- result_cu129.json"
cat result_cu129.json
echo "--- stderr 尾部"
tail -25 err_cu129.log
echo "=== DONE $(date)"
