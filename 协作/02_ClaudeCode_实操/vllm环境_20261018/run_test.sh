#!/bin/bash
# vLLM + Qwen3.5-9B 在本机（驱动 570 / CUDA 12.8）上的可用启动方式
#
# 关键：不能用 vLLM 预编译的 FlashAttention kernel —— 它们是 CUDA 12.9 PTX，
# 570 驱动的 JIT 不支持（cudaErrorUnsupportedPtxVersion）。改用运行时 JIT 的路径：
#   - 视觉编码器: mm_encoder_attn_backend = TORCH_SDPA
#   - LLM 注意力: attention_backend        = TRITON_ATTN
#   - 采样器:     VLLM_USE_FLASHINFER_SAMPLER=0（FlashInfer 对 sm_120 误判）
#   - enforce_eager = False（开 CUDA graph：启动多花 ~2 分钟，纯文本快 3.2 倍）
#     要关掉就设 VLLM_TEST_EAGER=1
#
# 用法: ./run_test.sh [GPU序号]
set -u
GPU="${1:-4}"
ENV=/data/renjinyao/envs/vllm026
D=/data/renjinyao/vllm_test

export CUDA_VISIBLE_DEVICES="$GPU"
export TMPDIR=/data/renjinyao/tmp
export HF_HUB_OFFLINE=1
export VLLM_MM_ATTN=TORCH_SDPA
export VLLM_LLM_ATTN=TRITON_ATTN
export VLLM_TEST_EAGER="${VLLM_TEST_EAGER:-0}"
export VLLM_USE_FLASHINFER_SAMPLER=0

exec "$ENV/bin/python" "$D/test_qwen35.py" \
  "/data/renjinyao/probe56/图/wafer_00044911_011.png" \
  "$D/prompt_v5.txt"
