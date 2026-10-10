#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""便宜的算力实测：bf16 4096^3 矩阵乘若干次并计时。

5090 上单次 4096x4096x4096 bf16 约几毫秒；若物理卡被别的容器严重争抢，
实测 TFLOPS 会明显低于预期（5090 bf16 稠密理论上百 TFLOPS 量级）。
只用来判断"这张卡我实际能用多少"，不做性能结论。
"""
import os
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES"), "未锁卡"
import torch

torch.cuda.init()
print("可见卡数", torch.cuda.device_count(), torch.cuda.get_device_name(0))
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
for _ in range(3):
    c = a @ b
torch.cuda.synchronize()
t0 = time.perf_counter()
N = 20
for _ in range(N):
    c = a @ b
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / N
flops = 2 * 4096 ** 3
print(f"单次 {dt * 1000:.2f} ms   等效 {flops / dt / 1e12:.1f} TFLOPS(bf16)")
print("本进程显存", torch.cuda.memory_allocated() / 2 ** 20, "MiB")
