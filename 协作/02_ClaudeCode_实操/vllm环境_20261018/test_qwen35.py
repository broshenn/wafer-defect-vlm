#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vLLM + Qwen3.5-9B 实跑自检。

只做三件事：加载模型、纯文本生成一次、带图生成一次。不改权重、不训练。
用法：python test_qwen35.py <图片绝对路径> <题面文件绝对路径>
"""
from __future__ import annotations

import json
import sys
import time

import torch
from PIL import Image
from vllm import LLM, SamplingParams

MODEL = "/data/renjinyao/models/Qwen3.5-9B"

img_path = sys.argv[1]
prompt_path = sys.argv[2]

result = {"model": MODEL, "torch": torch.__version__, "cuda_built": torch.version.cuda,
          "cuda_available": torch.cuda.is_available(),
          "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

import vllm
result["vllm"] = vllm.__version__

try:
    from vllm.model_executor.models.registry import ModelRegistry
    result["qwen3_5_registered"] = "Qwen3_5ForConditionalGeneration" in ModelRegistry.get_supported_archs()
except Exception as exc:
    result["qwen3_5_registered"] = f"查询失败 {type(exc).__name__}: {str(exc)[:150]}"

t0 = time.time()
llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    enforce_eager=False,
    limit_mm_per_prompt={"image": 1},
)
result["load_seconds"] = round(time.time() - t0, 2)
result["gpu_mem_used_GiB_after_load"] = round(torch.cuda.memory_allocated() / 1024**3, 2)

sp = SamplingParams(temperature=0.0, max_tokens=256, seed=3407)

# ---------- 1) 纯文本
t = time.time()
out = llm.generate(["用一句话说明什么是晶圆 BIN 图。"], sp)
result["text_ok"] = True
result["text_seconds"] = round(time.time() - t, 2)
result["text_out"] = out[0].outputs[0].text[:300]

# ---------- 2) 带图：走 chat 模板
prompt = open(prompt_path, encoding="utf-8").read().strip()
img = Image.open(img_path).convert("RGB")
result["image"] = {"path": img_path, "size": list(img.size), "mode": img.mode}
t = time.time()
try:
    out2 = llm.chat(
        [{"role": "user", "content": [
            {"type": "image_pil", "image_pil": img},
            {"type": "text", "text": prompt},
        ]}],
        sp,
    )
    result["mm_mode"] = "chat+image_pil"
    result["mm_out"] = out2[0].outputs[0].text[:500]
    result["mm_ok"] = True
except Exception as exc:
    result["mm_ok"] = False
    result["mm_error_chat"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    # 退化路径：字符串提示 + multi_modal_data，占位符用 Qwen3.5 的图像 token
    try:
        ph = "<|vision_start|><|image_pad|><|vision_end|>"
        out2 = llm.generate([{"prompt": ph + prompt, "multi_modal_data": {"image": img}}], sp)
        result["mm_mode"] = "generate+multi_modal_data"
        result["mm_out"] = out2[0].outputs[0].text[:500]
        result["mm_ok"] = True
    except Exception as exc2:
        result["mm_error_generate"] = f"{type(exc2).__name__}: {str(exc2)[:300]}"
result["mm_seconds"] = round(time.time() - t, 2)

print(json.dumps(result, ensure_ascii=False, indent=1))
