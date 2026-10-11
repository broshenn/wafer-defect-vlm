#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gate_sft 检测逻辑测试（纯函数；本机无 torch/safetensors 也可跑）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gate_sft import lora_b_ratio, visual_key_count  # noqa: E402


def main():
    results = []
    add = lambda n, c, x="": results.append((n, bool(c), x))  # noqa: E731
    keys_s2 = ["base_model.model.model.visual.blocks.0.attn.q_proj.lora_A.weight",
               "base_model.model.model.visual.blocks.0.attn.q_proj.lora_B.weight",
               "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
               "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    keys_s1 = [k for k in keys_s2 if "visual" not in k]
    add("S2 键含 visual 数=2", visual_key_count(keys_s2) == 2)
    add("S1 键含 visual 数=0", visual_key_count(keys_s1) == 0)
    nz = {"a.lora_B.weight": True, "b.lora_B.weight": False, "c.lora_A.weight": True}
    total, nz_ok = lora_b_ratio(nz)
    add("lora_B 只统计 B 键", (total, nz_ok) == (2, 1), (total, nz_ok))
    add("全零 B → 不过半", lora_b_ratio({"a.lora_B": False, "b.lora_B": False}) == (2, 0))
    add("无 lora_B → total=0", lora_b_ratio({"a.lora_A": True}) == (0, 0))
    # 真实读写：本机有 safetensors.numpy，写一个假的小文件测 load_adapter_tensors
    try:
        import numpy as np
        from safetensors.numpy import save_file
        from gate_sft import load_adapter_tensors
        FIX = HERE / "fixtures"
        FIX.mkdir(exist_ok=True)
        fake = FIX / "fake_s2.safetensors"
        save_file({"model.visual.x.lora_B.weight": np.ones((2, 2), dtype=np.float32),
                   "model.layers.0.lora_B.weight": np.zeros((2, 2), dtype=np.float32)}, str(fake))
        keys, nz = load_adapter_tensors(fake)
        add("真实读取：visual=1 且全零键=False",
            visual_key_count(keys) == 1 and nz["model.layers.0.lora_B.weight"] is False
            and nz["model.visual.x.lora_B.weight"] is True)
    except ImportError as e:
        print(f"  SKIP 真实读取（本机无 safetensors: {e}）—— 已由纯函数覆盖")

    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, extra in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name} {extra if not ok else ''}")
    print(f"{n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
