#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SFT 两步门槛检查（B 卡 5）。
用法：python gate_sft.py --config configs/S2.json --ckpt <probe 输出里的 checkpoint-2> [--out gate_sft.json]
检查：①adapter_model.safetensors 键名含 visual 的数量（S2 必须 >0，其余必须 =0）；
      ②lora_B 张量至少一半非全零；③checkpoint 的 args.json：adapters 为空/不存在、freeze_vit 与 config 一致。
任一不过 → 退出码 3。检测逻辑拆成纯函数便于本机无 safetensors 时测试。
"""
import argparse
import json
import sys
from pathlib import Path


def visual_key_count(keys):
    return sum(1 for k in keys if "visual" in k)


def lora_b_ratio(nonzero_by_key):
    """输入 {键名: 是否非全零}，返回 (lora_B 总数, 非零数)。"""
    bs = [v for k, v in nonzero_by_key.items() if "lora_B" in k]
    return len(bs), sum(1 for v in bs if v)


def load_adapter_tensors(path):
    """读 adapter_model.safetensors，返回 {键: 非全零 bool} 与键列表。torch 优先，numpy 兜底。"""
    try:
        from safetensors.torch import load_file
        import torch
        tensors = {k: bool(torch.any(v != 0).item()) for k, v in load_file(str(path)).items()}
        return list(tensors.keys()), tensors
    except Exception:
        from safetensors import safe_open
        out = {}
        keys = []
        with safe_open(str(path), framework="np") as f:
            for k in f.keys():
                keys.append(k)
                a = f.get_tensor(k)
                out[k] = bool((a != 0).any())
        return keys, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    ckpt = Path(a.ckpt)
    out_path = Path(a.out) if a.out else ckpt.parent / "gate_sft.json"
    report = {"config": str(a.config), "ckpt": str(a.ckpt), "line": cfg["line"]}

    ok = True
    adapter = ckpt / "adapter_model.safetensors"
    if not adapter.exists():
        report["adapter_exists"] = False
        ok = False
    else:
        keys, nz = load_adapter_tensors(adapter)
        vcount = visual_key_count(keys)
        report["adapter_exists"] = True
        report["n_keys"] = len(keys)
        report["visual_key_count"] = vcount
        expect_visual = not cfg["freeze_vit"]
        report["visual_expect"] = ">0" if expect_visual else "=0"
        if expect_visual:
            if vcount <= 0:
                ok = False
        else:
            if vcount != 0:
                ok = False
        nb_total, nb_nz = lora_b_ratio(nz)
        report["lora_B_total"] = nb_total
        report["lora_B_nonzero"] = nb_nz
        report["lora_B_half_nonzero"] = bool(nb_total > 0 and nb_nz >= (nb_total + 1) // 2)
        if not report["lora_B_half_nonzero"]:
            ok = False
    args_json = ckpt / "args.json"
    if args_json.exists():
        targs = json.loads(args_json.read_text(encoding="utf-8"))
        adapters = targs.get("adapters", [])
        report["args_json_adapters"] = adapters
        report["args_json_freeze_vit"] = targs.get("freeze_vit")
        if adapters:
            ok = False
        if targs.get("freeze_vit") != cfg["freeze_vit"]:
            ok = False
    else:
        report["args_json_adapters"] = "缺失"
        ok = False
    report["passed"] = ok
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
