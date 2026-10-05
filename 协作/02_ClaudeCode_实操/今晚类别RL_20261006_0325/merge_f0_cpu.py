#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F0：把 A1 的 LoRA 合并进基座（**优先纯 CPU**）。

坑：`AutoModelForCausalLM` 载入的是**纯文本**模型，模块路径里没有
`model.language_model.*`，peft 会报 "Target modules ... not found"。
必须用 ms-swift 自己的多模态加载器拿到与训练时**同名同结构**的模型。

失败即回退提示，不硬撑。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import io
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/WS/models/Qwen3.5-9B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--meta", default="/WS/logs/f0_merge_meta.json")
    a = ap.parse_args()

    print("=" * 78)
    print(f"F0 合并（CPU）  {datetime.datetime.now().isoformat()}")
    print("=" * 78)
    if os.path.exists(a.outdir):
        print("!! 输出目录已存在，拒绝覆盖"); return 3

    import torch
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}（应为 False）")

    from swift.model import get_model_processor
    print("\nget_model_processor 签名:")
    print(" ", inspect.signature(get_model_processor))

    model = None
    used = None
    for label, kw in [
        ("dtype+device_map", dict(torch_dtype=torch.bfloat16, device_map={"": "cpu"})),
        ("dtype only", dict(torch_dtype=torch.bfloat16)),
        ("device_map only", dict(device_map={"": "cpu"})),
        ("bare", dict()),
    ]:
        try:
            out = get_model_processor(a.model, **kw)
            model = out[0] if isinstance(out, (tuple, list)) else out
            used = label
            print(f"\n加载成功：{label}  -> {type(model).__name__}")
            break
        except Exception as e:
            print(f"  {label} 失败: {type(e).__name__}: {str(e)[:120]}")
    if model is None:
        print("!! 四种调用都失败，需改用 GPU 合并"); return 4

    names = [n for n, _ in model.named_parameters()]
    print(f"参数张量 {len(names)} 个；样例: {names[0]}")
    has_lm = any(n.startswith("model.language_model.") for n in names)
    print(f"含 'model.language_model.' 前缀: {has_lm}")
    if not has_lm:
        print("!! 模块路径与训练时不一致，peft 会报 target not found"); return 4

    ac = json.load(open(os.path.join(a.adapter, "adapter_config.json"), encoding="utf-8"))
    print(f"adapter: r={ac['r']} alpha={ac['lora_alpha']} "
          f"dropout={ac['lora_dropout']}")
    print(f"  target_modules = {ac['target_modules']}")

    t0 = datetime.datetime.now()
    from peft import PeftModel
    print("\n挂 adapter…", flush=True)
    m = PeftModel.from_pretrained(model, a.adapter)
    print("merge_and_unload…", flush=True)
    m = m.merge_and_unload()
    os.makedirs(a.outdir, exist_ok=True)
    print("保存…", flush=True)
    m.save_pretrained(a.outdir, safe_serialization=True)
    try:
        from swift.model import get_model_processor as g
        o = g(a.model)
        proc = o[1] if isinstance(o, (tuple, list)) and len(o) > 1 else None
        if proc is not None and hasattr(proc, "save_pretrained"):
            proc.save_pretrained(a.outdir)
            print("processor 已保存")
    except Exception as e:
        print("processor 保存跳过:", type(e).__name__)
    dt = (datetime.datetime.now() - t0).total_seconds()

    files = sorted(os.listdir(a.outdir))
    print(f"\n完成 {dt:.1f}s  文件 {len(files)} 个")
    tot = 0
    for f in files:
        p = os.path.join(a.outdir, f)
        if os.path.isfile(p):
            sz = os.path.getsize(p)
            tot += sz
            print(f"  {f:46s} {sz/2**20:9.1f} MiB")
    print(f"  合计 {tot/2**30:.2f} GiB")

    meta = {
        "base_model": a.model, "adapter": a.adapter, "outdir": a.outdir,
        "loader": used, "device": "cpu", "ran_on_gpu": False,
        "adapter_config_sha256": sha(os.path.join(a.adapter, "adapter_config.json")),
        "adapter_model_sha256": sha(os.path.join(a.adapter, "adapter_model.safetensors")),
        "seconds": round(dt, 1),
        "files": {f: sha(os.path.join(a.outdir, f)) for f in files
                  if f.endswith((".safetensors", ".json"))},
        "note": "纯 CPU 合并，不占 GPU 分钟；未覆盖基座与 A1 LoRA。",
        "finished_at": datetime.datetime.now().isoformat(),
    }
    with io.open(a.meta, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"\n元数据: {a.meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
