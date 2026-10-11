#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实 collator 的监督掩码检查（**CPU，只用 processor + template + collator，不加载 9B 权重**）。

目的：回答"assistant 的六字段答案到底哪些 token 参与监督、问题与图片是否被正确排除"，
并比较 SFT 的 `loss_scale=default` 与 GRPO 的 `loss_scale=last_round` 在**同一批真实单问单答**上的差别。

做法（**用框架自己的 collator，不手造 mask**）：
  1. 用与训练同一套 `get_model_processor(..., load_model=False)` 取 processor；
  2. `get_template(proc, template_type='qwen3_5', loss_scale=...)` 建两个模板；
  3. 读真实训练样本（SFT 用 adapt512，GRPO 用 group256_grpo），
     先 `encode()` 再 `data_collator()` —— **这就是训练时真正走的路径**；
  4. 逐 token 打印：文本、是否被监督（labels != -100）、loss_scale 值；
  5. 另外用一对**合成短对话**核对问答边界。

输出：`mask_check.json`（含逐 token 记录与区间摘要）。
**不加载 9B 权重、不做前向、不训练。**
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

MODEL = "/root/autodl-tmp/ws/models/Qwen3.5-9B-Dmerged"
V3 = Path("/root/autodl-tmp/ws/v3run")
OUT = V3 / "out" / "mask_check.json"

from swift.model import get_model_processor  # noqa: E402
from swift.template import get_template  # noqa: E402

# ── CPU 检查所需的最小适配（**不改框架源码，只在本次进程内打补丁**）────────────
# collator 要为 position_ids 建 dummy model，内部调用 get_model_processor(model_dir)
# **不带 model_type**，而本目录能匹配到 4 个候选类型 → ValueError。
# 这里把所有 swift 模块里按名导入的那个函数包一层，缺 model_type 时补上 qwen3_5。
import sys as _sys  # noqa: E402
import swift.model.register as _reg  # noqa: E402

_orig_gmp = _reg.get_model_processor


def _gmp_with_type(model_id_or_path, *args, **kwargs):
    if "model_type" not in kwargs and not args:
        kwargs["model_type"] = "qwen3_5"
    return _orig_gmp(model_id_or_path, *args, **kwargs)


for _name, _mod in list(_sys.modules.items()):
    if not _name.startswith("swift"):
        continue
    if getattr(_mod, "get_model_processor", None) is _orig_gmp:
        _mod.get_model_processor = _gmp_with_type
_reg.get_model_processor = _gmp_with_type


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_template(proc, loss_scale: str, mode: str = "train"):
    """建模板并**显式打开训练态**。

    关键：swift `Template._encode()` 结尾有
        `if not self.is_training: labels / loss_scale 置 None`
    所以不设 `is_training=True` 时，collator 拿到的 batch 里**根本没有 labels** ——
    我第一次跑就是这样（batch 键里没有 labels），一度误以为"CPU 查不了"。
    这是检查工具的开关问题，**不是训练没有监督**。
    """
    try:
        t = get_template(proc, template_type="qwen3_5", loss_scale=loss_scale)
    except TypeError:
        t = get_template(proc, template_type="qwen3_5")
        t._loss_scale = loss_scale
    # 关键：is_training 是**只读属性**，由 mode 推导：
    #     is_training = (mode not in {'transformers','vllm','lmdeploy','sglang'})
    # 实测 swift 的映射是 `mode_mapping = {'kto':'kto','gkd':'train','ppo':'transformers','grpo':'train'}`
    # —— **GRPO 与 SFT 用的是同一个 mode='train'**，不是 rlhf。
    # 两者真正差别只在 `loss_scale` 的 base strategy（default vs last_round）。
    t.set_mode("train")
    return t


def row_to_template_input(row: dict):
    return {"messages": row["messages"], "images": row["images"]}


def inspect(tmpl, row, label, tok, n_tok_keep=0):
    """编码 + 走真实 collator，返回逐 token 记录。"""
    ti = tmpl.encode(row_to_template_input(row), return_length=True)
    batch = tmpl.data_collator([ti])
    ids = batch["input_ids"][0].tolist()
    labels = batch.get("labels")
    labels = labels[0].tolist() if labels is not None else None
    ls = batch.get("loss_scale")
    ls = ls[0].tolist() if torch.is_tensor(ls) else (ls[0] if ls else None)
    rec = []
    for i, tid in enumerate(ids):
        rec.append({
            "pos": i,
            "token": tok.decode([tid], skip_special_tokens=False),
            "label": (labels[i] if labels is not None else None),
            "supervised": (labels is not None and labels[i] != -100),
            "loss_scale": (ls[i] if isinstance(ls, list) and i < len(ls) else None),
        })
    sup = [r["pos"] for r in rec if r["supervised"]]
    decoded = tok.decode(ids, skip_special_tokens=False)
    return {
        "标签": label,
        "sample_id": row.get("sample_id"),
        "总token": len(ids),
        "被监督token数": len(sup),
        "被监督区间": ([min(sup), max(sup)] if sup else None),
        "答案文本长度(应等于被监督数,若含结束符则+1)":
            len(tok.encode(row["messages"][-1]["content"], add_special_tokens=False)),
        "batch键": sorted(batch.keys()),
        "有labels": labels is not None,
        "loss_scale口径": getattr(tmpl, "_loss_scale", None),
        "mode": getattr(tmpl, "mode", None),
        "解码全文": decoded,
        "逐token": rec,
    }


def main() -> int:
    got = get_model_processor(MODEL, model_type="qwen3_5", load_model=False)
    proc = got[1] if isinstance(got, tuple) else got
    tok = proc.tokenizer
    res = {"模型目录": MODEL, "processor": type(proc).__name__,
           "输入SHA": {}, "检查": []}

    # 合成短对话：先核对问答边界（不掺图片，隔离出"边界"这一件事）
    syn = {"messages": [{"role": "user", "content": "PIN"},
                        {"role": "assistant", "content": "PONG"}], "images": []}
    for ls, mode in (("default", "train"), ("last_round", "train")):
        t = build_template(proc, ls, mode)
        res["检查"].append(inspect(t, syn, f"合成短对话 loss_scale={ls} mode={mode}", tok))

    # 真实样本：覆盖不同答案长度 / unknown 字段 / 方向形式
    sft_rows = [json.loads(l) for l in (V3 / "数据/adapt512_sft.jsonl").open(encoding="utf-8") if l.strip()]
    grpo_rows = [json.loads(l) for l in (V3 / "数据/group256_grpo.jsonl").open(encoding="utf-8") if l.strip()]
    res["输入SHA"]["adapt512_sft.jsonl"] = sha(V3 / "数据/adapt512_sft.jsonl")
    res["输入SHA"]["group256_grpo.jsonl"] = sha(V3 / "数据/group256_grpo.jsonl")

    # 选：最短答案、最长答案、含 angular_type=unknown、含 single/opposed
    def ans_of(r):
        return r["messages"][-1]["content"]
    sft_sorted = sorted(sft_rows, key=lambda r: len(ans_of(r)))
    picks = [("最短答案", sft_sorted[0]), ("最长答案", sft_sorted[-1])]
    for tag, key, val in (("含unknown", "angular_type", '"unknown"'),
                          ("含single", "angular_type", '"single"')):
        for r in sft_rows:
            if val in ans_of(r):
                picks.append((tag, r)); break
    seen = set()
    for tag, r in picks:
        if r["sample_id"] in seen:
            continue
        seen.add(r["sample_id"])
        for ls, mode in (("default", "train"), ("last_round", "train")):
            res["检查"].append(inspect(build_template(proc, ls, mode), r,
                                       f"{tag} / adapt512 / loss_scale={ls} mode={mode}", tok))
    # GRPO 一行（单问无答案）——看它怎么处理"没有 assistant"
    for ls, mode in (("default", "train"), ("last_round", "train")):
        res["检查"].append(inspect(build_template(proc, ls, mode), grpo_rows[0],
                                   f"GRPO单问样本 / group256 / loss_scale={ls} mode={mode}", tok))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=== 摘要（逐 token 明细在 mask_check.json）===")
    for c in res["检查"]:
        print(f"[{c['标签']}] id={c['sample_id']} 总{c['总token']} token；"
              f"被监督 {c['被监督token数']}（区间 {c['被监督区间']}）；"
              f"答案本应监督 {c['答案文本长度(应等于被监督数,若含结束符则+1)']}；"
              f"有labels={c['有labels']} batch键={c['batch键']}")
    print(f"\n写出 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
