#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 模板 / labels 门槛（**CPU，不占 GPU**）。

做三件事：
  1. 用真实 ms-swift 模板把 v3 训练样本编码，检查图片占位、长度、答案 token 数；
  2. **模板级证据**：证明 assistant 段前面的 `<think>\\n\\n</think>\\n\\n` 是**模板自己发出的** ——
     这解释了为什么 360/360 原答都带这个前缀，也说明 v3 原件的 `parse()` 必然解析失败；
  3. 如实报告：`encode()` **不返回 labels**（监督掩码在 collator 里做），
     所以 CPU 这一层**无法**直接验证 mask；该项**交由 2 步 GPU 门槛**
     （runtime_guard 会数可训练张量、非零梯度与真实前向）。

退出码 0 = 模板级检查通过；5 = 不通过。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

MODEL = "/root/autodl-tmp/ws/models/Qwen3.5-9B-Dmerged"
DATA = Path("/root/autodl-tmp/ws/v3run/数据/adapt512_sft.jsonl")
MAX_LEN = 2048
MAX_COMPLETION = 256
N = 3

from swift.model import get_model_processor  # noqa: E402
from swift.template import get_template  # noqa: E402

# 实测：load_model=False 返回 (None, processor)，processor 在第二个位置且带 model_info
got = get_model_processor(MODEL, model_type="qwen3_5", load_model=False)
proc = got[1] if isinstance(got, tuple) else got
tmpl = get_template(proc, template_type="qwen3_5")
tok = proc.tokenizer
print("模板类型:", type(tmpl).__name__)

# ── 2. 模板级证据：assistant 前缀由模板发出 ────────────────────────────
probe = tmpl.encode({"messages": [{"role": "user", "content": "x"},
                                  {"role": "assistant", "content": "ANSWER"}]},
                    return_length=True)
pids = probe["input_ids"]
pids = pids[0].tolist() if torch.is_tensor(pids) and pids.dim() > 1 else list(pids)
probe_text = tok.decode(pids, skip_special_tokens=False)
think_from_template = "<|im_start|>assistant\n<think>\n\n</think>\n\n" in probe_text
print("\n[模板级证据] assistant 段原文:", repr(probe_text))
print("[模板级证据] 空 think 前缀来自模板:", think_from_template)

# ── 1. 真实样本编码检查 ───────────────────────────────────────────────
rows = [json.loads(l) for l in DATA.open(encoding="utf-8") if l.strip()][:N]
fails, details = [], []
for i, r in enumerate(rows):
    img = Path(r["images"][0])
    if not img.exists():
        fails.append(f"[{i}] 图片不存在 {img}")
        continue
    out = tmpl.encode({"messages": r["messages"], "images": [str(img)]},
                      return_length=True)
    if i == 0:
        print("encode 返回键（注意：没有 labels）:", sorted(out.keys()))
    ids = out["input_ids"]
    ids = ids[0].tolist() if torch.is_tensor(ids) and ids.dim() > 1 else list(ids)
    decoded = tok.decode(ids, skip_special_tokens=False)
    ans = r["messages"][-1]["content"]
    ans_tokens = len(tok.encode(ans, add_special_tokens=False))
    has_img = "<|vision_start|>" in decoded
    n_tot = len(ids)
    details.append({"sample_id": r["sample_id"], "总长": n_tot, "图片占位": has_img,
                    "答案token": ans_tokens, "解码尾部与目标一致": decoded.endswith(ans)})
    print(f"[{i}] {r['sample_id']} 总长 {n_tot} 图片占位 {'有' if has_img else '**无**'} "
          f"答案token {ans_tokens} 尾部与目标一致 {decoded.endswith(ans)}")
    if not has_img:
        fails.append(f"[{i}] 没看到图片占位符")
    if ans_tokens >= MAX_COMPLETION:
        fails.append(f"[{i}] 答案 {ans_tokens} ≥ max_completion_length={MAX_COMPLETION}")
    if n_tot >= MAX_LEN:
        fails.append(f"[{i}] 总长 {n_tot} ≥ max_length={MAX_LEN}")
    if not decoded.endswith(ans):
        fails.append(f"[{i}] 解码尾部与训练目标不一致")

if not think_from_template:
    fails.append("模板级证据不成立：assistant 前缀里没有空 think")

verdict = "PASS" if not fails else fails
print("\n判定:", verdict)
print("说明：labels（监督掩码）不在 encode() 返回里，本层无法验证；"
      "该项交由 2 步 GPU 门槛（runtime_guard 数可训练张量/非零梯度/真实前向）。")

outp = Path("/root/autodl-tmp/ws/v3run/out/template_gate.json")
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps({
    "think_prefix_from_template": think_from_template,
    "assistant_prefix_example": probe_text,
    "samples": details,
    "labels_checked_here": False,
    "labels_note": "encode() 不返回 labels；掩码由 collator 生成，交 2 步 GPU 门槛",
    "verdict": verdict,
}, ensure_ascii=False, indent=1), encoding="utf-8")
sys.exit(0 if not fails else 5)
