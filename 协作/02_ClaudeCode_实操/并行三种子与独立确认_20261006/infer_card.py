#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单卡分类推理：基座或 基座+Adapter，跑任意一份样本清单。

与训练同源：题面固定、temperature=0、max_tokens=64、推理 seed 统一 3407、
解析规则与旧 dev18 一致（只认九类 + unknown，解析失败不剔除、单独记 parse_how）。

隔离：**不在 Python 里改 CUDA_VISIBLE_DEVICES**，断言父进程已设且只见 1 张卡。

用法：
    CUDA_VISIBLE_DEVICES=<uuid> python infer_card.py --data x.jsonl --out y.jsonl
    CUDA_VISIBLE_DEVICES=<uuid> python infer_card.py --data x.jsonl --out y.jsonl \
        --adapters <checkpoint-60>
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import re
import sys
import time
from collections import Counter

os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")

CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]

QUESTION = ("这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。"
            "请从 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_full、Random、Scratch、none "
            "中选一个最符合主要图案的类别。只输出一个 JSON 对象，形如 "
            '{"defect_class":"类别"}；无法可靠判断时填 {"defect_class":"unknown"}。')


def parse_class(text):
    """与旧 dev18 完全相同的解析规则。解析失败不剔除，记录 parse_how。"""
    if not text:
        return None, "empty"
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            o = json.loads(m.group(0))
            v = o.get("defect_class")
            if isinstance(v, str):
                return (v if v in CLASSES + ["unknown"] else None), \
                       ("ok" if v in CLASSES + ["unknown"] else "bad_enum")
        except json.JSONDecodeError:
            pass
    for c in CLASSES + ["unknown"]:
        if c in text:
            return c, "loose_match"
    return None, "unparsed"


def metrics(rows):
    y = [r["label"] for r in rows]
    p = [r["parsed"] for r in rows]
    n = len(rows)
    correct = sum(1 for a, b in zip(y, p) if a == b)
    acc = correct / n if n else 0.0
    # macro-F1（只在有真值的类上算）
    labels = sorted(set(y) | {x for x in p if x})
    per = {}
    for c in labels:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"n": sum(1 for a in y if a == c), "tp": tp, "precision": round(prec, 4),
                  "recall": round(rec, 4), "f1": round(f1, 4)}
    macro_f1 = sum(v["f1"] for v in per.values()) / len(per) if per else 0.0
    conf = {}
    for a, b in zip(y, p):
        conf.setdefault(a, Counter())[b or "<未解析>"] += 1
    return {"n": n, "correct": correct, "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4), "per_class": per,
            "confusion": {k: dict(v) for k, v in sorted(conf.items())},
            "pred_distribution": dict(Counter(p)),
            "parse_how": dict(Counter(r["parse_how"] for r in rows))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/WS/models/Qwen3.5-9B")
    ap.add_argument("--adapters", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import torch
    print("=" * 78)
    print(f"单卡推理  {datetime.datetime.now().isoformat()}  tag={a.tag}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')!r}（父进程设置）")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("!! 父进程未设置 CUDA_VISIBLE_DEVICES —— 拒绝运行")
        return 3
    torch.cuda.init()
    n = torch.cuda.device_count()
    print(f"可见 GPU 数 = {n}   device 0 = {torch.cuda.get_device_name(0)}")
    if n != 1:
        print(f"!! 可见 {n} 张卡，不是 1 张 —— 拒绝运行")
        return 3

    from swift import InferRequest, RequestConfig
    from swift.infer_engine import TransformersEngine

    rows = [json.loads(l) for l in io.open(a.data, encoding="utf-8") if l.strip()]
    print(f"样本 {len(rows)} 条   adapters={a.adapters}")

    t0 = time.time()
    engine = TransformersEngine(a.model, adapters=a.adapters)
    load_s = time.time() - t0
    cfg = RequestConfig(max_tokens=a.max_new_tokens, temperature=0.0, seed=3407)

    out = io.open(a.out, "w", encoding="utf-8", newline="\n")
    t1 = time.time()
    for i, r in enumerate(rows, 1):
        req = InferRequest(messages=[{"role": "user", "content": "<image>" + QUESTION}],
                           images=[r["image_path"]])
        resp = engine.infer([req], cfg)[0]
        raw = resp.choices[0].message.content
        pred, why = parse_class(raw)
        out.write(json.dumps({
            "sample_id": r["sample_id"], "label": r["label"],
            "raw": raw, "parsed": pred, "parse_how": why,
            "correct": bool(pred == r["label"]), "image_path": r["image_path"],
        }, ensure_ascii=False) + "\n")
        out.flush()
        if i <= 3 or i % 20 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] {r['sample_id']} 真值={r['label']:10s} 预测={pred}", flush=True)
    out.close()
    infer_s = time.time() - t1

    res = [json.loads(l) for l in io.open(a.out, encoding="utf-8") if l.strip()]
    m = metrics(res)
    peak = torch.cuda.max_memory_allocated(0) / 2**30
    m.update({"tag": a.tag, "adapters": a.adapters, "data": a.data,
              "load_seconds": round(load_s, 1), "infer_seconds": round(infer_s, 1),
              "peak_allocated_GiB_own_process": round(peak, 3),
              "device_name": torch.cuda.get_device_name(0),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "finished_at": datetime.datetime.now().isoformat()})
    with io.open(a.out + ".metrics.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(m, ensure_ascii=False, indent=2) + "\n")

    print(f"\n=== 结果 {a.tag} ===")
    print(f"  Accuracy   {m['correct']}/{m['n']} = {m['accuracy']}")
    print(f"  Macro-F1   {m['macro_f1']}")
    print(f"  预测分布   {m['pred_distribution']}")
    print(f"  解析方式   {m['parse_how']}")
    print(f"  逐类召回   " + "  ".join(f"{c}:{v['recall']}" for c, v in m["per_class"].items()))
    print(f"  加载 {load_s:.1f}s  推理 {infer_s:.1f}s  峰值 {peak:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
