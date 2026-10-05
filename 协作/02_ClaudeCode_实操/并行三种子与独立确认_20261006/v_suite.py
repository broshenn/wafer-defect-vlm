#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V 任务：单卡上把「基座 / 旧 Adapter」在「原 dev18 / 新 val90 / 信息移除派生图」上跑齐。

设计目标是把模型加载次数压到 2 次（基座一次、基座+Adapter 一次），
5 轮推理共用，好在 8 分钟预留内跑完：

    1) 基座        × dev18         （排除旧推理协议差异）
    2) 基座        × val90
    3) 旧Adapter   × dev18
    4) 旧Adapter   × val90
    5) 旧Adapter   × dev18-red2green（信息移除诊断）

隔离：CUDA_VISIBLE_DEVICES 必须由父进程设置，断言只见 1 张卡。
"""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
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
    per = {}
    for c in sorted(set(y)):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"n": sum(1 for a in y if a == c), "tp": tp,
                  "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    macro = sum(v["f1"] for v in per.values()) / len(per) if per else 0.0
    conf = {}
    for a_, b_ in zip(y, p):
        conf.setdefault(a_, Counter())[b_ or "<未解析>"] += 1
    return {"n": n, "correct": correct,
            "accuracy": round(correct / n, 4) if n else 0.0,
            "macro_f1": round(macro, 4), "per_class": per,
            "confusion": {k: dict(v) for k, v in sorted(conf.items())},
            "pred_distribution": dict(Counter(p)),
            "parse_how": dict(Counter(r["parse_how"] for r in rows))}


def build_red2green(rows, outdir):
    """只把红色(2)改成绿色(1)，其他像素与分辨率不动。派生图另存并记 hash。"""
    from PIL import Image
    import numpy as np
    os.makedirs(outdir, exist_ok=True)
    out_rows, meta = [], []
    for r in rows:
        im = Image.open(r["image_path"])
        arr = np.array(im)
        before = int((arr == 2).sum()) if arr.ndim == 3 else 0
        # 取最接近纯红的通道判断：渲染是 [255,0,0]
        if arr.ndim == 3:
            mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 60) & (arr[:, :, 2] < 60)
            arr = arr.copy()
            arr[mask] = [0, 255, 0]
        p = os.path.join(outdir, os.path.basename(r["image_path"]))
        Image.fromarray(arr).save(p)
        out_rows.append({"sample_id": r["sample_id"], "label": r["label"], "image_path": p})
        meta.append({"sample_id": r["sample_id"], "red_pixels_before": before,
                     "derived": p, "derived_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
    return out_rows, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/WS/models/Qwen3.5-9B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dev18", default="/WS/datasets/dev_18.server.jsonl")
    ap.add_argument("--val90", default="/WS/datasets/val90/val90.server.jsonl")
    ap.add_argument("--red2green-dir", default="/WS/datasets/dev18_red2green")
    ap.add_argument("--outdir", default="/WS/checkpoints/v_suite")
    a = ap.parse_args()

    import torch
    print("=" * 78)
    print(f"V 任务  {datetime.datetime.now().isoformat()}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')!r}（父进程设置）")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("!! 父进程未设置 —— 拒绝运行"); return 3
    torch.cuda.init()
    n = torch.cuda.device_count()
    print(f"可见 GPU 数 = {n}  device 0 = {torch.cuda.get_device_name(0)}")
    if n != 1:
        print("!! 不是 1 张卡 —— 拒绝运行"); return 3
    os.makedirs(a.outdir, exist_ok=True)

    from swift import InferRequest, RequestConfig
    from swift.infer_engine import TransformersEngine

    dev18 = [json.loads(l) for l in io.open(a.dev18, encoding="utf-8") if l.strip()]
    val90 = [json.loads(l) for l in io.open(a.val90, encoding="utf-8") if l.strip()]
    print(f"dev18 {len(dev18)} 条   val90 {len(val90)} 条")

    # 派生图（信息移除诊断）
    red_rows, red_meta = build_red2green(dev18, a.red2green_dir)
    print(f"信息移除派生图 {len(red_rows)} 张 -> {a.red2green_dir}")
    with io.open(os.path.join(a.outdir, "red2green_meta.json"), "w", encoding="utf-8",
                 newline="\n") as f:
        f.write(json.dumps({"transform": "仅把红色(2)失效像素改为绿色(1)；轮廓/分辨率/其他像素不变",
                            "note": "原标签不冒称改图后的 gold；下降只是视觉信息依赖线索",
                            "images": red_meta}, ensure_ascii=False, indent=2) + "\n")

    cfg = RequestConfig(max_tokens=64, temperature=0.0, seed=3407)
    results = {}
    t_all = time.time()

    def run(engine, rows, tag):
        out = os.path.join(a.outdir, f"{tag}.jsonl")
        t0 = time.time()
        fo = io.open(out, "w", encoding="utf-8", newline="\n")
        for i, r in enumerate(rows, 1):
            resp = engine.infer([InferRequest(
                messages=[{"role": "user", "content": "<image>" + QUESTION}],
                images=[r["image_path"]])], cfg)[0]
            raw = resp.choices[0].message.content
            pred, why = parse_class(raw)
            fo.write(json.dumps({"sample_id": r["sample_id"], "label": r["label"], "raw": raw,
                                 "parsed": pred, "parse_how": why,
                                 "correct": bool(pred == r["label"]),
                                 "image_path": r["image_path"]}, ensure_ascii=False) + "\n")
            fo.flush()
        fo.close()
        res = [json.loads(l) for l in io.open(out, encoding="utf-8") if l.strip()]
        m = metrics(res)
        m["seconds"] = round(time.time() - t0, 1)
        m["out"] = out
        results[tag] = m
        print(f"  [{tag}] Acc {m['correct']}/{m['n']}={m['accuracy']}  "
              f"MacroF1 {m['macro_f1']}  预测分布 {m['pred_distribution']}  "
              f"{m['seconds']}s", flush=True)
        return m

    print("\n=== 1) 基座 ===", flush=True)
    e_base = TransformersEngine(a.model)
    run(e_base, dev18, "base_dev18")
    run(e_base, val90, "base_val90")
    del e_base
    gc.collect(); torch.cuda.empty_cache()
    print(f"  释放基座后 allocated {torch.cuda.memory_allocated(0)/2**30:.2f} GiB", flush=True)

    print("\n=== 2) 基座 + 旧 Adapter ===", flush=True)
    e_old = TransformersEngine(a.model, adapters=a.adapter)
    run(e_old, dev18, "oldadapter_dev18")
    run(e_old, val90, "oldadapter_val90")
    run(e_old, red_rows, "oldadapter_dev18_red2green")

    peak = torch.cuda.max_memory_allocated(0) / 2**30
    summary = {"adapter": a.adapter,
               "adapter_config_sha256": hashlib.sha256(
                   open(os.path.join(a.adapter, "adapter_config.json"), "rb").read()).hexdigest(),
               "device_name": torch.cuda.get_device_name(0),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "total_seconds": round(time.time() - t_all, 1),
               "peak_allocated_GiB_own_process": round(peak, 3),
               "results": results,
               "dev18_n": len(dev18), "val90_n": len(val90),
               "finished_at": datetime.datetime.now().isoformat()}
    with io.open(os.path.join(a.outdir, "v_summary.json"), "w", encoding="utf-8",
                 newline="\n") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    # 信息移除：预测分布变化
    base18 = [json.loads(l) for l in io.open(results["oldadapter_dev18"]["out"], encoding="utf-8")]
    red18 = [json.loads(l) for l in io.open(results["oldadapter_dev18_red2green"]["out"], encoding="utf-8")]
    changed = [{"sample_id": x["sample_id"], "label": x["label"],
                "orig_pred": y["parsed"], "red2green_pred": z["parsed"]}
               for x, y, z in zip(dev18, base18, red18) if y["parsed"] != z["parsed"]]
    print(f"\n=== 信息移除诊断 ===")
    print(f"  原图正确 {sum(1 for x in base18 if x['correct'])}/{len(base18)}")
    print(f"  改图后正确 {sum(1 for x in red18 if x['correct'])}/{len(red18)}")
    print(f"  预测发生变化 {len(changed)}/{len(dev18)} 条")
    print(f"  预测分布 原图 {results['oldadapter_dev18']['pred_distribution']}")
    print(f"          改图 {results['oldadapter_dev18_red2green']['pred_distribution']}")
    with io.open(os.path.join(a.outdir, "red2green_diff.json"), "w", encoding="utf-8",
                 newline="\n") as f:
        f.write(json.dumps({"changed": changed,
                            "n_changed": len(changed),
                            "acc_orig": results["oldadapter_dev18"]["accuracy"],
                            "acc_red2green": results["oldadapter_dev18_red2green"]["accuracy"],
                            "note": "原标签不冒称改图后的 gold；下降只是视觉信息依赖线索，"
                                    "不称能力充分证明"},
                           ensure_ascii=False, indent=2) + "\n")
    print(f"\n总耗时 {summary['total_seconds']}s  峰值 {peak:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
