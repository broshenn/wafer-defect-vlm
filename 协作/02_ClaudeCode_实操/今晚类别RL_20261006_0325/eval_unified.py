#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一评测：F0 / F1 / F2 在 dev18 + 既有 val90(84) 上，同题面、同解码。

照任务书 §6：
  * 原 dev18 + 既有 val90 的 84 条，**不增补 Near_full**，不碰 test/冻结 benchmark；
  * 推理 seed 3407、temperature 0、max_tokens 64、原题；
  * **严格解析为主结果**，另报旧解析桥接结果，以及非法/空/unknown/截断计数；
  * F0 用已验收的 A1 基线原答，**不重跑整个基座**（但 F0=合并模型需与 A1 原答核对一致性）；
  * 主报告九类固定 Macro-F1、Accuracy、逐类召回、混淆、同题新增答对/丢对、F2−F0 与 F2−F1。

用法：
    CUDA_VISIBLE_DEVICES=<uuid> python eval_unified.py --models <a>=<dir> [<b>=<dir> ...]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import re
import sys
import time
from collections import Counter

os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_v1 import parse_answer, CLASSES      # noqa: E402

QUESTION = ("这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。"
            "请从 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_full、Random、Scratch、none "
            "中选一个最符合主要图案的类别。只输出一个 JSON 对象，形如 "
            '{"defect_class":"类别"}；无法可靠判断时填 {"defect_class":"unknown"}。')


def old_parse(text):
    """旧解析器（严格版之前的那个），用于桥接对照。"""
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


def metrics(rows, parse_key):
    y = [r["label"] for r in rows]
    p = [r[parse_key] for r in rows]
    n = len(rows)
    ok = sum(1 for a, b in zip(y, p) if a == b)
    per = {}
    for c in CLASSES:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"n": sum(1 for a in y if a == c), "tp": tp,
                  "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    present = [v for k, v in per.items() if v["n"] > 0]
    macro = sum(v["f1"] for v in present) / len(present) if present else 0.0
    conf = {}
    for a_, b_ in zip(y, p):
        conf.setdefault(a_, Counter())[b_ or "<未解析>"] += 1
    return {"n": n, "correct": ok, "accuracy": round(ok / n, 4) if n else 0.0,
            "macro_f1_fixed9": round(macro, 6), "per_class": per,
            "confusion": {k: dict(v) for k, v in sorted(conf.items())},
            "pred_distribution": dict(Counter(p))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="形如 f2=<dir>；也接受 f0=FILE 直接读已有原答")
    ap.add_argument("--dev18", default="/WS/datasets/dev_18.server.jsonl")
    ap.add_argument("--val90", default="/WS/datasets/val90/val90.server.jsonl")
    ap.add_argument("--outdir", default="/WS/checkpoints/eval_unified")
    ap.add_argument("--model-type", default="qwen3_5")
    ap.add_argument("--template-type", default="qwen3_5")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    import torch
    print("=" * 78)
    print(f"统一评测  {datetime.datetime.now().isoformat()}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')!r}（父进程设置）")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("!! 父进程未设置 —— 拒绝运行"); return 3
    torch.cuda.init()
    n = torch.cuda.device_count()
    print(f"可见 GPU 数 = {n}  device 0 = {torch.cuda.get_device_name(0)}")
    if n != 1:
        print("!! 不是 1 张卡 —— 拒绝运行"); return 3
    os.makedirs(a.outdir, exist_ok=True)

    dev = [json.loads(l) for l in io.open(a.dev18, encoding="utf-8") if l.strip()]
    val = [json.loads(l) for l in io.open(a.val90, encoding="utf-8") if l.strip()]
    if a.limit:
        dev, val = dev[:a.limit], val[:a.limit]
    print(f"dev18 {len(dev)}   val90 {len(val)}")

    from swift import InferRequest, RequestConfig
    from swift.infer_engine import TransformersEngine

    results = {}
    for spec in a.models:
        tag, _, path = spec.partition("=")
        # 支持 base::adapter：checkpoint 目录是 LoRA adapter，没有 config.json，
        # 必须挂在 F0 合并模型上加载。
        base, _, adapter = path.partition("::")
        is_adapter = bool(adapter)
        for dsname, rows in (("dev18", dev), ("val90", val)):
            out = os.path.join(a.outdir, f"{tag}_{dsname}.jsonl")
            if a.skip_existing and os.path.exists(out):
                print(f"  跳过已存在的 {out}")
                continue
            if os.path.exists(path) and path.endswith(".jsonl"):
                # 已有原答：只做重新解析，不跑推理
                src = [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]
                if len(src) != len(rows):
                    print(f"  !! {tag}/{dsname} 原答 {len(src)} 条 != 数据 {len(rows)} 条，跳过")
                    continue
                recs = []
                for r, s in zip(rows, src):
                    assert r["sample_id"] == s["sample_id"], (r["sample_id"], s["sample_id"])
                    sp, how = parse_answer(s["raw"] if "raw" in s else s.get("response"))
                    op, ohow = old_parse(s.get("raw"))
                    recs.append({"sample_id": r["sample_id"], "label": r["label"],
                                 "raw": s.get("raw"), "strict_parsed": sp, "strict_how": how,
                                 "old_parsed": op, "old_how": ohow,
                                 "strict_correct": bool(sp == r["label"]),
                                 "old_correct": bool(op == r["label"])})
                print(f"\n--- {tag} / {dsname}（复用已存原答，重解析）---")
                run_s = None
            else:
                print(f"\n--- {tag} / {dsname}（推理"
                      f"{'，adapter=' + adapter if is_adapter else ''}）---", flush=True)
                t0 = time.time()
                eng = TransformersEngine(base, model_type=a.model_type,
                                         template_type=a.template_type,
                                         adapters=[adapter] if is_adapter else None)
                run_s = time.time() - t0
                cfg = RequestConfig(max_tokens=64, temperature=0.0, seed=3407)
                recs = []
                for i, r in enumerate(rows, 1):
                    resp = eng.infer([InferRequest(
                        messages=[{"role": "user", "content": "<image>" + QUESTION}],
                        images=[r["image_path"]])], cfg)[0]
                    raw = resp.choices[0].message.content
                    sp, how = parse_answer(raw)
                    op, ohow = old_parse(raw)
                    recs.append({"sample_id": r["sample_id"], "label": r["label"], "raw": raw,
                                 "strict_parsed": sp, "strict_how": how,
                                 "old_parsed": op, "old_how": ohow,
                                 "strict_correct": bool(sp == r["label"]),
                                 "old_correct": bool(op == r["label"])})
                    if i % 20 == 0 or i == len(rows):
                        print(f"  [{i}/{len(rows)}]", flush=True)
                del eng
                import gc; gc.collect(); torch.cuda.empty_cache()
            with io.open(out, "w", encoding="utf-8", newline="\n") as f:
                for x in recs:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            m_strict = metrics(recs, "strict_parsed")
            m_old = metrics(recs, "old_parsed")
            bad = dict(Counter(x["strict_how"] for x in recs if x["strict_parsed"] is None))
            results[f"{tag}/{dsname}"] = {
                "strict": m_strict, "old_bridge": {"accuracy": m_old["accuracy"],
                                                   "macro_f1_fixed9": m_old["macro_f1_fixed9"]},
                "strict_reject_reasons": bad,
                "strict_rejected": sum(bad.values()),
                "unknown_count": sum(1 for x in recs if x["strict_parsed"] == "unknown"),
                "model_path": path, "n": len(recs), "load_seconds": run_s}
            print(f"  严格 {m_strict['correct']}/{m_strict['n']}={m_strict['accuracy']} "
                  f"MacroF1 {m_strict['macro_f1_fixed9']} | 旧解析 {m_old['accuracy']} / "
                  f"{m_old['macro_f1_fixed9']} | 拒绝 {sum(bad.values())} {bad}", flush=True)

    # 配对比较
    pair = {}
    for ds in ("dev18", "val90"):
        for x, y in (("f2", "f0"), ("f2", "f1"), ("f1", "f0")):
            A, B = results.get(f"{x}/{ds}"), results.get(f"{y}/{ds}")
            if not A or not B:
                continue
            ra = [json.loads(l) for l in io.open(f"{a.outdir}/{x}_{ds}.jsonl", encoding="utf-8")]
            rb = [json.loads(l) for l in io.open(f"{a.outdir}/{y}_{ds}.jsonl", encoding="utf-8")]
            gained = [p["sample_id"] for p, q in zip(ra, rb)
                      if p["strict_correct"] and not q["strict_correct"]]
            lost = [p["sample_id"] for p, q in zip(ra, rb)
                    if not p["strict_correct"] and q["strict_correct"]]
            pair[f"{x}-{y}/{ds}"] = {"gained": len(gained), "lost": len(lost),
                                     "net": len(gained) - len(lost),
                                     "gained_ids": gained[:12], "lost_ids": lost[:12]}

    summary = {
        "date": "2026-10-06", "dev18_n": len(dev), "val90_n": len(val),
        "results": results, "paired": pair,
        "note": "严格解析为主结果；旧解析仅作桥接对照。val90 是开发确认集，非冻结 Benchmark。",
        "finished_at": datetime.datetime.now().isoformat(),
    }
    with io.open(os.path.join(a.outdir, "eval_summary.json"), "w", encoding="utf-8",
                 newline="\n") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print("\n" + "=" * 78)
    print(f"{'run':22s} {'严格Acc':>10s} {'MacroF1':>9s} {'旧桥接':>9s} {'拒绝':>5s}")
    for k, v in results.items():
        print(f"{k:22s} {v['strict']['accuracy']:>10} {v['strict']['macro_f1_fixed9']:>9} "
              f"{v['old_bridge']['accuracy']:>9} {v['strict_rejected']:>5}")
    print("\n配对：")
    for k, v in pair.items():
        print(f"  {k:22s} 新增答对 {v['gained']:3d}  丢对 {v['lost']:3d}  净 {v['net']:+d}")
    print(f"\nsummary -> {a.outdir}/eval_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
