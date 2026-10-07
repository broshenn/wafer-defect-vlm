#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D0 / D1 同题描述评测：dev18 + val90(84) = 102 张，原**七字段**题面。

题面取自 `描述评测协议.json`（已核 SHA256），**不是训练用的分型题面**——
所以本轮分类成绩不能套用旧的类别-only 结果。

生成：temperature 0 / max_tokens 256 / seed 3407 / thinking 关闭（模板默认 False，已核）。
保存每一个原始回答；解析失败、截断、空答一概原样保留，不重试、不临时放大 max_tokens。

报告分两类，严格分开：
  * 合规指标（七字段合法率、字段缺失、空 Caption、截断、未验证数值/根因关键词）
    —— 这些**不等于事实正确率**；
  * 分类辅助指标（九类 Accuracy 与固定 Macro-F1）—— 新题面，不与旧成绩混写。
有独立 gold 的字段才自动判事实；无独立依据的字段一律列**未测**。
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

R = "/WS"
BASE = R + "/models/Qwen3.5-9B"

QUESTION = (
    "这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。\n"
    "请**只根据图中实际可见的内容**描述主要缺陷图案。不要推测工艺根因，"
    "不要把没有看到的模式写进答案，不要给出未经验证的数值尺寸、比例或覆盖率。\n"
    "只输出一个 JSON 对象，不写推理过程或 Markdown，必须包含以下七个字段：\n"
    '{"defect_class":"九类之一或 unknown",'
    '"morphology":"主要可见形态的简短中文描述",'
    '"radial_zone":"center/middle/edge/global/none/unknown",'
    '"clock_direction":"可靠识别局部缺陷方向时写简短钟点方向，否则 null",'
    '"extent_r":null,'
    '"caption_zh":"一句话客观描述主要形态与位置",'
    '"uncertainty":"不确定之处，无则空字符串"}\n'
    "defect_class 从 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_full、Random、"
    "Scratch、none 中选一个；无法可靠判断时填 \"unknown\"。\n"
    "图片上方为 12 点钟，右方为 3 点钟。中央、整圈、全局、正常或不确定时 "
    "clock_direction 填 null。extent_r 固定为 null（本轮不提供尺寸依据）。"
)
QUESTION_SHA = "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda"

FIELDS = ["defect_class", "morphology", "radial_zone", "clock_direction",
          "extent_r", "caption_zh", "uncertainty"]
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
ZONES = ["center", "middle", "edge", "global", "none", "unknown"]

EMPTY_THINK = re.compile(r"^\s*<think\b[^>]*>\s*</think\s*>\s*", re.S)
NUMRE = re.compile(r"\d+(?:\.\d+)?\s*%|0?\.\d+\s*[Rr]\b|百分之[一二三四五六七八九十百千\d]+"
                   r"|约?\s*\d+(?:\.\d+)?\s*(?:倍|成)")
ROOTRE = re.compile(r"根因|工艺|制程|刻蚀|沉积|光刻|机台|设备故障|污染源|导致")


class DupKey(ValueError):
    pass


def _nodup(pairs):
    k = [p[0] for p in pairs]
    if len(k) != len(set(k)):
        raise DupKey("duplicate_key")
    return dict(pairs)


def parse7(text):
    """严格解析七字段对象。返回 (obj 或 None, 判定码)。"""
    if text is None:
        return None, "none_input"
    if not isinstance(text, str):
        return None, "not_str"
    s = EMPTY_THINK.sub("", text).strip()
    if not s:
        return None, "empty"
    try:
        obj, end = json.JSONDecoder(object_pairs_hook=_nodup).raw_decode(s)
    except DupKey:
        return None, "duplicate_key"
    except json.JSONDecodeError:
        if s.startswith("{") and not s.rstrip().endswith("}"):
            return None, "truncated"
        if s.count("{") > 1:
            return None, "multiple_objects"
        return None, "not_json"
    if s[end:].strip():
        return None, "trailing_text"
    if not isinstance(obj, dict):
        return None, "not_object"
    return obj, "ok"


def metrics(pairs):
    """pairs: [(gold, pred)] → Accuracy 与固定九类 Macro-F1。"""
    n = len(pairs)
    ok = sum(1 for g, p in pairs if g == p)
    f1s = []
    for c in CLASSES:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"n": n, "correct": ok, "accuracy": round(ok / n, 4) if n else 0.0,
            "macro_f1": round(sum(f1s) / len(f1s), 4),
            "per_class_f1": {c: round(v, 4) for c, v in zip(CLASSES, f1s)}}


def summarise(recs):
    """合规指标：只描述输出形态，不判断事实。"""
    n = len(recs)
    how = Counter(r["how"] for r in recs)
    miss = Counter()
    for r in recs:
        if r["obj"] is None:
            continue
        for f in FIELDS:
            if f not in r["obj"]:
                miss[f] += 1
    badnum = [r["sample_id"] for r in recs
              if (r["obj"] or {}).get("extent_r") not in (None, "null")
              or NUMRE.search(str((r["obj"] or {}).get("morphology") or "")
                              + str((r["obj"] or {}).get("caption_zh") or ""))]
    root = [r["sample_id"] for r in recs
            if ROOTRE.search(str((r["obj"] or {}).get("morphology") or "")
                             + str((r["obj"] or {}).get("caption_zh") or ""))]
    emptycap = [r["sample_id"] for r in recs
                if not str((r["obj"] or {}).get("caption_zh") or "").strip()]
    return {"n": n, "解析判定": dict(how),
            "七字段合法率": round(how.get("ok", 0) / n, 4) if n else 0.0,
            "字段缺失计数": dict(miss),
            "空Caption": len(emptycap), "空Caption比例": round(len(emptycap) / n, 4) if n else 0.0,
            "截断": how.get("truncated", 0),
            "未经验证的数值断言": len(badnum), "未经验证的数值断言样例": badnum[:5],
            "根因关键词断言": len(root), "根因关键词断言样例": root[:5],
            "说明": ("均为**合规指标**（输出形态），不是事实正确率；"
                     "数值/根因用关键词启发式，会漏会误，未做人工核对。")}


def load_rows(dev18, val90):
    dev = [json.loads(l) for l in io.open(dev18, encoding="utf-8") if l.strip()]
    val = [json.loads(l) for l in io.open(val90, encoding="utf-8") if l.strip()]
    rows = [{"sample_id": r["sample_id"], "label": r["label"],
             "image_path": r["image_path"], "part": "dev18"} for r in dev]
    rows += [{"sample_id": r["sample_id"], "label": r["label"],
              "image_path": r["image_path"], "part": "val90"} for r in val]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="形如 d0=<dir>；也接受 d1=<base>::<adapter>")
    ap.add_argument("--dev18", default=R + "/datasets/dev_18.server.jsonl")
    ap.add_argument("--val90", default=R + "/datasets/val90/val90.server.jsonl")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model-type", default="qwen3_5")
    ap.add_argument("--template-type", default="qwen3_5")
    a = ap.parse_args()

    import torch
    print("=" * 78)
    print(f"D0/D1 同题描述评测  {datetime.datetime.now().isoformat()}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')!r}（父进程设置）")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("!! 父进程未设置 —— 拒绝运行"); return 3
    torch.cuda.init()
    n = torch.cuda.device_count()
    print(f"可见 GPU 数 = {n}  device 0 = {torch.cuda.get_device_name(0)}")
    if n != 1:
        print("!! 不是 1 张卡 —— 拒绝运行"); return 3
    import torch.distributed as dist
    if dist.is_initialized() and dist.get_world_size() != 1:
        print("!! world_size != 1 —— 拒绝运行"); return 3
    os.makedirs(a.outdir, exist_ok=True)

    qsha = hashlib.sha256(QUESTION.encode()).hexdigest()
    print(f"题面 {len(QUESTION)} 字符  sha256 {qsha}")
    if qsha != QUESTION_SHA:
        print(f"!! 题面与 描述评测协议.json 记录不一致（协议为 {QUESTION_SHA}）—— 拒绝运行")
        return 4
    print("   题面与描述评测协议.json 记录一致")
    rows = load_rows(a.dev18, a.val90)
    print(f"评测集 dev18 18 + val90 84 = {len(rows)} 张")

    from swift import InferRequest, RequestConfig
    from swift.infer_engine import TransformersEngine

    summary = {}
    for spec in a.models:
        tag, _, path = spec.partition("=")
        base, _, adapter = path.partition("::")
        print(f"\n--- {tag}（{'adapter=' + adapter if adapter else '基座'}）---", flush=True)
        t0 = time.time()
        eng = TransformersEngine(base, model_type=a.model_type,
                                 template_type=a.template_type,
                                 adapters=[adapter] if adapter else None)
        load_s = time.time() - t0
        cfg = RequestConfig(max_tokens=256, temperature=0.0, seed=3407)
        recs = []
        for i, r in enumerate(rows, 1):
            t1 = time.time()
            resp = eng.infer([InferRequest(
                messages=[{"role": "user", "content": "<image>" + QUESTION}],
                images=[r["image_path"]])], cfg)[0]
            raw = resp.choices[0].message.content
            obj, how = parse7(raw)
            recs.append({"sample_id": r["sample_id"], "part": r["part"],
                         "label": r["label"], "raw": raw, "obj": obj, "how": how,
                         "pred": (obj or {}).get("defect_class")
                         if isinstance((obj or {}).get("defect_class"), str) else None,
                         "gen_seconds": round(time.time() - t1, 2)})
            if i % 20 == 0 or i == len(rows):
                print(f"  [{i}/{len(rows)}]", flush=True)
        del eng
        import gc
        gc.collect(); torch.cuda.empty_cache()

        with io.open(f"{a.outdir}/{tag}_raw.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sm = summarise(recs)
        sm["模型加载秒"] = round(load_s, 1)
        sm["生成总秒"] = round(sum(r["gen_seconds"] for r in recs), 1)
        sm["分类辅助"] = metrics([(r["label"], r["pred"]) for r in recs])
        sm["分类辅助_dev18"] = metrics([(r["label"], r["pred"]) for r in recs if r["part"] == "dev18"])
        sm["分类辅助_val90"] = metrics([(r["label"], r["pred"]) for r in recs if r["part"] == "val90"])
        summary[tag] = sm
        print(f"  七字段合法率 {sm['七字段合法率']:.4f}  截断 {sm['截断']}  "
              f"空Caption {sm['空Caption']}  分类 Acc {sm['分类辅助']['accuracy']:.4f} "
              f"MacroF1 {sm['分类辅助']['macro_f1']:.4f}")

    # dev18 全 18 张 D0/D1 配对（不挑图）
    out = {"题面": QUESTION, "题面_sha256": hashlib.sha256(QUESTION.encode()).hexdigest(),
           "评测集": f"dev18 {sum(1 for r in rows if r['part']=='dev18')} + "
                     f"val90 {sum(1 for r in rows if r['part']=='val90')} = {len(rows)}",
           "生成": {"max_tokens": 256, "temperature": 0.0, "seed": 3407,
                    "thinking": "模板 enable_thinking 默认 False（已核）"},
           "逐模型": summary,
           "口径": ["合规指标 ≠ 事实正确率。",
                    "有独立 gold 的字段才自动判事实；无依据字段列**未测**。",
                    "extent_r=null 不代表尺寸能力已验证。",
                    "分类为辅助指标，题面已变，**不与旧类别-only 成绩混写**。"],
           "未测": ["形态/位置/方向的事实一致性（无独立 gold，需人工或独立依据）",
                    "尺寸/覆盖率", "跨样本泛化"]}
    with io.open(f"{a.outdir}/eval_summary.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"\n写出 {a.outdir}/eval_summary.json 与各模型 *_raw.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
