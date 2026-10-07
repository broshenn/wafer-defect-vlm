#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D0 / D1 同题描述评测 —— 修正版 v2（Codex 2026-10-07 §3）。

相对 v1 的改动：
  1. **JSON 可解析判定 与 七字段合法判定 分开计**，绝不混用一个通过率。
     v1 里 `{}`、仅 defect_class 的对象、七字段全是非法类型的对象都会返回 how='ok'
     并被计成「七字段合法率 1.0」——这是错的。
  2. 新增 `schema7()`：七键集合、值类型、类别/区域枚举、extent_r 必须是**真 JSON null**、
     方向为 string 或 null、morphology/caption/uncertainty 为 string；
     字符串 `"null"` 不算 JSON null，bool 不算类别。空 Caption 另计。
  3. **每条推理完成即写 JSONL 并 flush**，保留原答、解析结果、schema 结果、finish_reason；
     中断时已完成答案不丢。
  4. 分类辅助从「可解析且类别有效」的 JSON 抽取，**明确不要求完整七字段**，
     与七字段合法率分开命名。

生成上限仍 256，不重试、不扩额。合规 ≠ 事实正确。
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
ENUM_CLASS = CLASSES + ["unknown"]
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


def parse_json(text):
    """只判「是不是一个合法 JSON 对象」。返回 (obj 或 None, 判定码)。"""
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


def schema7(obj):
    """七字段 schema 判定。返回 (ok, 问题列表)。只在 parse_json 成功时调用。"""
    p = []
    if obj is None:
        return False, ["未解析"]
    missing = [f for f in FIELDS if f not in obj]
    if missing:
        p.append("缺字段:" + ",".join(missing))
    extra = [k for k in obj if k not in FIELDS]
    if extra:
        p.append("多字段:" + ",".join(sorted(extra)))
    if "defect_class" in obj:
        v = obj["defect_class"]
        if not isinstance(v, str):
            p.append("defect_class非字符串")
        elif v not in ENUM_CLASS:
            p.append(f"defect_class越界:{v[:20]}")
    if "radial_zone" in obj:
        v = obj["radial_zone"]
        if not isinstance(v, str):
            p.append("radial_zone非字符串")
        elif v not in ZONES:
            p.append(f"radial_zone越界:{v[:20]}")
    if "clock_direction" in obj:
        v = obj["clock_direction"]
        if not (v is None or isinstance(v, str)):
            p.append("clock_direction类型错")
    if "extent_r" in obj:
        v = obj["extent_r"]
        if v is not None:                       # 字符串 "null" 不算 JSON null
            p.append(f"extent_r非null:{str(v)[:12]}")
    for f in ("morphology", "caption_zh", "uncertainty"):
        if f in obj and not isinstance(obj[f], str):
            p.append(f"{f}非字符串")
    return (not p), p


def metrics(pairs):
    n = len(pairs)
    ok = sum(1 for g, x in pairs if g == x)
    f1s = []
    for c in CLASSES:
        tp = sum(1 for g, x in pairs if g == c and x == c)
        fp = sum(1 for g, x in pairs if g != c and x == c)
        fn = sum(1 for g, x in pairs if g == c and x != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"n": n, "correct": ok, "accuracy": round(ok / n, 4) if n else 0.0,
            "macro_f1": round(sum(f1s) / len(f1s), 4),
            "per_class_f1": {c: round(v, 4) for c, v in zip(CLASSES, f1s)}}


def summarise(recs):
    n = len(recs)
    how = Counter(r["how"] for r in recs)
    parsed = how.get("ok", 0)
    sch = [r for r in recs if r["schema_ok"]]
    miss = Counter()
    reasons = Counter()
    for r in recs:
        for x in r["schema_problems"]:
            reasons[x.split(":")[0]] += 1
        if r["obj"]:
            for f in FIELDS:
                if f not in r["obj"]:
                    miss[f] += 1
    badnum = [r["sample_id"] for r in recs
              if (r["obj"] or {}).get("extent_r") is not None
              and (r["obj"] or {}).get("extent_r") != "__absent__"
              or NUMRE.search(str((r["obj"] or {}).get("morphology") or "")
                              + str((r["obj"] or {}).get("caption_zh") or ""))]
    root = [r["sample_id"] for r in recs
            if ROOTRE.search(str((r["obj"] or {}).get("morphology") or "")
                             + str((r["obj"] or {}).get("caption_zh") or ""))]
    emptycap = [r["sample_id"] for r in recs
                if not str((r["obj"] or {}).get("caption_zh") or "").strip()]
    return {
        "n": n,
        "JSON可解析率": round(parsed / n, 4) if n else 0.0,
        "JSON解析判定": dict(how),
        "七字段合法率": round(len(sch) / n, 4) if n else 0.0,
        "七字段合法数": len(sch),
        "schema问题分类": dict(reasons),
        "字段缺失计数": dict(miss),
        "空Caption": len(emptycap), "空Caption比例": round(len(emptycap) / n, 4) if n else 0.0,
        "截断": how.get("truncated", 0),
        "finish_reason": dict(Counter(r.get("finish_reason") for r in recs)),
        "未经验证的数值断言": len(badnum), "样例": badnum[:5],
        "根因关键词断言": len(root), "根因样例": root[:5],
        "说明": ("**JSON可解析率**与**七字段合法率是两个不同的数，不可互相替代、不可混名。**"
                 "两者都是**合规指标**（输出形态），不等于事实正确率；"
                 "数值/根因用关键词启发式，会漏会误，未做人工核对。"),
    }


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
    print(f"D0/D1 同题描述评测（修正版 v2）  {datetime.datetime.now().isoformat()}")
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
        out_path = f"{a.outdir}/{tag}_raw.jsonl"
        fh = io.open(out_path, "w", encoding="utf-8", newline="\n")
        recs = []
        try:
            for i, r in enumerate(rows, 1):
                t1 = time.time()
                resp = eng.infer([InferRequest(
                    messages=[{"role": "user", "content": "<image>" + QUESTION}],
                    images=[r["image_path"]])], cfg)[0]
                ch = resp.choices[0]
                raw = ch.message.content
                fr = getattr(ch, "finish_reason", None)
                obj, how = parse_json(raw)
                sok, sprobs = (schema7(obj) if how == "ok" else (False, ["JSON未解析"]))
                cls = obj.get("defect_class") if isinstance(obj, dict) else None
                pred = cls if isinstance(cls, str) and cls in ENUM_CLASS else None
                rec = {"sample_id": r["sample_id"], "part": r["part"], "label": r["label"],
                       "raw": raw, "obj": obj, "how": how,
                       "schema_ok": bool(sok), "schema_problems": sprobs,
                       "class_valid": pred is not None, "pred": pred,
                       "finish_reason": fr, "gen_seconds": round(time.time() - t1, 2)}
                recs.append(rec)
                # 每条完成即落盘并 flush：中断时已完成答案不丢
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 20 == 0 or i == len(rows):
                    print(f"  [{i}/{len(rows)}]", flush=True)
        finally:
            fh.close()
        del eng
        import gc
        gc.collect(); torch.cuda.empty_cache()

        sm = summarise(recs)
        sm["模型加载秒"] = round(load_s, 1)
        sm["生成总秒"] = round(sum(r["gen_seconds"] for r in recs), 1)
        # 分类辅助：**可解析且类别有效**即可，不要求七字段齐全（与上面分开命名）
        pairs = [(r["label"], r["pred"]) for r in recs if r["class_valid"]]
        sm["分类辅助_口径"] = ("从**可解析且类别枚举有效**的 JSON 抽取，"
                               "**不要求七字段齐全**；缺答案的按答错计入")
        sm["分类辅助"] = metrics([(r["label"], r["pred"]) for r in recs])
        sm["分类辅助_dev18"] = metrics([(r["label"], r["pred"]) for r in recs if r["part"] == "dev18"])
        sm["分类辅助_val90"] = metrics([(r["label"], r["pred"]) for r in recs if r["part"] == "val90"])
        sm["可解析且类别有效条数"] = len(pairs)
        summary[tag] = sm
        print(f"  JSON可解析率 {sm['JSON可解析率']:.4f}  七字段合法率 {sm['七字段合法率']:.4f}"
              f"  截断 {sm['截断']}  空Caption {sm['空Caption']}"
              f"  分类Acc {sm['分类辅助']['accuracy']:.4f}", flush=True)

    out = {"题面": QUESTION, "题面_sha256": qsha,
           "评测集": f"dev18 18 + val90 84 = {len(rows)}",
           "生成": {"max_tokens": 256, "temperature": 0.0, "seed": 3407,
                    "thinking": "模板 enable_thinking 默认 False（已核）",
                    "重试": "无；不临时扩额"},
           "逐模型": summary,
           "口径": ["**JSON可解析率 与 七字段合法率是两个不同的数**，不可互相替代或混名。",
                    "两者都是合规指标 ≠ 事实正确率。",
                    "分类辅助从可解析且类别有效的 JSON 抽取，不要求七字段齐全。",
                    "题目已变，分类成绩**不与旧类别-only 结果混写**。",
                    "extent_r=null 不代表尺寸能力已验证。"],
           "未测": ["形态/位置/方向的事实一致性（无独立 gold，需人工或独立依据）",
                    "尺寸/覆盖率", "跨样本泛化"]}
    with io.open(f"{a.outdir}/eval_summary.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"\n写出 {a.outdir}/eval_summary.json 与各模型 *_raw.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
