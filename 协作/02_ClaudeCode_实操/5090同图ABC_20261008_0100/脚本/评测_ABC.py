#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A / B / C 信息对照评测 —— 同一题面、同一解码，只比训练信息量。

三组训练数据的**题面完全相同**（sha256 已核），区别只在监督目标：
  A 只类别 · B 加几何 · C 加描述。
所以这里的评测就是**同一个问题下比类别准确率**，附加字段另算。

解析按 `题面与解析协议.json`：只从**唯一**的 `defect_class` 取类别，
**允许附加字段**——不能复用「对象只能有一个字段」的旧严格解析器，那会把 C 全判错。

生成：temperature 0 / max_tokens 256 / seed 3407 / thinking 关闭。
每条推理完成即写 JSONL 并 flush；不重试、不扩额。
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
    "请判断图中**主要缺陷图案**的类别，从 Center、Donut、Edge_Loc、Edge_Ring、Loc、"
    "Near_full、Random、Scratch、none 中选一个。\n"
    "只输出一个 JSON 对象，不写推理过程或 Markdown。\n"
    "`defect_class` 为必填，取值为上述九类之一；无法可靠判断时填 \"unknown\"。\n"
    "可以在同一个对象里附加你**从图中实际看到**的判读依据字段；没有把握就不要写。"
)
QUESTION_SHA = "6f710f5a63ed6f646ea5a16b9d7e1111392b671c2859c8a571c19d6e98c4ae6b"

CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
ENUM_CLASS = CLASSES + ["unknown"]
ZONES = ["center", "middle", "edge", "global", "none", "unknown"]

EMPTY_THINK = re.compile(r"^\s*<think\b[^>]*>\s*</think\s*>\s*", re.S)


class DupKey(ValueError):
    pass


class NonFinite(ValueError):
    """JSON 标准不允许 NaN/Infinity/-Infinity；单独一类，避免被 JSONDecodeError 吞掉。"""
    pass


def _nodup(pairs):
    k = [p[0] for p in pairs]
    if len(k) != len(set(k)):
        raise DupKey("duplicate_key")
    return dict(pairs)


def _reject_const(x):
    raise NonFinite(x)


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
        obj, end = json.JSONDecoder(object_pairs_hook=_nodup,
                                    parse_constant=_reject_const).raw_decode(s)
    except DupKey:
        return None, "duplicate_key"
    except NonFinite:
        return None, "non_finite"
    except json.JSONDecodeError:
        if s.startswith("{") and not s.rstrip().endswith("}"):
            return None, "truncated"
        if s.count("{") > 1:
            return None, "multiple_objects"
        return None, "not_json"
    rest = s[end:].strip()
    if rest:
        # 尾随内容里还有对象 → 判多对象，否则才是普通尾随文本
        return None, ("multiple_objects" if "{" in rest else "trailing_text")
    if not isinstance(obj, dict):
        return None, "not_object"
    return obj, "ok"


def class_of(obj):
    """只取唯一 defect_class；合法则返回类别，否则 None。"""
    if not isinstance(obj, dict):
        return None, "无对象"
    v = obj.get("defect_class")
    if not isinstance(v, str):
        return None, "defect_class非字符串"
    if v not in ENUM_CLASS:
        return None, f"越界:{v[:20]}"
    return v, ("ok_unknown" if v == "unknown" else "ok")


def geometry_ok(obj):
    """B 组口径：附加几何是否成形（radial_zone 合法枚举 + defect_ratio 是 0-1 数）。"""
    if not isinstance(obj, dict):
        return False
    g = obj.get("geometry") if isinstance(obj.get("geometry"), dict) else obj
    rz, dr = g.get("radial_zone"), g.get("defect_ratio")
    return (isinstance(rz, str) and rz in ZONES
            and isinstance(dr, (int, float)) and not isinstance(dr, bool)
            and 0.0 <= dr <= 1.0)


def caption_ok(obj):
    """C 组口径：附加描述存在且非空。"""
    if not isinstance(obj, dict):
        return False
    c = obj.get("caption_zh") or obj.get("morphology")
    return isinstance(c, str) and bool(c.strip())


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


FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n?\s*```\s*$", re.S)


def strip_fence(text):
    """离线格式诊断用：剥离完整外层 Markdown 围栏后重新解析。主分不使用。"""
    if not isinstance(text, str):
        return None
    m = FENCE.match(text.strip())
    if not m:
        return None
    return parse_json(m.group(1))


def load_rows(evalset):
    rows = [json.loads(l) for l in io.open(evalset, encoding="utf-8") if l.strip()]
    return [{"sample_id": r["sample_id"], "label": r["label"],
             "image_path": r.get("image_path") or r.get("images", [None])[0],
             "part": r.get("part", "unknown")} for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="形如 base=<dir> · a=<base>::<adapter> · c=<base>::<adapter>")
    ap.add_argument("--evalset", default=R + "/datasets/eval_102.server.jsonl")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model-type", default="qwen3_5")
    ap.add_argument("--template-type", default="qwen3_5")
    a = ap.parse_args()

    import torch
    print("=" * 78)
    print(f"A/B/C 信息对照评测  {datetime.datetime.now().isoformat()}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')!r}（父进程设置）")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("!! 父进程未设置 —— 拒绝运行"); return 3
    torch.cuda.init()
    n = torch.cuda.device_count()
    print(f"可见 GPU 数 = {n}  device 0 = {torch.cuda.get_device_name(0)}")
    if n != 1:
        print("!! 不是 1 张卡 —— 拒绝运行"); return 3
    os.makedirs(a.outdir, exist_ok=True)

    qsha = hashlib.sha256(QUESTION.encode()).hexdigest()
    print(f"题面 {len(QUESTION)} 字符  sha256 {qsha}")
    if qsha != QUESTION_SHA:
        print(f"!! 题面与 题面与解析协议.json 记录不一致（协议为 {QUESTION_SHA}）—— 拒绝运行")
        return 4
    print("   与 题面与解析协议.json 记录一致")
    rows = load_rows(a.evalset)
    print(f"评测集 {len(rows)} 张（dev18 {sum(1 for r in rows if r['part']=='dev18')} + "
      f"val84 {sum(1 for r in rows if r['part']=='val90')}）")

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
        fh = io.open(f"{a.outdir}/{tag}_raw.jsonl", "w", encoding="utf-8", newline="\n")
        recs = []
        try:
            for i, r in enumerate(rows, 1):
                t1 = time.time()
                resp = eng.infer([InferRequest(
                    messages=[{"role": "user", "content": "<image>" + QUESTION}],
                    images=[r["image_path"]])], cfg)[0]
                ch = resp.choices[0]
                raw = ch.message.content
                obj, how = parse_json(raw)
                pred, code = (class_of(obj) if how == "ok" else (None, "JSON未解析"))
                rec = {"sample_id": r["sample_id"], "part": r["part"], "label": r["label"],
                       "raw": raw, "obj": obj, "how": how, "pred_code": code,
                       "pred": pred, "extra_fields": sorted(
                           k for k in (obj or {}) if k != "defect_class"),
                       "geometry_ok": geometry_ok(obj), "caption_ok": caption_ok(obj),
                       "fence_stripped": (lambda t: (lambda r: (r[0] is not None, r[1]))(strip_fence(t)) if strip_fence(t) else (False, "no_fence"))(raw),
                       "finish_reason": getattr(ch, "finish_reason", None),
                       "gen_seconds": round(time.time() - t1, 2)}
                recs.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 20 == 0 or i == len(rows):
                    print(f"  [{i}/{len(rows)}]", flush=True)
        finally:
            fh.close()
        del eng
        import gc
        gc.collect(); torch.cuda.empty_cache()

        how = Counter(r["how"] for r in recs)
        codes = Counter(r["pred_code"] for r in recs)
        sm = {"n": len(recs),
              "JSON可解析率": round(how.get("ok", 0) / len(recs), 4),
              "JSON解析判定": dict(how),
              # 主结果：同一个问题下的类别准确率
              "类别Acc": metrics([(r["label"], r["pred"]) for r in recs]),
              "类别Acc_dev18": metrics([(r["label"], r["pred"]) for r in recs if r["part"] == "dev18"]),
              "类别Acc_val90": metrics([(r["label"], r["pred"]) for r in recs if r["part"] == "val90"]),
              "类别取值判定": dict(codes),
              "answered_unknown": sum(1 for r in recs if r["pred"] == "unknown"),
              # 附加字段（B/C 组的实际产出）
              "含附加字段条数": sum(1 for r in recs if r["extra_fields"]),
              "附加字段种类": dict(Counter(f for r in recs for f in r["extra_fields"])),
              "geometry成形率": round(sum(1 for r in recs if r["geometry_ok"]) / len(recs), 4),
              "caption非空率": round(sum(1 for r in recs if r["caption_ok"]) / len(recs), 4),
              "截断": how.get("truncated", 0),
              "finish_reason": dict(Counter(r.get("finish_reason") for r in recs)),
              "模型加载秒": round(load_s, 1),
              "生成总秒": round(sum(r["gen_seconds"] for r in recs), 1)}
        summary[tag] = sm
        print(f"  类别Acc {sm['类别Acc']['accuracy']:.4f}  MacroF1 {sm['类别Acc']['macro_f1']:.4f}"
              f"  可解析 {sm['JSON可解析率']:.4f}  截断 {sm['截断']}"
              f"  附加字段 {sm['含附加字段条数']}", flush=True)

    out = {"题面": QUESTION, "题面_sha256": qsha,
           "说明": "三组训练数据题面完全相同，只比训练信息量（A 只类别 / B 加几何 / C 加描述）。",
           "评测集": f"{len(rows)} 张（dev18 + val84）",
           "生成": {"max_tokens": 256, "temperature": 0.0, "seed": 3407, "重试": "无"},
           "逐模型": summary,
           "口径": ["主结果是**同一个问题下的类别准确率**；附加字段是设计的一部分，不是合规指标。",
                    "分类成绩来自**新题面**，**不与旧类别-only 结果混写**。",
                    "解析只取唯一 defect_class，允许附加字段。",
                    "合规 ≠ 事实正确；几何/描述字段的事实一致性无独立 gold，列未测。"]}
    with io.open(f"{a.outdir}/abc_eval_summary.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"\n写出 {a.outdir}/abc_eval_summary.json 与各模型 *_raw.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
