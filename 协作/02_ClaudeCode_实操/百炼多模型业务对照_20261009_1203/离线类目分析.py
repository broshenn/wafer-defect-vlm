#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""百炼对照的**离线类目分析**（请求端不读 gold，本脚本才读）。

口径：
  · **三层解析分开**：严格 JSON / 完整 schema / **剥完整外层围栏**诊断；
    Codex 要求「围栏诊断另列，**对所有候选统一实施**」；
  · 类别真值取**公开 `failure_type`**（`样本清单.json` 的 `公开类别`），
    **不是模型标注**；
  · 指标：Accuracy、**固定九类 Macro-F1**、逐类 P/R/F1、混淆矩阵；
  · **不把某个模型的最优数拼成单个检查点**；每模型独立成行。
  · 未解析的**按类别错误计**（与前面几轮一致），同时单列可解析率。

用法：python 离线类目分析.py
"""
from __future__ import annotations
import io, json, re
from collections import Counter, defaultdict
from pathlib import Path
import importlib.util

D = Path(__file__).resolve().parent
REPO = D.parent.parent.parent
BLIND = REPO / "协作/01_Codex_指挥/百炼多模型对照_20261009/blind36.jsonl"
CHECKER = REPO / "协作/01_Codex_指挥/WorkBuddy外部102_执行补包_20261008/schema_check.py"
LABELS = REPO / "协作/02_ClaudeCode_实操/部署原型_20261009_0120/样本清单.json"
LOG = D / "请求日志.jsonl"
OUT = D / "类目结果.json"

CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
ENUM = CLASSES + ["unknown"]
EMPTY_THINK = re.compile(r"^\s*<think\b[^>]*>\s*</think\s*>\s*", re.S)
FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n?\s*```\s*$", re.S)


class NonFinite(ValueError):
    pass


def parse_json(text):
    if not isinstance(text, str):
        return None, "not_str"
    s = EMPTY_THINK.sub("", text.strip())
    if not s:
        return None, "empty"

    def _nodup(pairs):
        k = [p[0] for p in pairs]
        if len(k) != len(set(k)):
            raise ValueError("dup")
        return dict(pairs)

    def _const(x):
        raise NonFinite(x)
    try:
        obj, end = json.JSONDecoder(object_pairs_hook=_nodup,
                                    parse_constant=_const).raw_decode(s)
    except NonFinite:
        return None, "non_finite"
    except json.JSONDecodeError:
        if s.startswith("{") and not s.rstrip().endswith("}"):
            return None, "truncated"
        if s.count("{") > 1:
            return None, "multiple_objects"
        return None, "not_json"
    except ValueError:
        return None, "duplicate_key"
    rest = s[end:].strip()
    if rest:
        return None, ("multiple_objects" if "{" in rest else "trailing_text")
    return (obj, "ok") if isinstance(obj, dict) else (None, "not_object")


def fence_parse(text):
    """**统一诊断口径**：剥完整外层围栏后重解析；剥不掉就退回原严格解析。"""
    if not isinstance(text, str):
        return None, "not_str"
    m = FENCE.match(EMPTY_THINK.sub("", text.strip()))
    if m:
        return parse_json(m.group(1))
    return parse_json(text)


def pred_of(obj):
    if not isinstance(obj, dict):
        return None
    v = obj.get("defect_class")
    return v if isinstance(v, str) and v in ENUM else None


def prf(pairs):
    out, f1s = {}, []
    for c in CLASSES:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        out[c] = {"P": round(pr, 4), "R": round(rc, 4), "F1": round(f, 4),
                  "n": sum(1 for g, _ in pairs if g == c)}
        f1s.append(f)
    return sum(f1s) / len(f1s), out


def main():
    spec = importlib.util.spec_from_file_location("ck", CHECKER)
    ck = importlib.util.module_from_spec(spec); spec.loader.exec_module(ck)
    blind = {json.loads(l)["sample_id"]: json.loads(l)
             for l in io.open(BLIND, encoding="utf-8") if l.strip()}
    labels = {s["sample_id"]: s["公开类别"]
              for s in json.loads(LABELS.read_text(encoding="utf-8"))["样本"]}

    rows = [json.loads(l) for l in io.open(LOG, encoding="utf-8") if l.strip()]
    by_model = defaultdict(list)
    for r in rows:
        # 百炼 API 的回答在 content 字段
        by_model[r["model"]].append({"sample_id": r["sample_id"], "content": r.get("content")})
    # 本地 GPU 跑的两个（同一台租机、同一题面、同一解码参数）
    # D 的现场 HTTP 36（**复用**，不重跑）
    dep = REPO / "协作/02_ClaudeCode_实操/部署原型_20261009_0120/性能与自测/http_replay.jsonl"
    if dep.exists():
        for l in io.open(dep, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                by_model["D-N3072-3407(部署HTTP)"].append(
                    {"sample_id": d["sample_id"],
                     "content": ((d.get("response") or {}).get("raw_answer"))})
    # GLM 外部原答（从已冻结的 102 里按 sample_id 取这 36 条）
    ext = REPO / "协作/04_WorkBuddy_复核/外部评测102_20261008_201229"
    if (ext / "原答清单.jsonl").exists():
        for l in io.open(ext / "原答清单.jsonl", encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                f = ext / "raw" / (d["item_id"] + ".txt")
                if f.exists() and d["sample_id"] in blind:
                    by_model["GLM(WorkBuddy客户端)"].append(
                        {"sample_id": d["sample_id"], "content": f.read_text(encoding="utf-8")})
    for tag, fn in (("Base(9B原生,GPU)", "Base_raw.jsonl"),
                    ("L-N3072-3407(GPU)", "L-N3072-3407_raw.jsonl")):
        gp = D / "原答_GPU" / fn
        if gp.exists():
            for l in io.open(gp, encoding="utf-8"):
                if l.strip():
                    d = json.loads(l)
                    by_model[tag].append({"sample_id": d["sample_id"], "content": d.get("raw")})

    res = {}
    print(f"{'模型':<22}{'条数':>5}{'严格ok':>8}{'schema':>8}{'围栏诊断':>10}"
          f"{'Acc':>9}{'MacroF1':>10}")
    for m in sorted(by_model):
        recs = by_model[m]
        strict, fence, schema_ok = 0, 0, 0
        pairs_s, pairs_f = [], []
        how_c = Counter()
        for r in recs:
            sid = r["sample_id"]; gold = labels.get(sid)
            if gold is None:
                continue
            raw = r.get("content") or ""
            o, how = parse_json(raw); how_c[how] += 1
            p = pred_of(o) if how == "ok" else None
            if how == "ok":
                strict += 1
                sok, _ = ck.schema7(o)
                if sok:
                    schema_ok += 1
            pairs_s.append((gold, p or "unknown"))
            fo, fh = fence_parse(raw)
            fp_ = pred_of(fo) if fh == "ok" else None
            if fp_ is not None:
                fence += 1
            pairs_f.append((gold, fp_ or "unknown"))
        mf_s, per_s = prf(pairs_s)
        mf_f, _ = prf(pairs_f)
        acc_s = sum(1 for g, p in pairs_s if g == p) / len(pairs_s)
        acc_f = sum(1 for g, p in pairs_f if g == p) / len(pairs_f)
        res[m] = {
            "条数": len(recs), "n_有原答": len(pairs_s),
            "严格JSON可解析": strict, "完整schema": schema_ok,
            "围栏诊断可解析": fence,
            "解析判定分布": dict(how_c),
            "严格_Accuracy": round(acc_s, 4), "严格_MacroF1": round(mf_s, 4),
            "围栏诊断_Accuracy": round(acc_f, 4), "围栏诊断_MacroF1": round(mf_f, 4),
            "逐类_严格": per_s,
            "乱用率需复核提示_见请求日志": None,
        }
        print(f"{m:<22}{len(recs):>5}{strict:>8}{schema_ok:>8}{fence:>10}"
              f"{acc_s:>9.4f}{mf_s:>10.4f}")

    if not res:
        print("（日志里还没有可分析的记录）")
        return 4
    OUT.write_text(json.dumps({
        "口径": {
            "类别真值": "公开 manifest 的 failure_type（ground_truth），非模型标注",
            "未解析": "按类别错误计（同时单列可解析率）",
            "三层解析": "严格 JSON / 完整 schema / 剥完整外层围栏诊断（对全部候选统一实施）",
            "禁止": "不把某模型最优数拼成单个检查点；每模型独立成行",
        },
        "逐模型": res}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n写出 {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
