#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 首轮结果分析（CPU，可复跑）。

输入：
  out_v3/服务器产物/out/评测_v3/{M0_v3,M1_v3,M2_v3}_raw.jsonl
  out_v3/服务器产物/out/M2_v3/reward_audit.jsonl
  协作/01_Codex_指挥/无卡CPU整备_20261010/data_v3/confirmation120_reference.jsonl
输出：
  out_v3/v3_指标.json / v3_指标.md

口径（与 v3 合同一致）：
  · 主指标 = 公开类别 Macro-F1，分母固定为确认池**实际覆盖类**；
  · 次指标 = 规则奖励（事实 .9 + 格式 .1），**不替换主指标**；
  · 方向拆三分母：正向(single/opposed) / 非方向(nondirectional) / unknown(不计分)；
  · 格式两口径分列：**原始全文严格 JSON** 与 **剥完整空 think 包装后**；
  · 不做"最好 checkpoint"挑选，末步 checkpoint 固定。
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODEX = Path("D:/pycode/晶圆图研究/协作/01_Codex_指挥/无卡CPU整备_20261010")
SRV = HERE / "out_v3" / "服务器产物" / "out"
sys.path.insert(0, str(HERE))
import core_v3_复核修复 as R  # noqa: E402

MODELS = ["M0_v3", "M1_v3", "M2_v3"]
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_full", "Random",
           "Scratch", "none"]


def _ok(fn, text):
    try:
        fn(text); return 1
    except Exception:
        return 0


def jl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def boot(diffs, n=10000, seed=3407):
    rng = random.Random(seed)
    k = len(diffs)
    m = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)], sum(diffs) / k


def macro_f1(ids, pred, gold, cls):
    tp = Counter(); fp = Counter(); fn = Counter()
    for s in ids:
        g, p = gold[s], pred.get(s)
        if p == g and g in cls:
            tp[g] += 1
        else:
            if p in cls:
                fp[p] += 1
            if g in cls:
                fn[g] += 1
    f1s = []
    for c in cls:
        pr = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1s) / len(cls)


refs = {r["sample_id"]: r["reference"] for r in
        jl(CODEX / "data_v3" / "confirmation120_reference.jsonl")}
gold = {s: r["defect_class"] for s, r in refs.items()}
ids = sorted(refs)
present = [c for c in CLASSES if any(gold[s] == c for s in ids)]
out = {"确认池": {"n": len(ids), "实际覆盖类": present,
               "每类条数": {c: sum(1 for s in ids if gold[s] == c) for c in present}},
       "主指标": "公开类别 Macro-F1（分母固定为实际覆盖类）",
       "逐模型": {}, "配对差": {}}

per = {}
for m in MODELS:
    path = SRV / "评测_v3" / f"{m}_raw.jsonl"
    if not path.exists():
        print("缺", path); continue
    rows = jl(path)
    pred, raw_ok, norm_ok, rewards, parts = {}, 0, 0, {}, {
        "class": 0, "coverage": 0, "zones": 0, "angular": 0}
    ang = {"positive": [0, 0], "nondirectional": [0, 0], "unknown": 0}
    for r in rows:
        sid = r["sample_id"]
        raw = r["raw"]
        pred[sid] = (R.parse(raw).get("defect_class") if _ok(R.parse, raw) else None)
        raw_ok += _ok(R.parse_raw_strict, raw)
        norm_ok += _ok(R.parse, raw)
        sc = R.score(raw, refs[sid])
        rewards[sid] = sc["reward"]
        for k in parts:
            parts[k] += sc["parts"][k]
        a = refs[sid]["angular_type"]
        if a == "unknown":
            ang["unknown"] += 1
        else:
            key = "positive" if a in ("single", "opposed") else "nondirectional"
            ang[key][1] += 1
            ang[key][0] += 1 if sc["parts"]["angular"] == 1.0 else 0
    n = len(rows)
    per[m] = {"pred": pred, "reward": rewards}
    out["逐模型"][m] = {
        "n": n,
        "类别正确": sum(1 for s in ids if pred.get(s) == gold[s]),
        "类别Accuracy": round(sum(1 for s in ids if pred.get(s) == gold[s]) / n, 4),
        "主指标MacroF1": round(macro_f1(ids, pred, gold, present), 4),
        "次指标规则奖励": round(sum(rewards.values()) / n, 4),
        "分项正确率": {k: round(parts[k] / n, 4) for k in parts},
        "格式": {"原始全文严格JSON": f"{raw_ok}/{n}",
               "剥完整空think后": f"{norm_ok}/{n}"},
        "方向分母": {"正向": f"{ang['positive'][0]}/{ang['positive'][1]}",
                  "非方向": f"{ang['nondirectional'][0]}/{ang['nondirectional'][1]}",
                  "unknown不计分": ang["unknown"]},
    }

for a, b in (("M1_v3", "M0_v3"), ("M2_v3", "M0_v3"), ("M2_v3", "M1_v3")):
    if a not in per or b not in per:
        continue
    d = [per[a]["reward"][s] - per[b]["reward"][s] for s in ids]
    dc = [(1 if per[a]["pred"].get(s) == gold[s] else 0) -
          (1 if per[b]["pred"].get(s) == gold[s] else 0) for s in ids]
    lo, hi, mean = boot(d)
    lo2, hi2, mean2 = boot(dc)
    out["配对差"][f"{a}-{b}"] = {
        "次指标差": round(mean, 4), "次指标95区间": [round(lo, 4), round(hi, 4)],
        "次指标跨0": bool(lo <= 0 <= hi),
        "类别正确数差": round(mean2, 4), "类别差95区间": [round(lo2, 4), round(hi2, 4)],
        "类别差跨0": bool(lo2 <= 0 <= hi2)}

# M2 训练组信号
aud = SRV / "M2_v3" / "reward_audit.jsonl"
if aud.exists():
    rows = jl(aud)
    by = Counter(r["sample_id"] for r in rows)
    out["M2训练组"] = {"审计条数": len(rows), "组数": len(by),
                     "每组候选数集合": sorted(set(by.values())),
                     "平均奖励": round(sum(r["reward"] for r in rows) / len(rows), 4),
                     "训练时候选带空前缀比例":
                         round(sum(1 for r in rows if r.get("had_empty_think_prefix"))
                               / len(rows), 4),
                     "训练时原始全文严格JSON可解析":
                         round(sum(1 for r in rows if r.get("raw_parse_ok")) / len(rows), 4)}

(HERE / "out_v3" / "v3_指标.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

L = ["# v3 首轮指标（确认池 120 图）", "",
     f"确认池实际覆盖 {len(present)} 类：{present}", "",
     "| 模型 | 类别Acc | **主指标 Macro-F1** | 次指标规则奖励 | 覆盖 | 区带 | 角向 |",
     "|---|---:|---:|---:|---:|---:|---:|"]
for m in MODELS:
    if m not in out["逐模型"]:
        continue
    a = out["逐模型"][m]
    L.append(f"| {m} | {a['类别Accuracy']:.4f}（{a['类别正确']}/{a['n']}） | "
             f"**{a['主指标MacroF1']:.4f}** | {a['次指标规则奖励']:.4f} | "
             f"{a['分项正确率']['coverage']:.3f} | {a['分项正确率']['zones']:.3f} | "
             f"{a['分项正确率']['angular']:.3f} |")
L += ["", "## 方向三分母", "", "| 模型 | 正向 single/opposed | 非方向 nondirectional | unknown（不计分） |",
      "|---|---|---|---|"]
for m in MODELS:
    if m not in out["逐模型"]:
        continue
    d = out["逐模型"][m]["方向分母"]
    L.append(f"| {m} | {d['正向']} | {d['非方向']} | {d['unknown不计分']} |")
L += ["", "## 格式两口径", "", "| 模型 | 原始全文严格 JSON | 剥完整空 think 后 |", "|---|---|---|"]
for m in MODELS:
    if m not in out["逐模型"]:
        continue
    f = out["逐模型"][m]["格式"]
    L.append(f"| {m} | {f['原始全文严格JSON']} | {f['剥完整空think后']} |")
L += ["", "## 配对差（同图 bootstrap 10000，seed 3407）", "",
      "| 对比 | 次指标差 | 95% 区间 | 跨0 | 类别正确数差 | 95% 区间 | 跨0 |",
      "|---|---:|---|---|---:|---|---|"]
for k, v in out["配对差"].items():
    L.append(f"| {k} | {v['次指标差']:+.4f} | "
             f"[{v['次指标95区间'][0]:+.4f}, {v['次指标95区间'][1]:+.4f}] | "
             f"{'是' if v['次指标跨0'] else '否'} | {v['类别正确数差']:+.4f} | "
             f"[{v['类别差95区间'][0]:+.4f}, {v['类别差95区间'][1]:+.4f}] | "
             f"{'是' if v['类别差跨0'] else '否'} |")
if "M2训练组" in out:
    t = out["M2训练组"]
    L += ["", "## M2 训练组审计", "",
          f"审计 {t['审计条数']} 条 / {t['组数']} 组，每组候选数 {t['每组候选数集合']}，"
          f"平均奖励 {t['平均奖励']}；训练时带空前缀比例 {t['训练时候选带空前缀比例']}，"
          f"原始全文严格 JSON 可解析 {t['训练时原始全文严格JSON可解析']}"]
(HERE / "out_v3" / "v3_指标.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))

