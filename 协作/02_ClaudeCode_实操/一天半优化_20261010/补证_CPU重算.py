#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 补证·CPU 重算：把「我的报告」与「Codex 独立核验」对到同一个分母上。

做三件事：
  A. 同一奖励分组：从 **奖励审计**（权威来源，256 组全覆盖）重算"四候选同分"组数，
     并区分满分/未满分；与我在报告里写的「34/124 日志步」对照，说明两个分母为什么不同。
  B. 主指标 Macro-F1 的配对差：**按 lot 配对 bootstrap**（不是按图），与 Codex 的
     −0.0143 [−0.0613, +0.0300] 对照。
  C. 次指标（规则奖励）按 lot 的配对差，与 Codex 的 −0.0371 [−0.0650, −0.0093] 对照。

只读已有原答与审计；不联网、不占 GPU、不改任何原件。
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRV = HERE / "out_v3" / "服务器产物" / "out"
CODEX = Path("D:/pycode/晶圆图研究/协作/01_Codex_指挥/无卡CPU整备_20261010")
sys.path.insert(0, str(HERE))
import core_v3_复核修复 as R  # noqa: E402

CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_full", "Random",
           "Scratch", "none"]
MODELS = ["M0_v3", "M1_v3", "M2_v3"]


def jl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


rep: dict = {"输入": {}}

# ── A. 同一奖励分组：以审计为准 ────────────────────────────────────────
aud = SRV / "M2_v3" / "reward_audit.jsonl"
rep["输入"]["re报审计_sha256"] = sha(aud)
rows = jl(aud)
by = defaultdict(list)
for r in rows:
    by[r["sample_id"]].append(r["reward"])
sizes = Counter(len(v) for v in by.values())
same = {k: v for k, v in by.items() if len(set(v)) == 1}
same_full = [k for k, v in same.items() if abs(v[0] - 1.0) < 1e-9]
same_notfull = [k for k, v in same.items() if abs(v[0] - 1.0) >= 1e-9]
rep["A_同一奖励分组"] = {
    "审计条数": len(rows),
    "组数": len(by),
    "每组候选数分布": dict(sizes),
    "四候选完全同分的组数": len(same),
    "其中满分(=1.0)": len(same_full),
    "其中未满分": len(same_notfull),
    "未满分组的分值分布": dict(Counter(round(v[0], 4) for k, v in same.items()
                                   if k in set(same_notfull))),
    "全部256组平均奖励": round(sum(r["reward"] for r in rows) / len(rows), 4),
    "说明": "分母是**256 组**（审计覆盖全部训练组）；我此前报告的 34/124 是"
            "**从训练日志行抓取**的口径，两者分母不同，不可互换。",
}

# ── B/C. 按 lot 的配对 bootstrap ──────────────────────────────────────
refs = {r["sample_id"]: r["reference"] for r in
        jl(CODEX / "data_v3" / "confirmation120_reference.jsonl")}
gold = {s: r["defect_class"] for s, r in refs.items()}
ids = sorted(refs)
present = [c for c in CLASSES if any(gold[s] == c for s in ids)]

# lot 从池清单里取（confirmation.jsonl 有 lot_name）
lot_of = {r["sample_id"]: r["lot_name"] for r in jl(CODEX / "pools" / "confirmation.jsonl")}
lots = sorted({lot_of[s] for s in ids})
by_lot = defaultdict(list)
for s in ids:
    by_lot[lot_of[s]].append(s)

pred, rews, acc = {}, {}, {}
for m in MODELS:
    pred[m], rews[m] = {}, {}
    for r in jl(SRV / "评测_v3" / f"{m}_raw.jsonl"):
        s = r["sample_id"]
        try:
            pred[m][s] = R.parse(r["raw"]).get("defect_class")
        except Exception:
            pred[m][s] = None
        rews[m][s] = R.score(r["raw"], refs[s])["reward"]


def macro_f1(p, subset):
    tp = Counter(); fp = Counter(); fn = Counter()
    for s in subset:
        g, q = gold[s], p.get(s)
        if q == g and g in present:
            tp[g] += 1
        else:
            if q in present:
                fp[q] += 1
            if g in present:
                fn[g] += 1
    f1s = []
    for c in present:
        pr = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1s) / len(f1s)


def boot_lot(fn, n=10000, seed=3407):
    """按 lot 重抽（同一次抽样三个模型共用同一批 lot）。"""
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        pick = [lots[rng.randrange(len(lots))] for _ in range(len(lots))]
        sub = [s for L in pick for s in by_lot[L]]
        vals.append(fn(sub))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)], sum(vals) / n


def boot_lot_diff(fa, fb, n=10000, seed=3407):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        pick = [lots[rng.randrange(len(lots))] for _ in range(len(lots))]
        sub = [s for L in pick for s in by_lot[L]]
        vals.append(fa(sub) - fb(sub))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)], sum(vals) / n


rep["B_主指标(lot配对)"] = {
    "确认池lot数": len(lots), "图数": len(ids), "覆盖类": present,
    "逐模型MacroF1": {m: round(macro_f1(pred[m], ids), 4) for m in MODELS},
}
for a, b in (("M2_v3", "M1_v3"), ("M2_v3", "M0_v3"), ("M1_v3", "M0_v3")):
    lo, hi, mean = boot_lot_diff(lambda S: macro_f1(pred[a], S),
                                 lambda S: macro_f1(pred[b], S))
    rep["B_主指标(lot配对)"][f"{a}-{b}"] = {
        "差": round(mean, 4), "95区间": [round(lo, 4), round(hi, 4)],
        "跨0": bool(lo <= 0 <= hi)}

rep["C_次指标(lot配对)"] = {}
for a, b in (("M2_v3", "M1_v3"), ("M2_v3", "M0_v3"), ("M1_v3", "M0_v3")):
    lo, hi, mean = boot_lot_diff(
        lambda S: sum(rews[a][s] for s in S) / len(S),
        lambda S: sum(rews[b][s] for s in S) / len(S))
    rep["C_次指标(lot配对)"][f"{a}-{b}"] = {
        "差": round(mean, 4), "95区间": [round(lo, 4), round(hi, 4)],
        "跨0": bool(lo <= 0 <= hi)}

# Donut 抽不中的比例（Codex 提到 441/10000）
rng = random.Random(3407)
miss = 0
for _ in range(10000):
    pick = [lots[rng.randrange(len(lots))] for _ in range(len(lots))]
    sub = [s for L in pick for s in by_lot[L]]
    if not any(gold[s] == "Donut" for s in sub):
        miss += 1
rep["D_Donut稀疏性"] = {"Donut在图里条数": sum(1 for s in ids if gold[s] == "Donut"),
                      "10000次lot重抽中Donut缺席次数": miss}

print(json.dumps(rep, ensure_ascii=False, indent=1))
(HERE / "out_v3" / "补证_CPU重算.json").write_text(
    json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
