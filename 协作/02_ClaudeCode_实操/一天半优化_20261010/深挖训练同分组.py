#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练侧同分组深挖（CPU，只读）。

对象：M2_v3 的 256 个训练组里，**四候选完全同分但未满分**的 42 组
（Codex 已算出 91 组同分，其中 49 满分、42 未满分——本脚本深挖那 42 组）。

数据源（全部本地现成，不连服务器）：
  · `out_v3/服务器产物/out/M2_v3/reward_audit.jsonl`  1024 条 = 256 组 × 4 候选，逐条含
    `response`（原文）/ `parts`（分项分）/ `eligible_weights`（适用掩码）/ `reward`
  · `…/无卡CPU整备_20261010/data_v3/group256_grpo.jsonl`  冻结训练参考（`reference_json`）
  · `补证_20261011/ev/logging/out_M2_v3_*_logging.jsonl`  **完整 256 步**训练信号（不是日志行子集）
  · `core_v3_复核修复.py`  冻结奖励实现（用于复算分项，不重写规则）

对每组回答四个问题：
  Q1 哪些字段**四候选答案完全相同且都错**（同一个错误）？
  Q2 哪些字段**答案不同但得分相同**？
  Q3 哪些字段因**参考为 unknown 被排除**（eligible_weight=0）？
  Q4 哪些组**存在正确候选却没拿到应有高分**，哪些**根本没生成正确候选**？
另外：覆盖混淆与区带错法 × 类别 × 训练步位置（早/中/晚）。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRV = HERE / "out_v3" / "服务器产物" / "out"
CODEX = Path("D:/pycode/晶圆图研究/协作/01_Codex_指挥/无卡CPU整备_20261010")
LOG = HERE / "补证_20261011" / "ev" / "logging/out_M2_v3_v0-20261011-000507_logging.jsonl"
sys.path.insert(0, str(HERE))
import core_v3_复核修复 as R  # noqa: E402

FIELDS = ["defect_class", "coverage_band", "red_mass_zones", "angular_type", "clock_sectors"]
WEIGHT = {"defect_class": "class", "coverage_band": "coverage",
          "red_mass_zones": "zones", "angular_type": "angular", "clock_sectors": "angular"}


def jl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


refs = {json.loads(l)["sample_id"]: json.loads(l)["reference_json"]
        for l in (CODEX / "data_v3/group256_grpo.jsonl").open(encoding="utf-8") if l.strip()}

aud = jl(SRV / "M2_v3/reward_audit.jsonl")
groups = defaultdict(list)          # sample_id → 4 条（顺序即候选 0..3）
group_order = []
for r in aud:
    if r["sample_id"] not in groups:
        group_order.append(r["sample_id"])
    groups[r["sample_id"]].append(r)

for sid, rows in groups.items():
    for i, r in enumerate(rows):
        r["_idx"] = i
        r["_obj"] = (R.parse(r["response"]) if isinstance(r["response"], str) else {})
        r["_ref"] = R.parse(refs[sid])

# 组顺序应与冻结数据顺序一致（dataset_shuffle=False，每步 1 组）
frozen_order = [json.loads(l)["sample_id"]
                for l in (CODEX / "data_v3/group256_grpo.jsonl").open(encoding="utf-8") if l.strip()]
order_matches = group_order == frozen_order

report: dict = {"组顺序与冻结一致": order_matches, "组数": len(groups)}

# 全量同分统计（先复现 Codex 的 91/49/42）
same_groups, same_full, same_imperfect = [], [], []
for sid in group_order:
    rw = [r["reward"] for r in groups[sid]]
    if len(set(rw)) == 1:
        same_groups.append(sid)
        (same_full if abs(rw[0] - 1.0) < 1e-9 else same_imperfect).append(sid)
report["同分复现"] = {"同分组": len(same_groups), "满分": len(same_full),
                   "未满分": len(same_imperfect)}

step_of = {sid: i + 1 for i, sid in enumerate(frozen_order)}   # 第几步（1-based）

per_group = []
for sid in same_imperfect:
    rows = groups[sid]
    ref = rows[0]["_ref"]
    preds = {f: [r["_obj"].get(f) for r in rows] for f in FIELDS}
    parts = [r["parts"] for r in rows]
    elig = rows[0]["eligible_weights"]
    entry = {"sample_id": sid, "step": step_of[sid], "reward": rows[0]["reward"],
             "class_true": ref["defect_class"], "parts": parts, "eligible": elig,
             "字段诊断": {}}
    for f in FIELDS:
        vals = preds[f]
        all_same = len({json.dumps(v, sort_keys=True) for v in vals}) == 1
        wt = elig.get(WEIGHT[f], 0)
        wrong_vals = [v for v in vals if v != ref.get(f)]
        has_correct_cand = any(v == ref.get(f) for v in vals)
        entry["字段诊断"][f] = {
            "参考": ref.get(f),
            "四候选取值": vals,
            "四候选全同": all_same,
            "四候选全同且都错": bool(all_same and vals[0] != ref.get(f)),
            "存在正确候选": has_correct_cand,
            "适用权重": wt,
            "被掩码排除": wt == 0,
        }
    # Q4：正确候选有没有拿到该拿的分
    for f in ("defect_class", "coverage_band", "red_mass_zones"):
        if f not in ref:
            continue
        ci = [i for i, v in enumerate(preds[f]) if v == ref[f]]
        if ci:
            got = [parts[i][WEIGHT[f]] for i in ci]
            entry["字段诊断"][f]["正确候选得分"] = got
            entry["字段诊断"][f]["正确候选是否拿满"] = [g == 1.0 for g in got]
    per_group.append(entry)

# 汇总
agg = {f: {"全同且都错": 0, "答案不同但同分": 0, "被掩码排除": 0,
           "有正确候选": 0, "有正确候选但未拿满": 0, "从未出现正确候选": 0} for f in FIELDS}
for e in per_group:
    for f, d in e["字段诊断"].items():
        if d["被掩码排除"]:
            agg[f]["被掩码排除"] += 1
            continue
        if d["四候选全同且都错"]:
            agg[f]["全同且都错"] += 1
        elif not d["四候选全同"]:
            scores = [p[WEIGHT[f]] for p in e["parts"]]
            if len(set(scores)) == 1:
                agg[f]["答案不同但同分"] += 1
        if d["存在正确候选"]:
            agg[f]["有正确候选"] += 1
            if not all(d.get("正确候选是否拿满", [True])):
                agg[f]["有正确候选但未拿满"] += 1
        else:
            agg[f]["从未出现正确候选"] += 1
report["42组按字段汇总"] = agg

# 覆盖与区带的错法
def band_idx(x):
    order = ["zero", "under_5pct", "5_to_under_20pct", "20_to_under_50pct", "half_or_more"]
    return order.index(x) if x in order else None

cov_moves = Counter(); zone_err = Counter(); cls_cov = Counter()
for e in per_group:
    d = e["字段诊断"]["coverage_band"]
    if d["被掩码排除"]:
        continue
    for v in d["四候选取值"]:
        a, b = band_idx(d["参考"]), band_idx(v)
        if a is not None and b is not None and a != b:
            cov_moves[(d["参考"], v)] += 1
    z = e["字段诊断"]["red_mass_zones"]
    if not z["被掩码排除"]:
        for v in z["四候选取值"]:
            if v != z["参考"]:
                zone_err[("缺" if isinstance(v, list) and isinstance(z["参考"], list)
                          and set(v) < set(z["参考"]) else
                          "多" if isinstance(v, list) and isinstance(z["参考"], list)
                          and set(v) > set(z["参考"]) else "其他")] += 1
    cls_cov[(e["class_true"], d["参考"])] += 1
report["覆盖错法"] = {f"{a} → {b}": n for (a, b), n in cov_moves.most_common()}
report["区带错法"] = dict(zone_err)
report["42组按类别×参考覆盖"] = {f"{a}|{b}": n for (a, b), n in sorted(cls_cov.items())}

# 早/中/晚
def third(s):
    return "早(1-85)" if s <= 85 else "中(86-170)" if s <= 170 else "晚(171-256)"
report["42组在训练步上的分布"] = dict(Counter(third(e["step"]) for e in per_group))
report["42组的步号列表"] = sorted(e["step"] for e in per_group)

# 完整 logging.jsonl 的早中晚信号
logrows = [json.loads(l) for l in LOG.open(encoding="utf-8") if l.strip()]
logrows = [r for r in logrows if "reward" in r and "reward_std" in r and "completions/mean_length" in r]
def seg(rows):
    rw = [r["reward"] for r in rows]
    sd = [r["reward_std"] for r in rows]
    zf = [r["frac_reward_zero_std"] for r in rows]
    ln = [r["completions/mean_length"] for r in rows]
    return {"步数": len(rows), "奖励均值": round(sum(rw) / len(rw), 4),
            "reward_std均值": round(sum(sd) / len(sd), 4),
            "零方差步占比": round(sum(1 for x in zf if x >= 1) / len(zf), 4),
            "平均生成长度": round(sum(ln) / len(ln), 1)}
n = len(logrows)
report["完整logging早中晚"] = {"总步数": n,
                          "早": seg(logrows[:n // 3]),
                          "中": seg(logrows[n // 3:2 * n // 3]),
                          "晚": seg(logrows[2 * n // 3:])}
def _canon(o):
    return json.dumps({k: o.get(k) for k in FIELDS}, ensure_ascii=False, sort_keys=True)

report["生成多样性"] = {
    "全池_四候选去重后平均不同答案数":
        round(sum(len({_canon(r["_obj"]) for r in groups[s]}) for s in group_order)
              / len(group_order), 3),
    "全池_四候选完全同文本的比例":
        round(sum(1 for s in group_order
                  if len({r["response"] for r in groups[s]}) == 1) / len(group_order), 4),
    "42组_四候选去重后平均不同答案数":
        round(sum(len({_canon(r["_obj"]) for r in groups[s]}) for s in same_imperfect)
              / max(len(same_imperfect), 1), 3),
    "42组_四候选完全同文本的比例":
        round(sum(1 for s in same_imperfect
                  if len({r["response"] for r in groups[s]}) == 1) / max(len(same_imperfect), 1), 4),
}
report["逐组明细"] = per_group

print(json.dumps({k: v for k, v in report.items() if k != "逐组明细"},
                 ensure_ascii=False, indent=1))
(HERE / "out_v3" / "训练组深挖.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
