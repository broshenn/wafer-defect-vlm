#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：逐字段区分「有值 / 显式 null（来源判为不适用）/ 键缺失（缺标注）」。

来源装载逻辑与生成 描述七字段.jsonl 的原脚本逐行一致，只额外记录键是否存在。
"""
from __future__ import annotations
import io, json, sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path("D:/pycode/晶圆图研究")
ABC = Path("D:/pycode/.ssh-tmp/abc")
OUT = REPO / "协作" / "02_ClaudeCode_实操" / "描述SFT与同题评测_20261007_1430"
FIELDS = ["defect_class", "morphology", "radial_zone", "clock_direction",
          "extent_r", "caption_zh", "uncertainty"]


def rd(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


# ── 来源装载（与原脚本一致）────────────────────────────
old158 = {}
for x in rd(REPO / "协作" / "01_Codex_指挥" / "训练数据就绪_20261003" / "caption_train_158.jsonl"):
    m = x.get("messages") or []
    if len(m) > 1:
        old158[x["sample_id"]] = {"caption_zh": m[1]["content"], "_src": "旧158描述候选"}

struct90 = {}
for x in rd(REPO / "协作" / "01_Codex_指挥" / "训练数据就绪_20261003" / "structured_candidates_90.jsonl"):
    a = x.get("answer") or {}
    struct90[x["sample_id"]] = {**a, "_src": "结构化候选90"}

blind = rd(REPO / "协作" / "02_ClaudeCode_实操" / "ZCode标注准备_20261006" / "zcode_blind_new.jsonl")
zc2sid = {b["item_id"]: b["sample_id"] for b in blind}
raws = {}
for root, tag in ((REPO / "协作" / "03_ZCode_标注" / "扩容198_20261006_165855", "GLM原始回答"),
                  (REPO / "协作" / "03_ZCode_标注" / "接续49_20261006_20261007-000123", "GLM接续原始回答"),
                  (REPO / "01_Codex_指挥" / "GPT补标91_20261006", "GPT原始回答")):
    for p in sorted(root.rglob("*_raw.json")):
        sid = zc2sid.get(p.name.replace("_raw.json", ""))
        if sid:
            try:
                raws[sid] = {**json.loads(p.read_text(encoding="utf-8")), "_src": tag}
            except Exception:
                pass
rev49 = {x["sample_id"]: x for x in
         rd(REPO / "协作" / "01_Codex_指挥" / "验收_20261007_接续49" / "模型辅助候选49.jsonl")}

print(f"来源可用：旧158 {len(old158)} · 结构化90 {len(struct90)} · 原答 {len(raws)} · 49复核 {len(rev49)}")

# ── 逐条判定来源与键存在性 ─────────────────────────────
rows = rd(ABC / "items356.jsonl")
man = {r["sample_id"]: r for r in rd(REPO / "data" / "manifest.jsonl")}

stat = {f: Counter() for f in FIELDS}
chosen = Counter()
recs = []
for r in rows:
    sid = r["sample_id"]
    a, tag = None, None
    if sid in raws:
        a, tag = raws[sid], raws[sid]["_src"]
        if sid in rev49:
            ra = (rev49[sid].get("answer") or {})
            if ra.get("caption_zh"):
                a, tag = {**ra, "_src": tag + "+49复核副本"}, raws[sid]["_src"] + "+49复核副本"
    if a is None and sid in struct90:
        a, tag = struct90[sid], struct90[sid]["_src"]
    if a is None and sid in old158:
        a, tag = old158[sid], old158[sid]["_src"]
    if a is None:
        chosen["无来源"] += 1
        continue
    chosen[tag] += 1
    rec = {"sample_id": sid, "label": man[sid]["failure_type"], "src": tag, "f": {}, "present": {}}
    for f in FIELDS:
        present = f in a
        v = a.get(f)
        if isinstance(v, str):
            v = v.strip() or None
        if v is not None:
            stat[f]["有值"] += 1
        elif present:
            stat[f]["显式null"] += 1
        else:
            stat[f]["键缺失"] += 1
        rec["f"][f] = v
        rec["present"][f] = present
    recs.append(rec)

print(f"\n来源分账：{dict(chosen)}")
print(f"\n{'字段':<18}{'有值':>7}{'显式null':>10}{'键缺失':>9}")
for f in FIELDS:
    s = stat[f]
    print(f"{f:<18}{s['有值']:>7}{s['显式null']:>10}{s['键缺失']:>9}")

# ── 任务分型 ────────────────────────────────────────────
print("\n" + "=" * 62)
print("任务分型")
print("=" * 62)
kind = Counter()
for rec in recs:
    f = rec["f"]
    has_m, has_z = f["morphology"] is not None, f["radial_zone"] is not None
    kind[(has_m, has_z)] += 1
print(f"有 morphology: {sum(1 for r in recs if r['f']['morphology'] is not None)}")
print(f"有 radial_zone:{sum(1 for r in recs if r['f']['radial_zone'] is not None)}")
print(f"两者都有: {sum(1 for r in recs if r['f']['morphology'] and r['f']['radial_zone'])}")
print(f"morphology∩radial_zone 一致: {sum(1 for r in recs if (r['f']['morphology'] is not None) == (r['f']['radial_zone'] is not None))}/{len(recs)}")

# 按来源看形态覆盖
by_src = Counter()
for rec in recs:
    by_src[(rec["src"], rec["f"]["morphology"] is not None, rec["f"]["uncertainty"] is not None)] += 1
print("\n（来源, 有形态, 有不确定）→ 条数")
for k, v in sorted(by_src.items()):
    print(f"  {k} → {v}")

# ── clock_direction 的协议规则影响 ─────────────────────
print("\n" + "=" * 62)
print("clock_direction：来源值与协议改写")
print("=" * 62)
ZONES = ["center", "middle", "edge", "global", "none", "unknown"]
proto_na, src_null, src_val = 0, 0, 0
for rec in recs:
    f, p = rec["f"], rec["present"]
    rz = f["radial_zone"]
    rule = (rz in ("global", "none", "unknown")) or (rec["label"] == "Edge_Ring")
    if rule:
        proto_na += 1
    elif f["clock_direction"] is not None:
        src_val += 1
    elif p["clock_direction"]:
        src_null += 1
    else:
        print(f"  !! 键缺失且非协议: {rec['sample_id']} src={rec['src']}")
print(f"协议判为不适用（改写为 null）: {proto_na}")
print(f"来源给了方向值:              {src_val}")
print(f"非协议但来源显式 null:        {src_null}")

# uncertainty 缺失的 44 条来自哪
print("\nuncertainty 键缺失的来源分布：",
      dict(Counter(r["src"] for r in recs if not r["present"]["uncertainty"])))
print("uncertainty 显式 null 的来源分布：",
      dict(Counter(r["src"] for r in recs if r["present"]["uncertainty"] and r["f"]["uncertainty"] is None)))

json.dump({"chosen": dict(chosen),
           "per_field": {f: dict(stat[f]) for f in FIELDS}},
          io.open(OUT / "诊断_字段状态.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n写出 {OUT/'诊断_字段状态.json'}")
