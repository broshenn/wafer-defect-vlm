#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩容准备 第 1 步：库存核对 + 冻结排除输入核对 + 各类有效候选盘点。"""
from __future__ import annotations
import hashlib, io, json, os, re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("D:/pycode/晶圆图研究")
MANIFEST = REPO / "data" / "manifest.jsonl"
IMAGES = REPO / "data" / "images"
BENCH = REPO / "benchmark"
SRC180 = REPO / "协作" / "02_ClaudeCode_实操" / "学校训练准备_20261002-025848"
RLPOOL = REPO / "协作" / "02_ClaudeCode_实操" / "今晚类别RL_20261006_0325" / "rl_pool90.jsonl"
VAL90 = Path("D:/pycode/.ssh-tmp/val90/val90.jsonl")
DERIVED = re.compile(r"^(wafer_\d{8}_\d{3})__.+$")
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
EXPECT_RLPOOL_SHA = "0569a20bb9967966f56234ffec6837d2fd03a840a05012757c09e3a33d59e0af"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rd(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


print("=" * 78)
print("第 1 步：库存与冻结输入核对")
print("=" * 78)

# ── 1) manifest 与 PNG 逐 ID 一致 ───────────────────────
rows = rd(MANIFEST)
ids = [r["sample_id"] for r in rows]
png = {p.stem for p in IMAGES.glob("*.png")}
print(f"\nmanifest {len(rows)} 行，唯一 ID {len(set(ids))}")
print(f"data/images PNG {len(png)} 个")
print(f"  manifest 里缺 PNG 的: {len(set(ids) - png)}")
print(f"  清单外的 PNG:         {len(png - set(ids))}")
sp = Counter(r["split"] for r in rows)
print(f"  split: {dict(sp)}")
print(f"  label_source: {dict(Counter(r['label_source'] for r in rows))}")
print(f"  manifest sha256 {sha(MANIFEST)}")

# ── 2) 冻结的 RL 池核对 ─────────────────────────────────
print("\n--- 冻结的 RL 池 ---")
if RLPOOL.exists():
    s = sha(RLPOOL)
    print(f"  {RLPOOL.name} sha256 {s}")
    print(f"  与任务书给的值一致: {s == EXPECT_RLPOOL_SHA}")
    rlp = rd(RLPOOL)
    print(f"  {len(rlp)} 条，类别 {dict(Counter(x['label'] for x in rlp))}")
else:
    print("  !! 找不到"); rlp = []

# ── 3) 排除集 ───────────────────────────────────────────
print("\n--- 排除集构建 ---")
excl = {}
for name, p in (("train180", SRC180 / "train_180.jsonl"),
                ("dev18", SRC180 / "dev_18.jsonl"),
                ("val90", VAL90)):
    if p.exists():
        excl[name] = {r["sample_id"] for r in rd(p)}
        print(f"  {name:14s} {len(excl[name]):5d} 条")
excl["rl_pool90"] = {x["sample_id"] for x in rlp}
print(f"  {'rl_pool90':14s} {len(excl['rl_pool90']):5d} 条（冻结清单）")

prov = REPO / "协作" / "01_Codex_指挥" / "训练数据就绪_20261003" / "provenance_180.jsonl"
if prov.exists():
    excl["provenance180"] = {r["sample_id"] for r in rd(prov)}
    print(f"  {'provenance180':14s} {len(excl['provenance180']):5d} 条")

bm = set()
bmfiles = sorted(BENCH.glob("*.jsonl"))
for f in bmfiles:
    for o in rd(f):
        if isinstance(o, dict):
            for k in ("sample_id", "query_id", "gallery_id", "variant_id"):
                v = o.get(k)
                if isinstance(v, str) and v.startswith("wafer_"):
                    m = DERIVED.match(v)
                    bm.add(m.group(1) if m else v)
excl["benchmark根"] = bm
print(f"  {'benchmark根':14s} {len(bm):5d} 个（来自 {len(bmfiles)} 个文件）")

all_excl_ids = set().union(*excl.values())
by_id = {r["sample_id"]: r for r in rows}
excl_lots = {by_id[i]["lot_name"] for i in all_excl_ids if i in by_id}
print(f"  合计排除 ID {len(all_excl_ids)}，lot {len(excl_lots)}")
unknown = sorted(i for i in all_excl_ids if i not in by_id)
if unknown:
    print(f"  !! {len(unknown)} 个排除 ID 不在 manifest: {unknown[:5]}")

# ── 4) train 可用候选 ───────────────────────────────────
print("\n--- train 划分可用候选（已排除）---")
cand = [r for r in rows
        if r["split"] == "train"
        and r.get("label_source") == "ground_truth"
        and r["failure_type"] in CLASSES
        and r["sample_id"] not in all_excl_ids
        and r["lot_name"] not in excl_lots]
print(f"  {len(cand)} 条 / {len({r['lot_name'] for r in cand})} lot")
per = defaultdict(list)
for r in cand:
    per[r["failure_type"]].append(r)
print(f"\n  {'类':10s} {'可用条':>6s} {'可用lot':>7s} {'每lot一张上限':>12s}")
tot_lot_cap = 0
for c in CLASSES:
    nl = len({x["lot_name"] for x in per[c]})
    tot_lot_cap += nl
    print(f"  {c:10s} {len(per[c]):6d} {nl:7d} {nl:12d}")
print(f"  {'合计':10s} {len(cand):6d} {len({r['lot_name'] for r in cand}):7d} {tot_lot_cap:12d}")

# 1440 是否可行
print(f"\n  目标 1440 条唯一晶圆（每 lot 最多一张）: "
      f"{'可行' if tot_lot_cap >= 1440 else '**不可行**，上限 ' + str(tot_lot_cap)}")
print(f"  每类 160 条（每 lot 一张）: "
      f"{[c for c in CLASSES if len({x['lot_name'] for x in per[c]}) < 160] or '全部可行'}")

for c in ("Near_full", "none", "Donut"):
    print(f"    {c}: train 原有 "
          f"{sum(1 for r in rows if r['split']=='train' and r['failure_type']==c)} 条，"
          f"排除后 {len(per[c])} 条 / {len({x['lot_name'] for x in per[c]})} lot")

# split 非空核对（检验器要用的）
print(f"\n  manifest val lot {len({r['lot_name'] for r in rows if r['split']=='val'})} 个"
      f"（非空: {bool({r['lot_name'] for r in rows if r['split']=='val'})}）")
print(f"  manifest test lot {len({r['lot_name'] for r in rows if r['split']=='test'})} 个"
      f"（非空: {bool({r['lot_name'] for r in rows if r['split']=='test'})}）")

json.dump({
    "manifest_rows": len(rows), "manifest_unique": len(set(ids)),
    "png": len(png), "missing_png": len(set(ids) - png), "extra_png": len(png - set(ids)),
    "splits": dict(sp), "manifest_sha256": sha(MANIFEST),
    "exclusions": {k: len(v) for k, v in excl.items()},
    "excl_ids": len(all_excl_ids), "excl_lots": len(excl_lots),
    "train_available": len(cand),
    "train_available_lots": len({r["lot_name"] for r in cand}),
    "per_class": {c: {"n": len(per[c]), "lots": len({x["lot_name"] for x in per[c]})}
                  for c in CLASSES},
    "max_unique_at_one_per_lot": tot_lot_cap,
}, io.open("D:/pycode/.ssh-tmp/exp_inventory.json", "w", encoding="utf-8"),
    ensure_ascii=False, indent=2)
print("\n写出 exp_inventory.json")
