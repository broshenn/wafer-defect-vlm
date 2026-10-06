#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩容 v2：补入历史接触排除，**只替换新增部分**，保留原 180。"""
from __future__ import annotations
import hashlib, io, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("D:/pycode/晶圆图研究")
V1 = Path("D:/pycode/.ssh-tmp/exp")
OUT = Path("D:/pycode/.ssh-tmp/exp2")
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = REPO / "data" / "manifest.jsonl"
BENCH = REPO / "benchmark"
SRC180 = REPO / "协作" / "02_ClaudeCode_实操" / "学校训练准备_20261002-025848"
RLPOOL = REPO / "协作" / "02_ClaudeCode_实操" / "今晚类别RL_20261006_0325" / "rl_pool90.jsonl"
VAL90 = Path("D:/pycode/.ssh-tmp/val90/val90.jsonl")
HIST = OUT / "historical_ids_v2.json"
DERIVED = re.compile(r"^(wafer_\d{8}_\d{3})__.+$")
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
SEED, TARGET, PER = 3407, 1440, 160
QUESTION = ("这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。"
            "请从 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_full、Random、Scratch、none "
            "中选一个最符合主要图案的类别。只输出一个 JSON 对象，形如 "
            '{"defect_class":"类别"}；无法可靠判断时填 {"defect_class":"unknown"}。')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rd(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


print("=" * 78)
print("扩容 v2：补入历史接触排除")
print("=" * 78)
rows = rd(MANIFEST)
by_id = {r["sample_id"]: r for r in rows}

# ── 排除集 v1 ───────────────────────────────────────────
e = {}
for n, p in (("train180", SRC180 / "train_180.jsonl"), ("dev18", SRC180 / "dev_18.jsonl"),
             ("val90", VAL90)):
    e[n] = {r["sample_id"] for r in rd(p)}
e["rl_pool90"] = {x["sample_id"] for x in rd(RLPOOL)}
e["provenance180"] = {r["sample_id"] for r in rd(
    REPO / "协作" / "01_Codex_指挥" / "训练数据就绪_20261003" / "provenance_180.jsonl")}
bm = set()
for f in sorted(BENCH.glob("*.jsonl")):
    for o in rd(f):
        if isinstance(o, dict):
            for k in ("sample_id", "query_id", "gallery_id", "variant_id"):
                v = o.get(k)
                if isinstance(v, str) and v.startswith("wafer_"):
                    m = DERIVED.match(v)
                    bm.add(m.group(1) if m else v)
e["benchmark根"] = bm
v1_excl = set().union(*e.values())
v1_lots = {by_id[i]["lot_name"] for i in v1_excl if i in by_id}
print(f"v1 排除 {len(v1_excl)} ID / {len(v1_lots)} lot")

# ── 新增：历史接触 ──────────────────────────────────────
hj = json.load(io.open(HIST, encoding="utf-8"))
hist = {i for i in hj["historical_ids"] if i in by_id}
print(f"历史接触 ID {len(hist)}（有实际回答证据，非计划清单）")
hist_lots = {by_id[i]["lot_name"] for i in hist}
t180 = e["train180"]
t180_lots = {by_id[i]["lot_name"] for i in t180 if i in by_id}
new_hist_lots = hist_lots - t180_lots
print(f"  历史 lot {len(hist_lots)}，其中 **非 train180 的 {len(new_hist_lots)} 个**须排新增选择")
print(f"  （原 A 裁决：仅计划未发送的两张保留，不扩大成所有计划都排除）")

v2_excl = v1_excl | hist
v2_lots = v1_lots | new_hist_lots
print(f"\nv2 排除 {len(v2_excl)} ID / {len(v2_lots)} lot")

# ── 检查 v1 新增里的违规 ────────────────────────────────
print("\n--- v1 新增部分的违规情况 ---")
for lv in (360, 720, 1440):
    ids = [x["sample_id"] for x in rd(V1 / f"scale_{lv}.jsonl")]
    new = [i for i in ids if i not in t180]
    bad_id = [i for i in new if i in hist]
    bad_lot = [i for i in new if by_id[i]["lot_name"] in new_hist_lots]
    print(f"  scale_{lv}: 新增 {len(new)}，**ID 在历史接触里 {len(bad_id)}**，"
          f"**落入历史 lot 的 {len(bad_lot)}**")

# ── 重选（v2）───────────────────────────────────────────
cand = [r for r in rows
        if r["split"] == "train" and r.get("label_source") == "ground_truth"
        and r["failure_type"] in CLASSES
        and r["sample_id"] not in v2_excl and r["lot_name"] not in v2_lots]
per = defaultdict(list)
for r in cand:
    per[r["failure_type"]].append(r)
avail_lots = {c: len({x["lot_name"] for x in per[c]}) for c in CLASSES}
print(f"\n--- v2 可用候选 ---")
print(f"  {len(cand)} 条 / {len({r['lot_name'] for r in cand})} lot")
for c in CLASSES:
    print(f"  {c:10s} {len(per[c]):5d} 条 / {avail_lots[c]:5d} lot")

quota = {c: min(PER, avail_lots[c]) for c in CLASSES}
short = TARGET - sum(quota.values())
room = {c: avail_lots[c] - quota[c] for c in CLASSES}
if short > 0:
    tot = sum(room.values())
    for c in CLASSES:
        quota[c] += short * room[c] // tot
    i = 0
    while sum(quota.values()) < TARGET:
        c = CLASSES[i % len(CLASSES)]
        if quota[c] < avail_lots[c]:
            quota[c] += 1
        i += 1


def strat(c):
    items = per[c]
    dens = sorted(x["features"]["defect_ratio"] for x in items
                  if x.get("features", {}).get("status") == "ok")

    def q(v):
        if not dens:
            return 0
        return min(3, int(sum(1 for d in dens if d <= v) / len(dens) * 4))

    b = defaultdict(list)
    for x in items:
        fr = x.get("features", {})
        b[(q(fr.get("defect_ratio", 0.0)) if fr.get("status") == "ok" else 4,
           (x.get("matrix_shape") or (0,))[0] // 20)].append(x)
    ks = sorted(b)
    r = random.Random(SEED + abs(hash(c)) % 10000)
    for k in ks:
        r.shuffle(b[k])
    seq, li = [], {k: 0 for k in ks}
    while any(li[k] < len(b[k]) for k in ks):
        for k in ks:
            if li[k] < len(b[k]):
                seq.append(b[k][li[k]]); li[k] += 1
    return seq


ordered_cls = {c: strat(c) for c in CLASSES}
used, pool, taken, cur = set(), [], {c: 0 for c in CLASSES}, {c: 0 for c in CLASSES}
while len(pool) < TARGET and any(cur[c] < len(ordered_cls[c]) for c in CLASSES):
    prog = False
    for c in CLASSES:
        if len(pool) >= TARGET or taken[c] >= quota[c]:
            continue
        while cur[c] < len(ordered_cls[c]):
            x = ordered_cls[c][cur[c]]; cur[c] += 1
            if x["lot_name"] not in used:
                used.add(x["lot_name"]); pool.append(x); taken[c] += 1; prog = True
                break
    if not prog:
        break
print(f"\n--- v2 池：{len(pool)} 条 / {len({x['lot_name'] for x in pool})} lot ---")
for c in CLASSES:
    print(f"  {c:10s} 配额 {quota[c]:4d} 实取 {taken[c]:4d}"
          + ("" if taken[c] == quota[c] else "   ← 未满"))

# 全局定序（保证嵌套稳定）
bc = defaultdict(list)
for x in pool:
    bc[x["failure_type"]].append(x)
ordered, idx = [], {c: 0 for c in CLASSES}
while len(ordered) < len(pool):
    for c in CLASSES:
        if idx[c] < len(bc[c]):
            ordered.append(bc[c][idx[c]]); idx[c] += 1

base_ids = [x["sample_id"] for x in rd(SRC180 / "train_180.jsonl")]
levels = {}
for lv in (180, 360, 720, 1440):
    if lv == 180:
        ids, src = base_ids, "旧 train180（保留，未改）"
    else:
        ids = base_ids + [x["sample_id"] for x in ordered[:lv - 180]]
        src = f"train180 + 新 {lv-180}（v2 池前缀）"
    parts = []
    for i, sid in enumerate(ids, 1):
        o = by_id[sid]
        parts.append({"item_id": f"exp_{i:04d}", "sample_id": sid, "label": o["failure_type"],
                      "lot_name": o["lot_name"], "split": o["split"],
                      "label_source": o["label_source"], "matrix_shape": o.get("matrix_shape"),
                      "defect_ratio": (o.get("features") or {}).get("defect_ratio"),
                      "image": f"data/images/{sid}.png",
                      "image_sha256": sha(REPO / "data" / "images" / f"{sid}.png"),
                      "is_old180": sid in set(base_ids)})
    p = OUT / f"scale_{lv}_v2.jsonl"
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        for x in parts:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    levels[lv] = {"n": len(parts), "lots": len({x["lot_name"] for x in parts}),
                  "classes": dict(Counter(x["label"] for x in parts)),
                  "sha256": sha(p), "source": src}
    print(f"  {lv:5d}: {len(parts)} 条 / {len({x['lot_name'] for x in parts})} lot  {src}")

for a, b in ((180, 360), (360, 720), (720, 1440)):
    A = {x["sample_id"] for x in rd(OUT / f"scale_{a}_v2.jsonl")}
    B = {x["sample_id"] for x in rd(OUT / f"scale_{b}_v2.jsonl")}
    print(f"  嵌套 {a} ⊂ {b}: {A <= B}")

json.dump({
    "v1_excl": len(v1_excl), "v1_lots": len(v1_lots),
    "hist_ids": len(hist), "hist_lots": len(hist_lots),
    "hist_lots_outside_t180": len(new_hist_lots),
    "v2_excl": len(v2_excl), "v2_lots": len(v2_lots),
    "quota": quota, "taken": taken, "pool": len(pool),
    "levels": levels, "seed": SEED,
    "规则": "v1 排除 ∪ 历史接触；历史 lot 中**非 train180 的**排除；原 train180 保留可复用",
}, io.open(OUT / "scale_meta_v2.json", "w", encoding="utf-8"),
    ensure_ascii=False, indent=2)
print(f"\n写出 {OUT}/scale_*_v2.jsonl + scale_meta_v2.json")
