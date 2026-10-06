#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩容准备 第 2 步：构建嵌套规模清单 180 / 360 / 720 / 1440。

配额规则（**事前锁定，写进产物**）：
  * 目标每类 160；**每 lot 至多一张**；
  * Donut（147 lot）与 Near_full（81 lot）达不到 160，**如实取全部可用 lot**；
  * 差额按"剩余可用 lot 数"比例补到其余 7 类；
  * 最终 1440 条 = 1440 个**互不相同**的 lot。

嵌套：180 = 旧 train180（规模曲线的起点，不动）；
      360 = 180 + 新 180；720 = 360 + 新 360；1440 = 720 + 新 720。
      新的三档取自同一个 1440 池的**前缀**，因此严格嵌套。

分层：类内按「缺陷密度四分位 × 原矩阵 shape 桶」分层，层内 seed 3407 打乱，
      跨层轮转取样，保证 shape 与密度都被覆盖。
"""

from __future__ import annotations
import hashlib, io, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("D:/pycode/晶圆图研究")
MANIFEST = REPO / "data" / "manifest.jsonl"
SRC180 = REPO / "协作" / "02_ClaudeCode_实操" / "学校训练准备_20261002-025848"
RLPOOL = REPO / "协作" / "02_ClaudeCode_实操" / "今晚类别RL_20261006_0325" / "rl_pool90.jsonl"
VAL90 = Path("D:/pycode/.ssh-tmp/val90/val90.jsonl")
BENCH = REPO / "benchmark"
OUT = Path("D:/pycode/.ssh-tmp/exp")
DERIVED = re.compile(r"^(wafer_\d{8}_\d{3})__.+$")
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
SEED = 3407
TARGET = 1440
PER_CLASS_TARGET = 160
QUESTION = ("这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。"
            "请从 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_full、Random、Scratch、none "
            "中选一个最符合主要图案的类别。只输出一个 JSON 对象，形如 "
            '{"defect_class":"类别"}；无法可靠判断时填 {"defect_class":"unknown"}。')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rd(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


print("=" * 78)
print("构建嵌套规模清单")
print("=" * 78)

rows = rd(MANIFEST)
by_id = {r["sample_id"]: r for r in rows}

# 排除集（与第 1 步一致）
excl = {}
for name, p in (("train180", SRC180 / "train_180.jsonl"), ("dev18", SRC180 / "dev_18.jsonl"),
                ("val90", VAL90)):
    excl[name] = {r["sample_id"] for r in rd(p)}
excl["rl_pool90"] = {x["sample_id"] for x in rd(RLPOOL)}
excl["provenance180"] = {r["sample_id"] for r in rd(
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
excl["benchmark根"] = bm
excl_ids = set().union(*excl.values())
excl_lots = {by_id[i]["lot_name"] for i in excl_ids if i in by_id}
print(f"排除 {len(excl_ids)} ID / {len(excl_lots)} lot")

cand = [r for r in rows
        if r["split"] == "train" and r.get("label_source") == "ground_truth"
        and r["failure_type"] in CLASSES
        and r["sample_id"] not in excl_ids and r["lot_name"] not in excl_lots]
per = defaultdict(list)
for r in cand:
    per[r["failure_type"]].append(r)

# ── 配额 ────────────────────────────────────────────────
avail_lots = {c: len({x["lot_name"] for x in per[c]}) for c in CLASSES}
quota = {c: min(PER_CLASS_TARGET, avail_lots[c]) for c in CLASSES}
short = TARGET - sum(quota.values())
print(f"\n基础配额（每类 ≤160，受 lot 数限制）: {quota}  小计 {sum(quota.values())}")
room = {c: avail_lots[c] - quota[c] for c in CLASSES}
print(f"各类剩余可用 lot: {room}  合计 {sum(room.values())}")
if short > 0:
    tot_room = sum(room.values())
    print(f"差额 {short} 按剩余 lot 比例补足：")
    added, rem = {}, []
    for c in CLASSES:
        add = short * room[c] // tot_room if tot_room else 0
        added[c] = add; rem.append((short * room[c]) % tot_room)
    i = 0
    while sum(added.values()) < short:
        c = CLASSES[i % len(CLASSES)]
        if added[c] < room[c] and sum(added.values()) < short:
            added[c] += 1
        i += 1
    for c in CLASSES:
        quota[c] += added[c]
    print(f"  补足 {added}")
print(f"最终配额: {quota}  合计 {sum(quota.values())}")

# ── 分层取样（**全局 lot 唯一**，跨类轮转分配）────────────────
pool = []
per_class_ordered = {}
for c in CLASSES:
    items = per[c]
    dens = sorted(x["features"]["defect_ratio"] for x in items
                  if x.get("features", {}).get("status") == "ok")

    def q(v):
        if not dens:
            return 0
        k = sum(1 for d in dens if d <= v) / len(dens)
        return min(3, int(k * 4))

    def shape(x):
        return tuple(x.get("matrix_shape") or (0, 0))

    buckets = defaultdict(list)
    for x in items:
        fr = x.get("features", {})
        b = (q(fr.get("defect_ratio", 0.0)) if fr.get("status") == "ok" else 4,
             shape(x)[0] // 20)
        buckets[b].append(x)
    keys = sorted(buckets)
    r = random.Random(SEED + hash(c) % 10000)
    for k in keys:
        r.shuffle(buckets[k])
    seq, li = [], {k: 0 for k in keys}
    while any(li[k] < len(buckets[k]) for k in keys):
        for k in keys:
            if li[k] < len(buckets[k]):
                seq.append(buckets[k][li[k]]); li[k] += 1
    per_class_ordered[c] = seq
    print(f"  {c:10s} 分层定序 {len(seq):4d} 条，"
          f"shape {len({shape(x) for x in seq})} 种，层 {len(keys)}")

used_lots = set()
taken = {c: 0 for c in CLASSES}
cur = {c: 0 for c in CLASSES}
while len(pool) < TARGET and any(cur[c] < len(per_class_ordered[c]) for c in CLASSES):
    progressed = False
    for c in CLASSES:
        if len(pool) >= TARGET or taken[c] >= quota[c]:
            continue
        while cur[c] < len(per_class_ordered[c]):
            x = per_class_ordered[c][cur[c]]; cur[c] += 1
            if x["lot_name"] not in used_lots:
                used_lots.add(x["lot_name"]); pool.append(x); taken[c] += 1
                progressed = True
                break
    if not progressed:
        break

print(f"\n  {'类':10s} {'配额':>5s} {'实取':>5s}")
for c in CLASSES:
    flag = "" if taken[c] == quota[c] else "  ← 未满"
    print(f"  {c:10s} {quota[c]:5d} {taken[c]:5d}{flag}")

print(f"\n池合计 {len(pool)} 条 / {len({x['lot_name'] for x in pool})} lot")
assert len(pool) == len({x["lot_name"] for x in pool}), "仍有 lot 重复"
print("  **全局每 lot 一张：成立**")

# 全局定序（保证前缀嵌套稳定）：按类的固定顺序 + 类内分层顺序轮转
by_class = defaultdict(list)
for x in pool:
    by_class[x["failure_type"]].append(x)
ordered, idx = [], {c: 0 for c in CLASSES}
while len(ordered) < len(pool):
    for c in CLASSES:
        if idx[c] < len(by_class[c]):
            ordered.append(by_class[c][idx[c]]); idx[c] += 1
print(f"全局定序完成 {len(ordered)} 条")

# ── 嵌套清单 ────────────────────────────────────────────
OUT.mkdir(parents=True, exist_ok=True)
train180 = rd(SRC180 / "train_180.jsonl")
base_ids = [r["sample_id"] for r in train180]

levels = {}
for lv in (180, 360, 720, 1440):
    if lv == 180:
        ids = base_ids
        src = "旧 train180（规模曲线起点，未改）"
    else:
        n_new = lv - 180
        ids = base_ids + [x["sample_id"] for x in ordered[:n_new]]
        src = f"train180 + 新 {n_new}（1440 池前缀）"
    assert len(ids) == len(set(ids)), lv
    parts = []
    for i, sid in enumerate(ids, 1):
        o = by_id[sid]
        parts.append({
            "item_id": f"exp_{i:04d}", "sample_id": sid,
            "label": o["failure_type"], "lot_name": o["lot_name"],
            "split": o["split"], "label_source": o["label_source"],
            "matrix_shape": o.get("matrix_shape"),
            "defect_ratio": (o.get("features") or {}).get("defect_ratio"),
            "image": f"data/images/{sid}.png",
            "image_sha256": sha(REPO / "data" / "images" / f"{sid}.png"),
            "is_old180": sid in set(base_ids),
        })
    p = OUT / f"scale_{lv}.jsonl"
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        for x in parts:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    cm = Counter(x["label"] for x in parts)
    levels[lv] = {"n": len(parts), "lots": len({x["lot_name"] for x in parts}),
                  "classes": dict(cm), "sha256": sha(p), "source": src}
    print(f"  {lv:5d}: {len(parts)} 条 / {len({x['lot_name'] for x in parts})} lot  {src}")
    print(f"         类别 {dict(cm)}")

# 嵌套核对
for a, b in ((180, 360), (360, 720), (720, 1440)):
    A = {x["sample_id"] for x in rd(OUT / f"scale_{a}.jsonl")}
    B = {x["sample_id"] for x in rd(OUT / f"scale_{b}.jsonl")}
    print(f"  {a} ⊂ {b}: {A <= B}  （{a} {len(A)}，{b} {len(B)}）")

meta = {
    "seed": SEED, "target": TARGET,
    "规则": "每类≤160；不足的类取全部可用 lot；差额按剩余 lot 比例补到其余类；每 lot 至多一张",
    "可用盘点": {c: {"n": len(per[c]), "lots": avail_lots[c]} for c in CLASSES},
    "配额": quota,
    "配额受限的类": {c: avail_lots[c] for c in CLASSES if avail_lots[c] < PER_CLASS_TARGET},
    "levels": levels,
    "排除": {k: len(v) for k, v in excl.items()},
    "题面_sha256": hashlib.sha256(QUESTION.encode()).hexdigest(),
    "manifest_sha256": sha(MANIFEST),
    "边界": "这是**准备**清单；不代表已训练、也不代表学校已传齐这些图。",
}
(OUT / "scale_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"\n写出 {OUT}/scale_*.jsonl 与 scale_meta.json")
