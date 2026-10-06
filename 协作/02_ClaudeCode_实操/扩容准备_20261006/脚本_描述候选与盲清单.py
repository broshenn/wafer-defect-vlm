#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩容准备 第 4 步：360 条同图描述候选（A/B/C）+ ZCode 无答案盲清单。"""
from __future__ import annotations
import hashlib, io, json, random
from collections import Counter
from pathlib import Path

REPO = Path("D:/pycode/晶圆图研究")
OUT = Path("D:/pycode/.ssh-tmp/exp")
SRC = REPO / "协作" / "01_Codex_指挥" / "训练数据就绪_20261003"
DERIVED = None
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
QUESTION = ("这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。"
            "请从 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_full、Random、Scratch、none "
            "中选一个最符合主要图案的类别。只输出一个 JSON 对象，形如 "
            '{"defect_class":"类别"}；无法可靠判断时填 {"defect_class":"unknown"}。')
BLIND_KEYS = ["item_id", "sample_id", "image_path", "image_sha256"]
LEAK_KEYS = ["label", "failure_type", "class", "defect_class", "lot_name", "lot",
             "split", "caption", "morphology", "answer", "prediction", "geometry"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rd(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


print("=" * 78)
print("360 条同图描述候选（A/B/C）")
print("=" * 78)

cap = rd(SRC / "caption_train_158.jsonl") if (SRC / "caption_train_158.jsonl").exists() else []
stc = rd(SRC / "structured_candidates_90.jsonl") if (SRC / "structured_candidates_90.jsonl").exists() else []
qua = rd(SRC / "caption_quarantine_22.jsonl") if (SRC / "caption_quarantine_22.jsonl").exists() else []
cap_ids = {x.get("sample_id") for x in cap}
stc_ids = {x.get("sample_id") for x in stc}
qua_ids = {x.get("sample_id") for x in qua}
print(f"  描述候选 {len(cap_ids)}  结构化候选 {len(stc_ids)}  描述隔离 {len(qua_ids)}")
print(f"  描述 ∩ 结构化: {len(cap_ids & stc_ids)}")

train180 = rd(REPO / "协作" / "02_ClaudeCode_实操" / "学校训练准备_20261002-025848" / "train_180.jsonl")
t180_ids = [x["sample_id"] for x in train180]
manifest = {r["sample_id"]: r for r in rd(REPO / "data" / "manifest.jsonl")}

# 360 候选从 **scale_1440 池**里取：该池已通过隔离校验（train 划分、ground_truth、
# 与 dev/val/test/benchmark/RL 池的 ID 与 lot 均零重叠），且包含旧 train180。
scale1440 = rd(OUT / "scale_1440.jsonl")
pool = [x["sample_id"] for x in scale1440]
t180_set = set(t180_ids)
have_cap_early = {i for i in pool if i in cap_ids}
print(f"  候选池 {len(pool)}（= scale_1440，已隔离）")
print(f"  其中旧 train180: {len(t180_set & set(pool))}")
print(f"  其中已有描述: {len(have_cap_early)}，已有结构化: {len({i for i in pool if i in stc_ids})}")
rng = random.Random(3407)
# 分层：**优先已有描述的**，再补其余，保持类别均衡
by_class = {}
for i in pool:
    by_class.setdefault(manifest[i]["failure_type"], []).append(i)
for c in by_class:
    # 已有描述的排前面（内部仍按 seed 打乱），使 360 候选尽量复用现成标注
    head = [i for i in by_class[c] if i in cap_ids]
    tail = [i for i in by_class[c] if i not in cap_ids]
    rng.shuffle(head); rng.shuffle(tail)
    by_class[c] = head + tail
ordered, idx = [], {c: 0 for c in CLASSES}
while len(ordered) < len(pool):
    for c in CLASSES:
        if idx.get(c, 0) < len(by_class.get(c, [])):
            ordered.append(by_class[c][idx[c]]); idx[c] += 1
sel = ordered[:360]
print(f"  选出 {len(sel)} 条 / {len({manifest[i]['lot_name'] for i in sel})} lot")
cm = Counter(manifest[i]["failure_type"] for i in sel)
print(f"  类别 {dict(cm)}")

rows = []
for k, sid in enumerate(sel, 1):
    r = manifest[sid]
    rows.append({
        "item_id": f"desc_{k:03d}", "sample_id": sid,
        "label": r["failure_type"], "lot_name": r["lot_name"],
        "image": f"data/images/{sid}.png",
        "image_sha256": sha(REPO / "data" / "images" / f"{sid}.png"),
        "has_caption": sid in cap_ids,
        "has_structured": sid in stc_ids,
        "in_quarantine": sid in qua_ids,
    })
p = OUT / "desc_candidates_360.jsonl"
with io.open(p, "w", encoding="utf-8", newline="\n") as f:
    for x in rows:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
have_cap = sum(1 for x in rows if x["has_caption"])
have_stc = sum(1 for x in rows if x["has_structured"])
print(f"  写出 {p}")
print(f"  **已有描述 {have_cap}/360，已有结构化 {have_stc}/360**")
print(f"  待标注描述的 {360-have_cap} 条 —— 目前**不是 360 条都齐**，不冒称已合格")

# ── ZCode 盲清单 ────────────────────────────────────────
print()
print("=" * 78)
print("ZCode 无答案盲清单")
print("=" * 78)
need = [x for x in rows if not x["has_caption"]]
blind = []
for k, x in enumerate(need, 1):
    blind.append({"item_id": f"bz_{k:03d}", "sample_id": x["sample_id"],
                  "image_path": f"/WS/datasets/expand/images/{x['sample_id']}.png",
                  "image_sha256": x["image_sha256"]})
bp = OUT / "zcode_blind_need_caption.jsonl"
with io.open(bp, "w", encoding="utf-8", newline="\n") as f:
    for x in blind:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(f"  需新标注描述的 {len(blind)} 条 -> {bp}")
# 泄漏检查：盲清单只能有那四个键，且不含任何标签信息
bad = [x for x in blind if set(x) != set(BLIND_KEYS)]
has_leak = [k for k in LEAK_KEYS if any(k in x for x in blind)]
print(f"  每行恰好四字段: {not bad}")
print(f"  无答案性字段: {not has_leak}  {has_leak}")
# 首批 20 与分片
first20 = blind[:20]
fp = OUT / "zcode_blind_batch_01.jsonl"
with io.open(fp, "w", encoding="utf-8", newline="\n") as f:
    for x in first20:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(f"  首批 20 -> {fp}  sha256 {sha(fp)[:16]}")

meta = {
    "目标": 360,
    "实得候选": len(rows),
    "已有描述": have_cap, "已有结构化": have_stc, "待标注描述": len(blind),
    "盲清单字段": BLIND_KEYS,
    "题面_sha256": hashlib.sha256(QUESTION.encode()).hexdigest(),
    "边界": "360 条候选已列出；**描述不齐**。" +
            "A/B/C 必须同图、同评测协议；筛选依赖与 GLM+Codex 修改来源需披露。" +
            "本任务不启动描述训练，不编造人工审核。",
    "desc_sha256": sha(p), "blind_sha256": sha(bp),
}
(OUT / "desc_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"\n写出 desc_meta.json")
