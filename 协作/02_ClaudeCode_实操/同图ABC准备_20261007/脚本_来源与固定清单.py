#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同图 A/B/C 整合 第 2 步：固定 356 条 ID/顺序，构造 A/B/C 三组。"""
from __future__ import annotations
import hashlib, io, json, re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("D:/pycode/晶圆图研究")
OUT = Path("D:/pycode/.ssh-tmp/abc")
COMMON = REPO / "协作" / "02_ClaudeCode_实操" / "ZCode标注准备_20261006" / "共同图片清单.jsonl"
BLIND = REPO / "协作" / "02_ClaudeCode_实操" / "ZCode标注准备_20261006" / "zcode_blind_new.jsonl"
OLD158 = REPO / "协作" / "01_Codex_指挥" / "训练数据就绪_20261003" / "caption_train_158.jsonl"
QUAR = REPO / "协作" / "01_Codex_指挥" / "训练数据就绪_20261003" / "caption_quarantine_22.jsonl"
GLM198 = REPO / "协作" / "03_ZCode_标注" / "扩容198_20261006_165855"
CONT49 = REPO / "协作" / "03_ZCode_标注" / "接续49_20261006_20261007-000123"
GPT91 = REPO / "协作" / "01_Codex_指挥" / "GPT补标91_20261006"
REV49 = REPO / "协作" / "01_Codex_指挥" / "验收_20261007_接续49"
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
SEED = 3407


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rd(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


man = {r["sample_id"]: r for r in rd(REPO / "data" / "manifest.jsonl")}
common = rd(COMMON)
blind = rd(BLIND)
quar = {x["sample_id"] for x in rd(QUAR)}
c158 = {x["sample_id"]: x for x in rd(OLD158)}

# item_id(zc_) -> sample_id
zc2sid = {b["item_id"]: b["sample_id"] for b in blind}
sid2zc = {v: k for k, v in zc2sid.items()}

# ── 新原答收集 ──────────────────────────────────────────
def collect(root: Path, tag: str):
    out = {}
    for p in sorted(root.rglob("*_raw.json")):
        item = p.name.replace("_raw.json", "")
        sid = zc2sid.get(item)
        if sid is None:
            ent = p.with_name(f"{item}_entry.json")
            if ent.exists():
                try:
                    sid = json.loads(ent.read_text(encoding="utf-8")).get("sample_id")
                except Exception:
                    sid = None
        try:
            ans = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            ans = None
        out[item] = {"sample_id": sid, "answer": ans, "raw_path": str(p.relative_to(REPO)),
                     "source": tag}
    return out


new = {}
for root, tag in ((GLM198, "GLM"), (CONT49, "GLM_接续"), (GPT91, "GPT")):
    for k, v in collect(root, tag).items():
        if k in new:
            print(f"  !! item 重复 {k}: {new[k]['source']} vs {tag}")
        new[k] = v
print(f"新原答 {len(new)} 条（GLM/GPT 混合）")
by_sid = {}
for k, v in new.items():
    if v["sample_id"]:
        by_sid.setdefault(v["sample_id"], []).append(v)
print(f"  映射到 sample_id 的 {len(by_sid)} 个")

# 49 复核副本（仅作修订来源，不当原始回答）
rev = {x["sample_id"]: x for x in rd(REV49 / "模型辅助候选49.jsonl")}
print(f"49 复核副本 {len(rev)} 条")

# ── 固定 356 条与顺序 ───────────────────────────────────
usable = [x for x in common if x["sample_id"] not in quar]
print(f"\n可构造 {len(usable)} 条（隔离 {len(common)-len(usable)} 条保持隔离）")
# 固定顺序：保持共同清单原顺序（该清单本身已是 seed 3407 的九类交错序）
rows = []
for x in usable:
    sid = x["sample_id"]
    m = man[sid]
    rows.append({"order": len(rows) + 1, "item_id": x["item_id"], "sample_id": sid,
                 "label": m["failure_type"], "lot_name": m["lot_name"],
                 "image": f"data/images/{sid}.png", "image_sha256": x["image_sha256"],
                 "has_old_caption": sid in c158, "in_class1440": True})
print(f"固定 {len(rows)} 条 / {len({r['lot_name'] for r in rows})} lot")
print(f"  类别 {dict(sorted(Counter(r['label'] for r in rows).items()))}")

# ── C 的来源分账 ────────────────────────────────────────
def desc_of(sid):
    """返回 (描述文本, 来源标签, 是否复核修订)."""
    if sid in c158:
        o = c158[sid]
        t = o.get("caption_zh") or o.get("caption") or ""
        return t, "旧158描述候选", False
    lst = by_sid.get(sid, [])
    if lst:
        v = lst[0]
        if sid in rev:
            a = rev[sid].get("answer") or {}
            t = a.get("caption_zh") or a.get("caption") or ""
            if t:
                return t, f"{v['source']}+Codex复核副本", True
        a = v["answer"] or {}
        return (a.get("caption_zh") or ""), f"{v['source']}原始回答", False
    return "", "无", False


src_count = Counter()
for r in rows:
    t, s, revised = desc_of(r["sample_id"])
    r["desc_source"] = s
    r["desc_revised_copy"] = revised
    r["desc_len"] = len(t)
    src_count[s] += 1
print(f"\nC 描述来源分账:")
for k, v in src_count.most_common():
    print(f"  {k:24s} {v:4d}")
no_desc = [r for r in rows if r["desc_len"] == 0]
print(f"  **无描述 {len(no_desc)} 条**")
if no_desc:
    print(f"    {[r['sample_id'] for r in no_desc[:5]]}")

meta = {
    "n": len(rows), "lots": len({r["lot_name"] for r in rows}),
    "classes": dict(sorted(Counter(r["label"] for r in rows).items())),
    "order": "沿用共同清单原顺序（其本身为 seed 3407 的九类交错序）",
    "quarantined_kept_out": sorted(quar & {x["sample_id"] for x in common}),
    "desc_sources": dict(src_count), "n_no_desc": len(no_desc),
    "note": "C 来源是 GLM/GPT/Codex 复核混合，**不能叫纯 GLM 教师或人工专家**。",
}
(OUT / "abc_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8", newline="\n")
with io.open(OUT / "items356.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\n写出 items356.jsonl（{len(rows)} 条）")
