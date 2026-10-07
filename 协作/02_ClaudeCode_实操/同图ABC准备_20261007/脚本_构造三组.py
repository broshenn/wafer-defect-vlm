#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同图 A/B/C 整合 第 3 步：构造三组监督目标 + 新题面 + 新解析器 + 长度统计。"""
from __future__ import annotations
import hashlib, io, json, re
from collections import Counter
from pathlib import Path

REPO = Path("D:/pycode/晶圆图研究")
OUT = Path("D:/pycode/.ssh-tmp/abc")
man = {r["sample_id"]: r for r in
       (json.loads(l) for l in io.open(REPO / "data" / "manifest.jsonl", encoding="utf-8") if l.strip())}
rows = [json.loads(l) for l in io.open(OUT / "items356.jsonl", encoding="utf-8") if l.strip()]
OLD158 = REPO / "协作" / "01_Codex_指挥" / "训练数据就绪_20261003" / "caption_train_158.jsonl"
QUAR = REPO / "协作" / "01_Codex_指挥" / "训练数据就绪_20261003" / "caption_quarantine_22.jsonl"
GLM198 = REPO / "协作" / "03_ZCode_标注" / "扩容198_20261006_165855"
CONT49 = REPO / "协作" / "03_ZCode_标注" / "接续49_20261006_20261007-000123"
GPT91 = REPO / "协作" / "01_Codex_指挥" / "GPT补标91_20261006"
REV49 = REPO / "协作" / "01_Codex_指挥" / "验收_20261007_接续49"
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]


def rd(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ── C 描述：三种来源，字段名各不相同 ────────────────────
old158 = {}
for x in rd(OLD158):
    msgs = x.get("messages") or []
    t = msgs[1]["content"] if len(msgs) > 1 else ""
    old158[x["sample_id"]] = t

blind = rd(REPO / "协作" / "02_ClaudeCode_实操" / "ZCode标注准备_20261006" / "zcode_blind_new.jsonl")
zc2sid = {b["item_id"]: b["sample_id"] for b in blind}
raws = {}
for root, tag in ((GLM198, "GLM"), (CONT49, "GLM_接续"), (GPT91, "GPT")):
    for p in sorted(root.rglob("*_raw.json")):
        item = p.name.replace("_raw.json", "")
        sid = zc2sid.get(item)
        if sid:
            raws[sid] = {"path": p, "tag": tag,
                         "obj": json.loads(p.read_text(encoding="utf-8"))}
rev49 = {x["sample_id"]: x for x in rd(REV49 / "模型辅助候选49.jsonl")}
print(f"旧158 {len(old158)} · 新原答 {len(raws)} · 49复核 {len(rev49)}")

# ── 新题面（三组相同输入）───────────────────────────────
PROMPT = (
    "这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。\n"
    "请判断图中**主要缺陷图案**的类别，从 Center、Donut、Edge_Loc、Edge_Ring、Loc、"
    "Near_full、Random、Scratch、none 中选一个。\n"
    "只输出一个 JSON 对象，不写推理过程或 Markdown。\n"
    "`defect_class` 为必填，取值为上述九类之一；无法可靠判断时填 \"unknown\"。\n"
    "可以在同一个对象里附加你**从图中实际看到**的判读依据字段；没有把握就不要写。"
)
print(f"\n统一题面 {len(PROMPT)} 字符")
print(PROMPT)

# ── B 的几何字段（定义明确、源码核过）──────────────────
GEOM_DOC = {
    "radial_zone": {
        "定义": "主要缺陷相对晶圆半径所处区域",
        "取值": ["center", "middle", "edge", "global", "none", "unknown"],
        "来源": "manifest.features.radial_zone",
        "分母/坐标": "以拟合圆心的归一化半径 r/R 划分；不涉及渲染像素",
    },
    "defect_ratio": {
        "定义": "失效 die 占**晶圆内有效 die** 的比例",
        "公式": "defect_count / valid_count（不是整个矩阵面积）",
        "来源": "manifest.features.defect_count / valid_count",
        "单位": "无量纲，保留 4 位小数",
    },
    "extent_r": {"值": None, "原因": "尺寸无可靠 gold，本轮不下发数值"},
    "clock_direction": {"值": None, "原因": "扇区边界数值敏感，未修复前不下发"},
}
print("\nB 的几何字段：")
for k, v in GEOM_DOC.items():
    print(f"  {k}: {v.get('定义') or v.get('原因')}")

# ── 构造三组 ────────────────────────────────────────────
def geom_of(sid):
    f = (man[sid].get("features") or {})
    if f.get("status") != "ok":
        return {"radial_zone": None, "defect_ratio": None,
                "extent_r": None, "clock_direction": None}, f.get("status")
    vc = f.get("valid_count") or 0
    dc = f.get("defect_count") or 0
    return ({"radial_zone": f.get("radial_zone"),
             "defect_ratio": round(dc / vc, 4) if vc else None,
             "extent_r": None, "clock_direction": None}, "ok")


def desc_of(sid):
    if sid in old158 and old158[sid].strip():
        return old158[sid].strip(), "旧158描述候选", False
    r = raws.get(sid)
    if r:
        if sid in rev49:
            a = rev49[sid].get("answer") or {}
            t = (a.get("caption_zh") or "").strip()
            if t:
                return t, f"{r['tag']}+Codex复核副本", True
        a = r["obj"] or {}
        return (a.get("caption_zh") or "").strip(), f"{r['tag']}原始回答", False
    return "", "无", False


A, B, C, stat = [], [], [], []
for i, x in enumerate(rows, 1):
    sid = x["sample_id"]
    lab = man[sid]["failure_type"]
    g, gstat = geom_of(sid)
    d, dsrc, drev = desc_of(sid)
    img = f"/WS/datasets/expand_v2/images/{sid}.png"
    base = {"item_id": f"abc_{i:03d}", "sample_id": sid, "order": i,
            "label": lab, "lot_name": man[sid]["lot_name"]}
    a_t = {"defect_class": lab}
    b_t = {"defect_class": lab, "geometry": g}
    c_t = {"defect_class": lab, "caption_zh": d} if d else {"defect_class": lab}
    A.append({**base, "messages": [{"role": "user", "content": "<image>" + PROMPT},
                                   {"role": "assistant",
                                    "content": json.dumps(a_t, ensure_ascii=False,
                                                          separators=(",", ":"))}],
              "images": [img]})
    B.append({**base, "messages": [{"role": "user", "content": "<image>" + PROMPT},
                                   {"role": "assistant",
                                    "content": json.dumps(b_t, ensure_ascii=False,
                                                          separators=(",", ":"))}],
              "images": [img]})
    C.append({**base, "messages": [{"role": "user", "content": "<image>" + PROMPT},
                                   {"role": "assistant",
                                    "content": json.dumps(c_t, ensure_ascii=False,
                                                          separators=(",", ":"))}],
              "images": [img]})
    stat.append({"order": i, "sample_id": sid, "label": lab, "desc_source": dsrc,
                 "desc_revised_copy": drev, "desc_len": len(d), "geom_status": gstat,
                 "radial_zone": g["radial_zone"], "defect_ratio": g["defect_ratio"]})

for name, data in (("A", A), ("B", B), ("C", C)):
    p = OUT / f"abc_{name}.jsonl"
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {name}: {len(data)} 行  sha256 {sha(p)[:16]}")

ds = Counter(s["desc_source"] for s in stat)
print(f"\nC 描述来源分账: {dict(ds)}")
nod = [s for s in stat if s["desc_len"] == 0]
print(f"  **无描述 {len(nod)} 条** —— C 组这些只有类别，与 A 相同")
if nod:
    print(f"    例如 {[s['sample_id'] for s in nod[:5]]}")

# ── 目标长度统计 ────────────────────────────────────────
import statistics
print("\n目标长度（字符 / 粗估 token≈字符数/1.6 只作相对比较）:")
for name, data in (("A", A), ("B", B), ("C", C)):
    L = [len(r["messages"][1]["content"]) for r in data]
    sup = [len(r["messages"][1]["content"]) - len(json.dumps({"defect_class": r["label"]},
           ensure_ascii=False, separators=(",", ":"))) for r in data]
    print(f"  {name}: 目标 中位 {int(statistics.median(L))} 字符 / 最大 {max(L)}；"
          f"**超出类别以外的监督字符** 中位 {int(statistics.median(sup))}")

with io.open(OUT / "abc_stat.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for s in stat:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
(OUT / "prompt_parser.json").write_text(json.dumps({
    "统一题面": PROMPT, "题面_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
    "几何字段定义": GEOM_DOC,
    "解析规则": {
        "主答案": "只从**唯一** defect_class 字段取类别",
        "允许": "对象可含附加字段（geometry / caption_zh / morphology / radial_zone 等）",
        "拒绝": ["重复键", "截断", "多对象", "尾随文本", "非九类且非 unknown 的取值"],
        "重要": "**不能**复用「对象只能有一个字段」的旧严格解析器 —— 那会把 C 的附加描述全部判错",
    },
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"\n写出 abc_A/B/C.jsonl、abc_stat.jsonl、prompt_parser.json")
