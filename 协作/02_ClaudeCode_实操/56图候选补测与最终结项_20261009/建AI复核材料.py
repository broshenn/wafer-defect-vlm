#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 Phase D 的**匿名复核材料**：56 图 × 3 模型的同题原答，逐图随机编号。

- 每张图的三个回答**独立洗牌**（同一模型在不同图上不固定同一个字母），
  随机种子固定 `3407`，映射写入 `AI复核_盲号对照_key.json`，**复核时看不到映射**。
- 材料里只放：图号、随机字母、七字段原答（逐字）。
- **复核者（本执行会话）亲自跑过这三个模型**，所以这不是严格盲评 —— 此限制
  写进材料与结论，不冒充独立盲审。
"""
from __future__ import annotations
import json, random
from pathlib import Path

D = Path(__file__).resolve().parent
OUT = D / "AI复核材料"
OUT.mkdir(exist_ok=True)
SEED = 3407
TAGS = ["Base", "L-N3072-3407", "D-N3072-3407"]

frozen = json.loads((D / "冻结清单.json").read_text(encoding="utf-8"))
order = [r["sample_id"] for r in frozen["逐图"]]
raw = {t: {json.loads(l)["sample_id"]: json.loads(l)["raw"]
           for l in (D / "out" / f"{t}_raw.jsonl").open(encoding="utf-8") if l.strip()}
       for t in TAGS}

rnd = random.Random(SEED)
key, material = {}, {}
for sid in order:
    labs = ["甲", "乙", "丙"]
    rnd.shuffle(labs)
    m = {}
    for t, lab in zip(TAGS, labs):
        m[lab] = t
    key[sid] = m
    material[sid] = {lab: raw[t][sid] for lab, t in m.items()}

(D / "AI复核_盲号对照_key.json").write_text(
    json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# 分 7 批，每批 8 图，便于逐批阅图评分
lines = []
for b in range(7):
    chunk = order[b * 8:(b + 1) * 8]
    sec = [f"# 批次 {b+1}/7", ""]
    for sid in chunk:
        item = next(r["item_id"] for r in frozen["逐图"] if r["sample_id"] == sid)
        sec.append(f"## {item}  ({sid})")
        for lab in ["甲", "乙", "丙"]:
            sec.append(f"**{lab}**：`{material[sid][lab]}`")
        sec.append("")
    (OUT / f"批次{b+1}.md").write_text("\n".join(sec) + "\n", encoding="utf-8")
    lines.append(f"批次{b+1}.md  {len(chunk)} 图")
print("\n".join(lines))
print(f"写出 AI复核_盲号对照_key.json（seed={SEED}）")
