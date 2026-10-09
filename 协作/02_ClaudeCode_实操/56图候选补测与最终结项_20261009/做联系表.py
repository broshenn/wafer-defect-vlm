#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 56 张候选图拼成 7 张联系表（每张 8 图，2 列 × 4 行），供执行会话逐张阅图。

只加编号条（item_id），**不改动像素**；拼完对每张原图 sha 复核一次。
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from PIL import Image, ImageDraw

D = Path(__file__).resolve().parent
SRC = D / "上传包" / "图"
OUT = D / "联系表"
OUT.mkdir(exist_ok=True)
frozen = json.loads((D / "冻结清单.json").read_text(encoding="utf-8"))
rows = frozen["逐图"]
assert len(rows) == 56

TILE, BAR, GAP = 448, 34, 8
PER = 8
for s in range(7):
    chunk = rows[s * PER:(s + 1) * PER]
    W = TILE * 2 + GAP * 3
    H = (TILE + BAR) * 4 + GAP * 5
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for k, r in enumerate(chunk):
        p = SRC / f"{r['sample_id']}.png"
        b = p.read_bytes()
        got = hashlib.sha256(b).hexdigest()
        assert got == r["image_sha256"], f"{r['sample_id']} sha 不符"
        im = Image.open(p).convert("RGB")
        assert im.size == (448, 448), f"{r['sample_id']} 尺寸 {im.size}"
        col, row = k % 2, k // 2
        x = GAP + col * (TILE + GAP)
        y = GAP + row * (TILE + BAR + GAP)
        dr.rectangle([x, y, x + TILE, y + BAR - 4], fill=(0, 0, 0))
        dr.text((x + 8, y + 8), f"{r['item_id']}   ({s+1}/7)", fill=(255, 255, 0))
        sheet.paste(im, (x, y + BAR))
    f = OUT / f"联系表_{s+1}.png"
    sheet.save(f)
    print(f"{f.name}  {sheet.size}  {len(chunk)} 图")
print("完成：7 张联系表，56 图 sha 全部复核一致")
