#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 数据落位：把 JSONL 里的相对图片路径改写为服务端绝对路径。

**只改 `images` 字段，不改 messages / 题面 / 目标文本 / 参考。**
改写前后各记一次 SHA，逐文件列在 `out_v3/数据改写.json`。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CODEX = Path("D:/pycode/晶圆图研究/协作/01_Codex_指挥/无卡CPU整备_20261010")
OUT = Path(__file__).resolve().parent / "out_v3"
REMOTE_DIR = "/root/autodl-tmp/ws/v3run/图"

FILES = ["adapt512_sft.jsonl", "group256_sft4.jsonl", "group256_grpo.jsonl",
         "confirmation120_requests.jsonl"]

OUT.mkdir(exist_ok=True)
report = {}
for name in FILES:
    src = CODEX / "data_v3" / name
    raw = src.read_bytes()
    rows = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        old = list(r.get("images") or [])
        r["images"] = [f"{REMOTE_DIR}/{Path(p).name}" for p in old]
        rows.append((old, r))
    dst = OUT / name
    with dst.open("w", encoding="utf-8", newline="\n") as fh:
        for _, r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 校验：除 images 外逐字段与原文相同
    orig = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    rewritten = [r for _, r in rows]
    same = all({k: v for k, v in o.items() if k != "images"} ==
               {k: v for k, v in n.items() if k != "images"}
               for o, n in zip(orig, rewritten)) and len(orig) == len(rewritten)
    report[name] = {
        "行数": len(rows),
        "改写前sha256": hashlib.sha256(raw).hexdigest(),
        "改写后sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        "除images外逐字段完全相同": bool(same),
        "images样例": rows[0][1]["images"],
    }
    if not same:
        raise SystemExit(f"{name} 除了 images 之外还有字段被改动了 —— 停止")

(OUT / "数据改写.json").write_text(json.dumps({
    "remote_image_dir": REMOTE_DIR,
    "只改images字段": True,
    "files": report,
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: {"行数": v["行数"], "字段一致": v["除images外逐字段完全相同"]}
                  for k, v in report.items()}, ensure_ascii=False, indent=1))
