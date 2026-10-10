#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 图片落位：把 888 张唯一图打包并生成逐张 SHA 清单（CPU，只读本地数据）。

产物：
  out_v3/imgs_v3.tgz        888 张 PNG 的打包（约 1.6 MiB）
  out_v3/img_manifest.json  逐张 sha256 + 服务端目标路径
说明：只读 `data/images/`，不改任何原图与 JSONL。
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

CODEX = Path("D:/pycode/晶圆图研究/协作/01_Codex_指挥/无卡CPU整备_20261010")
REPO = Path("D:/pycode/晶圆图研究")
OUT = Path(__file__).resolve().parent / "out_v3"
REMOTE_DIR = "/root/autodl-tmp/ws/v3run/图"

need: set[str] = set()
for f in ("data_v3/adapt512_sft.jsonl", "data_v3/group256_sft4.jsonl",
          "data_v3/group256_grpo.jsonl", "data_v3/confirmation120_requests.jsonl"):
    for line in (CODEX / f).open(encoding="utf-8"):
        if line.strip():
            for p in (json.loads(line).get("images") or []):
                need.add(Path(p).name)

OUT.mkdir(exist_ok=True)
missing, manifest = [], {}
for name in sorted(need):
    src = REPO / "data" / "images" / name
    if not src.exists():
        missing.append(name)
        continue
    manifest[name] = {"sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                      "bytes": src.stat().st_size,
                      "remote": f"{REMOTE_DIR}/{name}"}
if missing:
    raise SystemExit(f"本地缺图 {len(missing)} 张：{missing[:5]}")

tgz = OUT / "imgs_v3.tgz"
with tarfile.open(tgz, "w:gz") as tf:
    for name in sorted(manifest):
        tf.add(REPO / "data" / "images" / name, arcname=f"图/{name}")

(OUT / "img_manifest.json").write_text(json.dumps({
    "count": len(manifest),
    "remote_dir": REMOTE_DIR,
    "tar_bytes": tgz.stat().st_size,
    "tar_sha256": hashlib.sha256(tgz.read_bytes()).hexdigest(),
    "images": manifest,
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"打包 {len(manifest)} 张，{tgz.stat().st_size/2**20:.3f} MiB → {tgz}")
