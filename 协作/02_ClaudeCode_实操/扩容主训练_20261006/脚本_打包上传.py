#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩容 v2：打包需要补传的 PNG，并生成服务端清单。"""
import io, json, hashlib, tarfile, os
from pathlib import Path

REPO = Path("D:/pycode/晶圆图研究")
OUT = Path("D:/pycode/.ssh-tmp/exp2")
IMG = REPO / "data" / "images"
SCHOOL_KNOWN = set()
for p in (REPO / "协作" / "02_ClaudeCode_实操" / "学校训练准备_20261002-025848" / "train_180.jsonl",
          Path("D:/pycode/.ssh-tmp/val90/val90.jsonl"),
          REPO / "协作" / "02_ClaudeCode_实操" / "今晚类别RL_20261006_0325" / "rl_pool90.jsonl"):
    for line in io.open(p, encoding="utf-8"):
        if line.strip():
            SCHOOL_KNOWN.add(json.loads(line)["sample_id"])
print(f"学校已知有的 ID: {len(SCHOOL_KNOWN)}")

rows = [json.loads(l) for l in io.open(OUT / "scale_1440_v2.jsonl", encoding="utf-8") if l.strip()]
need = [x["sample_id"] for x in rows if x["sample_id"] not in SCHOOL_KNOWN]
total = sum((IMG / f"{s}.png").stat().st_size for s in need)
print(f"scale_1440_v2: {len(rows)} 条，需补传 {len(need)} 张，{total} 字节 = {total/2**20:.2f} MiB")

tar = OUT / "expand_v2_images.tar.gz"
with tarfile.open(tar, "w:gz") as t:
    for s in need:
        t.add(IMG / f"{s}.png", arcname=f"{s}.png")
print(f"打包 {tar}  {tar.stat().st_size} 字节  sha256 {hashlib.sha256(tar.read_bytes()).hexdigest()}")

# 服务端清单（真实路径）
srv = OUT / "sft_scale_1440_v2.server.jsonl"
Q = ("这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。"
     "请从 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_full、Random、Scratch、none "
     "中选一个最符合主要图案的类别。只输出一个 JSON 对象，形如 "
     '{"defect_class":"类别"}；无法可靠判断时填 {"defect_class":"unknown"}。')
with io.open(srv, "w", encoding="utf-8", newline="\n") as f:
    for x in rows:
        f.write(json.dumps({"messages": [
            {"role": "user", "content": "<image>" + Q},
            {"role": "assistant", "content": json.dumps({"defect_class": x["label"]},
                                                        ensure_ascii=False,
                                                        separators=(",", ":"))}],
            "images": [f"/WS/datasets/expand_v2/images/{x['sample_id']}.png"],
            "sample_id": x["sample_id"]}, ensure_ascii=False) + "\n")
print(f"服务端 1440 清单 -> {srv}  sha256 {hashlib.sha256(srv.read_bytes()).hexdigest()[:16]}")

# 每档的服务端清单
for lv in (360, 720):
    rr = [json.loads(l) for l in io.open(OUT / f"scale_{lv}_v2.jsonl", encoding="utf-8") if l.strip()]
    p = OUT / f"sft_scale_{lv}_v2.server.jsonl"
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        for x in rr:
            f.write(json.dumps({"messages": [
                {"role": "user", "content": "<image>" + Q},
                {"role": "assistant", "content": json.dumps({"defect_class": x["label"]},
                                                            ensure_ascii=False,
                                                            separators=(",", ":"))}],
                "images": [f"/WS/datasets/expand_v2/images/{x['sample_id']}.png"],
                "sample_id": x["sample_id"]}, ensure_ascii=False) + "\n")
    print(f"服务端 {lv} 清单 -> {p}  {len(rr)} 行")

# 180 的服务端清单（复用已有图，指向已有目录）
r180 = [json.loads(l) for l in io.open(OUT / "scale_180_v2.jsonl", encoding="utf-8") if l.strip()]
p = OUT / "sft_scale_180_v2.server.jsonl"
with io.open(p, "w", encoding="utf-8", newline="\n") as f:
    for x in r180:
        f.write(json.dumps({"messages": [
            {"role": "user", "content": "<image>" + Q},
            {"role": "assistant", "content": json.dumps({"defect_class": x["label"]},
                                                        ensure_ascii=False,
                                                        separators=(",", ":"))}],
            "images": [f"/WS/datasets/images/{x['sample_id']}.png"],
            "sample_id": x["sample_id"]}, ensure_ascii=False) + "\n")
print(f"服务端 180 清单 -> {p}  {len(r180)} 行（指向 datasets/images）")
