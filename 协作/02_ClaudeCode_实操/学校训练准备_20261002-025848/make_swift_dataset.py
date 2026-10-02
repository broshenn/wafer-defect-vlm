#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把分类清单转成 ms-swift 的 `messages + images` 训练格式。

输入（已在服务器上）：train_180.server.jsonl / smoke_20.server.jsonl / dev_18.server.jsonl
每行: {sample_id, image_path, label, ...}

输出（ms-swift 格式）：
  {"messages":[{"role":"user","content":"<image>+统一分类问题"},
               {"role":"assistant","content":"{\"defect_class\":\"类别\"}"}],
   "images":["<服务端绝对路径>"]}

设计约束（照任务书 §3）：
  * 输入**只给图片和统一分类问题**，不含原类别、不含答案提示；
  * 回答**只含可信类别**，就是 `{"defect_class":"..."}` ——
    不伪造 morphology / 方向 / 尺寸 / Caption / 专家审核；
  * `label_source` 必须是 ground_truth（脚本会拒收其它来源）。

用法（本地，产出带服务端路径的文件）：
    python make_swift_dataset.py --remote-root "$HOME/data/datasets"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]

# 统一分类题面 —— **必须列出九类候选**。
# 实测：不列候选集时基座 18/18 全输出兜底值 unknown（题目本身不成立）。
QUESTION = ("这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。"
            "请从 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_full、Random、Scratch、none "
            "中选一个最符合主要图案的类别。只输出一个 JSON 对象，形如 "
            '{"defect_class":"类别"}；无法可靠判断时填 {"defect_class":"unknown"}。')


def convert(src: Path, dst: Path, remote_root: str, require_split: str | None) -> dict:
    n_in = n_out = 0
    rejected = {"label_source": 0, "bad_class": 0, "split": 0}
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8", newline="\n") as fo:
        for line in fi:
            if not line.strip():
                continue
            n_in += 1
            r = json.loads(line)
            if r.get("label_source") != "ground_truth":
                rejected["label_source"] += 1
                continue
            if require_split and r.get("split") != require_split:
                rejected["split"] += 1
                continue
            label = r["label"]
            if label not in CLASSES:
                rejected["bad_class"] += 1
                continue
            name = Path(r["image_path"]).name
            rec = {
                "messages": [
                    {"role": "user", "content": "<image>" + QUESTION},
                    {"role": "assistant",
                     "content": json.dumps({"defect_class": label}, ensure_ascii=False,
                                           separators=(",", ":"))},
                ],
                "images": [f"{remote_root.rstrip('/')}/images/{name}"],
                "sample_id": r["sample_id"],      # 自定义列，便于追溯；ms-swift 会忽略
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    return {"in": n_in, "out": n_out, "rejected": rejected}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcdir", default=str(HERE))
    ap.add_argument("--outdir", default=str(HERE / "swift_fmt"))
    ap.add_argument("--remote-root", required=True,
                    help="服务端 images 的父目录；必须显式传入，不设默认值。"
                         "远端真实路径不进版本库；本脚本设计为在服务器上运行，"
                         "输出留在服务器，不回传仓库。")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    jobs = [("train_180.jsonl", "train_180.swift.jsonl", "train"),
            ("smoke_20.jsonl", "smoke_20.swift.jsonl", "train"),
            ("dev_18.jsonl", "dev_18.swift.jsonl", "val")]

    for src_name, dst_name, split in jobs:
        src = Path(a.srcdir) / src_name
        if not src.exists():
            print(f"  跳过 {src_name}（不存在）")
            continue
        if a.dry_run:
            print(f"  [dry-run] {src_name} -> {dst_name}（split={split}）")
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        st = convert(src, outdir / dst_name, a.remote_root, split)
        flag = "" if st["out"] == st["in"] else f"  ⚠ 拒收 {st['rejected']}"
        print(f"  {src_name} -> {dst_name}: {st['out']}/{st['in']} 行{flag}")

    if a.dry_run:
        print("\n[dry-run] 未写文件")
        return 0

    print("\n=== 抽查一行 ===")
    first = next(iter(outdir.glob("*.jsonl")), None)
    if first:
        line = open(first, encoding="utf-8").readline()
        o = json.loads(line)
        print(" 文件:", first.name)
        print(" messages[user] 前 120 字:", o["messages"][0]["content"][:120])
        print(" messages[assistant]:", o["messages"][1]["content"])
        print(" images:", o["images"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
