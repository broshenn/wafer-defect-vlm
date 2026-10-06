#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提取历史接触过的 ID —— v2：覆盖文件名模式与 item_id 映射。"""
from __future__ import annotations
import io, json, os, re
from collections import defaultdict
from pathlib import Path

REPO = Path("D:/pycode/晶圆图研究")
OUT = Path("D:/pycode/.ssh-tmp/exp2")
WAFER = re.compile(r"(wafer_\d{8}_\d{3})")
DERIVED = re.compile(r"^(wafer_\d{8}_\d{3})__.+$")
ANS_KEYS = ["raw", "answer", "response", "content", "text", "caption_zh", "caption",
            "morphology", "message", "output", "completion", "reply", "defect_class"]
ID_KEYS = ["sample_id", "query_id", "gallery_id", "variant_id", "item_id", "id"]
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv"}
# 明确不是「历史标注接触」的目录（是本项目自己的推理产物，另行处理）
SELF_INFER = ("今晚类别RL_20261006_0325", "RL对照补证_20261006", "并行三种子与独立确认_20261006",
              "学校标签训练_20261006")


def root_of(v):
    m = DERIVED.match(v)
    return m.group(1) if m else v


def main() -> int:
    print("=" * 78)
    print("提取历史接触过的 ID（v2：文件名 + 内容 + item 映射）")
    print("=" * 78)

    # item_id -> sample_id 映射（三教师/十八张用的是 item_001..018）
    mp = {}
    for name in ("dev_18.jsonl", "train_180.jsonl", "smoke_20.jsonl"):
        p = REPO / "协作" / "02_ClaudeCode_实操" / "学校训练准备_20261002-025848" / name
        if p.exists():
            for k, line in enumerate(io.open(p, encoding="utf-8"), 1):
                if line.strip():
                    mp[f"item_{k:03d}"] = json.loads(line)["sample_id"]
    # 十八张/九张批次也各有 item 映射
    for p in sorted(REPO.glob("协作/**/*blind*.json*")) + sorted(REPO.glob("协作/**/*inputs*.json*")):
        try:
            if p.suffix == ".jsonl":
                for line in io.open(p, encoding="utf-8"):
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    if "item_id" in o and "sample_id" in o:
                        mp.setdefault(o["item_id"], o["sample_id"])
            else:
                o = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(o, list):
                    for x in o:
                        if isinstance(x, dict) and "item_id" in x and "sample_id" in x:
                            mp.setdefault(x["item_id"], x["sample_id"])
        except Exception:
            pass
    print(f"item_id → sample_id 映射 {len(mp)} 条")

    per_source = defaultdict(set)
    for root, dirs, files in os.walk(REPO / "协作"):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel_root = str(Path(root).relative_to(REPO))
        is_self_infer = any(k in rel_root for k in SELF_INFER)
        for f in files:
            p = Path(root) / f
            src = rel_root
            # (a) 文件名里的 wafer_ ID
            for m in WAFER.finditer(f):
                if f.endswith("_raw.json") or "raw" in f.lower() or f.endswith("_check.json"):
                    per_source[src].add(m.group(1))
            if not f.endswith((".json", ".jsonl", ".md", ".txt", ".tsv")):
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if "wafer_" not in txt and "item_" not in txt:
                continue
            objs = []
            if f.endswith(".jsonl"):
                for line in txt.splitlines():
                    if line.strip():
                        try:
                            objs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            else:
                try:
                    objs.append(json.loads(txt))
                except json.JSONDecodeError:
                    pass

            def walk(o):
                if isinstance(o, dict):
                    has_ans = any(isinstance(o.get(k), (str, dict, list)) and o.get(k)
                                  for k in ANS_KEYS)
                    for k in ID_KEYS:
                        v = o.get(k)
                        if not isinstance(v, str):
                            continue
                        r = root_of(v)
                        if WAFER.match(r):
                            if has_ans:
                                per_source[src].add(r)
                        elif v in mp and has_ans:
                            per_source[src].add(mp[v])
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
            for o in objs:
                walk(o)

    all_ids = set().union(*per_source.values()) if per_source else set()
    print(f"\n{'来源目录':60s} {'ID 数':>6s}")
    for k in sorted(per_source, key=lambda x: -len(per_source[x])):
        tag = "  [本项目自身推理产物]" if any(s in k for s in SELF_INFER) else ""
        print(f"  {k:58s} {len(per_source[k]):6d}{tag}")
    print(f"\n合计唯一 ID **{len(all_ids)}**")

    hist = {k: sorted(v) for k, v in per_source.items()
            if not any(s in k for s in SELF_INFER)}
    hist_ids = set().union(*hist.values()) if hist else set()
    print(f"其中**历史标注接触**（不含本项目自身推理）: **{len(hist_ids)}**")

    (OUT / "historical_ids_v2.json").write_text(json.dumps({
        "all_unique_ids": len(all_ids), "historical_unique_ids": len(hist_ids),
        "historical_ids": sorted(hist_ids),
        "per_source_historical": hist,
        "per_source_all": {k: sorted(v) for k, v in per_source.items()},
        "item_map_size": len(mp),
        "note": "只取有实际回答内容的记录；仅计划的文件不算接触。",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"写出 {OUT/'historical_ids_v2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
