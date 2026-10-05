#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GLM 标注准备的独立检查器 —— **不 import 准备器**。

准备器和检查器是两份独立代码：同一个 bug 不会在两处同时犯，
所以"两份得出同一结论"才算证据。

查七件事：
  1. 盲清单每行**只有**四个字段，且不含任何答案性字段
  2. 全局 item_id 唯一且连续；sample_id 唯一
  3. 九个批次并集 == 全量 180，两两不重叠
  4. 首批四个子清单各 5 条、两两不重叠、并集**精确等于**首批 20
  5. 图片存在、SHA256 与清单一致、全局指纹不重复
  6. 与 manifest 对应（标签/lot/split/来源）、train/dev/test/benchmark 的 lot 隔离
  7. zcode_batch_01_inputs.json 里的路径与 hash **实测复核**，且不含标签等泄露字段

全程只读。退出码 0 = 全过。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MANIFEST = REPO / "data" / "manifest.jsonl"
BENCHMARK = REPO / "benchmark"

BLIND_KEYS = {"item_id", "sample_id", "image_path", "image_sha256"}
LEAK_KEYS = {"label", "failure_type", "class", "defect_class", "lot_name", "lot",
             "split", "difficulty", "geometry", "radial_zone", "clock_direction",
             "extent_r", "caption", "morphology", "answer", "prediction"}
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    return bool(ok)


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE))
    a = ap.parse_args()
    d = Path(a.dir).resolve()

    print("=" * 78)
    print(f"GLM 标注准备 · 独立检查：{d}")
    print("=" * 78)

    need = ["blind_inputs_all_180.jsonl", "audit_selected.jsonl", "plan.json",
            "zcode_batch_01_inputs.json"] + \
           [f"batch_{i:02d}_blind.jsonl" for i in range(1, 10)] + \
           [f"batch_01_worker_{w:02d}.jsonl" for w in range(1, 5)]
    missing = [n for n in need if not (d / n).exists()]
    if missing:
        print(f"[无法检查] 缺文件：{missing}")
        return 2

    allb = read_jsonl(d / "blind_inputs_all_180.jsonl")
    batches = {i: read_jsonl(d / f"batch_{i:02d}_blind.jsonl") for i in range(1, 10)}
    workers = {w: read_jsonl(d / f"batch_01_worker_{w:02d}.jsonl") for w in range(1, 5)}
    audit = read_jsonl(d / "audit_selected.jsonl")
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    zcode = json.loads((d / "zcode_batch_01_inputs.json").read_text(encoding="utf-8"))

    # 1 字段
    print("\n[1] 盲清单字段")
    for label, rows in [("全部", allb)] + [(f"batch_{i:02d}", b) for i, b in batches.items()] \
            + [(f"worker_{w:02d}", x) for w, x in workers.items()]:
        check(f"{label} 每行恰好四字段", all(set(r) == BLIND_KEYS for r in rows),
              str(set().union(*[set(r) for r in rows])))
        check(f"{label} 无答案性字段",
              not (set().union(*[set(r) for r in rows]) & LEAK_KEYS))

    # 2 ID
    print("\n[2] ID 唯一性")
    ids = [r["item_id"] for r in allb]
    check("全局 item_id 唯一", len(set(ids)) == len(ids))
    check("item_id 连续 item_001..item_180",
          ids == [f"item_{i:03d}" for i in range(1, 181)])
    sids = [r["sample_id"] for r in allb]
    check("sample_id 唯一", len(set(sids)) == len(sids))

    # 3 批次
    print("\n[3] 批次并集与不重叠")
    bun = [r["item_id"] for i in range(1, 10) for r in batches[i]]
    check("九批每批 20 条", all(len(b) == 20 for b in batches.values()))
    check("九批并集 == 全量（顺序一致）", bun == ids)
    check("九批两两不重叠",
          len({r["sample_id"] for i in range(1, 10) for r in batches[i]}) == 180)

    # 4 子清单
    print("\n[4] 首批子清单")
    check("四份各 5 条", all(len(x) == 5 for x in workers.values()))
    wun = [r["item_id"] for w in range(1, 5) for r in workers[w]]
    check("并集精确等于首批 20（顺序一致）", wun == [r["item_id"] for r in batches[1]])
    check("四份两两不重叠",
          len({r["sample_id"] for w in range(1, 5) for r in workers[w]}) == 20)

    # 5 图片
    print("\n[5] 图片实体")
    shas, probs = [], []
    for r in allb:
        p = Path(r["image_path"])
        if not p.exists():
            probs.append(f"{r['item_id']} 缺图")
            continue
        got = sha_file(p)
        if got != r["image_sha256"]:
            probs.append(f"{r['item_id']} 指纹不符")
        shas.append(got)
    check("180 张图全部存在且指纹一致", not probs, str(probs[:3]))
    check("全局图片指纹不重复", len(set(shas)) == len(shas))
    check("image_path 不含类别名",
          not any(c in Path(r["image_path"]).stem for r in allb for c in CLASSES))

    # 6 manifest 与隔离
    print("\n[6] manifest 对应与 lot 隔离")
    man, mline = {}, {}
    with open(MANIFEST, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                o = json.loads(line)
                man[o["sample_id"]] = o
                mline[o["sample_id"]] = ln
    bad = []
    for r in allb:
        m = man.get(r["sample_id"])
        if m is None:
            bad.append(f"{r['item_id']} 不在 manifest")
            continue
        if m["label_source"] != "ground_truth":
            bad.append(f"{r['item_id']} 非 ground_truth")
    check("180 个样本都在 manifest 且来源为 ground_truth", not bad, str(bad[:3]))
    check("audit 与盲清单逐项对齐",
          [x["sample_id"] for x in audit] == sids)
    bad2 = [x["item_id"] for x in audit
            if man[x["sample_id"]]["failure_type"] != x["label"]
            or man[x["sample_id"]]["lot_name"] != x["lot_name"]
            or mline[x["sample_id"]] != x["manifest_line"]]
    check("audit 的标签/lot/行号与 manifest 一致", not bad2, str(bad2[:3]))

    lots = {s: {man[r["sample_id"]]["lot_name"] for r in allb
                if man[r["sample_id"]]["split"] == s} for s in ("train", "val", "test")}
    bm_ids = set()
    for p in sorted(BENCHMARK.glob("*.jsonl")):
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in ("sample_id", "query_id", "gallery_id"):
                v = o.get(k) if isinstance(o, dict) else None
                if isinstance(v, str) and v.startswith("wafer_"):
                    bm_ids.add(v)
    bm_lots = {man[i]["lot_name"] for i in bm_ids if i in man}
    tlots = {man[r["sample_id"]]["lot_name"] for r in allb}
    for nm, s in (("val", lots["val"]), ("test", lots["test"]), ("benchmark", bm_lots)):
        check(f"本批 lot 与 {nm} 无交集", not (tlots & s), str(sorted(tlots & s)[:3]))
    check("audit 记录的 lot 与清单一致",
          all(man[x["sample_id"]]["lot_name"] == x["lot_name"] for x in audit))

    # 7 zcode 索引
    print("\n[7] ZCode 输入索引")
    check("batch_id / total_limit / max_concurrency",
          zcode.get("batch_id") == "batch_01" and zcode.get("total_limit") == 20
          and zcode.get("max_concurrency") == 4)
    check("索引里不含标签/lot/split 等泄露字段",
          not (set(zcode) & LEAK_KEYS) and
          not any(k in json.dumps(zcode, ensure_ascii=False) for k in ('"label"', '"lot_name"', '"split"')))
    for key in ("batch_manifest", "prompt", "checker"):
        e = zcode.get(key, {})
        p = Path(e.get("path", ""))
        check(f"{key} 路径存在", p.exists(), str(p))
        if p.exists():
            check(f"{key} hash 实测一致", sha_file(p) == e.get("sha256"),
                  f"实测 {sha_file(p)[:16]}… 记录 {str(e.get('sha256'))[:16]}…")
    check("workers 恰好四条", len(zcode.get("workers", [])) == 4)
    wsum = 0
    for w in zcode.get("workers", []):
        p = Path(w.get("input_path", ""))
        ok = p.exists()
        check(f"{w.get('worker_id')} 路径存在且 hash 一致",
              ok and sha_file(p) == w.get("input_sha256"),
              str(p))
        check(f"{w.get('worker_id')} count 与文件行数一致",
              ok and len(read_jsonl(p)) == w.get("count"))
        wsum += w.get("count", 0)
    check("四 worker 的 count 合计 = 20", wsum == 20)
    # 索引指向的 worker 文件必须与目录里的同名文件一致
    for w in zcode.get("workers", []):
        p = Path(w.get("input_path", ""))
        if p.exists():
            i = int(w["worker_id"].split("_")[1])
            check(f"{w['worker_id']} 内容与目录内同名文件一致",
                  read_jsonl(p) == workers[i])

    print()
    fails = [x for x in results if not x[1]]
    for n, ok, det in results:
        if not ok:
            print(f"  FAIL  {n}   {det}")
    print(f"\n通过 {len(results) - len(fails)}/{len(results)}")
    print("结论：" + ("全部通过" if not fails else f"{len(fails)} 项未通过"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
