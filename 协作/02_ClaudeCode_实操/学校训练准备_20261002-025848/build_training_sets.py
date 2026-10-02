#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标签监督 SFT-A 技术基线的数据集构建（本地，只读公开 WM811K 派生数据）。

产出三份清单：
  train_180.jsonl   九类各 20 张，共 180 张，来自 `split=train`
  dev_18.jsonl      九类各 2 张，共 18 张，来自 `split=val`，与训练/冻结 benchmark 的 lot 隔离
  smoke_20.jsonl    九类各 2 张 + 固定种子补 2 张 = 20 张，**是 train_180 的子集**

硬约束（任一不满足即**失败退出**，不静默放宽）：
  1. 只用 `label_source=ground_truth` 的原始 `failure_type`，不读预测标签；
  2. train / val / test / benchmark 四者的 lot 交集必须为空；
  3. 排除正在进行的 ZCode 十八张开发样本**及其 lot**；
  4. 每张图唯一、指纹互不重复。

用法：
    python build_training_sets.py --dry-run     # 只算不写（默认）
    python build_training_sets.py --write
幂等：输出已存在时报错退出，不覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

SEED = 3407
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
TRAIN_PER_CLASS = 20
DEV_PER_CLASS = 2
SMOKE_PER_CLASS = 2
SMOKE_EXTRA = 2

MANIFEST = REPO / "data" / "manifest.jsonl"
IMAGES = REPO / "data" / "images"
BENCHMARK = REPO / "benchmark"
# ZCode 正在标注的十八张（**只取 ID，不读它的回答**）
ZCODE_BLIND = (REPO / "协作" / "02_ClaudeCode_实操" / "口径修正与准备_20261002-021207"
               / "blind_inputs.jsonl")

# 固定分类题面。**不含原类别、不含答案提示。** 与 GLM 七字段试标题面不同，
# 所以本训练**不能**冒称做过 GLM 同题能力对照。
QUESTION = ("这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。"
            "请判断主要图案属于哪一类，只输出一个 JSON 对象，形如 "
            '{"defect_class":"类别"}；无法可靠判断时填 {"defect_class":"unknown"}。')


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_manifest() -> list[dict]:
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                r["__line__"] = i
                rows.append(r)
    return rows


def benchmark_ids() -> set[str]:
    ids = set()
    for p in sorted(BENCHMARK.glob("*.jsonl")):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(o, dict):
                for k in ("sample_id", "query_id", "gallery_id"):
                    v = o.get(k)
                    if isinstance(v, str) and v.startswith("wafer_"):
                        ids.add(v)
    return ids


def draw(pool: list[dict], n: int, used_lots: set[str], rng: random.Random) -> list[dict]:
    """按 sample_id 升序，在未占用 lot 中逐个抽 n 张；抽中的 lot 立即占用。"""
    picked = []
    for _ in range(n):
        cands = [r for r in pool if r["lot_name"] not in used_lots]
        if not cands:
            return picked
        r = cands[rng.randrange(len(cands))]
        used_lots.add(r["lot_name"])
        picked.append(r)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    if a.write and a.dry_run:
        print("[停止] --write 与 --dry-run 互斥")
        return 2
    out_dir = Path(a.out).resolve()

    print("=" * 78)
    print("SFT-A 标签监督基线：数据集构建")
    print("=" * 78)

    rows = load_manifest()
    by_id = {r["sample_id"]: r for r in rows}
    man_sha = sha256_file(MANIFEST)
    print(f"\n[1] manifest {len(rows)} 行  SHA256 {man_sha}")
    bad_source = [r["sample_id"] for r in rows if r["label_source"] != "ground_truth"]
    print(f"    非 ground_truth 的行: {len(bad_source)}（必须为 0）")
    if bad_source:
        print("[停止] 存在非真值标签来源的行")
        return 3

    # ── 排除集合 ───────────────────────────────────────
    zcode_ids = set()
    if ZCODE_BLIND.exists():
        zcode_ids = {json.loads(l)["sample_id"]
                     for l in ZCODE_BLIND.read_text(encoding="utf-8").splitlines() if l.strip()}
    zcode_lots = {by_id[i]["lot_name"] for i in zcode_ids if i in by_id}
    print(f"\n[2] 排除 ZCode 十八张：{len(zcode_ids)} 个样本、{len(zcode_lots)} 个 lot")

    bm_ids = benchmark_ids()
    bm_lots = {by_id[i]["lot_name"] for i in bm_ids if i in by_id}
    print(f"    冻结 benchmark：{len(bm_ids)} 个样本、{len(bm_lots)} 个 lot")

    lots = {s: {r["lot_name"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    inter = {
        "train∩val": sorted(lots["train"] & lots["val"]),
        "train∩test": sorted(lots["train"] & lots["test"]),
        "val∩test": sorted(lots["val"] & lots["test"]),
        "train∩benchmark": sorted(lots["train"] & bm_lots),
        "val∩benchmark": sorted(lots["val"] & bm_lots),
    }
    print(f"\n[3] lot 交集检查（**任一非空即失败**）")
    for k, v in inter.items():
        print(f"    {k:20s} {len(v)}")
    if any(inter.values()):
        print("[停止] lot 泄漏，不放宽")
        return 3

    # ── 抽样 ───────────────────────────────────────────
    rng = random.Random(SEED)
    train_used: set[str] = set()
    train, dropped = [], {}
    for c in CLASSES:
        pool = sorted([r for r in rows
                       if r["split"] == "train" and r["failure_type"] == c
                       and r["sample_id"] not in zcode_ids
                       and r["lot_name"] not in zcode_lots],
                      key=lambda r: r["sample_id"])
        got = draw(pool, TRAIN_PER_CLASS, train_used, rng)
        train += got
        if len(got) < TRAIN_PER_CLASS:
            dropped[c] = f"候选不足，只得 {len(got)}/{TRAIN_PER_CLASS}"
    print(f"\n[4] train_180：抽到 {len(train)} 张（目标 {len(CLASSES) * TRAIN_PER_CLASS}）")
    if dropped:
        print(f"    ⚠ 不足的类：{dropped}")

    dev_used = set(train_used)
    dev = []
    for c in CLASSES:
        pool = sorted([r for r in rows
                       if r["split"] == "val" and r["failure_type"] == c
                       and r["sample_id"] not in zcode_ids
                       and r["lot_name"] not in zcode_lots],
                      key=lambda r: r["sample_id"])
        got = draw(pool, DEV_PER_CLASS, dev_used, rng)
        dev += got
    print(f"    dev_18：抽到 {len(dev)} 张（目标 {len(CLASSES) * DEV_PER_CLASS}）")

    # ── 二十张小测试（train 的子集）────────────────────
    smoke_rng = random.Random(SEED)
    smoke_used: set[str] = set()
    smoke = []
    for c in CLASSES:
        pool = sorted([r for r in train if r["failure_type"] == c], key=lambda r: r["sample_id"])
        smoke += draw(pool, SMOKE_PER_CLASS, smoke_used, smoke_rng)
    pool = sorted([r for r in train if r["sample_id"] not in {x["sample_id"] for x in smoke}],
                  key=lambda r: r["sample_id"])
    smoke += draw(pool, SMOKE_EXTRA, smoke_used, smoke_rng)
    print(f"    smoke_20：抽到 {len(smoke)} 张（目标 {len(CLASSES) * SMOKE_PER_CLASS + SMOKE_EXTRA}）")
    assert {x["sample_id"] for x in smoke} <= {x["sample_id"] for x in train}, "smoke 必须是 train 子集"

    # ── 校验与统计 ─────────────────────────────────────
    def summarize(name: str, sel: list[dict]) -> dict:
        ids = [r["sample_id"] for r in sel]
        shas, missing = [], []
        for r in sel:
            p = IMAGES / f"{r['sample_id']}.png"
            if p.exists():
                shas.append(sha256_file(p))
            else:
                missing.append(r["sample_id"])
        return {
            "name": name, "n": len(sel),
            "unique_sample_ids": len(set(ids)),
            "unique_lots": len({r["lot_name"] for r in sel}),
            "unique_png_sha256": len(set(shas)),
            "missing_images": missing,
            "per_class": {c: sum(1 for r in sel if r["failure_type"] == c) for c in CLASSES},
            "png_sha256": dict(zip(ids, shas)),
        }

    stats = [summarize("train_180", train), summarize("dev_18", dev), summarize("smoke_20", smoke)]
    print(f"\n[5] 校验")
    problems = []
    for s in stats:
        print(f"    {s['name']:10s} n={s['n']:3d} 唯一ID={s['unique_sample_ids']:3d} "
              f"lot={s['unique_lots']:3d} 唯一指纹={s['unique_png_sha256']:3d} 缺图={len(s['missing_images'])}")
        if s["unique_sample_ids"] != s["n"]:
            problems.append(f"{s['name']} 有重复 sample_id")
        if s["unique_png_sha256"] != s["n"]:
            problems.append(f"{s['name']} 有重复图片指纹")
        if s["missing_images"]:
            problems.append(f"{s['name']} 缺图 {s['missing_images'][:5]}")
    train_lots = {r["lot_name"] for r in train}
    dev_lots = {r["lot_name"] for r in dev}
    if train_lots & dev_lots:
        problems.append(f"train 与 dev lot 重叠 {len(train_lots & dev_lots)} 个")
    if train_lots & zcode_lots:
        problems.append("train 与 ZCode lot 重叠")
    if any(v for v in inter.values()):
        problems.append("lot 交集非空")
    if problems:
        print(f"\n[!] {len(problems)} 个问题：")
        for x in problems:
            print(f"    - {x}")
    else:
        print("    无问题")

    # ── 落盘 ───────────────────────────────────────────
    def jsonl(sel):
        return "\n".join(json.dumps({
            "sample_id": r["sample_id"],
            "image_path": str(IMAGES / f"{r['sample_id']}.png"),
            "label": r["failure_type"],
            "label_source": r["label_source"],
            "split": r["split"],
            "lot_name": r["lot_name"],
            "manifest_line": r["__line__"],
        }, ensure_ascii=False) for r in sorted(sel, key=lambda x: x["sample_id"])) + "\n"

    plan = {
        "task": "SFT-A 标签监督技术基线：数据集",
        "prepared_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "seed": SEED,
        "question": QUESTION,
        "question_note": "固定分类题面；与 GLM 七字段试标题面不同，本训练不做 GLM 同题能力对照。",
        "manifest_sha256": man_sha,
        "excluded_zcode_samples": sorted(zcode_ids),
        "excluded_zcode_lots": sorted(zcode_lots),
        "benchmark_lots": len(bm_lots),
        "lot_intersections": {k: len(v) for k, v in inter.items()},
        "stats": [{k: v for k, v in s.items() if k != "png_sha256"} for s in stats],
        "problems": problems,
        "counts": {"train": TRAIN_PER_CLASS, "dev": DEV_PER_CLASS,
                   "smoke": SMOKE_PER_CLASS, "smoke_extra": SMOKE_EXTRA},
    }

    if not a.write:
        print("\n" + "=" * 78)
        print("DRY-RUN：未写文件。加 --write 落盘。")
        print("=" * 78)
        return 0 if not problems else 1

    targets = ["train_180.jsonl", "dev_18.jsonl", "smoke_20.jsonl", "dataset_plan.json"]
    exist = [t for t in targets if (out_dir / t).exists()]
    if exist:
        print(f"\n[停止] 已存在，不覆盖：{exist}")
        return 3
    for name, sel in (("train_180", train), ("dev_18", dev), ("smoke_20", smoke)):
        (out_dir / f"{name}.jsonl").write_text(jsonl(sel), encoding="utf-8", newline="\n")
    plan["files"] = {t: sha256_file(out_dir / t) for t in targets[:3]}
    (out_dir / "dataset_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\n[6] 已写出到 {out_dir}")
    for t in targets:
        print(f"    {t:22s} {sha256_file(out_dir / t)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
