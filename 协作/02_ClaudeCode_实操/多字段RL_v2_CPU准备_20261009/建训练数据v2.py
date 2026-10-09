#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多字段 RL v2：由数据池清单生成 ms-swift 可用的 JSONL（CPU，确定性）。

产出（全部写到 out_v2/）：
  adapt_sft.jsonl        适配池 256 图 × v2 题面 → 目标 JSON（M0 的 schema 适配 SFT）
  group60.jsonl          首轮 60 图组（M1 连续 SFT 与 M2 GRPO **共用同一组图与顺序**）
  group60_sft4.jsonl     每图正确答案重复 4 行（M1 用；对应 GA4 → 1 图/更新步）
  group60_grpo.jsonl     每图 1 行（M2 用；G=4 由 num_generations 决定）
  confirm120.jsonl       确认池 120 图 × v2 题面（评测用；不带答案）
  confirm120_gold.jsonl  确认池的参考（评分侧单独保存，不与题面同文件）
  imglist.txt            需要上传到服务器的图片清单

**caption_zh 的处理（重要）**：本轮没有可信的教师文本，也不把模型自己的描述升格为 gold，
因此适配目标里的 caption 用**程序模板**生成（只由类别、径向区带、覆盖档位、方向这些
程序事实拼出，不引入新事实）。它只用于让模型学会「caption 字段要写什么位置」，
**不用于评价描述能力**；描述质量另做 AI 复核并单独报告。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "out_v2"

PROMPT = (HERE / "题面_prompt_v2.txt").read_text(encoding="utf-8").strip()

CLASS_PHRASE = {
    "Center": "中心区域失效", "Donut": "环形失效", "Edge_Loc": "边缘局部块状失效",
    "Edge_Ring": "边缘环形失效", "Loc": "局部块状失效", "Near_full": "接近整片失效",
    "Random": "随机散布失效", "Scratch": "线状划痕", "none": "未见明显缺陷图案",
}
ZONE_PHRASE = {"center": "晶圆中心区域", "middle": "晶圆中部", "edge": "晶圆边缘区域"}
COVER_PHRASE = {"none": "无红色失效 die", "low": "覆盖比例较低", "medium": "覆盖比例中等",
                "high": "覆盖比例较高", "near_all": "几乎覆盖全片"}


def caption_template(row: dict) -> str:
    """由程序事实拼出的模板句。**不引入任何图上没有的新事实。**

    注意 `none` 类在公开数据里的语义是「没有可辨别的缺陷图案」，不等于「一个红点都没有」；
    所以这一类不写「位于…」，改写成散在分布，避免拼出自相矛盾的句子。
    """
    cls = row["failure_type"]
    zones = row["ref_red_mass_zones"]
    cov = row["ref_coverage_level"]
    dt = row["ref_direction_type"]
    secs = row["ref_clock_sectors"]
    parts = [CLASS_PHRASE.get(cls, "失效图案")]
    if cls == "none":
        parts.insert(0, "红色失效 die 散在分布")
    elif isinstance(zones, list) and zones:
        parts.append("位于" + "、".join(ZONE_PHRASE[z] for z in zones))
    if dt == "single" and secs:
        parts.append(f"集中在 {secs[0]} 点钟方向")
    elif dt == "axis" and len(secs) == 2:
        parts.append(f"沿 {secs[0]} 点与 {secs[1]} 点方向成带状")
    if cov in COVER_PHRASE and cov != "none":
        parts.append(COVER_PHRASE[cov])
    return "，".join(parts) + "。"


def gold_json(row: dict) -> str:
    obj = {
        "defect_class": row["failure_type"],
        "coverage_level": row["ref_coverage_level"],
        "red_mass_zones": row["ref_red_mass_zones"],
        "direction_type": row["ref_direction_type"],
        "clock_sectors": row["ref_clock_sectors"],
        "caption_zh": caption_template(row),
        "uncertainty": "" if row["ref_direction_type"] != "unknown" else "方向证据不足，未给出具体钟点",
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def gold_columns(row: dict) -> dict:
    return {
        "gold_class": row["failure_type"],
        "gold_coverage_level": row["ref_coverage_level"],
        "gold_zones": (json.dumps(row["ref_red_mass_zones"], ensure_ascii=False)
                       if isinstance(row["ref_red_mass_zones"], list) else "unknown"),
        "gold_direction_type": row["ref_direction_type"],
        "gold_sectors": json.dumps(row["ref_clock_sectors"], ensure_ascii=False),
    }


def ms_row(row: dict, with_answer: bool, extra: dict | None = None) -> dict:
    msgs = [{"role": "user", "content": "<image>\n" + PROMPT}]
    if with_answer:
        msgs.append({"role": "assistant", "content": gold_json(row)})
    out = {"sample_id": row["sample_id"], "messages": msgs,
           "images": [f"/root/autodl-tmp/ws/v2run/图/{row['sample_id']}.png"]}
    if extra:
        out.update(extra)
    return out


def load(name: str) -> list[dict]:
    p = HERE / f"清单_{name}.jsonl"
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]
    return rows


def pick_group60(rl: list[dict]) -> list[dict]:
    """从 RL 池挑 60 图。规则：先保证方向可判样本全部入选，再按类别轮转补齐。"""
    import collections
    elig = sorted([r for r in rl if r["ref_direction_type"] in ("single", "axis")],
                  key=lambda r: r["sample_id"])
    rest = [r for r in rl if r not in elig]
    by_cls = collections.defaultdict(list)
    for r in rest:
        by_cls[r["failure_type"]].append(r)
    for k in by_cls:
        by_cls[k].sort(key=lambda r: r["sample_id"])
    order = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_full", "Random",
             "Scratch", "none"]
    out = list(elig)
    i = 0
    while len(out) < 60:
        k = order[i % len(order)]
        i += 1
        if by_cls.get(k):
            out.append(by_cls[k].pop(0))
        if i > 10000:
            break
    return sorted(out, key=lambda r: r["sample_id"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print("dry-run：不写文件")
    OUT.mkdir(exist_ok=True)

    adapt = load("适配池")
    rl = load("RL池")
    confirm = load("确认池")
    group60 = pick_group60(rl)

    files = {
        "adapt_sft.jsonl": [ms_row(r, True) for r in adapt],
        "group60.jsonl": [ms_row(r, True) for r in group60],
        "group60_sft4.jsonl": [ms_row(r, True) for r in group60 for _ in range(4)],
        "group60_grpo.jsonl": [ms_row(r, False, gold_columns(r)) for r in group60],
        "confirm120.jsonl": [ms_row(r, False) for r in confirm],
        "confirm120_gold.jsonl": [{**gold_columns(r), "sample_id": r["sample_id"],
                                   "lot_name": r["lot_name"],
                                   "ref_zones_raw": r["ref_red_mass_zones"],
                                   "ref_sectors_raw": r["ref_clock_sectors"]}
                                  for r in confirm],
    }
    if not a.dry_run:
        for fn, rows in files.items():
            p = OUT / fn
            with p.open("w", encoding="utf-8", newline="\n") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        ids = sorted({r["sample_id"] for r in adapt} | {r["sample_id"] for r in rl}
                     | {r["sample_id"] for r in confirm})
        (OUT / "imglist.txt").write_text(
            "\n".join(f"data/images/{i}.png" for i in ids) + "\n", encoding="utf-8")

    import collections
    print(f"适配池 {len(adapt)} 行 → adapt_sft.jsonl")
    print(f"60 图组 {len(group60)} 图（方向可判 "
          f"{sum(1 for r in group60 if r['ref_direction_type'] in ('single','axis'))}）")
    print("  类别:", dict(collections.Counter(r["failure_type"] for r in group60)))
    print("  方向:", dict(collections.Counter(r["ref_direction_type"] for r in group60)))
    print(f"确认池 {len(confirm)} 图 → confirm120.jsonl")
    print(f"需上传图片 {len({r['sample_id'] for r in adapt} | {r['sample_id'] for r in rl} | {r['sample_id'] for r in confirm})} 张")
    print("\n示例目标答案：\n ", gold_json(adapt[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
