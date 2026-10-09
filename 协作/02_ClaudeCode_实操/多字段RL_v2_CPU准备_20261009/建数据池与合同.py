#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多字段 RL v2 首轮的数据池划分与可追溯合同（CPU，确定性）。

产出四份清单，**lot 两两互斥**，全部只从公开数据 train 侧取：

  规则开发池 rule_dev   —— 定几何阈值、跑程序×AI 交叉诊断；不进训练
  适配池     adapt      —— M0 的 schema 适配 SFT（256 张，目标）
  RL 池      rl_pool    —— M2 的 GRPO 题池（256 张，目标）
  确认池     confirmation —— 首轮评测的留出卷；**冻结后不得用于调规则或调奖励**

排除规则（用于隔离，逐项在输出里计数）：
  1. 非 train split；
  2. 旧训练见过的：N3072 / 1440 / 720 / 360 / 180（如实记录：这些图被旧 D 接触过）；
  3. 旧开发与评测：开发 36 图、56 图候选、val90/dev18、外部 102、1200 复核包；
  4. 轮廓异常（椭圆解释不了的格 > 阈值）或状态非 ok。

用法: python 建数据池与合同.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

SEED = 3407
TARGET_ADAPT = 256
TARGET_RL = 256
TARGET_CONFIRM = 120
TARGET_RULE_DEV = 30

# ── 排除清单的来源文件（相对仓库根）──────────────────────────────────────
EXCLUDE_SOURCES = {
    "旧训练_N3072": ["协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/数据/D_N3072.jsonl"],
    "旧训练_1440": ["协作/02_ClaudeCode_实操/扩容主训练_20261006/scale_1440_v2.jsonl",
                    "协作/02_ClaudeCode_实操/扩容主训练_20261006/sft_scale_1440_v2.jsonl"],
    "旧训练_720": ["协作/02_ClaudeCode_实操/扩容主训练_20261006/scale_720_v2.jsonl",
                   "协作/02_ClaudeCode_实操/扩容主训练_20261006/sft_scale_720_v2.jsonl"],
    "旧训练_360": ["协作/02_ClaudeCode_实操/扩容主训练_20261006/scale_360_v2.jsonl",
                   "协作/02_ClaudeCode_实操/扩容主训练_20261006/sft_scale_360_v2.jsonl"],
    "旧训练_180": ["协作/01_Codex_指挥/训练数据就绪_20261003/sft_a_train_180.jsonl"],
    "旧开发_36图": ["协作/02_ClaudeCode_实操/百炼多模型业务对照_20261009_1203/盲号对照_key.json"],
    "旧候选_56图": ["协作/02_ClaudeCode_实操/56图候选补测与最终结项_20261009/冻结清单.json"],
    "旧验证_val90": ["协作/02_ClaudeCode_实操/收尾_RL90与报告_20261006/固定顺序90.jsonl"],
    "旧验证_dev18": ["协作/02_ClaudeCode_实操/RL对照补证_20261006/f0_dev18_原答.jsonl"],
}
GLOB_EXCLUDES = ["协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/客户端盲包/**/*.jsonl"]


def _ids_from(path: Path) -> tuple[set[str], str | None]:
    """尽量稳地从 json/jsonl 里取出 sample_id 集合；同时记一条可读的规模说明。"""
    txt = path.read_text(encoding="utf-8")
    ids: set[str] = set()
    if path.suffix == ".json":
        d = json.loads(txt)
        if isinstance(d, dict) and "逐图" in d:
            ids = {r["sample_id"] for r in d["逐图"]}
        elif isinstance(d, list):
            for r in d:
                if isinstance(r, dict) and "sample_id" in r:
                    ids.add(r["sample_id"])
        elif isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) and "sample_id" in v:
                    ids.add(v["sample_id"])
                elif isinstance(v, str) and v.startswith("wafer_"):
                    ids.add(v)
    else:
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(o, dict) and "sample_id" in o:
                ids.add(o["sample_id"])
    return ids, None


def build_exclusions() -> dict:
    out = {}
    for name, paths in EXCLUDE_SOURCES.items():
        ids = set()
        for p in paths:
            fp = REPO / p
            if not fp.exists():
                raise FileNotFoundError(f"排除来源缺失：{p}")
            got, _ = _ids_from(fp)
            ids |= got
        out[name] = ids
    for pat in GLOB_EXCLUDES:
        for fp in REPO.glob(pat):
            got, _ = _ids_from(fp)
            out.setdefault("外部评测包", set()).update(got)
    return out


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pick(rows: list[dict], n: int, key_pool: list[dict], rng) -> list[dict]:
    """从 key_pool 里按 key 分层抽取 n 条；不足时返回全部可用。"""
    by_key = defaultdict(list)
    for r in key_pool:
        by_key[key(r)].append(r)
    for k in by_key:
        by_key[k].sort(key=lambda r: r["sample_id"])
    keys = sorted(by_key)
    out, i = [], 0
    while len(out) < n and any(by_key.values()):
        k = keys[i % len(keys)]
        if by_key[k]:
            out.append(by_key[k].pop(0))
        i += 1
        if i > 100000:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import random
    rng = random.Random(SEED)

    geom = {}
    for line in (HERE / "几何特征表.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        geom[d["sample_id"]] = d
    manifest = [json.loads(l) for l in (REPO / "data" / "manifest.jsonl").open(encoding="utf-8")]
    for r in manifest:
        r["geom"] = geom.get(r["sample_id"], {})

    exclusions = build_exclusions()
    excl_all = set().union(*exclusions.values()) if exclusions else set()

    pool, dropped = [], Counter()
    for r in manifest:
        sid = r["sample_id"]
        if r["split"] != "train":
            dropped["非train"] += 1
            continue
        if sid in excl_all:
            dropped["已被旧训练或旧评测用过"] += 1
            continue
        g = r["geom"]
        if g.get("status") != "ok":
            dropped["几何状态非ok"] += 1
            continue
        if g.get("outline", {}).get("anomalous"):
            dropped["轮廓异常"] += 1
            continue
        pool.append(r)

    print(f"可用池（train 侧、未被旧训练/旧评测用过、几何正常）：{len(pool)}")
    print("排除计数：", dict(dropped))
    by_class = Counter(r["failure_type"] for r in pool)
    print("可用池类别分布：", dict(by_class))

    # ── 三池按 lot 分组，保证 lot 互斥；同一 lot 整体进同一个池 ───────────
    lot_of = defaultdict(list)
    for r in pool:
        lot_of[r["lot_name"]].append(r)
    lots = sorted(lot_of)
    rng.shuffle(lots)

    # 给每个 lot 一个「主类别」与方向标签，便于分层
    def lot_signature(lot: str) -> dict:
        rs = lot_of[lot]
        return {"n": len(rs),
                "classes": Counter(x["failure_type"] for x in rs),
                "dirs": Counter(x["geom"]["direction_type"] for x in rs)}

    def fill_to(picked: list[dict], target_n: int, chosen_lots: set[str]) -> list[dict]:
        for lot in lots:
            if len(picked) >= target_n:
                break
            if lot in chosen_lots:
                continue
            picked.extend(lot_of[lot])
            chosen_lots.add(lot)
        return picked

    used_lots: set[str] = set()
    # 方向可判的 lot 只有二十来个，显式按 2:1 分给 RL 池与确认池，
    # 否则会被先挑的池整批拿走（上一版确认池就一个都没分到）。
    elig_lots = sorted(l for l in lots
                       if lot_signature(l)["dirs"]["single"] + lot_signature(l)["dirs"]["axis"])
    rl_lots = [l for i, l in enumerate(elig_lots) if i % 3 != 2]
    confirm_lots = [l for i, l in enumerate(elig_lots) if i % 3 == 2]
    used_lots |= set(elig_lots)
    rl = [r for l in rl_lots for r in lot_of[l]]
    confirm = [r for l in confirm_lots for r in lot_of[l]]
    confirm = fill_to(confirm, TARGET_CONFIRM, used_lots)
    rl = fill_to(rl, TARGET_RL, used_lots)
    adapt = fill_to([], TARGET_ADAPT, used_lots)
    rule_dev = fill_to([], TARGET_RULE_DEV, used_lots)

    def summarize(name: str, rs: list[dict]) -> dict:
        return {
            "名称": name, "行数": len(rs),
            "类别分布": dict(Counter(x["failure_type"] for x in rs)),
            "方向分布": dict(Counter(x["geom"]["direction_type"] for x in rs)),
            "覆盖档位分布": dict(Counter(x["geom"]["coverage_level"] for x in rs)),
            "唯一lot数": len({x["lot_name"] for x in rs}),
            "唯一PNG数": len({x["sample_id"] for x in rs}),
        }

    pools = {"规则开发池": rule_dev, "适配池": adapt, "RL池": rl, "确认池": confirm}
    for name, rs in pools.items():
        print(name, summarize(name, rs))

    # ── lot / PNG 两两互斥核验 ─────────────────────────────────────────
    lot_sets = {k: {x["lot_name"] for x in v} for k, v in pools.items()}
    png_sets = {k: {x["sample_id"] for x in v} for k, v in pools.items()}
    overlap = {}
    names = list(pools)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap[f"{a}×{b}"] = {
                "lot交集": len(lot_sets[a] & lot_sets[b]),
                "png交集": len(png_sets[a] & png_sets[b]),
            }
    print("互斥核验：", json.dumps(overlap, ensure_ascii=False))

    result = {
        "seed": SEED,
        "生成脚本sha256": sha256_file(Path(__file__)),
        "几何定义版本": geom[next(iter(geom))]["definition_version"],
        "几何特征表sha256": sha256_file(HERE / "几何特征表.jsonl"),
        "manifest_sha256": sha256_file(REPO / "data" / "manifest.jsonl"),
        "排除来源计数": {k: len(v) for k, v in exclusions.items()},
        "排除后可用池": len(pool),
        "排除计数": dict(dropped),
        "池摘要": {k: summarize(k, v) for k, v in pools.items()},
        "互斥核验": overlap,
        "边界声明": [
            "四池只从公开数据 train 侧取；val/test 未被触碰。",
            "适配池与 RL 池中的图**不是全新未见数据**：它们来自同一公开数据集，"
            "旧 N3072 等训练集已排除，但同源同分布，不声称模型从未见过该分布的图像。",
            "确认池冻结后不得用于调阈值、调奖励或选 checkpoint。",
        ],
        "未做": [
            "几何阈值在 train 侧规则开发池上复核（见 阈值诊断.md）",
            "任何 GPU 运行（not run）",
        ],
    }

    if not args.dry_run:
        (HERE / "数据池合同.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        for name, rs in pools.items():
            fn = {"规则开发池": "清单_规则开发池.jsonl", "适配池": "清单_适配池.jsonl",
                  "RL池": "清单_RL池.jsonl", "确认池": "清单_确认池.jsonl"}[name]
            with (HERE / fn).open("w", encoding="utf-8", newline="\n") as fh:
                for r in sorted(rs, key=lambda x: x["sample_id"]):
                    g = r["geom"]
                    fh.write(json.dumps({
                        "sample_id": r["sample_id"], "lot_name": r["lot_name"],
                        "failure_type": r["failure_type"],
                        "image_path": f"data/images/{r['sample_id']}.png",
                        "image_sha256": g.get("png_sha256"),
                        "matrix_shape": r["matrix_shape"],
                        # 评分侧参考（模型输入里没有这些字段）
                        "ref_coverage_level": g.get("coverage_level"),
                        "ref_red_mass_zones": g.get("red_mass_zones"),
                        "ref_direction_type": g.get("direction_type"),
                        "ref_clock_sectors": g.get("clock_sectors"),
                        "ref_direction_eligible":
                            g.get("direction_type") in ("single", "axis")
                            and not g.get("direction", {}).get("sector_tie", False),
                        "ref_cov_band_edge": g.get("coverage_band_edge"),
                    }, ensure_ascii=False, sort_keys=True) + "\n")
        print("已写出 数据池合同.json 与四份清单")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
