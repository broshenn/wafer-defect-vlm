#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对 lot 结构与训练/评测隔离（纯 CPU）。

这条脚本存在的理由见 `报告/事实与错误清单.md` **E6**：
我曾经只看评测 102 行的 lot 分布，就推断「lot 由 sample_id 派生 = 一图一 lot
= 隔离同义反复」。**那是错的** —— 翻到更宽的 manifest 才发现
单个 lot 最多含 24 张晶圆。所以把这个检查**固化成能跑的脚本**，
而不是靠「读一遍觉得对」。

跑法：python 核对_lot隔离.py
"""
from __future__ import annotations
import io, json, sys, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent          # …/描述事实验收_20261008_1911
ROOT = HERE.parent                              # …/02_ClaudeCode_实操
REPO = ROOT.parent.parent                       # 仓库根
NIGHT = ROOT / "夜间扩容_20261008_0244"
MANIFEST = REPO / "data" / "manifest.jsonl"
TRAIN = NIGHT / "数据" / "L_N3072.jsonl"
EVAL = NIGHT / "客户端盲包" / "WorkBuddy_外部评测102" / "WB_eval102_索引.jsonl"


def jl(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def main() -> int:
    for p in (MANIFEST, TRAIN, EVAL):
        if not p.exists():
            print(f"!! 缺文件 {p}"); return 3
    man = {r["sample_id"]: r for r in jl(MANIFEST)}

    print("=" * 70)
    print("一、历史 manifest 的 lot 结构")
    print("=" * 70)
    lc = collections.Counter(r["lot_name"] for r in man.values())
    print(f"  晶圆 {len(man)}  /  唯一 lot {len(lc)}  /  单 lot 最多 {max(lc.values())} 张晶圆")
    deriv = sum(1 for r in man.values()
                if r["lot_name"] == f"lot{int(r['sample_id'].split('_')[1])}")
    print(f"  lot 名可由 sample_id 推出：{deriv}/{len(man)}"
          f"   （因为 sample_id 本身是 wafer_<lot号>_<lot内序号>）")

    l2s = collections.defaultdict(set)
    for r in man.values():
        l2s[r["lot_name"]].add(r["split"])
    cross = [k for k, v in l2s.items() if len(v) > 1]
    print(f"  **跨 split 的 lot：{len(cross)} / {len(l2s)}**"
          f"   {'✓ 隔离成立' if not cross else '✗ 存在泄漏'}")
    for s in sorted({r['split'] for r in man.values()}):
        sub = [r for r in man.values() if r["split"] == s]
        c = collections.Counter(r["lot_name"] for r in sub)
        print(f"    {s:<6} {len(sub):>5} 行 / {len(c):>5} lot / 最多 {max(c.values())} 张每 lot")

    print()
    print("=" * 70)
    print("二、训练 3072 与评测 102 的隔离")
    print("=" * 70)
    trs = {r["sample_id"] for r in jl(TRAIN)}
    evs = {r["sample_id"] for r in jl(EVAL)}
    print(f"  训练 {len(trs)} 张  /  评测 {len(evs)} 张")
    print(f"  样本交集：{len(trs & evs)}   {'✓' if not trs & evs else '✗'}")
    miss = [s for s in trs | evs if s not in man]
    print(f"  在历史 manifest 中缺失：{len(miss)}   {'✓ 两边都能对上' if not miss else miss[:5]}")
    if miss:
        return 4
    trL = {man[s]["lot_name"] for s in trs}
    evL = {man[s]["lot_name"] for s in evs}
    print(f"  训练 {len(trL)} 个 lot  /  评测 {len(evL)} 个 lot")
    print(f"  **lot 交集：{len(trL & evL)}**   {'✓' if not trL & evL else '✗'}")

    print()
    print("=" * 70)
    print("三、为什么「按 lot 重抽」在这 102 张上不等于第二个独立视角")
    print("=" * 70)
    print(f"  评测 102 张 → {len(evL)} 个 lot"
          f"（{'两两不同' if len(evL) == len(evs) else '有重复'}）")
    print("  但**这只是这批子集的属性**：manifest 里单 lot 最多含 "
          f"{max(lc.values())} 张晶圆。")
    print("  → 别拿子集的形状去断言全局结构（见 事实与错误清单.md E6）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
