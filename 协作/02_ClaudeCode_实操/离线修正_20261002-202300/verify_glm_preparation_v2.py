#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GLM 标注准备的检查器 **v2** —— 修 B49。

**不覆盖 v1**（`verify_glm_preparation.py`）。v1 保留作历史版本。

## v2 修的什么

**B49 缺陷 1：验证/测试 lot 取自"全是 train"的候选清单，比较的是空集合。**
v1 的 `:159`：

```python
lots = {s: {man[r["sample_id"]]["lot_name"] for r in allb
            if man[r["sample_id"]]["split"] == s} for s in ("train","val","test")}
```

`allb` 是**180 个 train 样本**，所以 `lots["val"]` 与 `lots["test"]` **恒为空集**，
后面的 `tlots & s` **永远为空** —— 这条检查**恒真**，证明不了任何外部划分隔离。

v2：val/test 的 lot 集合**从完整 manifest 取**（5904 行里的全部 val / test 行），
不是从本批候选里取。并且**显式断言这两个集合非空** —— 集合为空就判失败，
因为"与空集无交集"是废话而不是证据。

**B49 缺陷 2：benchmark 派生 ID 未映射回原晶圆，未知 ID 被静默跳过。**
v1 的 `:173`：

```python
bm_lots = {man[i]["lot_name"] for i in bm_ids if i in man}   # ← if i in man 静默吞掉未知
```

实测 benchmark 有 1008 个 ID，其中 **756 个带派生后缀**（`__recolor_v1` /
`__resize_224` / `__rotate_90_ccw`），它们**不在 manifest 里**，被 `if i in man` 默默丢掉。

v2：先把 `wafer_XXXXXXXX_YYY__suffix` **映射回根 ID**；**去重后再逐条核对**；
**根 ID 不在 manifest 里 → 显式失败**，不静默跳过。

**新增：合成同 lot 跨 split 冲突，证明检查非零。**
构造一个"某 train 样本的 lot 同时出现在 test 里"的合成输入，跑同一段检查逻辑，
要求它**报出冲突**。检查器如果对合成冲突仍然说"无交集"，那就是恒真检查。

**新增：输出样本与顺序精确等于锁定的 train_180 与 smoke_20。**
v1 只查了自洽性，没查"产出的是不是那批数据"。

## 与 v1 的关系

两份实现**不 import 彼此**。但如 B49 所说，两份实现也可能共享同一种概念错误 ——
所以 v2 额外用**合成冲突**去证明"这段检查真的会失败"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MANIFEST = REPO / "data" / "manifest.jsonl"
BENCHMARK = REPO / "benchmark"
SRC = REPO / "协作" / "02_ClaudeCode_实操" / "学校训练准备_20261002-025848"

SOURCE_SHA = {
    "train_180.jsonl": "04febd99674f70bf4619c3b529fe4c1313a5b8d739e89c69c390fe686324db47",
    "smoke_20.jsonl": "f59ae6ba5ba917fae410a9a82f6747c886abde20df189878093467242dd98eec",
}

BLIND_KEYS = {"item_id", "sample_id", "image_path", "image_sha256"}
LEAK_KEYS = {"label", "failure_type", "class", "defect_class", "lot_name", "lot",
             "split", "difficulty", "geometry", "radial_zone", "clock_direction",
             "extent_r", "caption", "morphology", "answer", "prediction"}
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]

DERIVED_SUFFIX = re.compile(r"^(wafer_\d{8}_\d{3})__.+$")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    return bool(ok)


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── 核心：这两段是 B49 的修复点，单独抽成函数，好被合成测试直接调用 ──

def full_split_lots(manifest_rows: list[dict]) -> dict[str, set[str]]:
    """**从完整 manifest** 取 train/val/test 的 lot 集合。

    这是 B49 缺陷 1 的修复：不能从"本批候选"里取 —— 那会把 val/test 取成空集。
    """
    out: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for r in manifest_rows:
        out.setdefault(r["split"], set()).add(r["lot_name"])
    return out


def benchmark_lots(bm_raw_ids: set[str], by_id: dict[str, dict]) -> tuple[set[str], list[str]]:
    """把 benchmark 的 ID **映射回根晶圆**后取 lot。返回 (lots, 未知根ID列表)。

    这是 B49 缺陷 2 的修复：带 `__suffix` 的派生 ID 要剥回根 ID；
    根 ID 不在 manifest 里 → **进"未知"列表并由调用方判失败**，不静默跳过。
    """
    roots, unknown = set(), []
    for i in bm_raw_ids:
        m = DERIVED_SUFFIX.match(i)
        root = m.group(1) if m else i
        roots.add(root)
    for root in sorted(roots):
        if root in by_id:
            continue
        unknown.append(root)
    return {by_id[r]["lot_name"] for r in roots if r in by_id}, unknown


def leakage_check(train_lots: set[str], other_lots: set[str],
                  other_name: str) -> tuple[bool, str]:
    """返回 (是否泄漏, 说明)。**对空集显式报警** —— 与空集无交集不是证据。"""
    if not other_lots:
        return True, f"{other_name} 的 lot 集合为空 —— 检查无效，不能当通过"
    inter = train_lots & other_lots
    return bool(inter), (f"{other_name} 交集 {sorted(inter)[:3]}" if inter else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE))
    ap.add_argument("--self-test", action="store_true",
                    help="只跑合成冲突与顺序核对，不查交付产物")
    a = ap.parse_args()
    d = Path(a.dir).resolve()

    print("=" * 78)
    print("GLM 标注准备 · 检查器 v2（修 B49）")
    print("=" * 78)

    # 完整 manifest
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                o = json.loads(line)
                o["__line__"] = ln
                rows.append(o)
    by_id = {r["sample_id"]: r for r in rows}
    print(f"\n[0] 完整 manifest {len(rows)} 行")

    # ── 合成冲突测试：证明这段检查真的会失败 ────────────
    print("\n[1] 合成同 lot 跨 split 冲突（证明检查非零）")
    splits = full_split_lots(rows)
    check("完整 manifest 的 val lot 非空", bool(splits["val"]), f"{len(splits['val'])} 个")
    check("完整 manifest 的 test lot 非空", bool(splits["test"]), f"{len(splits['test'])} 个")

    # 取一个真实存在的 val lot，合成一个"train 也用它"的场景
    victim_lot = sorted(splits["val"])[0]
    ok_clean, _ = leakage_check({"lotA", "lotB"}, splits["val"], "val")
    check("对照：无重叠时判为不泄漏", ok_clean is False)
    ok_leak, why = leakage_check(splits["val"] | {"lotZ"}, splits["val"], "val")
    check("合成冲突时判为泄漏（**检查真的会失败**）", ok_leak is True, why)
    ok_empty, why_empty = leakage_check({"lotA"}, set(), "val")
    check("对空集显式报警（不把'与空集无交集'当通过）", ok_empty is True, why_empty)
    print(f"      （合成用例借用了真实存在的 {victim_lot} 所在划分，仅用于测试逻辑）")

    # ── 输出样本与顺序 ─────────────────────────────────
    print("\n[2] 输出样本与顺序精确等于锁定来源")
    for name, want in SOURCE_SHA.items():
        p = SRC / name
        check(f"{name} 指纹仍为锁定值", sha_file(p) == want, sha_file(p)[:16] + "…")
    train = read_jsonl(SRC / "train_180.jsonl")
    smoke = read_jsonl(SRC / "smoke_20.jsonl")
    train_ids = [r["sample_id"] for r in train]
    smoke_ids = [r["sample_id"] for r in smoke]
    print(f"      train_180 {len(train_ids)} 条；smoke_20 {len(smoke_ids)} 条")

    if a.self_test:
        print("\n[self-test] 到此为止，不查交付产物")
        return _report()

    need = ["blind_inputs_all_180.jsonl", "audit_selected.jsonl"] + \
           [f"batch_{i:02d}_blind.jsonl" for i in range(1, 10)] + \
           [f"batch_01_worker_{w:02d}.jsonl" for w in range(1, 5)]
    missing = [n for n in need if not (d / n).exists()]
    if missing:
        print(f"\n[无法检查] 缺文件：{missing}")
        return 2

    allb = read_jsonl(d / "blind_inputs_all_180.jsonl")
    batches = {i: read_jsonl(d / f"batch_{i:02d}_blind.jsonl") for i in range(1, 10)}
    workers = {w: read_jsonl(d / f"batch_01_worker_{w:02d}.jsonl") for w in range(1, 5)}
    audit = read_jsonl(d / "audit_selected.jsonl")

    got_ids = [r["sample_id"] for r in allb]
    want_ids = smoke_ids + [s for s in train_ids if s not in set(smoke_ids)]
    check("全量清单的 sample 顺序 == smoke_20 原序 + train_180 原序剔除首批",
          got_ids == want_ids,
          f"首个不符位置 {next((i for i,(x,y) in enumerate(zip(got_ids,want_ids)) if x!=y), '无')}")
    check("首批 20 == smoke_20 原顺序",
          [r["sample_id"] for r in batches[1]] == smoke_ids)
    rest = [r["sample_id"] for i in range(2, 10) for r in batches[i]]
    check("其余 160 == train_180 原顺序剔除首批",
          rest == [s for s in train_ids if s not in set(smoke_ids)])
    check("audit 与盲清单逐项对齐", [x["sample_id"] for x in audit] == got_ids)

    # ── B49 修复后的隔离检查 ───────────────────────────
    print("\n[3] lot 隔离（**已修：val/test 取自完整 manifest**）")
    train_lots = {by_id[s]["lot_name"] for s in got_ids}
    for nm in ("val", "test"):
        leak, why = leakage_check(train_lots, splits[nm], nm)
        check(f"本批 lot 与 {nm} 无交集", not leak, why)
        print(f"      {nm}: manifest 里有 {len(splits[nm])} 个 lot（**非空**），本批与之交集 "
              f"{len(train_lots & splits[nm])}")

    bm_raw = set()
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
                    bm_raw.add(v)
    bm_lots, bm_unknown = benchmark_lots(bm_raw, by_id)
    n_derived = sum(1 for i in bm_raw if DERIVED_SUFFIX.match(i))
    print(f"\n[4] benchmark（**已修：派生 ID 映射回根晶圆**）")
    n_roots = len({(DERIVED_SUFFIX.match(i).group(1) if DERIVED_SUFFIX.match(i) else i)
                   for i in bm_raw})
    print(f"      原始 ID {len(bm_raw)} 个，其中派生 {n_derived} 个；"
          f"映射回根后 **{n_roots}** 个唯一根晶圆，取到 {len(bm_lots)} 个 lot")
    check("benchmark 根 ID 全部能在 manifest 里找到", not bm_unknown,
          f"未知根 {len(bm_unknown)} 个：{bm_unknown[:3]}")
    check("benchmark lot 集合非空", bool(bm_lots), f"{len(bm_lots)} 个")
    leak, why = leakage_check(train_lots, bm_lots, "benchmark")
    check("本批 lot 与 benchmark 无交集", not leak, why)

    # ── 其余沿用 v1 的检查（字段/ID/图片）───────────────
    print("\n[5] 字段与 ID")
    check("每行恰好四字段", all(set(r) == BLIND_KEYS for r in allb))
    check("无答案性字段", not (set().union(*[set(r) for r in allb]) & LEAK_KEYS))
    ids = [r["item_id"] for r in allb]
    check("item_id 连续 item_001..180", ids == [f"item_{i:03d}" for i in range(1, 181)])
    check("sample_id 唯一", len(set(got_ids)) == len(got_ids))

    print("\n[6] 图片")
    shas, probs = [], []
    for r in allb:
        p = Path(r["image_path"])
        if not p.exists():
            probs.append(f"{r['item_id']} 缺图")
            continue
        g = sha_file(p)
        if g != r["image_sha256"]:
            probs.append(f"{r['item_id']} 指纹不符")
        shas.append(g)
    check("180 张图存在且指纹一致", not probs, str(probs[:3]))
    check("图片指纹不重复", len(set(shas)) == len(shas))

    print("\n[7] 子清单")
    check("四份各 5 条", all(len(x) == 5 for x in workers.values()))
    wun = [r["item_id"] for w in range(1, 5) for r in workers[w]]
    check("并集精确等于首批 20", wun == [r["item_id"] for r in batches[1]])
    check("四份两两不重叠",
          len({r["sample_id"] for w in range(1, 5) for r in workers[w]}) == 20)

    return _report()


def _report() -> int:
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
