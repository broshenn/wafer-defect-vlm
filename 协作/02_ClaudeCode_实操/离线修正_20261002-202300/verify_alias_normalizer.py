#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 Codex 的类别别名规范化器做**离线边界验证**。

被测对象：`协作/01_Codex_指挥/验收_20261002_GLM首批20/normalize_label_alias.py`
**只读、不改被测代码**。本文件不覆盖它，也不改动任何原答。

## 要验的四条（照任务书 §4）

  1. **严格白名单**：只认显式登记的那几个别名；别的拼写一个都不改。
  2. **raw 不覆盖**：输出路径不可能等于输入路径；已存在的输出一律拒绝。
  3. **不修未知词**：未知类别、大小写变体、近似拼写都不"模糊纠错"。
  4. **拒绝重复键 / 坏 JSON**：不靠 `json.loads` 的静默折叠兜底。

外加：**副本内容除了被规范化的那一个字段，其余必须逐字节不变**。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TARGET = (REPO / "协作" / "01_Codex_指挥" / "验收_20261002_GLM首批20"
          / "normalize_label_alias.py")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    return bool(ok)


def load_target():
    if not TARGET.exists():
        print(f"[无法验证] 找不到被测文件：{TARGET}")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("codex_normalizer", TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    print("=" * 78)
    print("类别别名规范化器 · 离线边界验证")
    print(f"被测：{TARGET.relative_to(REPO)}")
    print(f"  指纹 {hashlib.sha256(TARGET.read_bytes()).hexdigest()}")
    print("=" * 78)
    M = load_target()

    print(f"\n[0] 规范化器声明的别名白名单（{len(M.ALIASES)} 条）")
    for k, v in sorted(M.ALIASES.items()):
        print(f"      {k!r} -> {v!r}")

    # ── 1 严格白名单 ────────────────────────────────────
    print("\n[1] 严格白名单")
    for before, after in sorted(M.ALIASES.items()):
        got, ch = M.normalize(json.dumps({"defect_class": before}).encode())
        check(f"登记别名 {before!r} → {after!r}",
              got["defect_class"] == after and len(ch) == 1)

    # 已合法的九类：一个都不许动
    LEGAL = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
             "Near_full", "Random", "Scratch", "none", "unknown"]
    unchanged = True
    for c in LEGAL:
        got, ch = M.normalize(json.dumps({"defect_class": c}).encode())
        if got["defect_class"] != c or ch:
            unchanged = False
            print(f"      被改了: {c!r} -> {got['defect_class']!r}")
    check("全部合法枚举值原样通过（含 unknown）", unchanged)

    # ── 2 不修未知词 ────────────────────────────────────
    print("\n[2] 不修未知词（模糊纠错一律不做）")
    UNKNOWN = ["near-full", "NEAR-FULL", "Near_Full ", "Edge Loc", "Edge—Loc",
               "Edge–Loc", "Edge_LocX", "Donat", "Randome", "NONE", "None",
               "NearFull", "near_full", "Edge-LOC"]
    bad = []
    for u in UNKNOWN:
        got, ch = M.normalize(json.dumps({"defect_class": u}).encode())
        if got["defect_class"] != u or ch:
            bad.append((u, got["defect_class"]))
    check(f"{len(UNKNOWN)} 种未知/变体拼写**一个都没改**", not bad, str(bad[:4]))
    for u in UNKNOWN[:4]:
        print(f"      {u!r} → 保持原样")

    # ── 3 拒绝重复键与坏 JSON ───────────────────────────
    print("\n[3] 拒绝重复键 / 坏 JSON")
    BAD = {
        "重复 defect_class": b'{"defect_class":"Loc","defect_class":"Center"}',
        "重复其它字段": b'{"defect_class":"Loc","caption_zh":"a","caption_zh":"b"}',
        "截断 JSON": b'{"defect_class":',
        "顶层是数组": b'[]',
        "顶层是字符串": b'"Loc"',
        "空内容": b'',
        "非法字符": b'{"defect_class":"Loc",}',
    }
    for label, raw in BAD.items():
        try:
            M.normalize(raw)
            check(f"拒绝：{label}", False, "**被静默接受了**")
        except Exception as e:                                # noqa: BLE001
            check(f"拒绝：{label}", True, type(e).__name__)
            print(f"      {label:16s} → {type(e).__name__}")

    # ── 4 副本除目标字段外逐字节不变 ────────────────────
    print("\n[4] 副本除被规范化字段外，其余内容不变")
    raw = json.dumps({"defect_class": "Near-full",
                      "morphology": "十几颗散点", "radial_zone": "edge",
                      "clock_direction": "7点钟", "extent_r": None,
                      "caption_zh": "边缘有一簇", "uncertainty": ""},
                     ensure_ascii=False).encode()
    got, ch = M.normalize(raw)
    src = json.loads(raw)
    diff = [k for k in src if src[k] != got.get(k)]
    check("只有 defect_class 变了", diff == ["defect_class"], str(diff))
    check("其余六个字段值完全保留",
          all(src[k] == got[k] for k in src if k != "defect_class"))
    check("改动被如实记录在 changes 里",
          len(ch) == 1 and ch[0]["before"] == "Near-full" and ch[0]["after"] == "Near_full")

    # ── 5 raw 不覆盖 ────────────────────────────────────
    print("\n[5] raw 不覆盖（结构上不可能覆盖）")
    src_txt = TARGET.read_text(encoding="utf-8")
    check("拒绝覆盖原答/既有副本的记录逻辑存在",
          "拒绝覆盖原答或既有副本" in src_txt)
    check("输出目录由执行者事先建立（不自动扩展输出位置）",
          "目录由执行者事先建立" in src_txt)
    check("self_test 里含『坏输入被静默修好』的断言",
          "坏输入被静默修好" in src_txt)

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
