#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文数量短语的**确定性提醒**（只提示，不改任何东西）。

## 为什么需要

`check_answer_v3.py:75`/`:150` 的数字检查只认 **ASCII 数字**。
所以「**十几颗**」这种**未经数量核验的中文计数**能一路通过格式检查 ——
如 B50 所记，**格式通过不能证明没有伪数值**。

## 三类要分开（任务书 §4 明确要求）

| 类别 | 例子 | 怎么处理 |
|---|---|---|
| **钟点方向** | `7点钟`、`八点方向` | **不是数量问题** —— 提示词明确允许，先剔除 |
| **形态/范围用语** | `一大片`、`一整圈`、`一簇`、`半个` | **不是计数** —— 描述形状与范围，单独归一类 |
| **未验证计数** | `十几颗`、`数十个`、`七八处` | **提示复核** —— 这是对失效 die **数量**的断言 |

## 三条自我约束（照任务书 §4）

  1. **不把词出现直接当视觉错** —— 本工具只产出"请人工核对这些短语"的清单，
     它**不判断**图里到底有几颗，也**不判断**答案对错。
  2. **不静默删答案** —— 一个字节都不改，输出只读。
  3. **只是提醒** —— 退出码 0 表示"扫描跑完了"，**不表示**"答案没问题"。
     有命中时退出码 1，语义是"**有待人工复核项**"，不是"答案错误"。

用法：
    python cn_quantity_audit.py <raw.json> [...]
    python cn_quantity_audit.py --dir <目录>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CN = "0-9〇零一二三四五六七八九十百千两"

# 1) 钟点方向：先剔除，避免把「7点钟」误当数量
CLOCK_RE = re.compile(rf"[{CN}]{{1,2}}\s*(?:[/\-~～、,，和及与]\s*[{CN}]{{1,2}}\s*)*点(?:钟)?(?:方向)?")

# 2) 形态/范围用语：描述形状与范围，不是对数量的断言
SHAPE_RE = re.compile(
    rf"(?<![{CN}])一(?:整|大|小|长|窄)?(?:片|块|团|簇|圈|条|带|层|侧|端|半|角)"   # 一大片/一整圈/一侧
    rf"|半(?:个|圆|片|边)"
    rf"|整(?:片|圈|块)"
)

# 3) 未验证计数：中文数字 + 数量词，且**是**在对失效 die 计数
#    含「几/数/多」这类约数标记，或明确的大于一的数目
COUNT_RE = re.compile(
    rf"(?:[{CN}]{{1,3}}|[数几多]十[{CN}]{{0,2}}|[数几多]百)\s*[几]?\s*(?:颗|枚|粒|个|处|簇|团|块|条)"
)

TEXT_FIELDS = ["morphology", "caption_zh", "uncertainty"]


def classify(text: str) -> dict:
    """把一段文本里的数字类短语分成三类。**只分类，不改写。**"""
    if not isinstance(text, str):
        return {"clock": [], "shape": [], "count": []}

    # 先剔除钟点，剩下的才对形态/计数判断
    clocks = CLOCK_RE.findall(text)
    rest = CLOCK_RE.sub("〔钟点〕", text)

    shapes = SHAPE_RE.findall(rest)
    rest2 = SHAPE_RE.sub("〔形态〕", rest)

    counts = [m.group(0).strip() for m in COUNT_RE.finditer(rest2)]
    return {"clock": [c.strip() for c in clocks], "shape": shapes, "count": counts}


def audit_file(p: Path) -> dict:
    raw = p.read_bytes()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"path": str(p), "error": f"JSON 解析失败：{e}"}
    if not isinstance(obj, dict):
        return {"path": str(p), "error": "顶层不是对象"}
    per_field = {k: classify(obj.get(k, "")) for k in TEXT_FIELDS}
    return {
        "path": str(p),
        "clock_expressions_excluded": {k: v["clock"] for k, v in per_field.items() if v["clock"]},
        "shape_extent_terms": {k: v["shape"] for k, v in per_field.items() if v["shape"]},
        "**unverified_count_claims**": {k: v["count"] for k, v in per_field.items() if v["count"]},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--dir")
    a = ap.parse_args()

    files = [Path(x) for x in a.paths]
    if a.dir:
        files += sorted(Path(a.dir).glob("*_raw.json"))
    if not files:
        ap.print_help()
        return 2

    print("=" * 78)
    print("中文数量短语 · 确定性提醒（**只提示，不改任何文件**）")
    print("=" * 78)
    print("退出码：0 = 扫描完成且无待复核项；1 = **有待复核项**（不是「答案错误」）")

    n_hit, n_clock, n_shape = 0, 0, 0
    out = []
    for f in files:
        r = audit_file(f)
        out.append(r)
        if r.get("error"):
            print(f"\n  {f.name}: {r['error']}")
            continue
        c = r["**unverified_count_claims**"]
        cl = r["clock_expressions_excluded"]
        sp = r["shape_extent_terms"]
        if cl:
            n_clock += 1
        if sp:
            n_shape += 1
        if c:
            n_hit += 1
            print(f"\n  ⚠ {f.name}")
            for k, v in c.items():
                print(f"      未验证计数 [{k}]：{v}")
            if cl:
                print(f"      （已剔除的钟点，不计入：{cl}）")

    print(f"\n--- 汇总 ---")
    print(f"  扫描 {len(files)} 个文件")
    print(f"  **有待复核计数的**: {n_hit}")
    print(f"  含钟点表达的（**不算问题**）: {n_clock}")
    print(f"  含形态/范围用语的（**不算计数**）: {n_shape}")
    print()
    print("  重申：本工具**不判断**图里有几颗，也**不判断**答案对错；")
    print("        它只把『对数量的中文断言』挑出来供人工复核。")

    (Path(__file__).resolve().parent / "cn_quantity_audit_result.json").write_text(
        json.dumps({"files": out, "n_with_count_claims": n_hit,
                    "n_with_clock": n_clock, "n_with_shape": n_shape,
                    "note": "只提示，未修改任何原答；命中不等于答案错误。"},
                   ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    return 0 if n_hit == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
