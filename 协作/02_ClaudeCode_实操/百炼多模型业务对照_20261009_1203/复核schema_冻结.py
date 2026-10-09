#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""描述事实复核的**冻结枚举 schema** 与**回归**。

Codex 点名禁止用自由文本正则给描述结论打分，理由是上一轮我的汇总脚本有三个缺口：
  1. `CLEAN_LEAD` 把「部分支持」列成干净；
  2. `classify()` 在没找到问题词时**默认干净** —— 于是负向维度只写「有」「轻微」
     就可能被错误放行；
  3. `cell()` 在有结论字段时**忽略证据文本**。
并举例：某条在「主要结构是否遗漏」写「有」、证据说漏掉主要团块，却因为没命中问题词被放行。

**本文件把判据从「读自由文本」改成「只接受枚举值」，并先用六种真实出现过的
自由文本状态跑回归，证明映射正确，然后才允许复核。**

冻结的枚举（**六维统一四值，极性按维度定义**）：
  · 正向维度（形态支持 / 位置吻合）：`SUPPORTED` 相符 / `PARTIAL` 部分 /
    `CONTRADICTED` 不符 / `UNCERTAIN` 判不了
  · 负向维度（是否遗漏 / 是否无依据断言 / 格式问题）：`SUPPORTED` **干净** /
    `PARTIAL` **轻微** / `CONTRADICTED` **有问题** / `UNCERTAIN` 判不了
  → **`SUPPORTED` 恒为「好」，`CONTRADICTED` 恒为「坏」**，跨维度可比。

方向**另立两列**（Codex 要求「没给方向」与「方向答对」分开算）：
  · `applicable` ∈ {true, false, unknown} —— 这张图**该不该**有方向
  · `verdict` ∈ {SUPPORTED, PARTIAL, CONTRADICTED, UNCERTAIN, **NO_ANSWER**}
    —— `NO_ANSWER` = 模型没给方向（与「给了但错」分开）

跑法：python 复核schema_冻结.py    → 期望 20/20 通过
"""
from __future__ import annotations
import json, sys

VERDICTS = ["SUPPORTED", "PARTIAL", "CONTRADICTED", "UNCERTAIN"]
DIRECTION_VERDICTS = VERDICTS + ["NO_ANSWER"]
APPLICABLE = [True, False, "unknown"]
DIMS = ["主要形态是否有图像支持", "位置是否吻合", "主要结构是否有遗漏",
        "是否无依据断言", "格式约定问题"]
NEGATIVE_DIMS = {"主要结构是否有遗漏", "是否无依据断言", "格式约定问题"}

# ---- 自由文本 → 枚举 的映射规则（**这是被回归的东西**） ----
# 规则按**维度极性**分别写；含否定/归属不明的**一律不予放行**，标 UNCERTAIN 或待核。
RULES = {
    # 正向维度
    "pos": [
        (r"^\s*支持|^\s*相符|^\s*吻合|^\s*是\s*$", "SUPPORTED"),
        (r"^\s*基本支持|^\s*基本吻合|^\s*大部分(支持|吻合)|^\s*大体(支持|吻合)", "SUPPORTED"),
        (r"^\s*部分支持|^\s*部分吻合|^\s*部分相符", "PARTIAL"),
        (r"^\s*(轻微|轻度)", "PARTIAL"),
        (r"^\s*不吻合|^\s*不符|^\s*不支持|^\s*相反|^\s*说反", "CONTRADICTED"),
        (r"^\s*不确定|^\s*无法判断|^\s*存疑|^\s*判不了", "UNCERTAIN"),
        (r"^\s*$", "MISSING"),
    ],
    # 负向维度：**「无/没有」才是 SUPPORTED**
    "neg": [
        (r"^\s*无重大遗漏|^\s*无遗漏|^\s*无\.?$|^\s*没有", "SUPPORTED"),
        (r"^\s*无", "SUPPORTED"),
        (r"^\s*有重大|^\s*有(?!效)|^\s*严重|^\s*是\s*$", "CONTRADICTED"),
        (r"^\s*(轻微|轻度|部分)", "PARTIAL"),
        (r"^\s*不确定|^\s*无法判断|^\s*存疑", "UNCERTAIN"),
        (r"^\s*$", "MISSING"),
    ],
}


def to_verdict(dim: str, text: str):
    """把自由文本映射成枚举。**匹配不到就返回 NEEDS_REVIEW，不默认放行。**"""
    import re
    t = (text or "").strip()
    kind = "neg" if dim in NEGATIVE_DIMS else "pos"
    for pat, val in RULES[kind]:
        if re.search(pat, t):
            return val
    return "NEEDS_REVIEW"


# ---------------- 回归：六种真实出现过的自由文本状态 ----------------
CASES = [
    # (维度, 原文, 期望)
    ("主要结构是否有遗漏", "无重大遗漏。",              "SUPPORTED"),   # ← 上一轮被误判成「有问题」
    ("主要结构是否有遗漏", "无重大结构遗漏（本图无环/线）。", "SUPPORTED"),
    ("主要结构是否有遗漏", "有重大遗漏——漏掉左缘红色块。",   "CONTRADICTED"),
    ("主要结构是否有遗漏", "有",                        "CONTRADICTED"),  # ← Codex 举的例子
    ("主要结构是否有遗漏", "有。",                      "CONTRADICTED"),
    ("主要结构是否有遗漏", "轻微",                       "PARTIAL"),
    ("主要结构是否有遗漏", "部分遗漏。",                  "PARTIAL"),
    ("主要结构是否有遗漏", "不确定",                      "UNCERTAIN"),
    ("主要结构是否有遗漏", "",                          "MISSING"),
    # 正向维度
    ("主要形态是否有图像支持", "支持——红色随机散布与图相符。", "SUPPORTED"),
    ("主要形态是否有图像支持", "基本支持。",                "SUPPORTED"),
    ("主要形态是否有图像支持", "部分支持——点状成立，但「稀疏」不符。", "PARTIAL"),
    ("主要形态是否有图像支持", "不吻合——红占 46.5% 说成绝大部分。", "CONTRADICTED"),
    ("主要形态是否有图像支持", "不确定",                    "UNCERTAIN"),
    ("位置是否吻合", "吻合。",                            "SUPPORTED"),
    ("位置是否吻合", "不吻合——实为外圈偏多。",              "CONTRADICTED"),
    ("格式约定问题", "无。",                              "SUPPORTED"),
    ("格式约定问题", "有——clock_direction 写成字符串 null。", "CONTRADICTED"),
    ("格式约定问题", "轻微——钟点写法不统一。",              "PARTIAL"),
    # 关键：**归属不明的评论不能放行**
    ("主要结构是否有遗漏", "不知道该怎么算，大概也许吧。",     "NEEDS_REVIEW"),
]


def main() -> int:
    print("=" * 78)
    print("冻结 schema 回归（六种状态 + 归属不明）")
    print("=" * 78)
    bad = 0
    for dim, txt, exp in CASES:
        got = to_verdict(dim, txt)
        flag = "✓" if got == exp else "✗"
        if got != exp:
            bad += 1
        print(f"  {flag} [{dim[:6]}] {txt[:32]:<34} 期望={exp:<12} 得到={got}")
    print(f"\n回归：{len(CASES)-bad}/{len(CASES)} 通过")
    if bad:
        print("**回归未过 —— 不得开始复核**")
        return 2

    print("\n" + "=" * 78)
    print("关键性质自检")
    print("=" * 78)
    checks = [
        ("「无重大遗漏」不再被判成有问题",
         to_verdict("主要结构是否有遗漏", "无重大遗漏。") == "SUPPORTED"),
        ("「有」不再默认放行",
         to_verdict("主要结构是否有遗漏", "有") == "CONTRADICTED"),
        ("「部分支持」不再被列成干净",
         to_verdict("主要形态是否有图像支持", "部分支持——但「稀疏」不符。") == "PARTIAL"),
        ("归属不明返回 NEEDS_REVIEW，不默认通过",
         to_verdict("主要结构是否有遗漏", "不知道") == "NEEDS_REVIEW"),
        ("空值显式记为 MISSING，不等同 SUPPORTED",
         to_verdict("主要结构是否有遗漏", "") == "MISSING"),
    ]
    for name, okk in checks:
        print(f"  {'✓' if okk else '✗'} {name}")
        if not okk:
            bad += 1

    S = {
        "版本": "v1-冻结",
        "冻结时间": __import__("time").strftime("%Y-%m-%d"),
        "六维（统一四值，SUPPORTED 恒为好，CONTRADICTED 恒为坏）": {
            **{d: VERDICTS for d in DIMS},
        },
        "方向（两列分列）": {"applicable": APPLICABLE, "verdict": DIRECTION_VERDICTS},
        "非法值与缺字段": "必须显式记为 NEEDS_REVIEW / MISSING，**不得默认通过**",
        "禁止": "不得用自由文本正则直接给结论打分；正则只用于本文件的**回归**，"
                "正式复核只接受枚举值",
        "极性说明": "负向维度的「无」= SUPPORTED（干净）；「有」= CONTRADICTED（有问题）",
    }
    p = __file__.replace("复核schema_冻结.py", "复核schema_冻结.json")
    open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(S, ensure_ascii=False, indent=2) + "\n")
    print(f"\n写出 {p}")
    print("回归通过 → schema 冻结，可以开始复核")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
