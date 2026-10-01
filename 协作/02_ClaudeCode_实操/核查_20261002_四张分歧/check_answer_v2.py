#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""七字段答案的格式检查器 v2 —— 修复踩雷记录 B35 与 C19。

不覆盖 ZCode 的 `协作/03_ZCode_标注/20261002-012138/check_answer.py`；那是原版，留着。

## 修的是什么

**B35**：旧检查器在 JSON 解析失败时 `sys.exit(0)`，其他字段失败也只打印 false 后正常退出。
退出码分不出「通过」和「格式失败」——用它做「失败即停」的判据会一路跑下去。

**C19**：另一个检查器把描述里**任何数字**都当尺寸，于是把合法的「7/8点钟」误报成尺寸错误。
本版先把钟点表达**剔掉**再查剩余数字。

## 退出码（非 0 即失败，可直接做失败即停的判据）

    0  全部通过
    2  不是合法 JSON（含截断）
    3  顶层不是对象（是数组/字符串/数字等）
    4  字段**集合**不符（缺字段或多字段；顺序不计入判定）
    5  字段类型错误
    6  枚举值未知
    7  extent_r 不是 null（本轮要求必须为 null）
    8  描述文本里出现未经验证的数值（钟点表达已剔除）

多个问题同时存在时，返回**编号最小的那个**，并在 `failures` 里列全。

## 本轮口径（写死在常量里，改口径要改代码并留痕）

- `defect_class` 合法值 = 九类 + `unknown`。**`unknown` 是合法的**，不算失败。
- `radial_zone` 合法值 = center/middle/edge/global/unknown/none。
- `extent_r` 必须为 `null`（任务书 §"本轮不验证数值尺寸能力"）。
- 描述中允许钟点方向（如「7点钟」「7/8点钟」「约7点钟方向」），不允许其余数字。

用法：
    python check_answer_v2.py <raw.json>              # 检查一个文件
    python check_answer_v2.py --dir <目录>            # 检查目录下所有 *_raw.json
    python check_answer_v2.py --self-test             # 跑内置失败路径自测
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_PARSE = 2
EXIT_NOT_OBJECT = 3
EXIT_FIELDS = 4
EXIT_TYPE = 5
EXIT_ENUM = 6
EXIT_EXTENT = 7
EXIT_NUMERIC = 8

CODE_NAME = {
    EXIT_PARSE: "不是合法 JSON（含截断）",
    EXIT_NOT_OBJECT: "顶层不是对象",
    EXIT_FIELDS: "字段名集合不符",
    EXIT_TYPE: "字段类型错误",
    EXIT_ENUM: "枚举值未知",
    EXIT_EXTENT: "extent_r 不是 null",
    EXIT_NUMERIC: "描述里出现数值（钟点已剔除）",
}

EXPECTED_FIELDS = ["defect_class", "morphology", "radial_zone", "clock_direction",
                   "extent_r", "caption_zh", "uncertainty"]
CLASS_ENUM = {"Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
              "Near_full", "Random", "Scratch", "none", "unknown"}
ZONE_ENUM = {"center", "middle", "edge", "global", "unknown", "none"}
TEXT_FIELDS = ["morphology", "caption_zh", "uncertainty"]

# ── 钟点表达（C19 的修复核心）────────────────────────────────
# 要剔除的形态：7点钟 / 7点 / 7/8点钟 / 7、8点 / 7-8点 / 7~8点 / 约7点钟方向 / 七点钟
# 数字用 ASCII 与中文都收；连接符含 / - ~ ～ 、 , ， 和 及 与
_CN_NUM = "0-9零一二三四五六七八九十两"
CLOCK_RE = re.compile(
    rf"[{_CN_NUM}]{{1,2}}\s*"
    rf"(?:[/\-~～、,，和及与]\s*[{_CN_NUM}]{{1,2}}\s*)*"
    rf"点(?:钟)?(?:方向)?"
)
# 剔除钟点后仍然允许出现的、非尺寸的数字/符号（本轮为空 —— 一律算失败）
ASCII_DIGIT_RE = re.compile(r"[0-9]")


def strip_clock(text: str) -> tuple[str, list[str]]:
    """把钟点表达换成占位符，返回 (剩余文本, 被剔除的片段)。"""
    removed: list[str] = []

    def _sub(m: re.Match) -> str:
        removed.append(m.group(0))
        return "〔钟点〕"

    return CLOCK_RE.sub(_sub, text), removed


def numeric_offenders(text: str) -> tuple[list[str], list[str]]:
    """返回 (违规的数字片段, 被剔除的钟点片段)。"""
    stripped, clocks = strip_clock(text)
    return ASCII_DIGIT_RE.findall(stripped), clocks


def check_object(obj) -> dict:
    """检查一个已解析的 JSON 对象。返回 result（含 exit_code）。"""
    failures: list[dict] = []
    detail: dict = {"clock_expressions_stripped": {}}

    if not isinstance(obj, dict):
        return {
            "exit_code": EXIT_NOT_OBJECT,
            "failures": [{"code": EXIT_NOT_OBJECT,
                          "message": f"顶层是 {type(obj).__name__}，要求 JSON 对象"}],
            "detail": detail,
        }

    # 4 字段名集合
    # 只查「集合」（缺字段/多字段），**不查顺序**：提示词第 6 行说的是
    # 「必须包含以下七字段」，那是包含要求，不是排列要求。
    # 注：两个历史检查器在这里不一致 —— ZCode 的 check_answer.py 查 keys_exact_order，
    #     Codex 复用的 kimi_k3_cost_test/run_kimi_k3_wafer.py:80-86 用集合。
    #     本版按提示词字面取集合口径，顺序只作为提示写进 detail。
    if set(obj.keys()) != set(EXPECTED_FIELDS):
        missing = sorted(set(EXPECTED_FIELDS) - set(obj.keys()))
        extra = sorted(set(obj.keys()) - set(EXPECTED_FIELDS))
        failures.append({
            "code": EXIT_FIELDS,
            "message": f"字段集合不符（缺 {missing}，多 {extra}）",
        })
        return {"exit_code": EXIT_FIELDS, "failures": failures, "detail": detail}

    detail["field_order_matches_prompt"] = list(obj.keys()) == EXPECTED_FIELDS

    # 5 类型
    # 非空要求逐字段定：提示词第 7 行写 uncertainty「不确定之处，无则空字符串」，
    # 所以它是**允许为空串**的；其余文本字段必须非空。
    nonempty = {"defect_class": True, "morphology": True,
                "radial_zone": True, "caption_zh": True, "uncertainty": False}
    for k, must_nonempty in nonempty.items():
        v = obj[k]
        if not isinstance(v, str) or (must_nonempty and not v.strip()):
            failures.append({"code": EXIT_TYPE,
                             "message": f"{k} 应为{'非空' if must_nonempty else ''}字符串，实为 {v!r}"})
    if not (obj["clock_direction"] is None or isinstance(obj["clock_direction"], str)):
        failures.append({"code": EXIT_TYPE,
                         "message": f"clock_direction 应为 null 或字符串，实为 {obj['clock_direction']!r}"})

    # 6 枚举（unknown 合法）
    if obj["defect_class"] not in CLASS_ENUM:
        failures.append({"code": EXIT_ENUM,
                         "message": f"defect_class={obj['defect_class']!r} 不在 {sorted(CLASS_ENUM)}"})
    if obj["radial_zone"] not in ZONE_ENUM:
        failures.append({"code": EXIT_ENUM,
                         "message": f"radial_zone={obj['radial_zone']!r} 不在 {sorted(ZONE_ENUM)}"})

    # 7 extent_r 本轮必须为 null
    if obj["extent_r"] is not None:
        failures.append({"code": EXIT_EXTENT,
                         "message": f"本轮 extent_r 必须为 null，实为 {obj['extent_r']!r}"})

    # 8 描述里的数值（先剔除钟点）
    for k in TEXT_FIELDS:
        v = obj[k]
        if not isinstance(v, str):
            continue
        digits, clocks = numeric_offenders(v)
        if clocks:
            detail["clock_expressions_stripped"][k] = clocks
        if digits:
            failures.append({"code": EXIT_NUMERIC,
                             "message": f"{k} 剔除钟点后仍含数字 {digits}：{v!r}"})

    code = min((f["code"] for f in failures), default=EXIT_OK)
    return {"exit_code": code, "failures": failures, "detail": detail}


def check_text(raw: str) -> dict:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"exit_code": EXIT_PARSE,
                "failures": [{"code": EXIT_PARSE, "message": f"JSON 解析失败：{e}"}],
                "detail": {}}
    return check_object(obj)


def check_file(p: Path) -> dict:
    return {"path": str(p), **check_text(p.read_text(encoding="utf-8"))}


# ── 自测：用真实失败路径验证，不发任何模型请求 ────────────────
GOOD_UNKNOWN = json.dumps({
    "defect_class": "unknown", "morphology": "形态不确定", "radial_zone": "unknown",
    "clock_direction": None, "extent_r": None, "caption_zh": "无法可靠判断。",
    "uncertainty": "图像信息不足"}, ensure_ascii=False)

GOOD_CLOCK = json.dumps({
    "defect_class": "Edge_Loc", "morphology": "左下边缘致密红簇",
    "radial_zone": "edge", "clock_direction": "约7点钟", "extent_r": None,
    "caption_zh": "晶圆左下边缘约7点钟方向有一小块致密失效簇，其余为稀疏红点。",
    "uncertainty": "边缘小簇与散点是否同源无法确认"}, ensure_ascii=False)

GOOD_CLOCK_RANGE = json.dumps({
    "defect_class": "Edge_Loc", "morphology": "7/8点钟边缘簇", "radial_zone": "edge",
    "clock_direction": "7/8点钟", "extent_r": None,
    "caption_zh": "边缘约7/8点钟方向有致密簇。", "uncertainty": ""}, ensure_ascii=False)

SELF_TEST_CASES: list[tuple[str, str, int]] = [
    ("合法：unknown 类别应通过", GOOD_UNKNOWN, EXIT_OK),
    ("合法：7点钟 描述应通过", GOOD_CLOCK, EXIT_OK),
    ("合法：7/8点钟 描述应通过", GOOD_CLOCK_RANGE, EXIT_OK),
    ("合法：字段顺序不同但集合完整（提示词只要求包含）",
     json.dumps({"uncertainty": "", "caption_zh": "中央簇。", "extent_r": None,
                 "clock_direction": None, "radial_zone": "center",
                 "morphology": "中心簇", "defect_class": "Center"}, ensure_ascii=False), EXIT_OK),
    ("合法：uncertainty 为空串（提示词写明「无则空字符串」）",
     json.dumps({"defect_class": "Center", "morphology": "中心簇", "radial_zone": "center",
                 "clock_direction": None, "extent_r": None, "caption_zh": "中央簇。",
                 "uncertainty": ""}, ensure_ascii=False), EXIT_OK),
    ("失败：截断 JSON",
     '{"defect_class":"Center","morphology":"中心簇"', EXIT_PARSE),
    ("失败：非法 JSON（单引号）",
     "{'defect_class':'Center'}", EXIT_PARSE),
    ("失败：顶层是数组", json.dumps(["Center"]), EXIT_NOT_OBJECT),
    ("失败：顶层是字符串", json.dumps("Center"), EXIT_NOT_OBJECT),
    ("失败：缺字段",
     json.dumps({"defect_class": "Center", "morphology": "x", "radial_zone": "center",
                 "clock_direction": None, "extent_r": None, "caption_zh": "y"},
                ensure_ascii=False), EXIT_FIELDS),
    ("失败：多字段（字段名集合不符）",
     json.dumps({"defect_class": "Center", "morphology": "x", "radial_zone": "center",
                 "clock_direction": None, "extent_r": None, "caption_zh": "y",
                 "uncertainty": "", "extra": 1}, ensure_ascii=False), EXIT_FIELDS),
    ("失败：类型错误（defect_class 是数字）",
     json.dumps({"defect_class": 3, "morphology": "x", "radial_zone": "center",
                 "clock_direction": None, "extent_r": None, "caption_zh": "y",
                 "uncertainty": ""}, ensure_ascii=False), EXIT_TYPE),
    ("失败：类型错误（morphology 为空串）",
     json.dumps({"defect_class": "Center", "morphology": "  ", "radial_zone": "center",
                 "clock_direction": None, "extent_r": None, "caption_zh": "y",
                 "uncertainty": ""}, ensure_ascii=False), EXIT_TYPE),
    ("失败：类型错误（clock_direction 是数字）",
     json.dumps({"defect_class": "Center", "morphology": "x", "radial_zone": "center",
                 "clock_direction": 7, "extent_r": None, "caption_zh": "y",
                 "uncertainty": ""}, ensure_ascii=False), EXIT_TYPE),
    ("失败：未知枚举（defect_class）",
     json.dumps({"defect_class": "Ring", "morphology": "x", "radial_zone": "center",
                 "clock_direction": None, "extent_r": None, "caption_zh": "y",
                 "uncertainty": ""}, ensure_ascii=False), EXIT_ENUM),
    ("失败：未知枚举（radial_zone）",
     json.dumps({"defect_class": "Center", "morphology": "x", "radial_zone": "outside",
                 "clock_direction": None, "extent_r": None, "caption_zh": "y",
                 "uncertainty": ""}, ensure_ascii=False), EXIT_ENUM),
    ("失败：extent_r 非 null（数值）",
     json.dumps({"defect_class": "Center", "morphology": "x", "radial_zone": "center",
                 "clock_direction": None, "extent_r": 0.65, "caption_zh": "y",
                 "uncertainty": ""}, ensure_ascii=False), EXIT_EXTENT),
    ("失败：extent_r 非 null（字符串）",
     json.dumps({"defect_class": "Center", "morphology": "x", "radial_zone": "center",
                 "clock_direction": None, "extent_r": "0.65R", "caption_zh": "y",
                 "uncertainty": ""}, ensure_ascii=False), EXIT_EXTENT),
    ("失败：描述里写数值尺寸（0.65R）",
     json.dumps({"defect_class": "Center", "morphology": "中心簇，跨度约0.65R",
                 "radial_zone": "center", "clock_direction": None, "extent_r": None,
                 "caption_zh": "y", "uncertainty": ""}, ensure_ascii=False), EXIT_NUMERIC),
    ("失败：描述里写覆盖率百分比",
     json.dumps({"defect_class": "Near_full", "morphology": "x", "radial_zone": "global",
                 "clock_direction": None, "extent_r": None,
                 "caption_zh": "失效 die 约占 85%。", "uncertainty": ""},
                ensure_ascii=False), EXIT_NUMERIC),
    ("边界：钟点 + 真实数值同时出现，仍要报失败",
     json.dumps({"defect_class": "Edge_Loc", "morphology": "x", "radial_zone": "edge",
                 "clock_direction": "8点钟", "extent_r": None,
                 "caption_zh": "8点钟方向边缘簇，跨度约 0.5R。", "uncertainty": ""},
                ensure_ascii=False), EXIT_NUMERIC),
]


def self_test() -> int:
    print("=" * 74)
    print(f"格式检查器 v2 自测（{len(SELF_TEST_CASES)} 条路径，不发任何模型请求）")
    print("=" * 74)
    bad = 0
    for name, raw, want in SELF_TEST_CASES:
        r = check_text(raw)
        ok = r["exit_code"] == want
        bad += not ok
        mark = "✓" if ok else "✗"
        got = r["exit_code"]
        print(f"  {mark} {name:44s} 期望 {want} 实得 {got}"
              f"{'' if ok else '   ← ' + (r['failures'][0]['message'][:60] if r['failures'] else '')}")
    print()
    print(f"通过 {len(SELF_TEST_CASES) - bad}/{len(SELF_TEST_CASES)}")
    # 关键断言：退出码必须能把失败和成功分开
    codes = {check_text(r)["exit_code"] for _, r, _ in SELF_TEST_CASES}
    print(f"出现过的退出码集合 = {sorted(codes)}  "
          f"({'可区分成功与失败' if 0 in codes and len(codes) > 1 else '⚠ 不可区分'})")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--dir")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    if a.dir:
        files = sorted(Path(a.dir).glob("*_raw.json"))
        if not files:
            print(f"目录里没有 *_raw.json：{a.dir}")
            return 2
        results = [check_file(f) for f in files]
        worst = max(r["exit_code"] for r in results)
        for r in results:
            name = Path(r["path"]).name
            print(f"  {'✓' if r['exit_code'] == 0 else '✗'} {name:34s} exit={r['exit_code']}"
                  f"  {CODE_NAME.get(r['exit_code'], '')}")
        print(f"\n{len(results)} 个文件；最差退出码 {worst}"
              f"（{CODE_NAME.get(worst, '全部通过')}）")
        return worst

    if not a.path:
        ap.print_help()
        return 2
    r = check_file(Path(a.path))
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        print(f"{r['path']}: exit={r['exit_code']} ({CODE_NAME.get(r['exit_code'], '通过')})")
        for f in r["failures"]:
            print(f"  - {f['message']}")
        if r["detail"].get("clock_expressions_stripped"):
            print(f"  已剔除的钟点表达：{r['detail']['clock_expressions_stripped']}")
    return r["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
