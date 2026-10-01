#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""七字段答案的格式检查器 v3 —— 在 v2 基础上修「类型错误未先阻断」。

不覆盖任何历史检查器：
  * `协作/03_ZCode_标注/20261002-012138/check_answer.py` —— ZCode 原版
  * `kimi_k3_cost_test/run_kimi_k3_wafer.py:56` `validate_answer` —— 更早的共用版
  * `核查_20261002_四张分歧/check_answer_v2.py` —— 我的 v2

## v3 修的 bug（Codex 验收裁决 §4 指出，我已复现）

`defect_class=[]` 或 `radial_zone={}` 时，v2 先记一条类型失败，**然后继续做集合成员判断**：

    obj["defect_class"] not in CLASS_ENUM      # [] 不可哈希 → TypeError

实测：确实抛 `TypeError: unhashable type: 'list'`，进程以退出码 **1** 崩掉，
**不是声明里的 5**，而且失败报告**根本没写出来**——只剩一个 traceback。

v3 的做法：**类型检查先跑，任何类型错误直接返回 5，不进枚举/尺寸/数值检查**。
后面的检查依赖字段类型成立，类型不成立时跳过它们是正确行为，不是省略。

## 检查顺序与退出码（非 0 即失败）

    0  全部通过
    2  不是合法 JSON（含截断）
    3  顶层不是对象
    4  字段**集合**不符（缺/多字段；顺序不计入判定）
    5  字段类型错误            ← v3 起**先阻断**，不再继续后续检查
    6  枚举值未知
    7  extent_r 不是 null（本轮要求）
    8  描述文本里出现未经验证的数值（钟点表达已剔除）

返回的是**最早发生的那一类**的码；全部失败明细仍写在 `failures` 里，供诊断。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EXIT_OK, EXIT_PARSE, EXIT_NOT_OBJECT = 0, 2, 3
EXIT_FIELDS, EXIT_TYPE, EXIT_ENUM, EXIT_EXTENT, EXIT_NUMERIC = 4, 5, 6, 7, 8

CODE_NAME = {
    EXIT_PARSE: "不是合法 JSON（含截断）",
    EXIT_NOT_OBJECT: "顶层不是对象",
    EXIT_FIELDS: "字段集合不符",
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

# 非空要求逐字段定：提示词第 7 行写 uncertainty「不确定之处，无则空字符串」，
# 所以它允许空串；其余文本字段必须非空。
NONEMPTY = {"defect_class": True, "morphology": True, "radial_zone": True,
            "caption_zh": True, "uncertainty": False}

_CN_NUM = "0-9零一二三四五六七八九十两"
CLOCK_RE = re.compile(
    rf"[{_CN_NUM}]{{1,2}}\s*"
    rf"(?:[/\-~～、,，和及与]\s*[{_CN_NUM}]{{1,2}}\s*)*"
    rf"点(?:钟)?(?:方向)?"
)
ASCII_DIGIT_RE = re.compile(r"[0-9]")


def strip_clock(text: str) -> tuple[str, list[str]]:
    removed: list[str] = []

    def _sub(m: re.Match) -> str:
        removed.append(m.group(0))
        return "〔钟点〕"

    return CLOCK_RE.sub(_sub, text), removed


def check_object(obj) -> dict:
    detail: dict = {"clock_expressions_stripped": {}}

    # ── 顺序 3：顶层类型 ────────────────────────────────
    if not isinstance(obj, dict):
        return {"exit_code": EXIT_NOT_OBJECT,
                "failures": [{"code": EXIT_NOT_OBJECT,
                              "message": f"顶层是 {type(obj).__name__}，要求 JSON 对象"}],
                "detail": detail}

    # ── 顺序 4：字段集合（不看顺序）──────────────────────
    if set(obj.keys()) != set(EXPECTED_FIELDS):
        missing = sorted(set(EXPECTED_FIELDS) - set(obj.keys()))
        extra = sorted(set(obj.keys()) - set(EXPECTED_FIELDS))
        return {"exit_code": EXIT_FIELDS,
                "failures": [{"code": EXIT_FIELDS,
                              "message": f"字段集合不符（缺 {missing}，多 {extra}）"}],
                "detail": detail}

    detail["field_order_matches_prompt"] = list(obj.keys()) == EXPECTED_FIELDS

    # ── 顺序 5：类型 —— **先跑，出错就返回，不往下走** ────
    type_failures = []
    for k, must_nonempty in NONEMPTY.items():
        v = obj[k]
        if not isinstance(v, str) or (must_nonempty and not v.strip()):
            kind = "非空字符串" if must_nonempty else "字符串"
            type_failures.append({"code": EXIT_TYPE,
                                  "message": f"{k} 应为{kind}，实为 {type(v).__name__} {v!r}"})
    cd = obj["clock_direction"]
    if not (cd is None or isinstance(cd, str)):
        type_failures.append({"code": EXIT_TYPE,
                              "message": f"clock_direction 应为 null 或字符串，实为 {type(cd).__name__}"})
    er = obj["extent_r"]
    if not (er is None or isinstance(er, (int, float)) or isinstance(er, str)):
        # extent_r 只允许 null（本轮）；这里只做"是不是标量"的类型判断，
        # "必须为 null" 留给顺序 7。dict/list 这种结构性错误归类型。
        type_failures.append({"code": EXIT_TYPE,
                              "message": f"extent_r 类型异常：{type(er).__name__}"})
    if type_failures:
        return {"exit_code": EXIT_TYPE, "failures": type_failures, "detail": detail}

    # 到这里五个文本字段都确认是 str，集合成员判断才安全
    failures: list[dict] = []

    # ── 顺序 6：枚举（unknown 合法）──────────────────────
    if obj["defect_class"] not in CLASS_ENUM:
        failures.append({"code": EXIT_ENUM,
                         "message": f"defect_class={obj['defect_class']!r} 不在 {sorted(CLASS_ENUM)}"})
    if obj["radial_zone"] not in ZONE_ENUM:
        failures.append({"code": EXIT_ENUM,
                         "message": f"radial_zone={obj['radial_zone']!r} 不在 {sorted(ZONE_ENUM)}"})

    # ── 顺序 7：extent_r 本轮必须为 null ────────────────
    if obj["extent_r"] is not None:
        failures.append({"code": EXIT_EXTENT,
                         "message": f"本轮 extent_r 必须为 null，实为 {obj['extent_r']!r}"})

    # ── 顺序 8：描述里的数值（先剔除钟点）───────────────
    for k in TEXT_FIELDS:
        v = obj[k]
        stripped, clocks = strip_clock(v)
        digits = ASCII_DIGIT_RE.findall(stripped)
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


# ── 自测（不发任何模型请求）──────────────────────────────────
def _obj(**kw):
    base = {"defect_class": "Center", "morphology": "中心簇", "radial_zone": "center",
            "clock_direction": None, "extent_r": None, "caption_zh": "中央簇。",
            "uncertainty": ""}
    base.update(kw)
    return json.dumps(base, ensure_ascii=False)


SELF_TEST_CASES: list[tuple[str, str, int]] = [
    # 合法（必须通过）
    ("合法：unknown 类别", _obj(defect_class="unknown", radial_zone="unknown"), EXIT_OK),
    ("合法：7点钟 描述", _obj(defect_class="Edge_Loc", radial_zone="edge",
                              clock_direction="约7点钟",
                              caption_zh="左下边缘约7点钟方向有致密簇。"), EXIT_OK),
    ("合法：7/8点钟 描述", _obj(clock_direction="7/8点钟", caption_zh="约7/8点钟边缘簇。"), EXIT_OK),
    ("合法：uncertainty 空串", _obj(), EXIT_OK),
    ("合法：字段顺序不同但集合完整",
     json.dumps({"uncertainty": "", "caption_zh": "中央簇。", "extent_r": None,
                 "clock_direction": None, "radial_zone": "center",
                 "morphology": "中心簇", "defect_class": "Center"}, ensure_ascii=False), EXIT_OK),
    # 2 / 3
    ("失败：截断 JSON", '{"defect_class":"Center","morphology":"中心簇"', EXIT_PARSE),
    ("失败：非法 JSON（单引号）", "{'defect_class':'Center'}", EXIT_PARSE),
    ("失败：顶层是数组", json.dumps(["Center"]), EXIT_NOT_OBJECT),
    ("失败：顶层是字符串", json.dumps("Center"), EXIT_NOT_OBJECT),
    # 4
    ("失败：缺字段",
     json.dumps({"defect_class": "Center", "morphology": "x", "radial_zone": "center",
                 "clock_direction": None, "extent_r": None, "caption_zh": "y"},
                ensure_ascii=False), EXIT_FIELDS),
    ("失败：多字段", _obj(**{"extra": 1}), EXIT_FIELDS),
    # 5 —— v3 的重点补测
    ("失败：类型 defect_class=[]（v2 在此崩溃）", _obj(defect_class=[]), EXIT_TYPE),
    ("失败：类型 radial_zone={}（v2 在此崩溃）", _obj(radial_zone={}), EXIT_TYPE),
    ("失败：类型 defect_class 是数字", _obj(defect_class=3), EXIT_TYPE),
    ("失败：类型 defect_class 是 null", _obj(defect_class=None), EXIT_TYPE),
    ("失败：类型 morphology 空串", _obj(morphology="  "), EXIT_TYPE),
    ("失败：类型 clock_direction 是数字", _obj(clock_direction=7), EXIT_TYPE),
    ("失败：类型 extent_r 是列表", _obj(extent_r=[0.6]), EXIT_TYPE),
    ("失败：类型 uncertainty 是 null（合法的是空串，不是 null）",
     _obj(uncertainty=None), EXIT_TYPE),
    # 6
    ("失败：未知枚举 defect_class", _obj(defect_class="Ring"), EXIT_ENUM),
    ("失败：未知枚举 radial_zone", _obj(radial_zone="outside"), EXIT_ENUM),
    # 7
    ("失败：extent_r 非 null（数值）", _obj(extent_r=0.65), EXIT_EXTENT),
    ("失败：extent_r 非 null（字符串）", _obj(extent_r="0.65R"), EXIT_EXTENT),
    # 8
    ("失败：描述写数值尺寸 0.65R", _obj(morphology="中心簇，跨度约0.65R"), EXIT_NUMERIC),
    ("失败：描述写覆盖率百分比", _obj(caption_zh="失效 die 约占 85%。"), EXIT_NUMERIC),
    ("边界：钟点 + 真实数值同时出现，仍要报 8",
     _obj(clock_direction="8点钟", caption_zh="8点钟方向边缘簇，跨度约 0.5R。"), EXIT_NUMERIC),
]


def self_test() -> int:
    print("=" * 78)
    print(f"格式检查器 v3 自测（{len(SELF_TEST_CASES)} 条路径，不发任何模型请求）")
    print("=" * 78)
    bad = 0
    for name, raw, want in SELF_TEST_CASES:
        r = check_text(raw)
        ok = r["exit_code"] == want
        bad += not ok
        tail = "" if ok else "   ← " + (r["failures"][0]["message"][:56] if r["failures"] else "")
        print(f"  {'✓' if ok else '✗'} {name:44s} 期望 {want} 实得 {r['exit_code']}{tail}")
    print(f"\n通过 {len(SELF_TEST_CASES) - bad}/{len(SELF_TEST_CASES)}")
    codes = sorted({check_text(r)["exit_code"] for _, r, _ in SELF_TEST_CASES})
    print(f"覆盖到的退出码 = {codes}")
    print(f"是否覆盖全部 8 个码: {codes == [0, 2, 3, 4, 5, 6, 7, 8]}")
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
            print(f"  {'✓' if r['exit_code'] == 0 else '✗'} {Path(r['path']).name:34s} "
                  f"exit={r['exit_code']}  {CODE_NAME.get(r['exit_code'], '')}")
        print(f"\n{len(results)} 个文件；最差退出码 {worst}（{CODE_NAME.get(worst, '全部通过')}）")
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
