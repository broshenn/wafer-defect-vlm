#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D 段存在性诊断的**解析器与离线回归**（纯 CPU，先于 GPU 跑）。

题面要求：`has_red_defect` 填 true / false；图片没有提供或无法可靠判断时填 **JSON null**。

口径（合同规定，不得放宽）：
  · **主口径**（严格）：唯一合法对象、唯一键 `has_red_defect`（允许附加键）、
    值必须是**真 JSON 布尔**；JSON null → `unknown`（单列未知，
    **不能当任意 gold 都满分**）；字符串 "true" / 数字 1 / 重复键 / NaN /
    空对象 / 截断 / 多对象 / 尾随文本 一律**拒**。
  · **副口径**（统一诊断）：剥完整外层 Markdown 围栏、去空 `<think></think>` 前缀后，
    再允许字符串 "true"/"false"（大小写不敏感）归一为布尔。**只在副口径生效。**
  · 必须**另写解析器**，不得复用要求 `defect_class` 的类别解析器——那会把全体判错。

跑法：python 解析回归.py   →  期望 30/30 通过
"""
from __future__ import annotations
import json, re, sys

EMPTY_THINK = re.compile(r"^\s*<think\b[^>]*>\s*</think\s*>\s*", re.S)
FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n?\s*```\s*$", re.S)
KEY = "has_red_defect"


class NonFinite(ValueError):
    pass


def _strip(text: str) -> str:
    return EMPTY_THINK.sub("", text.strip())


def parse_strict(text):
    """返回 (值, 判定)。值 ∈ {True, False, None(未知), _REJ}。"""
    if not isinstance(text, str):
        return _REJ, "not_str"
    s = _strip(text)
    if not s:
        return _REJ, "empty"

    def _nodup(pairs):
        k = [p[0] for p in pairs]
        if len(k) != len(set(k)):
            raise ValueError("dup")
        return dict(pairs)

    def _const(x):
        raise NonFinite(x)
    try:
        obj, end = json.JSONDecoder(object_pairs_hook=_nodup,
                                    parse_constant=_const).raw_decode(s)
    except NonFinite:
        return _REJ, "non_finite"
    except json.JSONDecodeError as e:
        # 注意顺序：JSONDecodeError 是 ValueError 的子类，
        # 若先 catch ValueError，截断会被误判成 duplicate_key。
        if isinstance(e, json.JSONDecodeError) and s.startswith("{") \
                and not s.rstrip().endswith("}"):
            return _REJ, "truncated"
        if s.count("{") > 1:
            return _REJ, "multiple_objects"
        return _REJ, "not_json"
    except ValueError:
        return _REJ, "duplicate_key"
    rest = s[end:].strip()
    if rest:
        return _REJ, ("multiple_objects" if "{" in rest else "trailing_text")
    if not isinstance(obj, dict):
        return _REJ, "not_object"
    if KEY not in obj:
        return _REJ, "missing_key"
    v = obj[KEY]
    if v is None:                       # 真 JSON null → 未知，单列
        return None, "unknown"
    if isinstance(v, bool):
        return v, "ok"
    return _REJ, f"bad_type:{type(v).__name__}"


def parse_diag(text):
    """统一诊断副口径：先剥围栏，再在严格口径上把字符串布尔归一。"""
    if not isinstance(text, str):
        return _REJ, "not_str"
    s = _strip(text)
    m = FENCE.match(s)
    if m:
        s = m.group(1).strip()
    val, how = parse_strict(s)
    if how.startswith("bad_type:str"):
        try:
            obj = json.loads(s)
            raw = obj.get(KEY)
        except Exception:
            return _REJ, "not_json"
        if isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
            return raw.strip().lower() == "true", "ok_str_bool"
    return val, how


_REJ = object()

# ---------------- 回归用例 ----------------
CASES = [
    ('{"has_red_defect": true}',                   True,  "ok"),
    ('{"has_red_defect": false}',                  False, "ok"),
    ('{"has_red_defect": null}',                   None,  "unknown"),
    ('<think>\n\n</think>\n{"has_red_defect": true}', True, "ok"),
    ('{"has_red_defect": true, "note": "x"}',      True,  "ok"),
    ('{"has_red_defect": "true"}',                 _REJ,  "bad_type:str"),
    ('{"has_red_defect": "false"}',                _REJ,  "bad_type:str"),
    ('{"has_red_defect": 1}',                      _REJ,  "bad_type:int"),
    ('{"has_red_defect": 0}',                      _REJ,  "bad_type:int"),
    ('{"has_red_defect": "null"}',                 _REJ,  "bad_type:str"),
    ('{"has_red_defect": true, "has_red_defect": false}', _REJ, "duplicate_key"),
    ('{"has_red_defect": NaN}',                    _REJ,  "non_finite"),
    ('{"has_red_defect": Infinity}',               _REJ,  "non_finite"),
    ('{}',                                         _REJ,  "missing_key"),
    ('{"other": true}',                            _REJ,  "missing_key"),
    ('{"has_red_defect": true',                    _REJ,  "truncated"),
    ('{"has_red_defect": true} {"x": 1}',          _REJ,  "multiple_objects"),
    ('{"has_red_defect": true} 说明如上',            _REJ,  "trailing_text"),
    ('',                                            _REJ,  "empty"),
    ('<think></think>',                             _REJ,  "empty"),
    ('true',                                        _REJ,  "not_object"),
    ('[true]',                                      _REJ,  "not_object"),
    ('null',                                        _REJ,  "not_object"),
    ('{"has_red_defect": [true]}',                 _REJ,  "bad_type:list"),
    ('{"has_red_defect": {}}',                     _REJ,  "bad_type:dict"),
    ('```json\n{"has_red_defect": true}\n```',      _REJ,  "not_json"),   # 主口径不剥围栏
]

DIAG_CASES = [
    ('```json\n{"has_red_defect": true}\n```',      True,  "ok"),
    ('```\n{"has_red_defect": false}\n```',         False, "ok"),
    ('{"has_red_defect": "True"}',                  True,  "ok_str_bool"),
    ('{"has_red_defect": "FALSE"}',                 False, "ok_str_bool"),
    ('{"has_red_defect": "yes"}',                   _REJ,  "bad_type:str"),
]


def main() -> int:
    bad = 0
    for txt, exp_v, exp_how in CASES:
        got_v, got_how = parse_strict(txt)
        ok = (got_v is exp_v or got_v == exp_v) and got_how == exp_how
        if not ok:
            bad += 1
            print(f"✗ 严格  {txt[:46]!r}\n   期望 {exp_v!r}/{exp_how}  得到 {got_v!r}/{got_how}")
    for txt, exp_v, exp_how in DIAG_CASES:
        got_v, got_how = parse_diag(txt)
        ok = (got_v is exp_v or got_v == exp_v) and got_how == exp_how
        if not ok:
            bad += 1
            print(f"✗ 副口径 {txt[:46]!r}\n   期望 {exp_v!r}/{exp_how}  得到 {got_v!r}/{got_how}")
    n = len(CASES) + len(DIAG_CASES)
    print(f"\n解析回归：{n - bad}/{n} 通过"
          f"（严格 {len(CASES)} + 副口径 {len(DIAG_CASES)}）")
    # 关键性质：null 必须落在 unknown，且**不得**被算成对 true 或 false 任一 gold 正确
    for g in (True, False):
        assert parse_strict('{"has_red_defect": null}')[0] is not g
    print("关键性质通过：JSON null 单列未知，对 true / false 两种 gold 都不判满分")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
