#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B/C 用的新解析器：只从**唯一** defect_class 取类别，**允许附加字段**。

与旧的严格解析器的关键差别：旧的要求对象只能有一个字段，
那会把 C 组的附加描述**全部判错**。本解析器只约束：
  * 必须有且只有一个 defect_class；
  * 取值必须是九类之一或 unknown；
  * 拒绝重复键、截断、多对象、尾随文本；
  * **允许**其它字段存在。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
VALID = set(CLASSES) | {"unknown"}
EMPTY_THINK = re.compile(r"^\s*<think>\s*</think>\s*", re.S)
FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.S)


class _DupKey(Exception):
    pass


def _no_dup(pairs):
    seen = set()
    for k, _ in pairs:
        if k in seen:
            raise _DupKey(k)
        seen.add(k)
    return dict(pairs)


def parse_class(text: Any) -> tuple[Optional[str], str]:
    """返回 (类别 或 None, 判定码)。"""
    if text is None:
        return None, "none_input"
    if not isinstance(text, str):
        return None, "not_str"
    s = text.strip()
    if not s:
        return None, "empty"
    s2 = EMPTY_THINK.sub("", s).strip()
    used_think = s2 != s
    if not s2:
        return None, "only_think"
    s = s2
    m = FENCE.match(s)
    if m:
        s = m.group(1).strip()
        if not s:
            return None, "empty_fence"
    try:
        obj, end = json.JSONDecoder(object_pairs_hook=_no_dup).raw_decode(s)
    except _DupKey as e:
        return None, f"duplicate_key:{e}"
    except json.JSONDecodeError:
        if s.startswith("{") and not s.rstrip().endswith("}"):
            return None, "truncated"
        if s.count("{") > 1:
            return None, "multiple_objects"
        return None, "not_json"
    if s[end:].strip():
        return None, "trailing_text"
    if not isinstance(obj, dict):
        return None, "not_object"
    if "defect_class" not in obj:
        return None, "missing_field"
    v = obj["defect_class"]
    if not isinstance(v, str):
        return None, "value_not_str"
    v2 = v.strip()
    if v2 not in VALID:
        return None, f"bad_enum:{v2[:24]}"
    keys = [k for k in obj if k != "defect_class"]
    tag = "ok" + ("_think" if used_think else "") + (f"_plus{len(keys)}f" if keys else "")
    if v2 == "unknown":
        tag += "_unknown"
    return v2, tag


# ── 自测 ────────────────────────────────────────────────
CASES = [
    ("A 只类别", '{"defect_class":"Random"}', "Random", "ok"),
    ("B 带 geometry", '{"defect_class":"Random","geometry":{"radial_zone":"global","defect_ratio":0.31,"extent_r":null,"clock_direction":null}}', "Random", "ok_plus1f"),
    ("C 带描述", '{"defect_class":"Edge_Ring","caption_zh":"边缘整圈红色失效 die，呈环状。"}', "Edge_Ring", "ok_plus1f"),
    ("空 think 包装", '<think>\n\n</think>\n\n{"defect_class":"Loc"}', "Loc", "ok_think"),
    ("围栏", '```json\n{"defect_class":"Donut"}\n```', "Donut", "ok"),
    ("unknown", '{"defect_class":"unknown"}', "unknown", "ok_unknown"),
    ("**旧解析器会误杀**：带附加字段的 C", '{"defect_class":"Loc","morphology":"散点","radial_zone":"global"}', "Loc", "ok_plus2f"),
    ("重复键", '{"defect_class":"Loc","defect_class":"Center"}', None, "duplicate_key:defect_class"),
    ("截断", '{"defect_class":"Edge_', None, "truncated"),
    ("多对象", '{"defect_class":"Loc"}{"defect_class":"Center"}', None, "trailing_text"),
    ("尾随文本", '{"defect_class":"Loc"} 以上。', None, "trailing_text"),
    ("缺字段", '{"morphology":"x"}', None, "missing_field"),
    ("非九类", '{"defect_class":"edge_ring"}', None, "bad_enum:edge_ring"),
    ("空", "", None, "empty"),
]
if __name__ == "__main__":
    print("=" * 78)
    print("新解析器自测")
    print("=" * 78)
    nf = 0
    for name, txt, want_p, want_tag in CASES:
        p, tag = parse_class(txt)
        ok = (p == want_p) and (tag == want_tag)
        nf += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {name:34s} pred={str(p):11s} tag={tag}")
    print(f"\n通过 {len(CASES)-nf}/{len(CASES)}")
    print("\n**关键**：第 7 条「带附加字段」在旧解析器下会被判 extra_fields → 0 分；")
    print("       新解析器只取 defect_class，因此 C 组的描述不会被误杀。")
