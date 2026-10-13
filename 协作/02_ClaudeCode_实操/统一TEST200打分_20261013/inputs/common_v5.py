#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5 公共常量与解析（合同 3.5）。
来源：`协作/02_ClaudeCode_实操/一天半优化_20261010/core_v3_复核修复.py`
（本地复制、不 import 旧文件；复制部分：EMPTY_THINK, FENCE, strip_envelope, _pairs,
_constant, parse_raw_strict, schema_errors, CLASSES, COVERAGE, ANGULAR, ZONES, FIELDS；
旧文件 SHA256=6bdff46b093fc7e5f19b2f5f64ba38a1f83156751e5742038ee8ffa60f9e8a5e）。
新增：MEASURE_RE、parse_v5、COV_ORDER、band_of_ratio、zones_of_frac（合同 3.5）。
"""
from __future__ import annotations

import json
import re

CLASSES = ['Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_full', 'Random',
           'Scratch', 'none']
COVERAGE = ['zero', 'under_5pct', '5_to_under_20pct', '20_to_under_50pct',
            'half_or_more', 'unknown']
ANGULAR = ['single', 'opposed', 'nondirectional', 'unknown']
ZONES = ['center', 'middle', 'edge']
FIELDS = ['defect_class', 'coverage_band', 'red_mass_zones', 'angular_type',
          'clock_sectors', 'uncertainty']

COV_ORDER = ["zero", "under_5pct", "5_to_under_20pct", "20_to_under_50pct", "half_or_more"]

# ── 信封（与 v3 复核修复版一致）──
EMPTY_THINK = re.compile(r'^\s*<think\b[^>]*>\s*</think\s*>\s*', re.S)
FENCE = re.compile(r'^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$', re.S)

# ── 合同 3.5 第 2 条：剥包装后的整体正则 ──
MEASURE_RE = re.compile(
    r'^<measure>red_ratio=(\d\.\d{2}); zone_frac=(\d\.\d),(\d\.\d),(\d\.\d); '
    r'R1=(\d\.\d); R2=(\d\.\d)</measure>\s*\n\s*(\{.*\})\s*$', re.S)


def strip_envelope(text: str) -> str:
    s = EMPTY_THINK.sub('', text.strip())
    m = FENCE.match(s)
    if m:
        s = m.group(1).strip()
    return s


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate_key')
        obj[key] = value
    return obj


def _constant(_):
    raise ValueError('non_finite')


def parse_raw_strict(text: str) -> dict:
    """原始全文严格 JSON：不剥任何包装。"""
    if not isinstance(text, str):
        raise ValueError('answer_not_string')
    obj = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    if not isinstance(obj, dict):
        raise ValueError('answer_not_object')
    return obj


def schema_errors(obj: dict) -> list:
    errors = []
    if not isinstance(obj, dict):
        return ['not_object']
    if set(obj) != set(FIELDS):
        errors.append('fields')
    if obj.get('defect_class') not in CLASSES + ['unknown']:
        errors.append('class')
    if obj.get('coverage_band') not in COVERAGE:
        errors.append('coverage')
    if obj.get('angular_type') not in ANGULAR:
        errors.append('angular')
    z = obj.get('red_mass_zones')
    if z != 'unknown' and (not isinstance(z, list) or any(x not in ZONES for x in z)
                           or len(z) != len(set(z))):
        errors.append('zones')
    sectors = obj.get('clock_sectors')
    valid_sectors = (isinstance(sectors, list) and
                     all(type(x) is int and 1 <= x <= 12 for x in sectors) and
                     len(sectors) == len(set(sectors)))
    if not valid_sectors:
        errors.append('sectors')
    else:
        a = obj.get('angular_type')
        if ((a == 'single' and len(sectors) != 1) or
            (a == 'opposed' and (len(sectors) != 2 or (sectors[0] - sectors[1]) % 12 != 6)) or
            (a in ['nondirectional', 'unknown'] and sectors)):
            errors.append('angular_sector_consistency')
    if not isinstance(obj.get('uncertainty'), str) or len(obj.get('uncertainty', '')) > 200:
        errors.append('uncertainty')
    return errors


def parse_v5(text):
    """合同 3.5：先剥空 think 前缀与单围栏，再整体匹配正则+严格 JSON+六字段 schema。
    返回 {"ok","raw_ok","measure","obj","errors"}；raw_ok = 原始全文（不剥包装）
    lstrip 后以 <measure> 开头且整体匹配同一正则（不参与奖励）。"""
    res = {"ok": False, "raw_ok": False, "measure": None, "obj": None, "errors": []}
    if not isinstance(text, str):
        res["errors"].append("answer_not_string")
        return res
    res["raw_ok"] = MEASURE_RE.match(text.lstrip()) is not None
    stripped = strip_envelope(text)
    m = MEASURE_RE.match(stripped)
    if not m:
        res["errors"].append("measure_re")
        return res
    measure = {"red_ratio": float(m.group(1)),
               "zone_frac": [float(m.group(2)), float(m.group(3)), float(m.group(4))],
               "R1": float(m.group(5)), "R2": float(m.group(6))}
    res["measure"] = measure
    try:
        obj = parse_raw_strict(m.group(7))
    except (ValueError, TypeError) as exc:
        res["errors"].append(f"json_{type(exc).__name__}:{exc}")
        return res
    errs = schema_errors(obj)
    res["obj"] = obj
    if errs:
        res["errors"].extend(errs)
        return res
    res["ok"] = True
    return res


def band_of_ratio(x: float) -> str:
    """合同 3.5：0 → zero；(0,0.05) → under_5pct；[0.05,0.20) → 5_to_under_20pct；
    [0.20,0.50) → 20_to_under_50pct；≥0.50 → half_or_more。"""
    if x == 0:
        return "zero"
    if x < 0.05:
        return "under_5pct"
    if x < 0.20:
        return "5_to_under_20pct"
    if x < 0.50:
        return "20_to_under_50pct"
    return "half_or_more"


def zones_of_frac(fr) -> list:
    """由 zone_frac 推区带集合：占比 ≥0.1 的区带（按 center,middle,edge 顺序）。"""
    return [z for z, f in zip(ZONES, fr) if f >= 0.1]
