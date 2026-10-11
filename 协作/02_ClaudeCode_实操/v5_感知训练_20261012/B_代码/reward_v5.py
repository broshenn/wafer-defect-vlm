#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5 奖励（合同第 7 节）。
格式不合法（3.5）→ 奖励 -1，不再算其他项。
合法时按分量加权：class/coverage/zones/angular/measure/consistency；
参考某字段为 unknown 时该分量从权重中移除后重新归一。
模式由环境变量 WAFER_REWARD_MODE ∈ {full, binary, noproc} 决定，未设置即抛异常。
奖励类签名与 v3 相同：__call__(completions, reference_json=None, sample_id=None, **kwargs)，
缺列或长度不齐直接抛异常；每条记录审计（含 had_empty_think_prefix 与 raw_ok）。
"""
from __future__ import annotations

import os

from common_v5 import (COV_ORDER, FIELDS, band_of_ratio, parse_v5,
                       parse_raw_strict, schema_errors, zones_of_frac)

WEIGHTS = {
    "full": {"class": 0.30, "coverage": 0.15, "zones": 0.15, "angular": 0.15,
             "measure": 0.15, "consistency": 0.10},
    "binary": {"class": 0.30, "coverage": 0.15, "zones": 0.15, "angular": 0.15,
               "measure": 0.15, "consistency": 0.10},
    "noproc": {"class": 0.30, "coverage": 0.15, "zones": 0.15, "angular": 0.15},
}
MODES = ("full", "binary", "noproc")


def _ref_six(ref: dict) -> dict:
    return {k: ref[k] for k in FIELDS}


def _coverage_part(model_band, ref_band, mode):
    if model_band == "unknown" or model_band not in COV_ORDER:
        return 0.0
    i, j = COV_ORDER.index(model_band), COV_ORDER.index(ref_band)
    if mode == "binary":
        return 1.0 if i == j else 0.0
    d = abs(i - j)
    return 1.0 if d == 0 else (0.5 if d == 1 else 0.0)


def _zones_part(model_zones, ref_zones, mode):
    if model_zones == "unknown":
        return 0.0
    ms, rs = set(model_zones), set(ref_zones)
    if mode == "binary":
        return 1.0 if ms == rs else 0.0
    union = ms | rs
    if not union:
        return 1.0
    return len(ms & rs) / len(union)


def _dist12(a, b):
    d = abs(a - b) % 12
    return min(d, 12 - d)


def _dist6(a, b):
    d = abs(a - b) % 6
    return min(d, 6 - d)


def _angular_part(obj, ref, mode):
    mt, rt = obj["angular_type"], ref["angular_type"]
    msec, rsec = obj["clock_sectors"], ref["clock_sectors"]
    if mode == "binary":
        return 1.0 if (mt == rt and set(msec) == set(rsec)) else 0.0
    if mt == "unknown":
        return 0.0
    type_ok = 1.0 if mt == rt else 0.0
    if rt == "single":
        if mt == "single" and len(msec) == 1 and len(rsec) == 1:
            d = _dist12(msec[0], rsec[0])
            s = 1.0 if d == 0 else (0.5 if d == 1 else 0.0)
        else:
            s = 0.0
    elif rt == "opposed":
        if mt == "opposed" and len(msec) == 2 and len(rsec) == 2:
            a, b = sorted(x % 6 for x in msec)
            c, d2 = sorted(x % 6 for x in rsec)
            dist = min(_dist6(a, c), _dist6(b, d2))
            # 允许两扇区对应配错顺序：取两种配对的较小距离
            dist_alt = min(_dist6(a, d2), _dist6(b, c))
            dist = min(dist, dist_alt)
            s = 1.0 if dist == 0 else (0.5 if dist == 1 else 0.0)
        else:
            s = 0.0
    elif rt == "nondirectional":
        s = 1.0 if (mt == "nondirectional" and not msec) else 0.0
    else:
        s = 0.0
    return 0.6 * type_ok + 0.4 * s


def _measure_part(model_m, ref_exact, mode):
    if model_m is None:
        return 0.0
    dr = abs(model_m["red_ratio"] - ref_exact["red_ratio"])
    l1 = sum(abs(a - b) for a, b in zip(model_m["zone_frac"], ref_exact["zone_frac"]))
    d1 = abs(model_m["R1"] - ref_exact["R1"])
    d2 = abs(model_m["R2"] - ref_exact["R2"])
    if mode == "binary":
        return sum([dr <= 0.02, l1 <= 0.2, d1 <= 0.1, d2 <= 0.1]) / 4.0
    parts = [max(0.0, 1.0 - dr / 0.10), 1.0 - l1 / 2.0,
             max(0.0, 1.0 - d1 / 0.3), max(0.0, 1.0 - d2 / 0.3)]
    return sum(parts) / 4.0


def _consistency_part(obj, model_m):
    if model_m is None:
        return 0.0
    c1 = 1.0 if band_of_ratio(model_m["red_ratio"]) == obj["coverage_band"] else 0.0
    if obj["red_mass_zones"] == "unknown":
        c2 = 0.0
    else:
        c2 = 1.0 if zones_of_frac(model_m["zone_frac"]) == list(obj["red_mass_zones"]) else 0.0
    return (c1 + c2) / 2.0


def score_v5(text, ref: dict, mode: str) -> dict:
    """单条打分。ref 为完整 reference 字典（六字段 + measure/measure_exact 等）。"""
    if mode not in MODES:
        raise ValueError(f"mode 非法: {mode}")
    if schema_errors(_ref_six(ref)):
        raise ValueError("invalid_reference")
    out = {"reward": -1.0, "ok": False, "raw_ok": False, "parts": {},
           "weights_used": {}, "errors": []}
    parsed = parse_v5(text)
    out["raw_ok"] = parsed["raw_ok"]
    if not parsed["ok"]:
        out["errors"] = parsed["errors"]
        return out
    obj, model_m = parsed["obj"], parsed["measure"]
    out["ok"] = True
    weights = dict(WEIGHTS[mode])
    if ref["coverage_band"] == "unknown":
        del weights["coverage"]
    if ref["red_mass_zones"] == "unknown":
        del weights["zones"]
    if ref["angular_type"] == "unknown":
        del weights["angular"]
    parts = {"class": (1.0 if obj["defect_class"] == ref["defect_class"]
                       else (0.2 if obj["defect_class"] == "unknown" else -1.0))}
    if "coverage" in weights:
        parts["coverage"] = _coverage_part(obj["coverage_band"], ref["coverage_band"], mode)
    if "zones" in weights:
        parts["zones"] = _zones_part(obj["red_mass_zones"], ref["red_mass_zones"], mode)
    if "angular" in weights:
        parts["angular"] = _angular_part(obj, ref, mode)
    if "measure" in weights:
        parts["measure"] = _measure_part(model_m, ref["measure_exact"], mode)
    if "consistency" in weights:
        parts["consistency"] = _consistency_part(obj, model_m)
    total_w = sum(weights.values())
    out["parts"] = parts
    out["weights_used"] = weights
    out["reward"] = sum(weights[k] * parts[k] for k in weights) / total_w
    return out


class WaferV5Reward:
    """ms-swift 奖励回调（合同 7/6.3）。"""

    def __init__(self):
        self.mode = os.environ.get("WAFER_REWARD_MODE", "")
        if self.mode not in MODES:
            raise RuntimeError(f"WAFER_REWARD_MODE 未设置或非法: {self.mode!r}")
        self.audit = []

    def __call__(self, completions, reference_json=None, sample_id=None, **kwargs):
        if reference_json is None or len(reference_json) != len(completions):
            raise RuntimeError("Missing or misaligned reference_json")
        if sample_id is None or len(sample_id) != len(completions):
            raise RuntimeError("Missing or misaligned sample_id")
        out = []
        for text, raw_ref, sid in zip(completions, reference_json, sample_id):
            if not isinstance(text, str):
                raise RuntimeError("Completion contract expects string")
            ref = parse_raw_strict(raw_ref)
            if schema_errors(_ref_six(ref)):
                raise RuntimeError(f"reference schema invalid: {sid}")
            result = score_v5(text, ref, self.mode)
            from common_v5 import strip_envelope
            self.audit.append({"sample_id": sid, "response": text,
                               "had_empty_think_prefix": strip_envelope(text) != text.strip(),
                               "mode": self.mode, **result})
            out.append(result["reward"])
        return out
