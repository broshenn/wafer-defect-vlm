#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reward_v2：多字段规则奖励（类别 / 覆盖档位 / 径向区带 / 方向），格式最多占一成。

**这是新模块，不覆盖也不修改 `收尾_RL90与报告_20261006/reward_v1.py`。**
reward_v1 只认单一 `defect_class` 字段，答案里多出任何一个字段都会被判非法 ——
拿它给 v2 题面评分会把所有回答都打 0。两者并存，按题面版本选用。

## 一、奖励构成（正式开跑前冻结，训练中不得改）

    事实分（占 0.9）＝ 各**适用**字段得分的加权平均，权重：
        defect_class   30%
        coverage_level 20%
        red_mass_zones 20%
        direction      20%（类型 0.6 + 扇区 0.4）
    格式分（占 0.1）＝ 结构、键集合、类型、枚举域、字段间一致性逐项检查的通过比例

    **适用性由参考端决定，不由模型声明。** 参考未知的字段权重归零并重新归一化；
    参考已知时回答 "unknown" 得 0 分，不给部分分；缺参考／接线错误显式抛错，不填 0 掩盖。

## 二、为什么这样设计（对着已知的投机路径）

  * 「全部答 unknown」→ 事实分 0（参考已知时 unknown 不算对），只剩格式分 ≤0.1。
  * 「格式全对、内容全错」→ 上限就是格式分 0.1。
  * 「一律答 none」→ 参考是 single/axis 的样本方向项得 0；
    且本版**参考 none 的样本占多数**，所以这条路径的分数完全由其余三项决定，
    不能靠方向项刷分 —— 这也是为什么方向项必须按参考端归一化。
  * 「编造方向」→ 参考是 none 时给单/双向 = 0；参考是 single 时给错扇区 = 0。

## 三、答案信封（与 v1 一致，题面里写明「不写围栏」，此处是容错不是要求）

  允许：开头一个**空** `<think></think>`；整体一个 ```json 围栏。
  其余一切从严：重复键、截断、多对象、尾随文本、数组、非对象 → 解析失败。

## 四、系统错误（必须显式失败，不能静默填 0）

  生成数 ≠ 参考数；缺参考列；参考值非法（类别不在九类内、扇区不是 1–12 整数、
  区间不是合法列表）；bool 冒充整数；NaN/Infinity 等非有限常量（由 strict 解析拒绝）。
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, List, Optional

CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
CLASS_VALID = set(CLASSES) | {"unknown"}
COVERAGE_LEVELS = ["none", "low", "medium", "high", "near_all"]
COVERAGE_VALID = set(COVERAGE_LEVELS) | {"unknown"}
ZONE_NAMES = ["center", "middle", "edge"]
DIRECTION_TYPES = ["single", "axis", "none"]
DIRECTION_VALID = set(DIRECTION_TYPES) | {"unknown"}

FIELDS = ["defect_class", "coverage_level", "red_mass_zones", "direction_type",
          "clock_sectors", "caption_zh", "uncertainty"]

W_FACT_TOTAL = 0.90
W_FORMAT = 0.10
W_CLASS, W_COVERAGE, W_ZONES, W_DIRECTION = 0.30, 0.20, 0.20, 0.20
W_DIR_TYPE, W_DIR_SECTORS = 0.60, 0.40

EMPTY_THINK = re.compile(r"^\s*<think\b[^>]*>\s*</think\s*>\s*", re.S)
FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.S)


class RewardSystemError(RuntimeError):
    """接线/数据层面的错误 —— 必须显式失败，不能填 0 掩盖。"""


class _DupKey(Exception):
    pass


class _NonFinite(Exception):
    pass


def _no_dup_pairs(pairs):
    seen = set()
    for k, _ in pairs:
        if k in seen:
            raise _DupKey(k)
        seen.add(k)
    return dict(pairs)


def _no_constant(value):
    raise _NonFinite(value)


def _is_int(x: Any) -> bool:
    """bool 是 int 的子类，必须显式排除（True 冒充扇区 1 是典型投机）。"""
    return isinstance(x, int) and not isinstance(x, bool)


def parse_answer(text: Any) -> tuple[Optional[dict], str]:
    """返回 (对象 或 None, 判定码)。严格：任何可疑形态一律判非法。"""
    if text is None:
        return None, "none_input"
    if not isinstance(text, str):
        return None, "not_str"
    s = text.strip()
    if not s:
        return None, "empty"
    s2 = EMPTY_THINK.sub("", s).strip()
    if not s2:
        return None, "only_think"
    s = s2
    m = FENCE.match(s)
    if m:
        s = m.group(1).strip()
        if not s:
            return None, "empty_fence"
    try:
        obj, end = json.JSONDecoder(object_pairs_hook=_no_dup_pairs,
                                     parse_constant=_no_constant).raw_decode(s)
    except _DupKey as e:
        return None, f"duplicate_key:{e}"
    except _NonFinite:
        return None, "non_finite"
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
    # 形状闸门：`clock_sectors` 必须是整数列表。**bool 是 int 的子类**，
    # `[true]` / `[false]` 会被 Python 当成 [1] / [0] 混过去，是典型投机形态 → 直接拒收。
    if "clock_sectors" in obj:
        cs = obj["clock_sectors"]
        if not isinstance(cs, list) or any(not _is_int(x) for x in cs):
            return None, "bad_clock_sectors"
        if any(not (1 <= x <= 12) for x in cs):
            return None, "clock_sectors_out_of_range"
    if "red_mass_zones" in obj:
        z = obj["red_mass_zones"]
        ok = isinstance(z, str) or (isinstance(z, list)
                                    and all(isinstance(x, str) for x in z))
        if not ok:
            return None, "bad_red_mass_zones"
    return obj, "ok"


def check_format(obj: Optional[dict], how: str) -> tuple[float, dict]:
    """格式分的逐项检查。返回 (通过比例, 明细)。"""
    checks: dict[str, bool] = {"可解析": obj is not None}
    if obj is None:
        checks.update({"键集合正确": False, "类型正确": False, "枚举域合法": False,
                       "字段间一致": False, "文本字段非空": False})
        return 0.0, {"checks": checks, "parse": how}

    keys = set(obj)
    checks["键集合正确"] = keys == set(FIELDS)

    types_ok = True
    if not isinstance(obj.get("defect_class"), str):
        types_ok = False
    if not isinstance(obj.get("coverage_level"), str):
        types_ok = False
    z = obj.get("red_mass_zones")
    if not (isinstance(z, str) or
            (isinstance(z, list) and all(isinstance(x, str) for x in z))):
        types_ok = False
    if not isinstance(obj.get("direction_type"), str):
        types_ok = False
    cs = obj.get("clock_sectors")
    if not (isinstance(cs, list) and all(_is_int(x) for x in cs)):
        types_ok = False
    for k in ("caption_zh", "uncertainty"):
        if not isinstance(obj.get(k), str):
            types_ok = False
    checks["类型正确"] = types_ok

    dom_ok = True
    if obj.get("defect_class") not in CLASS_VALID:
        dom_ok = False
    if obj.get("coverage_level") not in COVERAGE_VALID:
        dom_ok = False
    if isinstance(z, str):
        if z != "unknown":
            dom_ok = False
    elif isinstance(z, list):
        if len(set(z)) != len(z) or any(x not in ZONE_NAMES for x in z):
            dom_ok = False
    if obj.get("direction_type") not in DIRECTION_VALID:
        dom_ok = False
    if isinstance(cs, list):
        if any((not _is_int(x)) or x < 1 or x > 12 for x in cs):
            dom_ok = False
    checks["枚举域合法"] = dom_ok

    cons_ok = True
    dt = obj.get("direction_type")
    if isinstance(cs, list) and all(_is_int(x) for x in cs):
        if dt == "single" and len(cs) != 1:
            cons_ok = False
        elif dt == "axis" and (len(set(cs)) != 2 or len(cs) != 2):
            cons_ok = False
        elif dt in ("none", "unknown") and len(cs) != 0:
            cons_ok = False
    else:
        cons_ok = False
    checks["字段间一致"] = cons_ok

    checks["文本字段非空"] = bool(obj.get("caption_zh", "").strip())
    passed = sum(1 for v in checks.values() if v)
    return passed / len(checks), {"checks": checks, "parse": how}


# ── 参考端（gold）解析 ─────────────────────────────────────────────────────
def _parse_ref_list(v: Any, field: str, i: int) -> Optional[list]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s == "unknown":
            return None
        try:
            v = json.loads(s)
        except json.JSONDecodeError as e:
            raise RewardSystemError(f"第 {i} 条 {field} 不是合法 JSON：{v!r}（{e}）") from None
    if not isinstance(v, list):
        raise RewardSystemError(f"第 {i} 条 {field} 不是列表：{v!r}")
    return v


def parse_reference(row: dict, i: int) -> dict:
    """把一行数据里的参考列解析成结构化参考 + 适用掩码。缺列=接线错误，显式抛错。"""
    need = ["gold_class", "gold_coverage_level", "gold_zones", "gold_direction_type",
            "gold_sectors"]
    missing = [k for k in need if k not in row]
    if missing:
        raise RewardSystemError(f"第 {i} 条缺少参考列 {missing}；现有列 {sorted(row)}")
    gc = row["gold_class"]
    if not isinstance(gc, str) or gc.strip() not in set(CLASSES):
        raise RewardSystemError(f"第 {i} 条 gold_class 非法：{gc!r}（必须是九类之一）")
    cov = row["gold_coverage_level"]
    if cov is not None and (not isinstance(cov, str) or cov not in set(COVERAGE_LEVELS)):
        raise RewardSystemError(f"第 {i} 条 gold_coverage_level 非法：{cov!r}")
    zones = _parse_ref_list(row["gold_zones"], "gold_zones", i)
    dt = row["gold_direction_type"]
    if dt is not None and dt not in set(DIRECTION_TYPES) | {"unknown"}:
        raise RewardSystemError(f"第 {i} 条 gold_direction_type 非法：{dt!r}")
    secs = _parse_ref_list(row["gold_sectors"], "gold_sectors", i)
    for s in (secs or []):
        if not _is_int(s) or not (1 <= s <= 12):
            raise RewardSystemError(f"第 {i} 条 gold_sectors 含非法扇区：{s!r}")
    if dt in ("single", "axis") and secs is None:
        raise RewardSystemError(f"第 {i} 条 gold_direction_type={dt} 但没有扇区参考")
    d_elig = dt in ("single", "axis", "none")
    return {
        "class": gc.strip(),
        "coverage_level": cov,                       # None = 不适用
        "zones": zones,                              # None = 不适用
        "direction_type": dt,                        # None/"unknown" = 不适用
        "sectors": secs or [],
        "elig": {
            "class": True,
            "coverage_level": cov is not None,
            "zones": zones is not None,
            "direction": d_elig,
        },
    }


# ── 奖励本体 ───────────────────────────────────────────────────────────────
def score_one(obj: Optional[dict], ref: dict) -> dict:
    """单条打分。返回各分量与最终分。obj=None 表示解析失败。"""
    parts: dict[str, Optional[float]] = {"class": None, "coverage_level": None,
                                         "zones": None, "direction": None}
    detail: dict[str, Any] = {}
    if obj is not None:
        # 类别
        pred = obj.get("defect_class")
        parts["class"] = 1.0 if (isinstance(pred, str) and pred == ref["class"]) else 0.0
        detail["class_pred"] = pred
        # 覆盖档位
        pc = obj.get("coverage_level")
        parts["coverage_level"] = 1.0 if pc == ref["coverage_level"] else 0.0
        detail["coverage_pred"] = pc
        # 径向区带（按集合比较；题面已规定顺序，顺序不额外扣分）
        pz = obj.get("red_mass_zones")
        if isinstance(pz, list) and all(isinstance(x, str) for x in pz):
            parts["zones"] = 1.0 if set(pz) == set(ref["zones"]) else 0.0
        else:
            parts["zones"] = 0.0            # 字符串 "unknown" 对已知参考 = 0
        detail["zones_pred"] = pz
        # 方向：类型 0.6 + 扇区 0.4
        pd = obj.get("direction_type")
        ps = obj.get("clock_sectors")
        t_ok = 1.0 if pd == ref["direction_type"] else 0.0
        if isinstance(ps, list) and all(_is_int(x) for x in ps):
            s_ok = 1.0 if sorted(ps) == sorted(ref["sectors"]) else 0.0
        else:
            s_ok = 0.0
        parts["direction"] = W_DIR_TYPE * t_ok + W_DIR_SECTORS * s_ok
        detail["direction_pred"] = {"type": pd, "sectors": ps,
                                    "type_ok": bool(t_ok), "sectors_ok": bool(s_ok)}
    else:
        for k in parts:
            parts[k] = 0.0

    # 事实分：只在适用字段上归一化
    w = {"class": W_CLASS * ref["elig"]["class"],
         "coverage_level": W_COVERAGE * ref["elig"]["coverage_level"],
         "zones": W_ZONES * ref["elig"]["zones"],
         "direction": W_DIRECTION * ref["elig"]["direction"]}
    total_w = sum(w.values())
    if total_w <= 0:
        fact = 0.0
        detail["no_eligible_facts"] = True
    else:
        fact = sum(w[k] * parts[k] for k in parts) / total_w
    detail["eligible_weights"] = w
    detail["fact_score"] = round(fact, 6)
    return {"parts": parts, "fact": fact, "detail": detail}


class WaferMultiFieldReward:
    """ms-swift 奖励回调。`completions` 是生成结果，其余列由数据集提供。"""

    def __init__(self, args=None, **kwargs):
        self.args = args
        self.audit: List[dict] = []
        self.audit_limit = 8000

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        n = len(completions)
        for key in ("gold_class", "gold_coverage_level", "gold_zones",
                    "gold_direction_type", "gold_sectors"):
            if key not in kwargs:
                raise RewardSystemError(f"奖励回调没有收到参考列 {key!r}；收到 {sorted(kwargs)}")
            if len(kwargs[key]) != n:
                raise RewardSystemError(
                    f"{key} 长度 {len(kwargs[key])} != 生成数 {n}；拒绝静默截短")
        sample_ids = kwargs.get("sample_id") or [None] * n
        rewards: List[float] = []
        for i, comp in enumerate(completions):
            row = {k: kwargs[k][i] for k in ("gold_class", "gold_coverage_level",
                                             "gold_zones", "gold_direction_type",
                                             "gold_sectors")}
            ref = parse_reference(row, i)
            obj, how = parse_answer(comp)
            fmt, fmt_detail = check_format(obj, how)
            res = score_one(obj, ref)
            reward = W_FACT_TOTAL * res["fact"] + W_FORMAT * fmt
            rewards.append(float(reward))
            if len(self.audit) < self.audit_limit:
                self.audit.append({
                    "i": i, "sample_id": sample_ids[i], "parse": how,
                    "format_score": round(fmt, 4), "format_checks": fmt_detail["checks"],
                    "parts": res["parts"], "fact_score": res["detail"]["fact_score"],
                    "eligible_weights": res["detail"]["eligible_weights"],
                    "reward": round(float(reward), 6),
                    "ref": {"class": ref["class"], "coverage_level": ref["coverage_level"],
                            "zones": ref["zones"], "direction_type": ref["direction_type"],
                            "sectors": ref["sectors"]},
                    "detail": res["detail"],
                    "response": comp,
                })
        return rewards


# ── 自测 ───────────────────────────────────────────────────────────────────
def _ref(class_="Edge_Ring", cov="medium", zones=("middle", "edge"),
         dt="single", secs=(4,)) -> dict:
    return {"gold_class": class_, "gold_coverage_level": cov,
            "gold_zones": json.dumps(list(zones)) if zones is not None else "unknown",
            "gold_direction_type": dt,
            "gold_sectors": json.dumps(list(secs)) if secs is not None else None}


def _ans(**kw) -> str:
    base = {"defect_class": "Edge_Ring", "coverage_level": "medium",
            "red_mass_zones": ["middle", "edge"], "direction_type": "single",
            "clock_sectors": [4], "caption_zh": "边缘环形缺陷。", "uncertainty": ""}
    base.update(kw)
    return json.dumps(base, ensure_ascii=False, separators=(",", ":"))


def self_test() -> dict:
    cases = []
    rw = WaferMultiFieldReward()

    def run(resp, refkw=None, **kw):
        ref = _ref(**(refkw or {}))
        out = rw(completions=[resp], **{k: [v] for k, v in ref.items()})
        return out[0], rw.audit[-1]

    # 1 全对 → 1.0
    r, a = run(_ans())
    cases.append(("全对=1.0", abs(r - 1.0) < 1e-9, r))

    # 2 格式全对但事实全错 → 只剩格式分（<=0.1）
    r, a = run(_ans(defect_class="Loc", coverage_level="low",
                    red_mass_zones=["center"], direction_type="axis",
                    clock_sectors=[1, 7]))
    cases.append(("格式全对事实全错<=0.1", r <= W_FORMAT + 1e-9, r))

    # 3 全部 unknown → 低分
    r, a = run(_ans(defect_class="unknown", coverage_level="unknown",
                    red_mass_zones="unknown", direction_type="unknown",
                    clock_sectors=[], caption_zh="不确定"))
    cases.append(("全unknown<=0.1", r <= W_FORMAT + 1e-9, r))

    # 4 编造方向：参考 none，模型给 single
    r, a = run(_ans(direction_type="single", clock_sectors=[4]),
               refkw={"dt": "none", "secs": ()})
    r_ok, _ = run(_ans(direction_type="none", clock_sectors=[]),
                  refkw={"dt": "none", "secs": ()})
    cases.append(("编造方向低于答对", r < r_ok, (r, r_ok)))

    # 5 方向类型对但扇区错 → 只拿类型分
    #    事实分 = (0.30 + 0.20 + 0.20 + 0.20*0.6) / 0.90 = 0.9111…
    #    总分 = 0.9*0.9111 + 0.1*1.0 = 0.92
    r, a = run(_ans(clock_sectors=[7]), refkw={"secs": (4,)})
    cases.append(("扇区错只拿类型分=0.92", abs(r - 0.92) < 1e-6, r))

    # 6 参考未知的字段被排除（方向 unknown → 权重归零后仍能满分）
    r, a = run(_ans(direction_type="unknown", clock_sectors=[]),
               refkw={"dt": "unknown", "secs": None})
    cases.append(("参考未知则该项不计分", abs(r - (W_FACT_TOTAL * 1.0 + W_FORMAT)) < 1e-6, r))

    # 7 参考已知时答 unknown 不得满分
    r, a = run(_ans(defect_class="unknown"))
    cases.append(("参考已知答unknown不得满分", r < 0.9, r))

    # 8 重复键
    r, a = run('{"defect_class":"Edge_Ring","defect_class":"Loc"}')
    cases.append(("重复键=0分", r == 0.0, r))

    # 9 bool 冒充整数扇区
    r, a = run(_ans(clock_sectors=[True]))
    cases.append(("bool冒充扇区不得分", a["parts"]["direction"] == 0.0, a["parts"]["direction"]))

    # 10 非有限常量
    r, a = run('{"defect_class":"Edge_Ring","coverage_level":NaN}')
    cases.append(("非有限常量=解析失败", a["parse"] == "non_finite", a["parse"]))

    # 11 尾随文本 / 多对象
    r, a = run(_ans() + " 以上。")
    cases.append(("尾随文本=0", r == 0.0, r))
    r, a = run(_ans() + _ans())
    cases.append(("多对象=0", r == 0.0, r))

    # 12 空 think + 围栏容错
    r, a = run("<think>\n\n</think>\n\n```json\n" + _ans() + "\n```")
    cases.append(("空think+围栏容错", abs(r - 1.0) < 1e-9, r))

    # 13 非空 think 不放过
    r, a = run("<think>我猜是环</think>\n" + _ans())
    cases.append(("非空think=0", r == 0.0, r))

    # 14 字段间不一致（single 却给两个扇区）
    r, a = run(_ans(clock_sectors=[1, 7]))
    cases.append(("single给两扇区=一致性不过", a["format_checks"]["字段间一致"] is False, a["format_checks"]))

    # 15 系统错误必须显式失败
    sys_cases = []
    try:
        rw(completions=[_ans()])
        sys_cases.append(("缺全部参考列", False))
    except RewardSystemError:
        sys_cases.append(("缺全部参考列", True))
    try:
        rw(completions=[_ans(), _ans()], **{k: [v] for k, v in _ref().items()})
        sys_cases.append(("数量不等", False))
    except RewardSystemError:
        sys_cases.append(("数量不等", True))
    try:
        bad = _ref(); bad["gold_class"] = "Donut2"
        rw(completions=[_ans()], **{k: [v] for k, v in bad.items()})
        sys_cases.append(("gold类别非法", False))
    except RewardSystemError:
        sys_cases.append(("gold类别非法", True))
    try:
        bad = _ref(); bad["gold_direction_type"] = "single"; bad["gold_sectors"] = "null"
        rw(completions=[_ans()], **{k: [v] for k, v in bad.items()})
        sys_cases.append(("single却无扇区参考", False))
    except RewardSystemError:
        sys_cases.append(("single却无扇区参考", True))
    for name, ok in sys_cases:
        cases.append((name + "（系统错误）", ok, None))

    # 16 审计留痕完整
    rw2 = WaferMultiFieldReward()
    rw2(completions=[_ans(), _ans(defect_class="Loc")],
        sample_id=["s1", "s2"], **{k: [v, v] for k, v in _ref().items()})
    cases.append(("审计两条", len(rw2.audit) == 2, len(rw2.audit)))
    cases.append(("审计含分量", all("parts" in x and "eligible_weights" in x for x in rw2.audit), None))

    passed = sum(1 for _, ok, _ in cases if ok)
    return {"total": len(cases), "passed": passed, "all_passed": passed == len(cases),
            "cases": [{"name": n, "passed": bool(ok), "value": v} for n, ok, v in cases]}


def main() -> int:
    res = self_test()
    for c in res["cases"]:
        print(f"  {'OK ' if c['passed'] else 'FAIL'} {c['name']:28s} {c['value']}")
    print(f"通过 {res['passed']}/{res['total']}")
    return 0 if res["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
