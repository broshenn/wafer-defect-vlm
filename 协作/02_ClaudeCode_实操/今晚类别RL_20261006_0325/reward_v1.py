#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reward_v1：**只奖励类别答对**。

规则（照任务书 §2）：
  * 唯一奖励：有效答案类别 == 可信类别 → 1.0，否则 0.0。
  * **不加**格式、几何、长度或 LLM 评分奖励。
  * 字段必须是 `defect_class`；只允许九类或 `unknown`。
  * unknown / 错类 / 非法答案 → 0。
  * 允许**已验证的空 think 包装**和**单个 JSON 围栏**（val 原答里两者都有）。
  * **不做**类别词子串搜索；**不从多个互相矛盾的答案里挑一个**。
  * 不奖励工艺推断或额外字段。
  * 拒绝：重复键、截断、多对象、缺字段、错 schema、尾随文本。
  * 空值不能与任何 gold 相等得分。
  * **缺 gold / 无效 gold / 输入输出数量不等 = 系统错误**，显式抛错，
    不 zip 静默截短、不填 0 掩盖接线故障。

为什么不能用旧插件：`code/tools/wafer_grpo_plugin.py:105` 比较的是 `defect_type`，
而当前答题字段是 `defect_class`。若答案与 gold 都改成新字段名，旧实现会把**双方缺失值
都转成空字符串**，于是"空 == 空"被错误地判为满分。这个模块显式区分"缺字段"与"答错"。
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
VALID = set(CLASSES) | {"unknown"}

# 只允许**空**的 think 包装；非空 think 里的内容不参与判分，但也不被当作答案
EMPTY_THINK = re.compile(r"^\s*<think>\s*</think>\s*", re.S)
# 单个 JSON 围栏，且必须覆盖到结尾
FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.S)


class RewardSystemError(RuntimeError):
    """接线/数据层面的错误 —— 必须显式失败，不能填 0 掩盖。"""


class _DupKey(Exception):
    pass


def _no_dup_pairs(pairs):
    seen = set()
    for k, _ in pairs:
        if k in seen:
            raise _DupKey(k)
        seen.add(k)
    return dict(pairs)


def parse_answer(text: Any) -> tuple[Optional[str], str]:
    """返回 (类别 或 None, 判定码)。**严格**：任何可疑形态一律判非法。"""
    if text is None:
        return None, "none_input"
    if not isinstance(text, str):
        return None, "not_str"
    s = text.strip()
    if not s:
        return None, "empty"

    # 1) 允许**空** think 包装（可以有，也可以没有）
    s2 = EMPTY_THINK.sub("", s).strip()
    used_think = s2 != s
    if not s2:
        return None, "only_think"
    s = s2

    # 2) 允许**单个** JSON 围栏，且必须正好包住剩余内容
    m = FENCE.match(s)
    if m:
        s = m.group(1).strip()
        if not s:
            return None, "empty_fence"

    # 3) 必须整体是一个 JSON 对象 —— 用 raw_decode 检查**没有尾随内容**
    try:
        obj, end = json.JSONDecoder(object_pairs_hook=_no_dup_pairs).raw_decode(s)
    except _DupKey as e:
        return None, f"duplicate_key:{e}"
    except json.JSONDecodeError:
        # 分清"截断"与"根本不是一个对象"
        if s.startswith("{") and not s.rstrip().endswith("}"):
            return None, "truncated"
        if s.count("{") > 1:
            return None, "multiple_objects"
        return None, "not_json"

    if s[end:].strip():
        return None, "trailing_text"
    if not isinstance(obj, dict):
        return None, "not_object"

    keys = list(obj.keys())
    if "defect_class" not in obj:
        return None, "missing_field"
    if len(keys) != 1:
        # 多个字段（含多个答案字段）→ 拒绝，**不挑一个**
        return None, f"extra_fields:{sorted(keys)}"

    v = obj["defect_class"]
    if not isinstance(v, str):
        return None, "value_not_str"
    v2 = v.strip()
    if v2 not in VALID:
        return None, f"bad_enum:{v2[:24]}"
    tag = "ok_think" if used_think else "ok"
    if v2 == "unknown":
        tag += "_unknown"
    return v2, tag


def _gold_of(row: dict, col: str) -> str:
    if col not in row:
        raise RewardSystemError(f"gold 列 {col!r} 不在数据行里；现有列 {sorted(row)}")
    g = row[col]
    if not isinstance(g, str) or g.strip() not in set(CLASSES):
        raise RewardSystemError(f"无效 gold {g!r}（必须是九类之一，不含 unknown）")
    return g.strip()


class WaferClassReward:
    """ms-swift 奖励回调。`completions` 是生成结果，其余列由数据集提供。"""

    def __init__(self, args=None, gold_col: str = "ground_truth", **kwargs):
        self.args = args
        self.gold_col = gold_col
        # 审计留痕：真实回调收到的 (sample_id, gold, response)
        self.audit: List[dict] = []
        self.audit_limit = 4000

    def __call__(self, completions: List[str], ground_truth: Any = None,
                 solution: Any = None, sample_id: Any = None, **kwargs) -> List[float]:
        golds = ground_truth if ground_truth is not None else solution
        if golds is None:
            raise RewardSystemError("奖励回调没有收到 gold 列（ground_truth/solution 都是 None）")
        if not isinstance(golds, (list, tuple)):
            raise RewardSystemError(f"gold 不是序列：{type(golds).__name__}")
        if len(completions) != len(golds):
            # 不 zip 静默截短
            raise RewardSystemError(
                f"生成数 {len(completions)} != gold 数 {len(golds)}；拒绝静默截短")

        rewards: List[float] = []
        for i, (comp, gold) in enumerate(zip(completions, golds)):
            g = str(gold).strip()
            if g not in set(CLASSES):
                raise RewardSystemError(f"第 {i} 条 gold 无效：{gold!r}")
            pred, how = parse_answer(comp)
            r = 1.0 if (pred is not None and pred != "unknown" and pred == g) else 0.0
            rewards.append(r)
            if len(self.audit) < self.audit_limit:
                self.audit.append({
                    "i": i,
                    "sample_id": (sample_id[i] if isinstance(sample_id, (list, tuple))
                                  and i < len(sample_id) else None),
                    "gold": g, "pred": pred, "parse": how, "reward": r,
                    "response": comp,
                })
        return rewards


# ── 自测 ────────────────────────────────────────────────────────────────
GOOD = '{"defect_class":"Edge_Ring"}'
GOOD_THINK = '<think>\n\n</think>\n\n' + GOOD
GOOD_FENCE = '```json\n{\n"defect_class":"Edge_Ring"\n}\n```'
GOOD_BOTH = '<think>\n\n</think>\n\n```json\n{"defect_class":"Edge_Ring"}\n```'

CASES = [
    # (标签, 响应, 期望类别, 期望奖励)
    ("正确", GOOD, "Edge_Ring", 1.0),
    ("正确-空think", GOOD_THINK, "Edge_Ring", 1.0),
    ("正确-围栏", GOOD_FENCE, "Edge_Ring", 1.0),
    ("正确-空think+围栏", GOOD_BOTH, "Edge_Ring", 1.0),
    ("错类", '{"defect_class":"Loc"}', "Edge_Ring", 0.0),
    ("unknown", '{"defect_class":"unknown"}', "Edge_Ring", 0.0),
    ("空串", "", "Edge_Ring", 0.0),
    ("只有think", '<think>\n\n</think>\n\n', "Edge_Ring", 0.0),
    ("旧字段 defect_type", '{"defect_type":"Edge_Ring"}', "Edge_Ring", 0.0),
    ("双方缺字段（旧插件的满分陷阱）", '{}', "Edge_Ring", 0.0),
    ("重复键", '{"defect_class":"Edge_Ring","defect_class":"Loc"}', "Edge_Ring", 0.0),
    ("截断", '{"defect_class":"Edge_', "Edge_Ring", 0.0),
    ("多对象", '{"defect_class":"Loc"}{"defect_class":"Edge_Ring"}', "Edge_Ring", 0.0),
    ("尾随文本", '{"defect_class":"Edge_Ring"} 以上。', "Edge_Ring", 0.0),
    ("多字段", '{"defect_class":"Edge_Ring","reasoning":"x"}', "Edge_Ring", 0.0),
    ("双答案字段", '{"defect_class":"Edge_Ring","defect_class2":"Loc"}', "Edge_Ring", 0.0),
    ("数组", '["Edge_Ring"]', "Edge_Ring", 0.0),
    ("值不是字符串", '{"defect_class":["Edge_Ring"]}', "Edge_Ring", 0.0),
    ("子串幻觉：文字里带类名但JSON是别的", '图里有 Edge_Ring 特征。{"defect_class":"Loc"}',
     "Edge_Ring", 0.0),
    ("子串幻觉：只有文字没有JSON", "这张图是 Edge_Ring。", "Edge_Ring", 0.0),
    ("非空think", '<think>我觉得是Edge_Ring</think>\n{"defect_class":"Loc"}', "Edge_Ring", 0.0),
    ("大小写错", '{"defect_class":"edge_ring"}', "Edge_Ring", 0.0),
]


def self_test() -> int:
    print("=" * 78)
    print("reward_v1 自测")
    print("=" * 78)
    rw = WaferClassReward()
    n_fail = 0
    for name, resp, gold, want in CASES:
        got = rw(completions=[resp], ground_truth=[gold], sample_id=["s0"])[0]
        pred, how = parse_answer(resp)
        ok = got == want
        n_fail += not ok
        print(f"  {'OK ' if ok else 'FAIL'} {name:34s} pred={str(pred):12s} "
              f"how={how:22s} reward={got}")

    print("\n--- 系统错误必须显式失败 ---")
    SYSCASES = [
        ("缺 gold 列", dict(completions=[GOOD])),
        ("gold 无效", dict(completions=[GOOD], ground_truth=["Donut2"])),
        ("gold 是 unknown", dict(completions=[GOOD], ground_truth=["unknown"])),
        ("数量不等", dict(completions=[GOOD, GOOD], ground_truth=["Edge_Ring"])),
    ]
    for name, kw in SYSCASES:
        try:
            rw(**kw)
            print(f"  FAIL {name}: **被静默接受了**")
            n_fail += 1
        except RewardSystemError as e:
            print(f"  OK  {name}: {type(e).__name__}: {str(e)[:60]}")

    print("\n--- 审计留痕 ---")
    rw2 = WaferClassReward()
    rw2(completions=[GOOD, '{"defect_class":"Loc"}', GOOD],
        ground_truth=["Edge_Ring", "Edge_Ring", "Edge_Ring"],
        sample_id=["w1", "w2", "w3"])
    for a in rw2.audit:
        print(f"  {a['sample_id']} gold={a['gold']} pred={a['pred']} "
              f"parse={a['parse']} reward={a['reward']} resp={a['response'][:40]!r}")

    print(f"\n通过 {len(CASES) + len(SYSCASES) - n_fail}/{len(CASES) + len(SYSCASES)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
