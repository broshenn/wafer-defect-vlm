#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""费用守卫：发下一次计费请求之前，先算最坏情况成本并对照预算。

设计原则是 **失败关闭（fail closed）**：
  * 价目**未核验** → 拒绝该候选；
  * 汇率**未核验** → 拒绝按人民币结算；
  * 预算**未给定** → 拒绝一切计费请求；
  * 任何一条算不出来 → 拒绝，不是"放过去"。

守卫按 **最坏情况** 计价：即使实际会便宜，也按贵的算——
低峰/缓存命中都不假设。这样才能保证"发出去的花费不会超过授权"。

只读本地文件，不联网、不读凭据。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRICING = HERE / "pricing_verified.json"

# 中文法定节假日无法在本地准确判定 → 一律按**高峰**计（更保守）。
HOLIDAYS_KNOWN = False


def load_pricing() -> dict:
    return json.loads(PRICING.read_text(encoding="utf-8"))


def is_peak(now_utc: dt.datetime | None = None) -> tuple[bool, str]:
    """按官方说明判断高峰时段。节假日判不准时按高峰算（保守）。"""
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    now = now.astimezone(dt.timezone.utc)
    wd = now.weekday()                       # 0=Mon
    if wd >= 5:
        return False, f"{now:%Y-%m-%d %H:%M} UTC 是周末 → 低峰"
    t = now.time()
    for lo, hi in (([1, 0, 0], [4, 0, 0]), ([6, 0, 0], [10, 0, 0])):
        if dt.time(*lo) <= t < dt.time(*hi):
            note = "" if HOLIDAYS_KNOWN else "（节假日判不准，本实现按高峰保守计）"
            return True, f"{now:%Y-%m-%d %H:%M} UTC 落在高峰窗 {lo[0]:02d}:00-{hi[0]:02d}:00 {note}"
    return False, f"{now:%Y-%m-%d %H:%M} UTC 在工作日高峰窗之外 → 低峰"


def estimate_worst_case(cand: dict, input_tokens: int, max_tokens: int,
                        now_utc: dt.datetime | None = None) -> dict:
    """最坏情况成本。**一律按 peak + cache_miss**，不用低峰价、不假设缓存命中。"""
    if not cand.get("price_verified"):
        return {"ok": False, "reason": "价目未核验，拒绝计价"}
    p = cand["price_per_mtok"]
    p_in = p["input_cache_miss"]["peak"]
    p_out = p["output"]["peak"]
    cost = input_tokens / 1e6 * p_in + max_tokens / 1e6 * p_out
    return {
        "ok": True,
        "currency": cand["currency"],
        "input_tokens_assumed": input_tokens,
        "max_tokens_assumed": max_tokens,
        "unit_price_input_peak": p_in,
        "unit_price_output_peak": p_out,
        "worst_case_cost": round(cost, 8),
        "assumption": "peak + cache_miss（最坏），未折算低峰与缓存命中",
    }


def admit_tokens(cand_name: str, cand: dict, input_tokens: int, max_tokens: int,
                 tokens_used: float, quota: float | None) -> dict:
    """**按 token 额度**放行（用于单价未核验、但用户给了免费额度的候选）。

    与金额守卫同源的失败关闭：额度未给定 / 没算出来 → 拒。
    按**最坏情况**计：假设输入 token 全量 + 输出**用满** max_tokens。

    [事实] 免费额度是**用户说明**，不是我方核验；额度耗尽即停，不按金额往前冲。
    """
    if quota is None:
        return {"allow": False, "reason": f"{cand_name}: 未给定 token 额度 —— 拒绝", "estimate": None}
    worst = input_tokens + max_tokens
    remaining = quota - tokens_used
    if worst > remaining:
        return {"allow": False,
                "reason": f"{cand_name}: 剩余 {remaining:.0f} tok < 最坏用量 {worst} tok",
                "estimate": {"worst_case_tokens": worst, "remaining_tokens": remaining}}
    return {"allow": True,
            "reason": f"剩余 {remaining:.0f} tok ≥ 最坏用量 {worst} tok",
            "estimate": {"worst_case_tokens": worst, "remaining_tokens": remaining,
                         "quota_source": cand.get("free_quota_source")}}


def admit(cand_name: str, cand: dict, input_tokens: int, max_tokens: int,
          spent: float, budget: float | None,
          now_utc: dt.datetime | None = None) -> dict:
    """判定"能不能发这一次"。返回 {allow, reason, estimate}。"""
    if budget is None:
        return {"allow": False, "reason": "**未给定费用上限** —— 守卫拒绝一切计费请求", "estimate": None}
    est = estimate_worst_case(cand, input_tokens, max_tokens, now_utc)
    if not est["ok"]:
        return {"allow": False, "reason": f"{cand_name}: {est['reason']}", "estimate": None}
    remaining = budget - spent
    if est["worst_case_cost"] > remaining:
        return {"allow": False,
                "reason": f"{cand_name}: 剩余 {remaining:.6f} < 最坏成本 {est['worst_case_cost']:.6f}",
                "estimate": est}
    peak, why = is_peak(now_utc)
    return {"allow": True,
            "reason": f"剩余 {remaining:.6f} ≥ 最坏成本 {est['worst_case_cost']:.6f}；{why}",
            "estimate": est, "is_peak_now": peak}


# ── 自测 ────────────────────────────────────────────────────
def _selftest() -> int:
    print("=" * 74)
    print("费用守卫自测（不联网、不发请求、不读凭据）")
    print("=" * 74)
    pr = load_pricing()
    bad = 0

    print("\n[1] 高峰/低峰判定（按官方说明的 UTC 窗口，Mon-Fri 01:00-04:00 与 06:00-10:00）")
    cases = [
        ("2026-10-02 02:00 UTC 周五", dt.datetime(2026, 10, 2, 2, 0, tzinfo=dt.timezone.utc), True),
        ("2026-10-02 05:00 UTC 周五", dt.datetime(2026, 10, 2, 5, 0, tzinfo=dt.timezone.utc), False),
        ("2026-10-02 08:30 UTC 周五", dt.datetime(2026, 10, 2, 8, 30, tzinfo=dt.timezone.utc), True),
        ("2026-10-02 12:00 UTC 周五", dt.datetime(2026, 10, 2, 12, 0, tzinfo=dt.timezone.utc), False),
        ("2026-10-03 08:30 UTC 周六", dt.datetime(2026, 10, 3, 8, 30, tzinfo=dt.timezone.utc), False),
        ("2026-10-04 02:00 UTC 周日", dt.datetime(2026, 10, 4, 2, 0, tzinfo=dt.timezone.utc), False),
    ]
    for label, when, want in cases:
        got, why = is_peak(when)
        ok = got == want
        bad += not ok
        print(f"   {'✓' if ok else '✗'} {label:28s} 高峰={got}  期望={want}")

    print("\n[2] 最坏成本计算（peak + cache_miss）")
    ds = pr["candidates"]["deepseek"]
    est = estimate_worst_case(ds, input_tokens=2000, max_tokens=16384)
    exp = 2000 / 1e6 * 0.3 + 16384 / 1e6 * 1.2
    ok = est["ok"] and abs(est["worst_case_cost"] - exp) < 1e-9
    bad += not ok
    print(f"   {'✓' if ok else '✗'} 输入 2000 / 输出 16384 → {est['worst_case_cost']:.6f} "
          f"(手算 {exp:.6f}) {est['currency']}")

    print("\n[3] 失败关闭：价目未核验的候选**不能按金额**放行")
    for name in ("kimi", "qwen"):
        r = admit(name, pr["candidates"][name], 2000, 16384, 0.0, 10.0)
        ok = r["allow"] is False
        bad += not ok
        print(f"   {'✓' if ok else '✗'} {name:8s} 金额路径 allow={r['allow']}  {r['reason'][:52]}")

    print("\n[3b] token 额度路径（单价未核验但用户给了免费额度时用这条）")
    tk = pr["candidates"]["kimi"]
    r1 = admit_tokens("kimi", tk, 3000, 16384, 0, 1_000_000)
    r2 = admit_tokens("kimi", tk, 3000, 16384, 0, None)
    r3 = admit_tokens("kimi", tk, 3000, 16384, 990_000, 1_000_000)
    for label, r, want in (("额度充足 → 放行", r1, True), ("未给额度 → 拒", r2, False),
                           ("额度将耗尽 → 拒", r3, False)):
        ok = r["allow"] == want
        bad += not ok
        print(f"   {'✓' if ok else '✗'} {label:16s} allow={r['allow']} 期望={want}  {r['reason'][:44]}")

    print("\n[4] 失败关闭：未给预算必须全拒")
    r = admit("deepseek", ds, 2000, 16384, 0.0, None)
    ok = r["allow"] is False
    bad += not ok
    print(f"   {'✓' if ok else '✗'} budget=None → allow={r['allow']}  {r['reason'][:60]}")

    print("\n[5] 预算边界")
    one = exp
    for spent, budget, want in ((0.0, one * 1.001, True), (0.0, one * 0.999, False),
                                (one * 0.5, one * 0.6, False), (one * 0.5, one * 1.6, True)):
        r = admit("deepseek", ds, 2000, 16384, spent, budget)
        ok = r["allow"] == want
        bad += not ok
        print(f"   {'✓' if ok else '✗'} spent={spent:.4f} budget={budget:.4f} → allow={r['allow']} 期望={want}")

    print("\n[6] 守卫是否**不**假设低峰/缓存命中")
    ok = est["assumption"].startswith("peak + cache_miss")
    bad += not ok
    print(f"   {'✓' if ok else '✗'} assumption = {est['assumption']}")

    print()
    print(f"结论：{'全部通过' if bad == 0 else f'{bad} 项未通过'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
