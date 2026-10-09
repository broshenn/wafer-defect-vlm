#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 交付协作者：只读复算部署原型的**服务实测**数字（不改主会话目录）。

复算：HTTP 状态分布、类别正确数/Macro-F1、needs_review 数、时延 p50/p95/max、
      `format_changes` 类型分布（D 的格式问题到底改了什么）。
输出：部署服务复算.json / .md
"""
from __future__ import annotations
import io, json, re
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent
DEP = D.parent / "部署原型_20261009_0120"


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round((len(xs) - 1) * q)))
    return xs[i]


def main():
    rows = [json.loads(l) for l in io.open(
        DEP / "性能与自测/http_replay.jsonl", encoding="utf-8") if l.strip()]
    lat = [r["client_seconds"] for r in rows if r.get("client_seconds")]
    status = Counter(r.get("http_status") for r in rows)
    nr = sum(1 for r in rows if (r.get("response") or {}).get("needs_review"))
    correct = sum(1 for r in rows
                  if ((r.get("response") or {}).get("result") or {}).get("defect_class")
                  == r.get("gold_class"))
    # 固定 9 类 Macro-F1
    labels = sorted({r["gold_class"] for r in rows})
    per = {}
    for L in labels:
        tp = sum(1 for r in rows if r["gold_class"] == L and
                 ((r.get("response") or {}).get("result") or {}).get("defect_class") == L)
        fp = sum(1 for r in rows if r["gold_class"] != L and
                 ((r.get("response") or {}).get("result") or {}).get("defect_class") == L)
        fn = sum(1 for r in rows if r["gold_class"] == L and
                 ((r.get("response") or {}).get("result") or {}).get("defect_class") != L)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        per[L] = {"n": sum(1 for r in rows if r["gold_class"] == L),
                  "P": round(p, 4), "R": round(rc, 4),
                  "F1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0}
    macro = round(sum(v["F1"] for v in per.values()) / len(per), 4) if per else None
    fc = Counter()
    for r in rows:
        for c in ((r.get("response") or {}).get("format_changes") or []):
            fc[f"{c.get('field')}: {c.get('reason')}"] += 1
    reasons = Counter()
    for r in rows:
        for x in ((r.get("response") or {}).get("review_reasons") or []):
            reasons[x] += 1

    rep = {
        "条数": len(rows),
        "HTTP状态分布": dict(status),
        "gold类别分布": dict(Counter(r["gold_class"] for r in rows)),
        "类别正确数": correct,
        "类别Acc": round(correct / len(rows), 4),
        "MacroF1_fixed9": macro,
        "逐类": per,
        "needs_review数": nr,
        "时延": {"首条": lat[0] if lat else None,
                 "p50": pct(lat, .5), "p95": pct(lat, .95), "max": max(lat) if lat else None,
                 "均值": round(sum(lat) / len(lat), 4) if lat else None},
        "format_changes分布": dict(fc),
        "review_reasons分布": dict(reasons),
        "与selftest_summary对照": json.loads(
            (DEP / "性能与自测/selftest_summary.json").read_text(encoding="utf-8")),
    }
    (D / "部署服务复算.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L = ["# 部署服务实测：只读复算\n",
         f"\n- 回放条数 {rep['条数']}，HTTP 状态 {rep['HTTP状态分布']}\n",
         f"- 类别正确 **{correct}/{len(rows)} = {rep['类别Acc']}**，"
         f"Macro-F1(固定9类) **{macro}**（自测摘要标 "
         f"{rep['与selftest_summary对照'].get('macro_f1_fixed9')}）\n",
         f"- `needs_review` {nr} 条（自测摘要标 "
         f"{rep['与selftest_summary对照'].get('needs_review')}）\n",
         f"- 时延：首条 {rep['时延']['首条']:.2f}s、p50 {rep['时延']['p50']:.2f}s、"
         f"p95 {rep['时延']['p95']:.2f}s、max {rep['时延']['max']:.2f}s\n",
         f"\n## 服务实际改动过哪些字段（`format_changes`）\n\n"]
    for k, v in fc.most_common():
        L.append(f"- {v} 次：{k}\n")
    L.append("\n## `needs_review` 触发原因\n\n")
    for k, v in reasons.most_common():
        L.append(f"- {v} 次：{k}\n")
    L.append("\n> 时延口径：同客户端 HTTP 提交到完整响应，**不是**外部网络时延；"
             "显存与加载时间见 `selftest_summary.json`。\n")
    (D / "部署服务复算.md").write_text("".join(L), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
