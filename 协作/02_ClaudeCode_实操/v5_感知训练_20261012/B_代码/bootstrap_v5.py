#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配对 lot bootstrap（B 卡 8；合同 8.3）。
用法：python bootstrap_v5.py --a per_sample_A.jsonl --b per_sample_B.jsonl \
      --metric geoscore|class_macro_f1|class_acc --n 10000 --seed 3407
只在两边都有的 sample_id 上比；按 lot_name 成组有放回抽样；每次重算整体值再相减；
输出点估计（原样本差）、bootstrap 均值、2.5%/97.5% 分位。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

import numpy as np

V5_CLASSES = ['Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_full', 'Random',
              'Scratch', 'none']


def load_per_sample(p):
    rows = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["sample_id"]] = r
    return rows


def metric_value(rows, metric):
    """rows: list of per-sample 字典（同一模型）。返回整体指标值。"""
    if metric == "geoscore":
        parts = []
        cov = [r for r in rows if "cov_exact" in r]
        zon = [r for r in rows if "zones_exact" in r]
        ang = [r for r in rows if "ang_ok" in r]
        if cov:
            parts.append(sum(1 for r in cov if r["cov_exact"]) / len(cov))
        if zon:
            parts.append(sum(1 for r in zon if r["zones_exact"]) / len(zon))
        if ang:
            parts.append(sum(1 for r in ang if r["ang_ok"]) / len(ang))
        return sum(parts) / len(parts) if parts else float("nan")
    if metric == "class_acc":
        rows2 = [r for r in rows if r.get("present")]
        return sum(1 for r in rows2 if r.get("class_ok")) / len(rows2) if rows2 else float("nan")
    if metric == "class_macro_f1":
        tp, fp, fn = Counter(), Counter(), Counter()
        for r in rows:
            if not r.get("present"):
                continue
            pred = r.get("class_pred")
            # per_sample 未存真值；用 class_ok 与 pred 推不出来 → 要求 build 时存了 true_class
            truth = r.get("true_class")
            if truth is None:
                raise RuntimeError("per_sample 缺 true_class，无法算 macro_f1")
            if pred == truth:
                tp[truth] += 1
            else:
                fn[truth] += 1
                if pred in V5_CLASSES:
                    fp[pred] += 1
        f1s = []
        for c in V5_CLASSES:
            p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
            rr = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
            f1s.append(2 * p * rr / (p + rr) if (p + rr) else 0.0)
        return sum(f1s) / len(f1s)
    raise ValueError(metric)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--metric", choices=["geoscore", "class_macro_f1", "class_acc"], required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A = load_per_sample(a.a)
    B = load_per_sample(a.b)
    common = sorted(set(A) & set(B))
    rows_a = [A[s] for s in common]
    rows_b = [B[s] for s in common]
    point = metric_value(rows_a, a.metric) - metric_value(rows_b, a.metric)
    by_lot = defaultdict(list)
    for s in common:
        lot = A[s].get("lot_name") or B[s].get("lot_name") or "__nolot__"
        by_lot[lot].append(s)
    lots = sorted(by_lot)
    rng = np.random.default_rng(a.seed)
    diffs = []
    for _ in range(a.n):
        pick = rng.integers(0, len(lots), size=len(lots))
        sids = [s for i in pick for s in by_lot[lots[i]]]
        va = metric_value([A[s] for s in sids], a.metric)
        vb = metric_value([B[s] for s in sids], a.metric)
        if np.isfinite(va) and np.isfinite(vb):
            diffs.append(va - vb)
    diffs = np.array(diffs)
    out = {"metric": a.metric, "n_common": len(common), "n_lots": len(lots),
           "point_estimate": round(float(point), 5),
           "bootstrap_mean": round(float(diffs.mean()), 5) if len(diffs) else None,
           "ci95": [round(float(np.percentile(diffs, 2.5)), 5),
                    round(float(np.percentile(diffs, 97.5)), 5)] if len(diffs) else None,
           "n_bootstrap_used": len(diffs), "seed": a.seed,
           "excludes_zero": bool(len(diffs) and
                                 (np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0)),
           "a_file": a.a, "b_file": a.b}
    print(json.dumps(out, ensure_ascii=False))
    if a.out:
        open(a.out, "w", encoding="utf-8", newline="\n").write(
            json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
