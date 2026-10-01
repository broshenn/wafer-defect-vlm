#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图案诊断 v2 —— 修正 v1 的统计解释（Codex 验收裁决 §4）。

**不改 `核查_20261002_四张分歧/图案诊断.json`**；本脚本另写 `图案诊断_v2.json`。

## v1 错在哪（我自己的错，已复现确认）

1. **结论与自己的产物矛盾。** v1 的 `banding_is_real_structure` 对 007、008、006 **三个样本**
   都标了 `true`（007 是 0.6269>0.2499、006 是 0.3042>0.2980），
   但我的报告只说了"只有 008 显著""对照样本均不显著" —— **读错了自己的输出**。
2. **事后加规则。** 看到 006 只超 p95 一点点之后，我在报告里临时引入"差 <0.05 就视为不显著"。
   这不是原先统一的 p95 规则，属于看结果改判据。

## v2 的更正

- 字段改名：`banding_is_real_structure` → **`departs_from_uniform_permutation_null`**。
  统计量是**逐列失效率序列的自相关峰**，它对**环状分布和局部簇也会阳性** ——
  它不是"有条带"的判据，更不是任何类别的判据。
- 新增 `statistic_specificity` 字段，**显式列出本批哪些样本阳性**，包括环（007）和局部簇（006）。
- 新增 `what_this_does_not_show`，写清阳性**只**意味着偏离"同掩膜、同数量、均匀随机置换"这一零假设。
- **不设 0.05 例外**，只用 p95 一条规则；观测与 p95 的差原样列出，由读者判断。

只读原矩阵（从 PNG 反解），不联网、不调模型。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wafer_png_v2 import REPO, recover_matrix  # noqa: E402

SAMPLES = [
    ("item_001", "wafer_00044523_017", "none", "Random"),
    ("item_003", "wafer_00025336_019", "Loc", "Edge_Loc"),
    ("item_007", "wafer_00018489_020", "Donut", "Loc"),
    ("item_008", "wafer_00017102_001", "Random", "Scratch"),
    ("item_006", "wafer_00013704_014", "Edge_Loc", None),
    ("item_009", "wafer_00011911_008", "Center", None),
]
NULL_DRAWS = 300
NULL_SEED = 3407

NOT_SHOWN = (
    "阳性只表示本图**偏离**“同掩膜、同有效 die 位置、同失效数量、均匀随机置换”这一个零假设。"
    "它**不**表示：该图不可能由别的随机机制生成；**不**表示存在条带；"
    "**不**表示类别是 Scratch；**不**说明原图或上游数据来源。"
)


def column_autocorr_peak(matrix: np.ndarray) -> float:
    mask = matrix > 0
    d = matrix == 2
    cols = np.array([(d[:, j].sum()) / max((mask[:, j]).sum(), 1)
                     for j in range(matrix.shape[1])])
    s = cols - cols.mean()
    if s.std() == 0:
        return 0.0
    n = len(s)
    ac = np.correlate(s, s, "full")[n - 1:] / (s @ s)
    return float(max(ac[2:12]))


def diagnose(matrix: np.ndarray) -> dict:
    mask = matrix > 0
    n_def, n_valid = int((matrix == 2).sum()), int(mask.sum())
    observed = column_autocorr_peak(matrix)

    rng = np.random.default_rng(NULL_SEED)
    idx = np.flatnonzero(mask.ravel())
    base = np.zeros(mask.size, dtype=np.uint8)
    base[mask.ravel()] = 1
    draws = np.empty(NULL_DRAWS)
    for k in range(NULL_DRAWS):
        sel = rng.choice(idx, size=n_def, replace=False)
        mm = base.copy()
        mm[sel] = 2
        draws[k] = column_autocorr_peak(mm.reshape(mask.shape))

    p95 = float(np.percentile(draws, 95))
    return {
        "statistic": "逐列失效率序列的自相关在周期 2..11 上的最大值",
        "statistic_caveat": "该统计量对**环状分布与局部簇也会阳性**，不是条带判据，也不是类别判据。",
        "defect_die": n_def,
        "valid_die": n_valid,
        "raw_defect_rate": round(n_def / max(n_valid, 1), 4),
        "observed": round(observed, 4),
        "null_model": {
            "hypothesis": "同掩膜、同有效 die 位置、同失效数量、均匀随机置换",
            "draws": NULL_DRAWS, "seed": NULL_SEED,
            "mean": round(float(draws.mean()), 4),
            "p95": round(p95, 4),
            "max": round(float(draws.max()), 4),
        },
        "exceeds_p95": bool(observed > p95),
        "margin_over_p95": round(observed - p95, 4),
        "observed_percentile_in_null": round(float((draws < observed).mean() * 100), 1),
        "departs_from_uniform_permutation_null": bool(observed > p95),
        "what_this_does_not_show": NOT_SHOWN,
    }


def main() -> int:
    out = {"version": "pattern-diagnostics-v2@2026-10-02",
           "supersedes": "核查_20261002_四张分歧/图案诊断.json 的解释字段（不覆盖该文件）",
           "corrections_vs_v1": [
               "字段 banding_is_real_structure 删除，改名 departs_from_uniform_permutation_null",
               "删除事后加入的“差<0.05 视为不显著”例外；只用 p95 一条规则",
               "显式列出全部阳性样本（含环与局部簇），不再只挑 008",
               "明确写出该统计量不能证明条带类别或 Scratch",
           ],
           "items": []}

    for item_id, sid, gt, pred in SAMPLES:
        m = recover_matrix(REPO / "data" / "images" / f"{sid}.png")["matrix"]
        out["items"].append({
            "item_id": item_id, "sample_id": sid,
            "ground_truth_class": gt, "glm_class": pred,
            "matrix_HW": list(m.shape),
            **diagnose(m),
        })

    positives = [e["item_id"] for e in out["items"] if e["exceeds_p95"]]
    out["statistic_specificity"] = {
        "positive_items": positives,
        "n_positive": len(positives),
        "n_total": len(out["items"]),
        "note": "阳性并不专属于 008；本批的 Donut（007）与 Edge_Loc 对照（006）同样阳性。",
    }

    (HERE / "图案诊断_v2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"写出 {HERE / '图案诊断_v2.json'}\n")
    print(f"{'item':9s} {'sample':22s} {'原标签':9s} {'GLM':9s} {'观测':8s} "
          f"{'零模型p95':9s} {'超出':9s} {'阳性'}")
    for e in out["items"]:
        print(f"{e['item_id']:9s} {e['sample_id']:22s} {str(e['ground_truth_class']):9s} "
              f"{str(e['glm_class']):9s} {e['observed']:8.4f} {e['null_model']['p95']:9.4f} "
              f"{e['margin_over_p95']:+9.4f} {e['departs_from_uniform_permutation_null']}")
    print(f"\n阳性样本 {len(positives)}/{len(out['items'])}：{positives}")
    print("注意：阳性含 Donut(007) 与 Edge_Loc 对照(006)，说明该统计量不是条带专属。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
