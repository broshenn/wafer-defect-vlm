#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四张分歧样本的图案诊断：径向剖面 + 条带性零模型检验。

在 `analyze_four.py` 的反解矩阵之上，回答任务书第 2 问里「环状分布趋势」和
「密集连通块是否造成假条带」两小问。结果写 `图案诊断.json`。

口径（写死在这里，便于复核）：
  * 分母：失效率 = 该环带/该行列上的失效 die 数 ÷ 同一集合里的有效 die 数（>0 的格）。
  * 环带：用**椭圆归一**半径 r = hypot(Δcol/b, Δrow/a)，a/b 是晶圆掩膜的外接半轴。
    用椭圆而不是历史圆拟合，是因为 die 栅格各向异性 —— 晶圆在栅格坐标里是椭圆、
    在渲染图里才是圆。方向定义不影响 r。
  * 条带性统计量：逐列失效率序列的自相关在周期 2..11 上的最大值。
    零模型 = 在同样的有效 die 位置里**随机置换**同样多的失效点，重算同一统计量。
    观测值超过零模型 95 分位才算「不是致密连通造成的假条带」。
  * 零模型种子固定 3407，可复跑。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wafer_png import REPO, recover_matrix  # noqa: E402

SAMPLE_SET = [
    ("item_001", "wafer_00044523_017", "none", "Random"),
    ("item_003", "wafer_00025336_019", "Loc", "Edge_Loc"),
    ("item_007", "wafer_00018489_020", "Donut", "Loc"),
    ("item_008", "wafer_00017102_001", "Random", "Scratch"),
]
CONTROLS = [("item_006", "wafer_00013704_014"), ("item_009", "wafer_00011911_008")]
NULL_DRAWS = 300
NULL_SEED = 3407


def radial_profile(matrix: np.ndarray) -> dict:
    mask = matrix > 0
    d = matrix == 2
    ys, xs = np.nonzero(mask)
    a = (ys.max() - ys.min() + 1) / 2.0
    b = (xs.max() - xs.min() + 1) / 2.0
    cy = (ys.max() + ys.min()) / 2.0
    cx = (xs.max() + xs.min()) / 2.0
    r = np.hypot((np.arange(matrix.shape[1])[None, :] - cx) / b,
                 (np.arange(matrix.shape[0])[:, None] - cy) / a)
    edges = np.linspace(0, 1, 11)
    rows = []
    for i in range(10):
        sel = mask & (r >= edges[i]) & (r < edges[i + 1])
        n_valid = int(sel.sum())
        n_def = int((d & sel).sum())
        rows.append({
            "r_lo": round(float(edges[i]), 2), "r_hi": round(float(edges[i + 1]), 2),
            "valid_die": n_valid, "defect_die": n_def,
            "defect_rate": round(n_def / n_valid, 4) if n_valid else None,
        })
    rates = [x["defect_rate"] for x in rows if x["defect_rate"] is not None]
    return {
        "bins": rows,
        "center_rate_r_lt_0_2": round(
            (sum(x["defect_die"] for x in rows[:2]) /
             max(sum(x["valid_die"] for x in rows[:2]), 1)), 4),
        "peak_bin": max(rows, key=lambda x: x["defect_rate"] or -1)["r_lo"],
        "peak_rate": max(rates),
        "shape_note": (
            "中心低 + 中径峰 = 环/甜甜圈特征" if rates[0] < rates[1] < max(rates) else
            "中心高 = 中心型" if rates[0] >= max(rates) * 0.8 else "无明显环特征"),
    }


def banding(matrix: np.ndarray, n_draws: int = NULL_DRAWS, seed: int = NULL_SEED) -> dict:
    """逐列失效率的自相关峰值，与随机零模型比较。"""
    mask = matrix > 0
    d = matrix == 2
    n_def = int(d.sum())
    n_valid = int(mask.sum())

    def stat(mm: np.ndarray) -> float:
        cols = np.array([
            (mm[:, j] == 2).sum() / max((mm[:, j] > 0).sum(), 1) for j in range(mm.shape[1])
        ])
        s = cols - cols.mean()
        if s.std() == 0:
            return 0.0
        n = len(s)
        ac = np.correlate(s, s, "full")[n - 1:] / (s @ s)
        return float(max(ac[2:12]))

    observed = stat(matrix)

    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask.ravel())
    base = np.zeros(mask.size, dtype=np.uint8)
    base[mask.ravel()] = 1
    draws = np.empty(n_draws)
    for k in range(n_draws):
        sel = rng.choice(idx, size=n_def, replace=False)
        mm = base.copy()
        mm[sel] = 2
        draws[k] = stat(mm.reshape(mask.shape))

    pct = float((draws < observed).mean())
    return {
        "defect_die": n_def, "valid_die": n_valid,
        "defect_rate": round(n_def / max(n_valid, 1), 4),
        "column_autocorr_peak_period_2_11": round(observed, 4),
        "null_model": {
            "draws": n_draws, "seed": seed,
            "mean": round(float(draws.mean()), 4),
            "p95": round(float(np.percentile(draws, 95)), 4),
            "max": round(float(draws.max()), 4),
        },
        "observed_percentile_in_null": round(pct * 100, 1),
        "banding_is_real_structure": bool(pct > 0.95),
        "verdict": ("观测值高于零模型 95 分位 → 条带是真结构，不是致密连通造成的假象"
                    if pct > 0.95 else
                    "观测值不高于零模型 95 分位 → 不能排除是致密连通造成的假条带"),
    }


def population_context() -> dict:
    """各原标签在训练集里的 defect_ratio 分布。**这是背景，不是判据阈值。**"""
    import collections
    pop = collections.defaultdict(list)
    with open(REPO / "data" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["split"] == "train":
                pop[r["failure_type"]].append(r["features"]["defect_ratio"])
    out = {}
    for c, v in sorted(pop.items()):
        v = np.array(v)
        out[c] = {"n": len(v), "median": round(float(np.median(v)), 4),
                  "p10": round(float(np.percentile(v, 10)), 4),
                  "p90": round(float(np.percentile(v, 90)), 4)}
    return out


def main() -> int:
    man = {}
    with open(REPO / "data" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                man[r["sample_id"]] = r

    pop = population_context()
    report = {
        "definitions_and_caveats": {
            "radial_bins": "10 段等宽 r∈[0,1]，r 用椭圆归一（见模块 docstring）",
            "denominator": "失效率分母 = 同一集合内值 >0 的格（有效 die）",
            "banding_statistic": "逐列失效率序列自相关在周期 2..11 的最大值",
            "banding_uses_raw_defects": "banding.defect_rate 用的是**未过滤**的失效 die（m==2）；"
                                        "而 defect_ratio_stored_in_manifest 用的是历史过滤后的口径"
                                        "（isolated_point_filter 之后）。两个数不同源，不要互相减。",
            "banding_p95_is_marginal_when_close": "观测值只略高于 p95 时（差 <0.05）应视为不显著；"
                                                  "本轮对照样本 item_006 就是这种边缘情形。",
            "null_model": f"在同样有效 die 位置随机置换同样多失效点，{NULL_DRAWS} 次，seed={NULL_SEED}",
            "population_context_is_not_a_threshold":
                "下表是各标签在整个训练集里的 defect_ratio 分布，用作**背景参照**，"
                "不是 WM-811K 的官方判据，也不得反过来当成阈值使用。",
        },
        "population_defect_ratio_by_label": pop,
        "items": [],
    }

    for item_id, sid, gt, pred in SAMPLE_SET + [(a, b, None, None) for a, b in CONTROLS]:
        m = recover_matrix(REPO / "data" / "images" / f"{sid}.png")["matrix"]
        stored = man[sid]["features"]["defect_ratio"]
        entry = {
            "item_id": item_id, "sample_id": sid,
            "ground_truth_class": gt, "glm_class": pred,
            "role": "分歧样本" if pred else "对照（标签与模型一致）",
            "matrix_HW": list(m.shape),
            "defect_ratio_stored_in_manifest": round(stored, 4),
            "radial": radial_profile(m),
            "banding": banding(m),
        }
        report["items"].append(entry)

    # 该图在「原标签」与「GLM 类别」两个人群里的密度分位
    import collections
    pop_vals = collections.defaultdict(list)
    with open(REPO / "data" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["split"] == "train":
                    pop_vals[r["failure_type"]].append(r["features"]["defect_ratio"])
    pop_vals = {k: np.array(v) for k, v in pop_vals.items()}
    for e in report["items"]:
        dr = e["defect_ratio_stored_in_manifest"]
        for label, key in ((e["ground_truth_class"], "ground_truth"),
                           (e["glm_class"], "glm_class")):
            if label:
                e[f"defect_ratio_percentile_in_{key}_label"] = round(
                    float((pop_vals[label] < dr).mean() * 100), 1)

    out = HERE / "图案诊断.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {out}\n")

    print(f"{'item':9s} {'sample':22s} {'原标签':9s} {'GLM':9s} {'失效率':8s} "
          f"{'原标签分位':10s} {'GLM类分位':10s} {'径向峰':7s} {'条带观测':9s} {'零模型p95':9s} {'真结构?'}")
    for e in report["items"]:
        b, r = e["banding"], e["radial"]
        print(f"{e['item_id']:9s} {e['sample_id']:22s} {str(e['ground_truth_class']):9s} "
              f"{str(e['glm_class']):9s} {b['defect_rate']*100:6.1f}%  "
              f"{str(e.get('defect_ratio_percentile_in_ground_truth_label')):10s} "
              f"{str(e.get('defect_ratio_percentile_in_glm_class_label')):10s} "
              f"{r['peak_bin']:6.1f}  {b['column_autocorr_peak_period_2_11']:9.3f} "
              f"{b['null_model']['p95']:9.3f} {b['banding_is_real_structure']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
