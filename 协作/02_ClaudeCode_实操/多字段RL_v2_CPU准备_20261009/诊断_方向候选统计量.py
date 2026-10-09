#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""方向判据候选统计量的全量诊断（只读，用于选规则，不写进奖励）。

对每个样本计算：
  R1           绕晶圆圆心的单向集中度（一阶谐波合成向量长度）
  R2           轴向集中度（二阶谐波）
  arc15_frac   任意 ±15° 扇带内的最大质量占比
  line_frac    过圆心任意直线 ±d 带内的最大质量占比（d=0.05 归一化单位）
  rim_frac     归一化半径 r>=0.9 的红点质量占比（边缘红圈占比，用于解释稀释）
按 failure_type 汇总分位，用于挑选「方向参考」的判据与阈值。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from 几何参考_v2 import _ellipse_of, _normalized_uv, recover_matrix  # noqa: E402

LINE_HALF_WIDTH = 0.05
ARC_HALF_DEG = 15.0
ANGLE_STEP_DEG = 2.0


def stats(m: np.ndarray) -> dict:
    green, red = (m == 1), (m == 2)
    valid = green | red
    n_red = int(red.sum())
    H, W = m.shape
    if n_red == 0 or valid.sum() == 0:
        return {"n_red": 0}
    cy, cx, a, b = _ellipse_of(valid)
    u, v = _normalized_uv(H, W, cy, cx, a, b)
    uu, vv = u[red], v[red]
    rr = np.sqrt(uu * uu + vv * vv)
    deg = np.mod(np.degrees(np.arctan2(vv, -uu)), 360.0)
    rad = np.radians(deg)
    R1 = float(np.hypot(np.cos(rad).mean(), np.sin(rad).mean()))
    R2 = float(np.hypot(np.cos(2 * rad).mean(), np.sin(2 * rad).mean()))

    # ±15° 扇带最大质量占比
    best_arc = 0.0
    for t in np.arange(0.0, 360.0, ANGLE_STEP_DEG):
        d = np.abs((deg - t + 180.0) % 360.0 - 180.0)
        best_arc = max(best_arc, float((d <= ARC_HALF_DEG).mean()))
    # 过圆心的直线（方向 φ，法向距离 = |u·cosφ - v·sinφ|... 用角度差等效实现）
    # 点与「过圆心、朝向 φ 的直线」的法向距离：|u*sin φ + v*(-cos φ)| 反向 → 用极角差
    best_line = 0.0
    for phi in np.arange(0.0, 180.0, ANGLE_STEP_DEG):
        # 直线两端在 phi 与 phi+180；点到该直线的法向距离 = r*|sin(θ-phi)|
        dist = rr * np.abs(np.sin(np.radians(deg - phi)))
        best_line = max(best_line, float((dist <= LINE_HALF_WIDTH).mean()))
    rim = float((rr >= 0.9).mean())
    return {"n_red": n_red, "R1": round(R1, 4), "R2": round(R2, 4),
            "arc15_frac": round(best_arc, 4), "line_frac": round(best_line, 4),
            "rim_frac": round(rim, 4)}


def main() -> int:
    rows = [json.loads(l) for l in open(REPO / "data" / "manifest.jsonl", encoding="utf-8")]
    out = []
    for i, r in enumerate(rows, 1):
        sid = r["sample_id"]
        rec = recover_matrix(REPO / "data" / "images" / f"{sid}.png",
                             hint_hw=tuple(r["matrix_shape"]))
        s = stats(rec["matrix"])
        s.update({"sample_id": sid, "failure_type": r["failure_type"], "split": r["split"]})
        out.append(s)
        if i % 1000 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    with open(HERE / "诊断_方向候选统计量.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for d in out:
            fh.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")

    cls = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_full", "Random",
           "Scratch", "none"]
    keys = ["R1", "R2", "arc15_frac", "line_frac", "rim_frac"]
    print(f"\n{'class':10s} {'n':>5s} " + " ".join(f"{k+' p50/p90/p99':>26s}" for k in keys))
    for c in cls:
        g = [d for d in out if d["failure_type"] == c and d.get("n_red")]
        line = [f"{c:10s} {len(g):5d} "]
        for k in keys:
            a = np.array([d[k] for d in g])
            line.append(f"{np.percentile(a,50):.2f}/{np.percentile(a,90):.2f}/{np.percentile(a,99):.2f}".rjust(26))
        print("".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
