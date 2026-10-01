#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""钟点角工具 v1 —— 在**渲染图方向**下算角度与扇区归属。

存在理由（Codex 验收裁决 §3③）：
  ≥B37 角度直方图（`utils.py:197-198` 的 `np.histogram`）在扇区边界上会随圆心 1e-15 的漂移翻转；
  ≥B38 历史 `clock_sector` 在**栅格坐标**里算角，而模型看到的是**归一化后的图像**。
本工具把参照系统一到图像方向，并把边界吸附、首尾环绕、并列规则写成显式约定
（参数与理由见同目录 `angle_config.json`）。

**不做类别判断，不声称物理角，不设类别阈值。**

用法：
    python angles.py --demo                 # 合成方向自检
    python angles.py --from-matrix <sample_id> [--center row,col]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOLUTION = 448


def load_config(path: Path | None = None) -> dict:
    return json.loads((path or (HERE / "angle_config.json")).read_text(encoding="utf-8"))


CFG = load_config()
N_SECTORS = CFG["sectors"]["n"]
SECTOR_WIDTH = 360.0 / N_SECTORS
SNAP_TOL_DEG = CFG["boundary_handling"]["snap_tol_deg"]


def cell_center_to_image(row: float, col: float, H: int, W: int,
                         resolution: int = RESOLUTION) -> tuple[float, float]:
    """栅格 (行,列) → 图像像素坐标。取格心 (i+0.5)。

    与 `wafer_png_v2.source_index_for` 互为前向/逆向：那边是 `x → col`，这边是 `col → x`。
    """
    return ((col + 0.5) * resolution / W, (row + 0.5) * resolution / H)


def clock_angle_deg(d_row: float, d_col: float, center_row: float, center_col: float,
                    H: int, W: int, resolution: int = RESOLUTION) -> float:
    """给定相对圆心的栅格偏移，返回图像空间下 12 点钟为 0、顺时针的角（度，[0,360)）。

    **注意**：必须走图像坐标再算角。直接在栅格坐标里 `atan2(d_col, -d_row)` 得到的是
    B38 说的那个角；当 H≠W 时两者不等。
    """
    x0, y0 = cell_center_to_image(center_row, center_col, H, W, resolution)
    x1, y1 = cell_center_to_image(center_row + d_row, center_col + d_col, H, W, resolution)
    dx, dy = x1 - x0, y1 - y0
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def grid_angle_deg(d_row: float, d_col: float) -> float:
    """历史口径：直接在栅格坐标里算角（`utils.py:189`）。只为对照，不推荐使用。"""
    return math.degrees(math.atan2(d_col, -d_row)) % 360.0


def sector_of(angle_deg: float, snap_tol_deg: float | None = None) -> dict:
    """角度 → 扇区归属，含边界吸附与环绕。返回 dict 便于审计。

    `snap_tol_deg` 应**逐点**给出（见 `sector_of_point`）：容差的单位是度，
    而圆心不确定度的单位是格，两者换算依赖该点到圆心的距离（踩雷记录 B43）。
    """
    tol = SNAP_TOL_DEG if snap_tol_deg is None else snap_tol_deg
    a = angle_deg % 360.0
    nearest_boundary = round(a / SECTOR_WIDTH) * SECTOR_WIDTH % 360.0
    dist_to_boundary = abs((a - nearest_boundary + 180.0) % 360.0 - 180.0)

    on_boundary = dist_to_boundary <= tol
    if on_boundary:
        a_snapped = nearest_boundary % 360.0
        index = int(math.floor(a_snapped / SECTOR_WIDTH)) % N_SECTORS
    else:
        a_snapped = a
        index = int(math.floor(a / SECTOR_WIDTH)) % N_SECTORS

    return {
        "angle_deg": round(a, 9),
        "angle_used_deg": round(a_snapped, 9),
        "on_boundary": bool(on_boundary),
        "distance_to_boundary_deg": round(dist_to_boundary, 12),
        "applied_tolerance_deg": tol,
        "sector_index": index,
        "sector_number": index + 1,
        "sector_range_deg": [round(index * SECTOR_WIDTH, 6),
                             round((index + 1) * SECTOR_WIDTH, 6)],
    }


def guard_tolerance_deg(distance_px: float, center_uncertainty_cells: float,
                        H: int, W: int, resolution: int = RESOLUTION) -> float:
    """点距圆心 d、圆心不确定度 δ（格）时，该点适用的角度容差（度）。

    推导：圆心沿垂直于该点的方向漂 δ_px，角度改变 ≈ δ_px / d_px 弧度。
    格 → 像素取两轴中较**大**的换算（保守）：
        δ_px = δ · resolution · max(1/W, 1/H)

    返回 max(基础容差, 该点的物理容差)。点的 d 越大，圆心漂移造成的角度改变越小，
    容差就越接近基础值；d 很小时容差会很大 —— 这正是"圆心附近方向本就不可靠"的如实反映。
    """
    delta_px = center_uncertainty_cells * resolution * max(1.0 / W, 1.0 / H)
    if distance_px <= 0:
        return 180.0                      # 点在圆心上，方向无定义，全部算边界
    return max(SNAP_TOL_DEG, math.degrees(delta_px / distance_px))


def sector_of_point(row: float, col: float, center_row: float, center_col: float,
                    H: int, W: int, center_uncertainty_cells: float = 0.0) -> dict:
    """**按绝对坐标**给一个点定扇区。这是推荐的入口。

    与旧用法的区别：圆心移动时**缺陷点不动**，这才是"圆心漂移下的稳定性"要测的东西。
    """
    x0, y0 = cell_center_to_image(center_row, center_col, H, W)
    x1, y1 = cell_center_to_image(row, col, H, W)
    d_px = math.hypot(x1 - x0, y1 - y0)
    angle = math.degrees(math.atan2(x1 - x0, -(y1 - y0))) % 360.0
    tol = guard_tolerance_deg(d_px, center_uncertainty_cells, H, W)
    out = sector_of(angle, tol)
    out["distance_to_center_px"] = round(d_px, 6)
    out["center_uncertainty_cells"] = center_uncertainty_cells
    return out


def angular_histogram(rows, cols, center_row, center_col, H, W,
                      center_uncertainty_cells: float = 0.0) -> dict:
    """一批缺陷的扇区直方图。**按绝对坐标**逐点算容差（B43）。

    并列取扇区号最小的（显式约定，不是 np.argmax 的隐含行为）。
    """
    counts = [0] * N_SECTORS
    flagged = 0
    angles = []
    for r, c in zip(rows, cols):
        s = sector_of_point(r, c, center_row, center_col, H, W,
                            center_uncertainty_cells)
        counts[s["sector_index"]] += 1
        flagged += s["on_boundary"]
        angles.append(s["angle_deg"])
    total = sum(counts)
    probs = [c / total if total else 0.0 for c in counts]
    best = max(range(N_SECTORS), key=lambda i: (counts[i], -i)) if total else None
    return {
        "counts": counts,
        "probabilities": [round(p, 8) for p in probs],
        "n": total,
        "n_on_boundary": flagged,
        "center_uncertainty_cells": center_uncertainty_cells,
        "modal_sector_number": (best + 1) if best is not None else None,
        "tie_break": CFG["histogram"]["tie_break"],
    }


# ── 合成自检 ────────────────────────────────────────────────
def _dir_cases() -> list[tuple[str, float, int]]:
    """(名字, 期望角, 期望扇区号) —— 正上/右/下/左 + 边界两侧 + 环绕。"""
    return [
        ("正上（12 点钟）", 0.0, 1),
        ("正右（3 点钟）", 90.0, 4),
        ("正下（6 点钟）", 180.0, 7),
        ("正左（9 点钟）", 270.0, 10),
        ("边界下方一点 29.9°", 29.9, 1),
        ("边界上方一点 30.1°", 30.1, 2),
        ("恰在边界 30°（左闭右开，归上界那一侧）", 30.0, 2),
        ("环绕：359.9°", 359.9, 12),
        ("环绕：0.1°", 0.1, 1),
    ]


def demo() -> int:
    print("=" * 78)
    print(f"钟点角工具自检（配置版本 {CFG['version']}）")
    print("=" * 78)
    bad = 0

    print("\n[1] 方向正确性：用 41×42（非方阵）验证图像空间换算")
    H, W = 41, 42
    cr, cc = 20.0, 21.0
    for name, want_deg, want_sector in _dir_cases():
        rad = math.radians(want_deg)
        # 在**图像空间**里放一个点：12 点钟 = 图像上方向
        # 先按期望角反推图像偏移，再换回栅格偏移
        x, y = math.sin(rad), -math.cos(rad)
        d_col = x * W / RESOLUTION * 10
        d_row = y * H / RESOLUTION * 10
        got = clock_angle_deg(d_row, d_col, cr, cc, H, W)
        s = sector_of(got)
        ok = abs(got - want_deg) < 1e-6 and s["sector_number"] == want_sector
        bad += not ok
        print(f"   {'✓' if ok else '✗'} {name:20s} 期望 {want_deg:7.2f}° 扇区{want_sector:2d}  "
              f"实得 {got:8.3f}° 扇区{s['sector_number']:2d}"
              f"{'  边界吸附' if s['on_boundary'] else ''}")

    print("\n[2] 非方阵下「图像角」与「栅格角」确实不同（B38）")
    for (H2, W2) in [(41, 42), (56, 41), (72, 72), (25, 27)]:
        d_row, d_col = -6.0, 6.0          # 右上方向
        ai = clock_angle_deg(d_row, d_col, H2 / 2, W2 / 2, H2, W2)
        ag = grid_angle_deg(d_row, d_col)
        print(f"   {H2}×{W2}: 图像角 {ai:8.3f}°  栅格角 {ag:8.3f}°  差 {abs(ai - ag):7.3f}°"
              f"{'   ← 两者不同' if abs(ai - ag) > 1e-9 else ''}")

    print("\n[3] 圆心漂移下的稳定性 —— **固定缺陷点，只动圆心**（B43 修正）")
    print("    v1 的写法保持 dr/dc 不变、圆心与点一起平移，测到的是平移不变性；")
    print("    这里改为固定点的绝对坐标，圆心单独漂移。")
    H3 = W3 = 72
    fixed_pt = (36.0, 46.0)              # 绝对栅格坐标，全程不动
    cases = [
        ("恰在边界（正右，d=10 格）", (36.0, 36.0), 0.0),
        ("同一点，圆心 ±1e-12 格", (36.0, 36.0), 1e-12),
        ("同一点，圆心 +1e-6 格（v1 会翻扇区）", (36.0, 36.0), 1e-6),
    ]
    for label, center, drift in cases:
        for sign in ((1,) if drift == 0 else (1, -1)):
            c = (center[0] + sign * drift, center[1])
            s = sector_of_point(fixed_pt[0], fixed_pt[1], c[0], c[1], H3, W3,
                                center_uncertainty_cells=drift)
            ok = s["sector_number"] == 4
            bad += not ok
            print(f"   {'✓' if ok else '✗'} {label:34s} 漂 {sign * drift:+.0e} 格："
                  f"角度 {s['angle_deg']:.10f}° 扇区 {s['sector_number']} "
                  f"容差 {s['applied_tolerance_deg']:.3e}° on_boundary={s['on_boundary']}")

    print("\n[3b] 容差的适用范围（解析式，非经验值）")
    print("    要让容差 τ 挡住 δ 格的圆心漂移，需 d > δ·(180/π)/τ。实例：")
    for tau, delta in ((1e-6, 1e-6), (1e-6, 1e-3), (1e-3, 1e-3)):
        need = delta * (180.0 / math.pi) / tau
        print(f"      τ={tau:.0e}°、δ={delta:.0e} 格 ⇒ 需 d > {need:8.2f} 格"
              f"{'   ← 远超本库晶圆半径(12–36 格)，即默认容差几乎无保护' if need > 36 else ''}")

    print("\n[3c] 显式声明圆心不确定度 δ 后的稳定范围")
    print("     语义：容差 = 漂移 δ 时的**最坏**角度偏移，所以保证范围是「漂移 **严格小于** δ」。")
    print("     漂移正好等于 δ 时由构造决定落在边界上 —— 这是定义使然，不靠放大容差去凑。")
    for delta in (0.0, 1e-6, 1e-3):
        for frac, must_stable in ((0.5, True), (0.99, True)):
            secs = set()
            for sign in (1, -1):
                c = (36.0 + sign * frac * delta, 36.0)
                s = sector_of_point(fixed_pt[0], fixed_pt[1], c[0], c[1], H3, W3,
                                    center_uncertainty_cells=delta)
                secs.add(s["sector_number"])
            ok = len(secs) == 1
            bad += not ok
            print(f"   {'✓' if ok else '✗'} δ={delta:.0e} 格、漂移 {frac:.0%}·δ："
                  f"扇区集合 = {sorted(secs)}")

    print("\n[4] 直方图并列取最小扇区号")
    # 绝对坐标：圆心 (36,36)；(36,46) 在正右 → 90° → 扇区 4；(46,36) 在正下 → 180° → 扇区 7
    h = angular_histogram([36.0, 36.0, 46.0, 46.0], [46.0, 46.0, 36.0, 36.0],
                          36.0, 36.0, 72, 72)
    tie_ok = (h["counts"][3] == h["counts"][6] == 2 and h["modal_sector_number"] == 4)
    bad += not tie_ok
    print(f"   {'✓' if tie_ok else '✗'} counts={h['counts']} 最大扇区={h['modal_sector_number']}"
          f"（扇区 4 与 7 并列，取扇区号最小的 4）")

    print()
    print(f"结论：{'全部通过' if bad == 0 else f'{bad} 项未通过'}")
    return 0 if bad == 0 else 1


def from_matrix(sample_id: str, center: str | None) -> int:
    sys.path.insert(0, str(HERE))
    from wafer_png_v2 import REPO, recover_matrix
    r = recover_matrix(REPO / "data" / "images" / f"{sample_id}.png")
    m = r["matrix"]
    H, W = m.shape
    if center:
        cr, cc = (float(x) for x in center.split(","))
        src = "调用方给定"
    else:
        ys, xs = (m > 0).nonzero()
        cr, cc = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
        src = "本工具按晶圆掩膜外接框中心临时算的（**仅为演示**，正式使用必须由调用方给定并记录）"
    ys, xs = (m == 2).nonzero()
    h = angular_histogram(ys, xs, cr, cc, H, W)
    print(json.dumps({
        "sample_id": sample_id, "matrix_HW": [H, W],
        "center_row_col": [cr, cc], "center_source": src,
        "n_defect": h["n"], "n_on_boundary": h["n_on_boundary"],
        "counts": h["counts"], "modal_sector_number": h["modal_sector_number"],
        "angle_config_version": CFG["version"],
        "warning": "圆心不确定度未声明（按 0 处理）。圆心若有漂移，边界点会在容差外翻转。",
    }, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--from-matrix")
    ap.add_argument("--center", help="栅格坐标下的 row,col")
    a = ap.parse_args()
    if a.from_matrix:
        return from_matrix(a.from_matrix, a.center)
    return demo()


if __name__ == "__main__":
    sys.exit(main())
