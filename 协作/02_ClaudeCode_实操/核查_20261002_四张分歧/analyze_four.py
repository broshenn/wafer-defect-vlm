#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四张类别分歧的离线核查：样本对应 + 几何诊断。

输入全部来自本机，只读：
  * data/images/<sample_id>.png —— 渲染图（可从它无损反解出原矩阵，见 wafer_png.py）
  * data/manifest.jsonl        —— 原标签与历史特征
  * 协作/01_Codex_指挥/复核_20261002_GLM九张/事实核验_v2.json —— 模型答案
  * 协作/03_ZCode_标注/20261002-012138/<sample_id>_raw.json

**原始矩阵（data/raw/）本机没有**。本脚本用的是「从 PNG 逐格反解、并用往返像素证明其无损」
的矩阵，报告中一律标为"反解矩阵"，不冒充原始数据复验。

只读、不联网、不调用模型。

名词（口径先写清楚，避免读错）：
  * 分母 defect_ratio = defect_count / valid_count，两者都在**过滤后**的矩阵上数。
  * 过滤 isolated_point_filter：8 邻域内缺陷数 ≤1 的缺陷格 → 改写为合格 die(1)。
  * 方向 12 点钟为上、顺时针；历史实现 x=(col-cx)/R, y=(cy-row)/R, angle=atan2(x,y) mod 2π。
  * **栅格角 vs 图像角**：历史角在栅格坐标里算；模型看到的图经过 448×448 归一化拉伸。
    晶圆在栅格坐标里是椭圆、在图像里是圆，两者只在 W==H 时同角。本脚本两个都算。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wafer_png import REPO, recover_matrix, roundtrip  # noqa: E402

sys.path.insert(0, str(REPO / "code" / "projects" / "wafer-defect-vlm" / "src"))
from wafer_vlm.utils import calculate_static_features, isolated_point_filter  # noqa: E402

FOUR = [
    ("item_001", "wafer_00044523_017", "none", "Random"),
    ("item_003", "wafer_00025336_019", "Loc", "Edge_Loc"),
    ("item_007", "wafer_00018489_020", "Donut", "Loc"),
    ("item_008", "wafer_00017102_001", "Random", "Scratch"),
]
CONTEXT = [
    ("item_002", "wafer_00043922_015"), ("item_004", "wafer_00015139_006"),
    ("item_005", "wafer_00047178_011"), ("item_006", "wafer_00013704_014"),
    ("item_009", "wafer_00011911_008"),
]


def load_manifest():
    rows = []
    with open(REPO / "data" / "manifest.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                r["__line__"] = i
                rows.append(r)
    return rows


def fit_extents(mask: np.ndarray) -> dict:
    """晶圆在栅格坐标里的外接范围与椭圆半轴（用于量出栅格各向异性）。"""
    ys, xs = np.nonzero(mask)
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    return {
        "mask_row_span": r1 - r0 + 1,
        "mask_col_span": c1 - c0 + 1,
        "ellipse_semi_a_rows": (r1 - r0 + 1) / 2.0,   # 纵向半轴（格）
        "ellipse_semi_b_cols": (c1 - c0 + 1) / 2.0,   # 横向半轴（格）
        "grid_aspect_rows_over_cols": (r1 - r0 + 1) / (c1 - c0 + 1),
    }


def components(matrix: np.ndarray, conn: int = 8):
    """缺陷连通块。conn=8 与 isolated_point_filter 的邻域定义一致。"""
    defects = matrix == 2
    st = np.ones((3, 3), int) if conn == 8 else None
    lab, n = ndimage.label(defects, structure=st)
    if n == 0:
        return lab, 0, []
    sizes = np.bincount(lab.ravel())[1:]
    order = np.argsort(-sizes)
    info = []
    ys, xs = np.nonzero(defects)
    for k in order:
        m = lab == (k + 1)
        yy, xx = np.nonzero(m)
        info.append({
            "label": int(k + 1),
            "size": int(sizes[k]),
            "rows": [int(yy.min()), int(yy.max())],
            "cols": [int(xx.min()), int(xx.max())],
        })
    return lab, int(n), info


def angle_sector(angle_rad: float) -> int:
    """12 点钟为 0、顺时针的 12 扇区，返回 1..12。"""
    return int(angle_rad / (2 * math.pi / 12)) + 1


def sector_center_clock(s: int) -> str:
    lo = (s - 1) * 30
    return f"{lo}-{lo + 30}°({s}扇区)"


def geometry(matrix: np.ndarray, feats: dict) -> dict:
    """在反解矩阵上做几何诊断。所有定义都写在这里，便于复核。"""
    H, W = matrix.shape
    raw_defects = matrix == 2
    filt = isolated_point_filter(matrix)
    defects = filt == 2
    valid = filt > 0

    out: dict = {}
    out["matrix_HW"] = [H, W]
    out["valid_count"] = int(valid.sum())
    out["raw_defect_count"] = int(raw_defects.sum())
    out["filtered_defect_count"] = int(defects.sum())
    out["removed_by_filter"] = int(raw_defects.sum() - defects.sum())
    out["removed_fraction_of_raw"] = round(
        (raw_defects.sum() - defects.sum()) / max(raw_defects.sum(), 1), 4)

    # ── 连通块 ──────────────────────────────────────────
    lab, n, comps = components(matrix, conn=8)
    out["components_8conn"] = n
    out["component_sizes_top5"] = [c["size"] for c in comps[:5]]
    singles = sum(1 for c in comps if c["size"] == 1)
    out["size1_components"] = singles
    in_singles = sum(c["size"] for c in comps if c["size"] == 1)
    out["defect_cells_in_size1"] = in_singles
    out["scatter_share_of_raw"] = round(in_singles / max(raw_defects.sum(), 1), 4)
    if comps:
        biggest = comps[0]
        out["largest_component_size"] = biggest["size"]
        out["largest_component_share_of_raw"] = round(
            biggest["size"] / max(raw_defects.sum(), 1), 4)
        out["largest_component_bbox_rows"] = biggest["rows"]
        out["largest_component_bbox_cols"] = biggest["cols"]
        bh = biggest["rows"][1] - biggest["rows"][0] + 1
        bw = biggest["cols"][1] - biggest["cols"][0] + 1
        out["largest_component_bbox_hw"] = [bh, bw]
        out["largest_component_fill_ratio"] = round(biggest["size"] / max(bh * bw, 1), 3)

    # ── 主簇是否接晶圆边缘（用掩膜边界，不用质心） ──────
    mask = matrix > 0
    boundary = mask & ~ndimage.binary_erosion(mask)
    edge_cells = set(zip(*np.nonzero(boundary)))
    if comps:
        m = lab == comps[0]["label"]
        cy_, cx_ = np.nonzero(m)
        d_edge = [min(abs(r - er) + abs(c - ec) for er, ec in edge_cells)
                  for r, c in zip(cy_, cx_)]
        out["main_cluster_min_dist_to_wafer_edge_cells"] = int(min(d_edge))
        out["main_cluster_touches_edge"] = bool(min(d_edge) <= 1)
        # 最近的边缘点落在哪个钟点
        k = int(np.argmin(d_edge))
        er, ec = min(edge_cells, key=lambda p: abs(p[0] - cy_[k]) + abs(p[1] - cx_[k]))
        out["main_cluster_nearest_edge_cell"] = [int(er), int(ec)]

    # ── 径向（历史圆拟合 vs 椭圆归一） ──────────────────
    cx, cy, rad = feats["wafer_center_xy"][0], feats["wafer_center_xy"][1], feats["wafer_radius"]
    ext = fit_extents(mask)
    a = ext["ellipse_semi_a_rows"]
    b = ext["ellipse_semi_b_cols"]
    ys, xs = np.nonzero(raw_defects)
    if len(ys):
        xg = xs - cx
        yg = cy - ys                      # 上为 +y
        r_circle = np.hypot(xg, yg) / rad
        r_ellipse = np.hypot(xg / b, yg / a)   # 椭圆归一：物理上各向同性
        out["wafer_radius_circle_fit_cells"] = round(rad, 3)
        out["ellipse_semi_axes_cells"] = [round(a, 2), round(b, 2)]
        out["grid_anisotropy_a_over_b"] = round(a / b, 4)
        out["centroid_r_circle"] = round(float(np.hypot(xg.mean(), yg.mean()) / rad), 4)
        out["centroid_r_ellipse"] = round(float(np.hypot(xg.mean() / b, yg.mean() / a)), 4)
        # 超出晶圆半径的缺陷比例（圆口径下常常 >1，说明口径不对）
        out["frac_defects_outside_r1_circle"] = round(float((r_circle > 1).mean()), 4)
        out["frac_defects_outside_r1_ellipse"] = round(float((r_ellipse > 1).mean()), 4)
        # 径向密度 10 段（椭圆口径）
        h, _ = np.histogram(r_ellipse, bins=10, range=(0, 1))
        ann = np.pi * (np.linspace(0, 1, 11)[1:] ** 2 - np.linspace(0, 1, 11)[:-1] ** 2)
        dens = h / np.maximum(ann, 1e-12)
        if dens.sum() > 0:
            dens = dens / dens.sum()
        out["radial_density_10_ellipse"] = [round(float(v), 4) for v in dens]
        out["radial_density_10_circle_asrecorded"] = feats.get("radial_density")

    # ── 方向：栅格角 vs 图像角 ──────────────────────────
    if len(ys):
        ang_grid = np.mod(np.arctan2(xg, yg), 2 * math.pi)
        # 图像角 = 栅格角按像素纵横比修正 → 等价于用椭圆归一坐标
        xp = xg / b
        yp = yg / a
        ang_img = np.mod(np.arctan2(xp, yp), 2 * math.pi)
        hg, _ = np.histogram(ang_grid, bins=12, range=(0, 2 * math.pi))
        hi, _ = np.histogram(ang_img, bins=12, range=(0, 2 * math.pi))
        sg = int(np.argmax(hg)) + 1
        si = int(np.argmax(hi)) + 1
        out["clock_sector_grid"] = sg
        out["clock_sector_image_corrected"] = si
        out["clock_sector_manifest"] = feats.get("clock_sector")
        out["clock_sector_differs_grid_vs_image"] = sg != si
        out["angular_hist_grid"] = [int(v) for v in hg]
        out["angular_hist_image"] = [int(v) for v in hi]
        out["centroid_clock_grid"] = sector_center_clock(sg)
        out["centroid_clock_image"] = sector_center_clock(si)
        out["centroid_clock_deg_grid"] = round(math.degrees(math.atan2(xg.mean(), yg.mean())) % 360, 1)
        out["centroid_clock_deg_image"] = round(
            math.degrees(math.atan2(xg.mean() / b, yg.mean() / a)) % 360, 1)

    # ── 假条带：整簇与最大块的伸长 ──────────────────────
    ys2, xs2 = np.nonzero(raw_defects)
    if len(ys2) >= 3:
        pts = np.column_stack([xs2.astype(float), -ys2.astype(float)])
        cov = np.cov(pts.T)
        ev = np.linalg.eigvalsh(cov)
        out["cloud_anisotropy_lambda_min_over_max"] = round(
            float(max(ev[0], 0) / max(ev[-1], 1e-12)), 4)
        out["cloud_principal_axis_deg_from_vertical"] = round(
            float(math.degrees(math.atan2(*np.linalg.eigh(cov)[1][:, -1][::-1]))), 1)
    out["extents"] = ext
    return out


def main() -> int:
    man = {r["sample_id"]: r for r in load_manifest()}
    v2 = json.loads((REPO / "协作" / "01_Codex_指挥" / "复核_20261002_GLM九张"
                     / "事实核验_v2.json").read_text(encoding="utf-8"))
    v2_by = {e["sample_id"]: e for e in v2["entries"]}
    blind = {json.loads(l)["sample_id"]: json.loads(l)
             for l in open(REPO / "协作" / "02_ClaudeCode_实操" / "准备_20261002_九类试标"
                           / "blind_inputs.jsonl", encoding="utf-8")}

    report = {
        "raw_matrix_available_locally": False,
        "raw_matrix_note": "data/raw/ 本机不存在（AGENTS.md §15）。本报告用的是从 PNG 逐格反解、"
                           "并用往返像素证明无损的**反解矩阵**，不是原始 pkl 复验。",
        "items": [],
    }

    for item_id, sid, gt, pred in FOUR + [(a, b, None, None) for a, b in CONTEXT]:
        png = REPO / "data" / "images" / f"{sid}.png"
        rec = recover_matrix(png)
        rt = roundtrip(rec["matrix"], png)
        row = man[sid]
        feats = row["features"]
        recomputed = calculate_static_features(rec["matrix"])

        # 历史特征 vs 复算：逐字段比对
        # 用相对容差而不是精确相等：复算与原计算可能跑在不同 BLAS/numpy 版本上，
        # 最小二乘求和顺序不同会带来 ~1e-15 的浮点噪声（实测正是如此），
        # 那是数值噪声不是分歧。容差取 1e-9 相对 / 1e-12 绝对，比噪声大 6 个数量级、
        # 又远小于任何有物理意义的差别。
        RTOL, ATOL = 1e-9, 1e-12

        def close(a, b):
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return abs(a - b) <= max(ATOL, RTOL * max(abs(a), abs(b)))
            if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                return all(close(x, y) for x, y in zip(a, b))
            return a == b

        def maxrel(a, b):
            if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                return max((maxrel(x, y) for x, y in zip(a, b)), default=0.0)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return abs(a - b) / max(abs(a), abs(b), 1e-30)
            return 0.0 if a == b else float("inf")

        diffs = {}
        for k, v in feats.items():
            rv = recomputed.get(k)
            diffs[k] = {"manifest": v, "recomputed": rv,
                        "max_rel_diff": maxrel(v, rv), "match": close(v, rv)}
        worst = max((d["max_rel_diff"] for d in diffs.values()
                     if isinstance(d["max_rel_diff"], float)), default=0.0)
        diffs["__worst_relative_difference__"] = {
            "manifest": None, "recomputed": None,
            "max_rel_diff": worst, "match": worst <= RTOL}
        diffs["__fields_compared__"] = {
            "manifest": len(feats), "recomputed": len(recomputed),
            "max_rel_diff": 0.0,
            "match": set(feats) == set(recomputed)}

        entry = {
            "item_id": item_id,
            "sample_id": sid,
            "manifest_line": row["__line__"],
            "ground_truth_class": row["failure_type"],
            "glm_class": pred,
            "label_source": row["label_source"],
            "split": row["split"],
            "source_index": row["source_index"],
            "lot_name": row["lot_name"],
            "wafer_index": row["wafer_index"],
            "sample_id_matches_lot_and_wafer": (
                sid == f"wafer_{int(str(row['lot_name']).lstrip('lot')):08d}_{row['wafer_index']:03d}"
            ),
            "manifest_matrix_shape": row["matrix_shape"],
            "png_sha256_actual": rec["png_sha256"],
            "png_sha256_in_blind_list": blind[sid]["image_sha256"],
            "png_sha256_matches_blind": rec["png_sha256"] == blind[sid]["image_sha256"],
            "png_sha256_in_codex_v2": v2_by[sid].get("image_sha256"),
            "recovered_HW": [rec["H"], rec["W"]],
            "recovered_matches_manifest_shape": [rec["H"], rec["W"]] == list(row["matrix_shape"]),
            "edge_fit_score_H": rec["edge_fit_score_H"],
            "edge_fit_score_W": rec["edge_fit_score_W"],
            "unexpected_colors_in_png": rec["unexpected_colors"],
            **rt,
            "feature_match_manifest_vs_recomputed": diffs,
            "all_features_match": all(d["match"] for d in diffs.values()),
            "geometry": geometry(rec["matrix"], feats),
        }
        report["items"].append(entry)

    out = HERE / "核查结果.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {out}")

    # 控制台摘要
    print(f"\n{'item':9s} {'sample':22s} {'原标签':9s} {'GLM':9s} {'反解':9s} {'形状符':4s} "
          f"{'往返像素差':7s} {'特征全符':5s}")
    for e in report["items"]:
        print(f"{e['item_id']:9s} {e['sample_id']:22s} {str(e['ground_truth_class']):9s} "
              f"{str(e['glm_class']):9s} {str(e['recovered_HW']):9s} "
              f"{'✓' if e['recovered_matches_manifest_shape'] else '✗':4s} "
              f"{e['roundtrip_pixels_differing']:7d} {'✓' if e['all_features_match'] else '✗':5s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
