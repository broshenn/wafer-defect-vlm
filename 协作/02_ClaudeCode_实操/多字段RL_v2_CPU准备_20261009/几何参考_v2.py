#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""晶圆几何参考 v2 —— 多字段 RL 的「事实测量」地基。

**这是新实现，不覆盖任何历史脚本。** 历史 `calculate_geo_param`（硬编码 52×52 / R=26）
与旧 features 里的 `angular_probability` / `angle_entropy_bits` / `clock_sector`
存在已知未修问题（AGENTS.md §6、§18.7），本模块不复用它们的公式，只把它们的
**测量对象**（圆心、半径、径向分布、方向）重新定义清楚。

## 一、测量对象与坐标（先把话说明白）

- 输入是**模型实际看到的 448×448 标准 PNG**（黑=片外，绿=合格 die，红=失效 die，
  三色离散图，渲染函数 = `code/projects/wafer-defect-vlm/src/wafer_vlm/utils.py:matrix_to_image`）。
- PNG 是把 (H,W) 的 BIN 矩阵**直接拉伸**到 448×448 得到的（纵横比会变）。
  测量在 **PNG 无损反解出的矩阵坐标** 上做：每个矩阵格恰好对应 PNG 里一整块同色像素，
  反解可做到逐像素往返一致，因此矩阵坐标不是近似，是等价表示。
- 矩阵坐标下：行号 i 向下、列号 j 向右；格 (i,j) 的中心记作 (i+0.5, j+0.5)。
- 钟点方向：**上方为 12 点、右方为 3 点、顺时针**，与冻结题面一致。
- 归一化：晶圆有效区（绿∪红）外接框定出中心 (cy,cx) 与半轴 (a,b)，
  横向按 b、纵向按 a 归一化后，`r = 1` 就是晶圆边缘。
  **非方阵矩阵里晶圆呈椭圆**（实测：45×48 的图，外接框半轴就是 22.5 / 24）——
  归一化正是把这条拉伸还原回圆，角度才可比。

## 二、这一版**不**声称的事

- 不声称矩阵格 = 物理 die 尺寸或间距（未经核验）；不声称归一化还原了真实物理几何。
- 不声称这是原始 WM811K 数据生成时用的真值圆：本模块给出的是**定义**，
  定义本身可复核、可复现，不是对数据生成过程的断言。
- 尺寸（extent）本轮只产出**候选定义下的诊断值**，不进奖励；见 `extent_candidates`。

## 三、失败状态

每个测量都有显式状态码；无缺陷、单点、共线、轮廓异常都各归其类，
**不把「没测出来」写成 0，也不把「不适用」当满分**。

只读图片与矩阵；不联网；不调用模型。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 定义版本：任何影响参考值或阈值的改动都必须提升它，并另存新文件
# ─────────────────────────────────────────────────────────────────────────────
DEFINITION_VERSION = "wafer_geom_v2.0"
THRESHOLD_SET_VERSION = "geom_thr_v2.0-train"

RES = 448
REPO = Path(__file__).resolve().parents[3]

# 反解器来源（已验证实现，按 SHA 钉住；不复制其代码以免分叉）
RECOVER_SRC = REPO / "协作" / "02_ClaudeCode_实操" / "口径修正与准备_20261002-021207" / "wafer_png_v2.py"
RECOVER_SRC_SHA256 = "b8214c0ad1388bbf6c54f06db95aaccbcf0865c7ac6fe2df48b192562f286800"

# ── 冻结的测量参数（阈值来自 train 侧诊断，见 阈值诊断.json；冻结后不得训练中改）──
COVERAGE_BANDS = [
    # (名称, 下界含, 上界不含)
    ("none", 0.0, 0.0),
    ("low", 0.0, 0.05),
    ("medium", 0.05, 0.20),
    ("high", 0.20, 0.50),
    ("near_all", 0.50, 1.0000001),
]
COVERAGE_NONE_EPS = 0.0          # red == 0 → none
ZONE_EDGES = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)   # center / middle / edge
ZONE_NAMES = ("center", "middle", "edge")
ZONE_MASS_TAU = 0.10             # 区带占红点质量 ≥10% 才算「存在」
ZONE_MIN_DIES = 2                # 且至少 2 个红格，避免单格噪声
ZONE_MIN_TOTAL_RED = 3           # 红格总数少于此值 → 区带参考记 unknown（证据太薄）

N_MIN_DIRECTION = 8              # 少于此红格数 → 方向参考 unknown（证据不足）
N_MIN_MEAN_R = 0.25              # 轴向判据要求红点平均半径够大（否则角度无意义）
TAU_R1_SINGLE = 0.60             # 绕晶圆中心的单向集中度 ≥ 此值 → single
TAU_R2_AXIS = 0.70               # 轴向集中度（二阶谐波）≥ 此值 → axis
R1_NONE_MAX = 0.20               # R1 < 此值 且 R2 < R2_NONE_MAX → 才敢判「无方向」
R2_NONE_MAX = 0.30
# ── 线状结构检测（应对「划痕 + 边缘红圈 + 散布」的稀释）──────────────────
LINE_HALF_WIDTH = 0.05           # 直线带半宽（归一化半径单位）
LINE_MIN_POINTS = 8              # 带内至少这么多红格
LINE_MASS_TAU = 0.25             # 带内红点占全部红点的比例 ≥ 此值才算「有线状结构」
                                 # 只用于**否决「无方向」**，不用来给出方向（见判据注释）
LINE_SPAN_MIN = 0.60             # 带内红点沿直线方向跨度 ≥ 此值（R 单位）
LINE_FILL_MIN = 0.55             # 跨度上被占的等分格比例 ≥ 此值（环会只占两端 → 被排除）
LINE_NBINS = 20
SECTOR_TIE_TOL_DEG = 3.0         # 方向落在扇区边界 ±3° 内 → 扇区并列，扇区参考不可用
SHAPE_ANOMALY_FRAC = 0.10        # 外接椭圆解释不了的有效格/黑格超过此比例 → 轮廓异常
                                 # 实测 400 张 train 侧样本：leak 最大 3.0%、hole 最大 6.4%
                                 # （残差来自原图圆盘在格点上量化的固有误差），
                                 # 阈值放在实测极值之外，只捕捉「严重非椭圆/被截断」的轮廓。

CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_full", "Random", "Scratch", "none"]
COVERAGE_LEVELS = ["none", "low", "medium", "high", "near_all", "unknown"]
DIRECTION_TYPES = ["single", "axis", "none", "unknown"]


# ─────────────────────────────────────────────────────────────────────────────
# 反解器装载
# ─────────────────────────────────────────────────────────────────────────────
def _load_recover_module():
    if not RECOVER_SRC.exists():
        raise FileNotFoundError(
            f"反解器不在 {RECOVER_SRC}；本模块不复制其代码以避免分叉，请恢复该文件")
    spec = importlib.util.spec_from_file_location("_wafer_png_v2_pinned", RECOVER_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def recover_source_sha256() -> str:
    return hashlib.sha256(RECOVER_SRC.read_bytes()).hexdigest()


def recover_matrix(png_path: Path, hint_hw: tuple[int, int] | None = None) -> dict:
    """PNG → BIN 矩阵（无损反解）。

    `hint_hw` 给出 (H,W) 时先按它反解并做像素级往返验证；通过即采用（快路径）。
    不通过或未给 hint 时回落到全搜索。**两条路都必须往返验证通过**，
    验证失败时 `roundtrip_validated=False`，调用方必须当失败处理。
    """
    mod = _load_recover_module()
    png_path = Path(png_path)
    if hint_hw is not None:
        h, w = int(hint_hw[0]), int(hint_hw[1])
        if 1 <= h <= RES and 1 <= w <= RES:
            raw = png_path.read_bytes()
            import io
            from PIL import Image
            arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            # 先做尺寸与颜色检查，与全搜索路径同一口径
            cand, unknown = mod._recover_at(arr, h, w)
            if not unknown:
                buf = io.BytesIO()
                mod.matrix_to_image(cand, RES).save(buf, format="PNG", optimize=True)
                buf.seek(0)
                if (np.asarray(Image.open(buf).convert("RGB")) == arr).all():
                    return {
                        "matrix": cand, "H": h, "W": w,
                        "roundtrip_validated": True, "path_used": "hint",
                        "full_score_candidates_H": [h], "full_score_candidates_W": [w],
                        "png_sha256": hashlib.sha256(raw).hexdigest(),
                        "unexpected_colors": [],
                        "recover_module_sha256": recover_source_sha256(),
                    }
    out = mod.recover_matrix(png_path)
    out["path_used"] = "search"
    out["recover_module_sha256"] = recover_source_sha256()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 纯测量：矩阵 → 参考字段（无 I/O，可单测）
# ─────────────────────────────────────────────────────────────────────────────
def _ellipse_of(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    cy = (float(ys.min()) + float(ys.max()) + 1.0) / 2.0
    cx = (float(xs.min()) + float(xs.max()) + 1.0) / 2.0
    a = (float(ys.max()) - float(ys.min()) + 1.0) / 2.0
    b = (float(xs.max()) - float(xs.min()) + 1.0) / 2.0
    return cy, cx, a, b


def _normalized_uv(H: int, W: int, cy: float, cx: float, a: float, b: float):
    ii, jj = np.mgrid[0:H, 0:W]
    u = (ii + 0.5 - cy) / a      # 纵向（向下为正）
    v = (jj + 0.5 - cx) / b      # 横向（向右为正）
    return u, v


def _clock_deg(u, v):
    """由归一化坐标求钟点角：12 点 = 0°，顺时针增大，[0,360)。"""
    deg = np.degrees(np.arctan2(v, -u))
    return np.mod(deg, 360.0)


def sector_of_deg(deg: float) -> int:
    """0°=12 点起顺时针，每 30° 一个扇区，返回 1..12。"""
    return int(math.floor((deg % 360.0) / 30.0)) + 1


def _sector_boundary_margin_deg(deg: float) -> float:
    """到最近扇区边界的角距（度）。0 = 正好压在边界上。"""
    m = (deg % 30.0)
    return min(m, 30.0 - m)


def coverage_level_of(ratio: float, n_red: int) -> tuple[str, bool]:
    """返回 (档位, 是否落在档位边界附近)。"""
    if n_red == 0:
        return "none", False
    band = None
    for name, lo, hi in COVERAGE_BANDS:
        if name == "none":
            continue
        if lo <= ratio < hi:
            band = name
            break
    if band is None:
        band = "near_all"
    # 距离所属档位任一界 0.004 以内算边界样本（≈1/250，1 个红格级别的抖动）
    for name, lo, hi in COVERAGE_BANDS:
        if name == "none":
            continue
        for edge in (lo, hi if hi < 1.0 else None):
            if edge is not None and abs(ratio - edge) <= 0.004:
                return band, True
    return band, False


def measure_matrix(matrix: np.ndarray) -> dict:
    """矩阵 → 参考字段。**不读标签，不读图片，不猜类别。**"""
    m = np.asarray(matrix)
    if m.ndim != 2:
        return {"status": "bad_matrix_shape", "definition_version": DEFINITION_VERSION}
    H, W = int(m.shape[0]), int(m.shape[1])
    green = m == 1
    red = m == 2
    valid = green | red
    n_green, n_red, n_valid = int(green.sum()), int(red.sum()), int(valid.sum())

    out: dict = {
        "definition_version": DEFINITION_VERSION,
        "threshold_set_version": THRESHOLD_SET_VERSION,
        "matrix_hw": [H, W],
        "n_green": n_green, "n_red": n_red, "n_valid": n_valid,
    }
    if n_valid == 0:
        out.update({"status": "no_valid_area", "coverage": None, "coverage_level": "unknown",
                    "red_mass_zones": "unknown", "direction_type": "unknown", "clock_sectors": []})
        return out

    # ── 晶圆轮廓（椭圆）与吻合度 ─────────────────────────────────────────
    cy, cx, a, b = _ellipse_of(valid)
    if a <= 0 or b <= 0:
        out.update({"status": "degenerate_outline", "coverage": None,
                    "coverage_level": "unknown", "red_mass_zones": "unknown",
                    "direction_type": "unknown", "clock_sectors": []})
        return out
    u, v = _normalized_uv(H, W, cy, cx, a, b)
    r = np.sqrt(u * u + v * v)
    inside = r <= 1.0
    leak = int((valid & ~inside).sum())        # 有效格落在椭圆外
    hole = int((~valid & inside).sum())        # 椭圆内却是黑格
    leak_frac, hole_frac = leak / n_valid, hole / max(1, int(inside.sum()))
    out["outline"] = {
        "model": "axis_aligned_ellipse_by_valid_bbox",
        "center_yx": [round(cy, 4), round(cx, 4)],
        "semi_axes_ab": [round(a, 4), round(b, 4)],
        "leak_count": leak, "hole_count": hole,
        "leak_frac_of_valid": round(leak_frac, 5), "hole_frac_of_inside": round(hole_frac, 5),
        "anomalous": bool(leak_frac > SHAPE_ANOMALY_FRAC or hole_frac > SHAPE_ANOMALY_FRAC),
        "bbox_is_full_matrix": bool(a == H / 2 and b == W / 2),
    }

    # ── 覆盖率：分母是有效格（绿∪红），黑格不进分母 ─────────────────────
    ratio = n_red / n_valid
    lvl, band_edge = coverage_level_of(ratio, n_red)
    out["coverage"] = round(ratio, 6)
    out["coverage_level"] = lvl
    out["coverage_band_edge"] = bool(band_edge)

    if n_red == 0:
        out.update({"status": "no_red", "red_mass_zones": [],
                    "direction_type": "none", "clock_sectors": [],
                    "radial": {"bins10": [0.0] * 10,
                               "zone_mass": {k: 0 for k in ZONE_NAMES},
                               "zone_frac": {k: 0.0 for k in ZONE_NAMES}}})
        out["direction"] = {"n_red": 0, "note": "无红点：direction_type=none 是事实，不是 unknown"}
        out["extent_candidates"] = {"not_computed": "no_red"}
        return out

    ru, rv = r[red], None
    rr = r[red]
    deg = _clock_deg(u[red], v[red])

    # ── 径向质量分布（固定 [0,1] 终点，10 个 bin）───────────────────────
    hist, _ = np.histogram(rr, bins=np.linspace(0.0, 1.0, 11))
    zone_mass = {}
    for name, lo, hi in zip(ZONE_NAMES, ZONE_EDGES[:-1], ZONE_EDGES[1:]):
        zone_mass[name] = int(((rr >= lo) & (rr < hi)).sum())
    over = int((rr > 1.0).sum())         # 有效格在椭圆外的红点（轮廓异常时可能非零）
    zone_frac = {k: zone_mass[k] / n_red for k in ZONE_NAMES}
    out["radial"] = {
        "bins10": [int(x) for x in hist],
        "bin_edges": [round(x, 4) for x in np.linspace(0.0, 1.0, 11)],
        "zone_mass": zone_mass, "zone_frac": {k: round(v, 5) for k, v in zone_frac.items()},
        "n_red_beyond_r1": over,
        "defect_density_by_zone": {
            k: round(zone_mass[k] / max(1, int(((r >= lo) & (r < hi) & valid).sum())), 5)
            for k, lo, hi in zip(ZONE_NAMES, ZONE_EDGES[:-1], ZONE_EDGES[1:])
        },
    }

    # ── 红点质量所在径向区带（有序去重）────────────────────────────────
    if n_red < ZONE_MIN_TOTAL_RED:
        out["red_mass_zones"] = "unknown"
        out["red_mass_zones_rule"] = {
            "why_unknown": f"n_red={n_red} < {ZONE_MIN_TOTAL_RED}，质量占比在如此小的样本上不稳定",
            "note": "不确定就记 unknown，不硬填区带；奖励端据此排除该项而不是按 0 计",
        }
    else:
        zones_present = [k for k in ZONE_NAMES
                         if zone_mass[k] >= ZONE_MIN_DIES and zone_frac[k] >= ZONE_MASS_TAU]
        out["red_mass_zones"] = zones_present
        out["red_mass_zones_rule"] = {
            "tau_mass": ZONE_MASS_TAU, "min_dies": ZONE_MIN_DIES,
            "note": "全体红点按归一化半径分带的质量占比；不是「主要结构」的语义区域",
        }

    # ── 方向：线状（轴）或单向（single）或非方向性 ──────────────────────
    uu, vv = u[red], v[red]
    r1 = float(np.hypot(np.cos(np.radians(deg)).mean(), np.sin(np.radians(deg)).mean()))
    r2 = float(np.hypot(np.cos(np.radians(2 * deg)).mean(), np.sin(np.radians(2 * deg)).mean()))
    mean_dir_deg = float(math.degrees(math.atan2(np.sin(np.radians(deg)).mean(),
                                                 np.cos(np.radians(deg)).mean())) % 360.0)
    # 轴向角由二阶谐波给出：直线两端 θ 与 θ+180 的二阶相位相同，2φ mod 360 → φ mod 180
    axis_deg_h = float(math.degrees(math.atan2(np.sin(np.radians(2 * deg)).mean(),
                                               np.cos(np.radians(2 * deg)).mean())) / 2.0 % 180.0)
    # 关于红点质心的 PCA（归一化空间），各向异性比 = λmin/λmax，越接近 0 越线状
    pts = np.stack([uu - uu.mean(), vv - vv.mean()], axis=1)
    cov = (pts.T @ pts) / max(1, len(pts))
    evals, evecs = np.linalg.eigh(cov)
    lam_min, lam_max = float(evals[0]), float(evals[1])
    linearity = 0.0 if lam_max <= 0 else lam_min / lam_max
    axis_vec = evecs[:, 1]                     # 主轴（对应 λmax）
    axis_deg = float(math.degrees(math.atan2(axis_vec[1], axis_vec[0])) % 180.0)

    line = _line_structure(uu, vv)
    dt, sectors, why, tie = _decide_direction(n_red, r1, r2, float(rr.mean()),
                                              axis_deg_h, mean_dir_deg, line)
    out["direction"] = {
        "n_red": n_red,
        "R1_concentration": round(r1, 5), "R2_axial": round(r2, 5),
        # 以下两项是**诊断**，不参与判据：整图 PCA 会被边缘红圈/放射状多线带偏
        # （实测：某张清晰的过中心竖线，因叠加边缘红圈，λmin/λmax 高达 0.79）
        "linearity_lambda_min_over_max_DIAG": round(linearity, 5),
        "mean_r": round(float(rr.mean()), 5),
        "lambda_max": round(lam_max, 6), "lambda_min": round(lam_min, 6),
        "mean_dir_deg_12clockwise": round(mean_dir_deg, 2),
        "axis_deg_mod180": round(axis_deg, 2),
        "centroid_angle_deg_DIAG_only": round(
            float(math.degrees(math.atan2(vv.mean(), -uu.mean())) % 360.0), 2),
        "direction_type": dt, "clock_sectors": sectors,
        "decided_by": why, "sector_tie": tie,
        "line_structure": line,
        "thresholds": {"N_min": N_MIN_DIRECTION, "N_min_mean_r": N_MIN_MEAN_R,
                       "tau_R1_single": TAU_R1_SINGLE, "tau_R2_axis": TAU_R2_AXIS,
                       "R1_none_max": R1_NONE_MAX, "R2_none_max": R2_NONE_MAX,
                       "line": {"half_width": LINE_HALF_WIDTH, "mass_tau": LINE_MASS_TAU,
                                "span_min": LINE_SPAN_MIN, "fill_min": LINE_FILL_MIN,
                                "n_bins": LINE_NBINS},
                       "sector_tie_tol_deg": SECTOR_TIE_TOL_DEG},
        "note": "方向由红点绕晶圆圆心的角向分布（R1/R2 圆统计量）决定；"
                "整图质心角度只作诊断，不作定义。模糊带一律记 unknown，不强塞方向答案。",
    }
    out["direction_type"] = dt
    out["clock_sectors"] = sectors

    # ── 尺寸候选（本轮不进奖励，只作为待定义对象的诊断）────────────────
    out["extent_candidates"] = {
        "not_for_reward": "本轮尺寸不进主奖励，见 schema v2 草案",
        "all_red_radial_span_r": round(float(rr.max() - rr.min()), 5),
        "all_red_max_r": round(float(rr.max()), 5),
        "all_red_min_r": round(float(rr.min()), 5),
        "cloud_major_axis_len_r": round(float(2.0 * math.sqrt(max(0.0, lam_max))), 5),
        "cloud_minor_axis_len_r": round(float(2.0 * math.sqrt(max(0.0, lam_min))), 5),
    }

    out["status"] = "ok"
    return out


def _line_structure(uu: np.ndarray, vv: np.ndarray) -> dict:
    """稳健的线状结构检测（**允许直线不过圆心**）：找「带内点数 × 跨度填充度」最大的直线。

    为什么需要它：一张「划痕 + 边缘红圈 + 散布红点」的图，谐波 R1/R2 会被稀释
    （实测 R2 只有 0.19），但直线带内的跨度与填充度仍能把它认出来。
    环形图案也会落进某条直线带（两端各一段弧），所以用 `fill`（跨度填充度）
    与 `span`（跨度）把环排除掉。

    算法：先在 180 个方向上对法向坐标做滑窗直方图，取带内点数最多的那个偏置，
    再对该带内点计算沿轴跨度与跨度填充度。
    """
    n = len(uu)
    if n < LINE_MIN_POINTS:
        return {"found": False, "why": "点数不足"}
    best = None
    for phi_deg in np.arange(0.0, 180.0, 1.0):
        phi = math.radians(phi_deg)
        s = -uu * math.sin(phi) + vv * math.cos(phi)      # 法向坐标
        lo_s, hi_s = float(s.min()), float(s.max())
        if hi_s - lo_s <= LINE_HALF_WIDTH:
            continue
        edges = np.arange(lo_s, hi_s + LINE_HALF_WIDTH, LINE_HALF_WIDTH)
        hist, _ = np.histogram(s, bins=edges)
        win = 3                                             # ±LINE_HALF_WIDTH
        if len(hist) < win:
            continue
        csum = np.concatenate([[0], np.cumsum(hist)])
        counts = csum[win:] - csum[:-win]
        j = int(np.argmax(counts))
        s0 = 0.5 * (edges[j] + edges[min(j + win, len(edges) - 1)])
        m = np.abs(s - s0) <= LINE_HALF_WIDTH
        k = int(m.sum())
        if k < LINE_MIN_POINTS:
            continue
        t = uu * math.cos(phi) + vv * math.sin(phi)
        t_in = t[m]
        lo, hi = float(t_in.min()), float(t_in.max())
        span = hi - lo
        if span <= 1e-9:
            continue
        h2, _ = np.histogram(t_in, bins=np.linspace(lo, hi, LINE_NBINS + 1))
        fill = float((h2 > 0).mean())
        score = k * fill
        if best is None or score > best["score"]:
            best = {"score": score, "phi_deg": float(phi_deg), "s0": float(s0),
                    "k": k, "span": span, "fill": fill, "t": t_in,
                    "u": uu[m], "v": vv[m]}
    if best is None:
        return {"found": False, "why": "无候选直线"}
    frac = best["k"] / n
    # 两端各取 10% 跨度的点，求其平均方向角 → 轴的两个扇区
    t, u_in, v_in = best["t"], best["u"], best["v"]
    lo, hi = float(t.min()), float(t.max())
    span = hi - lo
    band = max(1e-9, 0.10 * span)
    ends = []
    for sel in (t <= lo + band, t >= hi - band):
        if sel.sum() == 0:
            continue
        du, dv = float(u_in[sel].mean()), float(v_in[sel].mean())
        ends.append(float(math.degrees(math.atan2(dv, -du)) % 360.0))
    ok = (frac >= LINE_MASS_TAU and span >= LINE_SPAN_MIN and best["fill"] >= LINE_FILL_MIN)
    return {"found": bool(ok), "phi_deg": round(best["phi_deg"], 2),
            "offset_from_center": round(best["s0"], 4),
            "in_band": best["k"], "mass_frac": round(frac, 4),
            "span": round(span, 4), "fill": round(best["fill"], 4),
            "end_angles_deg": [round(x, 2) for x in ends],
            "why": ("达标" if ok else
                    f"未达标（mass_frac={frac:.2f}<{LINE_MASS_TAU} 或 span={span:.2f}<{LINE_SPAN_MIN}"
                    f" 或 fill={best['fill']:.2f}<{LINE_FILL_MIN}）")}


def _decide_direction(n_red: int, r1: float, r2: float, mean_r: float,
                      axis_deg: float, mean_dir_deg: float, line: dict):
    """方向判定。返回 (direction_type, sectors, 依据, 是否扇区并列)。

    判据来自**绕晶圆圆心的角向分布**（R1/R2 圆统计量）与**稳健线状结构检测**，
    不依赖整图 PCA：PCA 会被「边缘红圈 + 一条线」这类混合模式带偏。
    判不出就记 unknown —— 参考未知的样本不参与该项计分，好过硬塞一个方向。
    """
    if n_red < N_MIN_DIRECTION:
        return "unknown", [], f"n_red<{N_MIN_DIRECTION}", False
    # 注意：线状结构检测（line_structure）**只作诊断，不参与判据**。
    # 实测它分不开「细线」与「实心团块」：一个实心盘的中央带同样有高填充度与高占比
    # （Center 类 mass_frac 中位数 0.15、fill p90 0.90）。把它接进判据会给出错误方向。
    if r1 >= TAU_R1_SINGLE:
        tie = _sector_boundary_margin_deg(mean_dir_deg) <= SECTOR_TIE_TOL_DEG
        return "single", [sector_of_deg(mean_dir_deg)], f"R1>={TAU_R1_SINGLE}", bool(tie)
    if r2 >= TAU_R2_AXIS and mean_r >= N_MIN_MEAN_R:
        s1, s2 = sector_of_deg(axis_deg), sector_of_deg(axis_deg + 180.0)
        tie = (_sector_boundary_margin_deg(axis_deg) <= SECTOR_TIE_TOL_DEG
               or _sector_boundary_margin_deg(axis_deg + 180.0) <= SECTOR_TIE_TOL_DEG)
        return "axis", sorted({s1, s2}), f"R2>={TAU_R2_AXIS} 且 mean_r>={N_MIN_MEAN_R}", bool(tie)
    if r1 < R1_NONE_MAX and r2 < R2_NONE_MAX:
        if line.get("found"):
            # 检测到线状结构时**不敢说「没有方向」**：即便谐波被稀释到低于阈值，
            # 图上仍可能有一条真实的划痕。此处一律记 unknown，不记 none。
            return "unknown", [], "谐波很低但检测到线状结构，不敢断言无方向", False
        return "none", [], f"R1<{R1_NONE_MAX} 且 R2<{R2_NONE_MAX} 且无线状结构（有把握说没有方向）", False
    return "unknown", [], "落在模糊带：不足以判方向，也不足以断言无方向", False


# ─────────────────────────────────────────────────────────────────────────────
# 便捷入口：PNG → 参考字段
# ─────────────────────────────────────────────────────────────────────────────
def reference_from_png(png_path: Path, hint_hw=None) -> dict:
    png_path = Path(png_path)
    try:
        rec = recover_matrix(png_path, hint_hw=hint_hw)
    except Exception as exc:  # 反解失败必须是显式状态，不能吞掉
        return {"status": "recover_failed", "error": f"{type(exc).__name__}: {exc}",
                "definition_version": DEFINITION_VERSION, "png_path": str(png_path)}
    if not rec.get("roundtrip_validated"):
        return {"status": "recover_not_lossless", "definition_version": DEFINITION_VERSION,
                "png_path": str(png_path),
                "detail": {k: rec.get(k) for k in
                           ("H", "W", "full_score_candidates_H", "full_score_candidates_W")}}
    ref = measure_matrix(rec["matrix"])
    ref["png_path"] = str(png_path)
    ref["png_sha256"] = rec["png_sha256"]
    ref["recover_path_used"] = rec["path_used"]
    ref["recover_module_sha256"] = rec["recover_module_sha256"]
    return ref


# ─────────────────────────────────────────────────────────────────────────────
# 自测：合成用例（离线，无真实标签）
# ─────────────────────────────────────────────────────────────────────────────
def _disc(n=41, red_mode="center") -> np.ndarray:
    """合成方形晶圆矩阵：1=绿、2=红、0=黑。"""
    m = np.zeros((n, n), dtype=np.uint8)
    c = (n - 1) / 2.0
    ii, jj = np.mgrid[0:n, 0:n]
    r = np.hypot(ii - c, jj - c)
    m[r <= n / 2 - 0.5] = 1
    if red_mode == "center":
        m[r <= n * 0.12] = 2
    elif red_mode == "edge_ring":
        m[(r <= n / 2 - 0.5) & (r >= n / 2 - 3.0)] = 2
    elif red_mode == "eccentric":
        # 右侧一个紧凑团块（半径 3 格），离晶圆中心约 0.55R
        cyy, cxx = c, c + n * 0.30
        m[(np.hypot(ii - cyy, jj - cxx) <= 3.0) & (r <= n / 2 - 0.5)] = 2
    elif red_mode == "line_through_center":
        m[np.abs(jj - c) <= 0.5] = 2
        m[r > n / 2 - 0.5] = 0
    elif red_mode == "chord_line":
        m[(np.abs(ii - c * 0.6) <= 0.5) & (r <= n / 2 - 0.5)] = 2
    elif red_mode == "single":
        m[int(c), int(c)] = 2
    elif red_mode == "scatter":
        rs = np.random.RandomState(3407)
        idx = rs.choice(n * n, 40, replace=False)
        m.reshape(-1)[idx] = 2
        m[r > n / 2 - 0.5] = 0
    elif red_mode == "collinear":
        m[int(c), int(c - 2):int(c + 3)] = 2
    elif red_mode == "none":
        pass
    return m


def self_test() -> dict:
    cases = []
    add = lambda name, got, want, extra=None: cases.append(  # noqa: E731
        {"name": name, "passed": bool(got == want), "got": got, "want": want, "extra": extra})

    # 1 中心块 → zones 只含 center；方向 none（不是一个钟点）
    r = measure_matrix(_disc(red_mode="center"))
    add("中心块_zones", r["red_mass_zones"], ["center"])
    add("中心块_方向", r["direction_type"], "none", r["direction"])
    add("中心块_状态", r["status"], "ok")

    # 2 边缘环 → 必须判「靠边缘」，不是中心
    r = measure_matrix(_disc(red_mode="edge_ring"))
    add("边缘环_zones", r["red_mass_zones"], ["edge"])
    add("边缘环_方向", r["direction_type"], "none", r["direction"])

    # 3 偏心块 → 单向 single，扇区在右侧（3 点方向 = 扇区 4）
    r = measure_matrix(_disc(red_mode="eccentric"))
    add("偏心块_方向", r["direction_type"], "single", r["direction"])
    add("偏心块_扇区在右", r["clock_sectors"], [4], r["direction"])

    # 4 穿中心的线 → 轴（axis），两个相对扇区
    r = measure_matrix(_disc(red_mode="line_through_center"))
    add("中心线_方向", r["direction_type"], "axis", r["direction"])
    add("中心线_两扇区", len(r["clock_sectors"]), 2, r["direction"])
    if len(r["clock_sectors"]) == 2:
        s = set(r["clock_sectors"])
        ok = any((abs(a - b) == 6) for a in s for b in s if a != b)
        add("中心线_扇区相对", ok, True, r["clock_sectors"])

    # 5 弦线（不过圆心）→ 按定义属于「一个方向的集中」= single（axis 只留给过圆心的直线）
    r = measure_matrix(_disc(red_mode="chord_line"))
    add("弦线_方向", r["direction_type"], "single", r["direction"])

    # 6 无红点 → coverage none，方向 none（是事实不是 unknown）
    r = measure_matrix(_disc(red_mode="none"))
    add("无红点_状态", r["status"], "no_red")
    add("无红点_档位", r["coverage_level"], "none")
    add("无红点_方向", r["direction_type"], "none")
    add("无红点_zones", r["red_mass_zones"], [])

    # 7 单点 → 方向证据不足 unknown；区带也因红格太少记 unknown（不硬填）
    r = measure_matrix(_disc(red_mode="single"))
    add("单点_方向", r["direction_type"], "unknown", r["direction"])
    add("单点_zones", r["red_mass_zones"], "unknown")

    # 8 共线（3 格）→ 仍不足 8 格 → unknown
    r = measure_matrix(_disc(red_mode="collinear"))
    add("共线少点_方向", r["direction_type"], "unknown", r["direction"])

    # 9 离散散点 → 方向 none
    r = measure_matrix(_disc(red_mode="scatter"))
    add("散点_方向", r["direction_type"], "none", r["direction"])

    # 10 全红（掩盖整片）→ near_all
    m = _disc(red_mode="none")
    m[m == 1] = 2
    r = measure_matrix(m)
    add("全红_档位", r["coverage_level"], "near_all")
    add("全红_方向", r["direction_type"], "none", r["direction"])

    # 11 全黑 → no_valid_area
    r = measure_matrix(np.zeros((10, 10), dtype=np.uint8))
    add("全黑_状态", r["status"], "no_valid_area")

    # 12 非方阵（拉伸）→ 椭圆归一化仍工作：把方形矩阵按列复制两倍模拟横向拉伸
    m = _disc(41, red_mode="center")
    m2 = np.repeat(m, 2, axis=1)
    r = measure_matrix(m2)
    add("拉伸_zones", r["red_mass_zones"], ["center"], r.get("outline"))
    add("拉伸_半轴成比例", (r["outline"]["semi_axes_ab"][1] /
                            r["outline"]["semi_axes_ab"][0]), 2.0, r["outline"])

    # 13 旋转不变性：绕中心旋转 90°（顺时针），single 扇区应平移 +3
    m = _disc(red_mode="eccentric")
    r0 = measure_matrix(m)
    m_rot = np.rot90(m, k=-1)          # 顺时针 90°
    r1 = measure_matrix(m_rot)
    add("旋转90_方向类型", r1["direction_type"], r0["direction_type"],
        {"before": r0["direction"], "after": r1["direction"]})
    if r0["direction_type"] == "single" and r1["direction_type"] == "single":
        want = (r0["clock_sectors"][0] - 1 + 3) % 12 + 1
        add("旋转90_扇区+3", r1["clock_sectors"][0], want,
            {"before": r0["clock_sectors"], "after": r1["clock_sectors"]})

    # 14 扇区边界并列：一个紧凑团块，其平均方向恰好压在 30°（12 点与 1 点的分界）上
    n = 121
    m = np.zeros((n, n), dtype=np.uint8)
    cc = (n - 1) / 2.0
    ii, jj = np.mgrid[0:n, 0:n]
    m[np.hypot(ii - cc, jj - cc) <= n / 2 - 0.5] = 1
    rad = math.radians(30.0)
    by, bx = cc - (n * 0.45) * math.cos(rad), cc + (n * 0.45) * math.sin(rad)
    blob = (np.hypot(ii - by, jj - bx) <= 3.5) & (m == 1)
    n_put = int(blob.sum())
    m[blob] = 2
    r = measure_matrix(m)
    add("边界并列_点已放置", n_put >= 20, True, {"n": n_put})
    add("边界并列_类型", r["direction_type"], "single", r["direction"])
    add("边界并列_被标出", r["direction"].get("sector_tie"), True,
        {k: r["direction"][k] for k in ("mean_dir_deg_12clockwise", "sector_tie")})
    # 纯函数边界口径（不依赖量化）
    add("扇区_0度是1", sector_of_deg(0.0), 1)
    add("扇区_29.9度是1", sector_of_deg(29.9), 1)
    add("扇区_30度是2", sector_of_deg(30.0), 2)
    add("扇区_359.9是12", sector_of_deg(359.9), 12)
    add("边界距_30度为0", _sector_boundary_margin_deg(30.0), 0.0)
    add("边界距_15度为15", _sector_boundary_margin_deg(15.0), 15.0)

    # 15b 线状结构检测器：环/团块不能触发，直线要触发
    add("线检测_环不触发",
        measure_matrix(_disc(red_mode="edge_ring"))["direction"]["line_structure"]["found"], False)
    add("线检测_中心块不触发",
        measure_matrix(_disc(red_mode="center"))["direction"]["line_structure"]["found"], False)
    add("线检测_偏心块不触发",
        measure_matrix(_disc(red_mode="eccentric"))["direction"]["line_structure"]["found"], False)
    add("线检测_中心线触发",
        measure_matrix(_disc(red_mode="line_through_center"))["direction"]["line_structure"]["found"], True)
    add("线检测_弦线触发",
        measure_matrix(_disc(red_mode="chord_line"))["direction"]["line_structure"]["found"], True)
    # 稀释场景：一条过中心竖线 + 一段边缘红弧 + 一些散布红点（模拟真实 Scratch 图）
    # 期望：谐波被稀释到阈值以下，但线检测应否决「无方向」→ 记 unknown（不硬给也不硬否）
    n = 101
    m = np.zeros((n, n), dtype=np.uint8)
    cc = (n - 1) / 2.0
    ii, jj = np.mgrid[0:n, 0:n]
    disc = np.hypot(ii - cc, jj - cc) <= n / 2 - 0.5
    m[disc] = 1
    m[(np.abs(jj - cc) <= 0.5) & disc] = 2                      # 过中心竖线
    ring = disc & (np.hypot(ii - cc, jj - cc) >= n / 2 - 1.2)
    m[ring] = 2                                                  # 一圈较窄的边缘红弧
    rs = np.random.RandomState(3407)
    idx = rs.choice(np.nonzero(disc.reshape(-1))[0], 90, replace=False)
    m.reshape(-1)[idx] = 2                                       # 散布
    r = measure_matrix(m)
    add("稀释竖线_不硬否", r["direction_type"] != "none" or
        not r["direction"]["line_structure"].get("found"), True, r["direction"])
    add("稀释竖线_不硬给", r["direction_type"] in ("axis", "unknown"), True,
        {"type": r["direction_type"], "sectors": r["clock_sectors"],
         "line": {k: r["direction"]["line_structure"].get(k)
                  for k in ("found", "mass_frac", "span", "fill")}})

    # 15 重复调用确定性
    a = measure_matrix(_disc(red_mode="eccentric"))
    b = measure_matrix(_disc(red_mode="eccentric"))
    add("确定性", json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), True)

    passed = sum(1 for c in cases if c["passed"])
    return {"total": len(cases), "passed": passed, "all_passed": passed == len(cases),
            "definition_version": DEFINITION_VERSION,
            "recover_module_sha256": recover_source_sha256(),
            "cases": cases}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        res = self_test()
        for c in res["cases"]:
            print(f"  {'OK ' if c['passed'] else 'FAIL'} {c['name']:22s} got={c['got']!r} want={c['want']!r}")
        print(f"通过 {res['passed']}/{res['total']}")
        return 0 if res["all_passed"] else 1
    if len(sys.argv) > 2 and sys.argv[1] == "--measure":
        out = reference_from_png(Path(sys.argv[2]))
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0
    print(__doc__)
    print("用法: 几何参考_v2.py --self-test | --measure <png>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
