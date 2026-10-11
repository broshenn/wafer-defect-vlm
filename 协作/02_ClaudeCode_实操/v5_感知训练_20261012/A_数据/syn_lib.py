#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成图工具库（合同 5.2）。被 step4_synthetic.py 与 test_synthetic.py 共用。

坐标：用几何 v2 的 _ellipse_of(valid) 与 _normalized_uv(H,W,cy,cx,a,b)；
r = sqrt(u^2+v^2)；钟点角 deg = degrees(atan2(v, -u)) mod 360（12 点 = 0°，顺时针）。
所有图案只在 valid（绿∪红）格内改值：1 -> 2。
"""
import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
_CORE_V3 = REPO / "协作/01_Codex_指挥/无卡CPU整备_20261010/core_v3.py"


_CORE = None


def load_core():
    global _CORE
    if _CORE is None:
        spec = importlib.util.spec_from_file_location("_core_v3", _CORE_V3)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CORE = mod
    return _CORE


def _val(params, key, rng, lo, hi):
    """参数取值：测试可传 params 固定；常规路径从 rng 均匀抽取。"""
    if params is not None and key in params:
        return float(params[key])
    return rng.uniform(lo, hi)


def uv_of(matrix):
    """返回 (u, v, r, deg, valid)：矩阵坐标 → 归一化极坐标。"""
    geom = load_core().load_geometry()
    m = np.asarray(matrix)
    valid = (m == 1) | (m == 2)
    cy, cx, a, b = geom._ellipse_of(valid)
    H, W = m.shape
    u, v = geom._normalized_uv(H, W, cy, cx, a, b)
    r = np.sqrt(u * u + v * v)
    deg = np.mod(np.degrees(np.arctan2(v, -u)), 360.0)
    return u, v, r, deg, valid


def _uv_from_polar(rr, deg_deg):
    """(r, 钟点角) → (u, v)。deg = atan2(v, -u) ⇒ u = -r cos, v = r sin。"""
    rad = np.radians(deg_deg)
    return -rr * np.cos(rad), rr * np.sin(rad)


def pattern_local_blob(u, v, r, deg, valid, rng, params=None):
    """r0∈[0.2,0.85]，θ∈[0,360)，半径 s∈[0.08,0.20]，填充 p∈[0.6,1.0]。"""
    r0 = _val(params, 'r0', rng, 0.2, 0.85)
    th = _val(params, 'theta_deg', rng, 0, 360)
    s = _val(params, 's', rng, 0.08, 0.20)
    p = _val(params, 'p', rng, 0.6, 1.0)
    u0, v0 = _uv_from_polar(r0, th)
    dist = np.hypot(u - u0, v - v0)
    cand = valid & (dist < s)
    hit = cand & (rng.random(u.shape) < p)
    return hit, {"r0": round(r0, 4), "theta_deg": round(th, 2), "s": round(s, 4), "p": round(p, 4)}


def pattern_edge_arc(u, v, r, deg, valid, rng, params=None):
    """θ 中心，跨度 30–120°，r∈[0.85,1.0]，p∈[0.6,1.0]。"""
    th = _val(params, 'theta_deg', rng, 0, 360)
    span = _val(params, 'span_deg', rng, 30, 120)
    p = _val(params, 'p', rng, 0.6, 1.0)
    half = span / 2
    d = np.abs(((deg - th + 180) % 360) - 180)      # 环形角距
    cand = valid & (r >= 0.85) & (r <= 1.0) & (d <= half)
    hit = cand & (rng.random(u.shape) < p)
    return hit, {"theta_deg": round(th, 2), "span_deg": round(span, 2), "p": round(p, 4)}


def pattern_ring(u, v, r, deg, valid, rng, params=None):
    """r_in∈[0.3,0.85]，宽 w∈[0.05,0.15]，p∈[0.6,1.0]。"""
    r_in = _val(params, 'r_in', rng, 0.3, 0.85)
    w = _val(params, 'w', rng, 0.05, 0.15)
    p = _val(params, 'p', rng, 0.6, 1.0)
    cand = valid & (r >= r_in) & (r <= r_in + w)
    hit = cand & (rng.random(u.shape) < p)
    return hit, {"r_in": round(r_in, 4), "w": round(w, 4), "p": round(p, 4)}


def pattern_line(u, v, r, deg, valid, rng, params=None):
    """过点 (r0∈[0,0.5], θ)，方向 φ∈[0,180)，半宽 0.03，长度 L∈[0.6,1.6]，p∈[0.7,1.0]。"""
    r0 = _val(params, 'r0', rng, 0, 0.5)
    th = _val(params, 'theta_deg', rng, 0, 360)
    phi = _val(params, 'phi_deg', rng, 0, 180)
    L = _val(params, 'L', rng, 0.6, 1.6)
    p = _val(params, 'p', rng, 0.7, 1.0)
    u0, v0 = _uv_from_polar(r0, th)
    dvec = np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
    wx = u - u0
    wy = v - v0
    t = wx * dvec[0] + wy * dvec[1]                 # 沿线投影（自过点起算）
    s = -wx * dvec[1] + wy * dvec[0]                # 法向距离
    cand = valid & (np.abs(s) < 0.03) & (np.abs(t) <= L / 2)
    hit = cand & (rng.random(u.shape) < p)
    return hit, {"r0": round(r0, 4), "theta_deg": round(th, 2), "phi_deg": round(phi, 2),
                 "L": round(L, 4), "p": round(p, 4)}


def pattern_center_blob(u, v, r, deg, valid, rng, params=None):
    """半径 s∈[0.15,0.35]，p∈[0.6,1.0]。"""
    s = _val(params, 's', rng, 0.15, 0.35)
    p = _val(params, 'p', rng, 0.6, 1.0)
    cand = valid & (r < s)
    hit = cand & (rng.random(u.shape) < p)
    return hit, {"s": round(s, 4), "p": round(p, 4)}


def pattern_random_scatter(u, v, r, deg, valid, rng, params=None):
    """每个有效格以 p∈[0.05,0.30] 变红。"""
    p = _val(params, 'p', rng, 0.05, 0.30)
    hit = valid & (rng.random(u.shape) < p)
    return hit, {"p": round(p, 4)}


PATTERNS = {
    "local_blob": pattern_local_blob,
    "edge_arc": pattern_edge_arc,
    "ring": pattern_ring,
    "line": pattern_line,
    "center_blob": pattern_center_blob,
    "random_scatter": pattern_random_scatter,
    "blob_plus_line": None,   # 特殊：local_blob + line 各一次
}
PATTERN_NAMES = list(PATTERNS.keys())


def draw_pattern(matrix, name, rng, params=None):
    """在底图矩阵上画一种图案；返回 (新矩阵, 参数dict)。只改 valid 内的 1→2。
    params 仅供测试固定参数；常规路径传 None，全部从 rng 抽。"""
    u, v, r, deg, valid = uv_of(matrix)
    m = matrix.copy()
    if name == "blob_plus_line":
        hit1, pr1 = pattern_local_blob(u, v, r, deg, valid, rng, (params or {}).get("local_blob"))
        hit2, pr2 = pattern_line(u, v, r, deg, valid, rng, (params or {}).get("line"))
        hit = hit1 | hit2
        pr = {"local_blob": pr1, "line": pr2}
    else:
        hit, pr = PATTERNS[name](u, v, r, deg, valid, rng, params)
    m[hit & valid] = 2
    return m, pr
