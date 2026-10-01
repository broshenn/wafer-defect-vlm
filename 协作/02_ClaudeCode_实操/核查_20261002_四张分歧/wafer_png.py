#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 448×448 渲染 PNG 逐格反解原始 BIN 矩阵。

为什么可能：历史渲染函数 `wafer_vlm.utils.matrix_to_image` 是
    Image.fromarray(rgb).resize((448,448), Image.Resampling.NEAREST)
NEAREST 放大是**纯复制**——每个源单元格映射成一块同色像素，不插值、不混合。
所以目标像素 x 与源列的关系是 src = floor(x * W / 448)，逐列一一对应，**可逆**。

反解两步：
  1. 用「颜色边界位置落在周期格点上」拟合出源分辨率 (H, W)；
  2. 每个单元格取块中心那个像素的颜色，得到原矩阵。

**无损性用往返证明**：把反解出的矩阵用同一个历史函数重新渲染，
与磁盘上的原 PNG 比 SHA256。相同 ⇒ 反解无损，且渲染函数就是它。（见 roundtrip()）

只读；不联网；不调用模型。
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code" / "projects" / "wafer-defect-vlm" / "src"))

from wafer_vlm.utils import matrix_to_image  # noqa: E402  历史固定渲染函数

RES = 448
# 颜色 → 值。按 AGENTS.md §5.2：0 黑 / 1 绿 / 2 红
COLOR_TO_VALUE = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2}


def _color_key(arr: np.ndarray) -> np.ndarray:
    """把 RGB 压成一个整数，便于比较与找边界。"""
    return arr[:, :, 0].astype(np.int32) * 1000 + arr[:, :, 1].astype(np.int32)


def _edge_positions(key: np.ndarray, axis: int) -> list[int]:
    """沿 axis 方向，所有出现过颜色变化的位置。"""
    a = key if axis == 1 else key.T
    pos = set()
    for line in a:
        diff = np.nonzero(line[1:] != line[:-1])[0]
        pos.update((diff + 1).tolist())
    return sorted(pos)


def fit_dimension(edges: list[int], lo: int = 16, hi: int = 200) -> tuple[int, float]:
    """在 [lo,hi] 里找最能解释这些边界位置的周期数 N。

    渲染把 N 个源单元格铺到 448 像素上，第 k 条格线落在 round(k*448/N)。
    对每个候选 N 打分 = 边界位置落在格线 ±1 px 内的比例（位置越少越易满分，
    所以并列时取**最大**的 N —— 多的那个才可能同时解释所有边界）。
    """
    if not edges:
        return 0, 0.0
    best_n, best_score, best_hit = 0, -1.0, -1
    for n in range(lo, hi + 1):
        lattice = np.array([round(k * RES / n) for k in range(1, n)])
        e = np.asarray(edges)
        # 每个边界位置到最近格线的距离
        d = np.abs(e[:, None] - lattice[None, :]).min(axis=1)
        hit = int((d <= 1).sum())
        score = hit / len(edges)
        if score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and hit > best_hit):
            best_n, best_score, best_hit = n, score, hit
    return best_n, best_score


def recover_matrix(png_path: Path) -> dict:
    """从 PNG 反解矩阵。返回矩阵、拟合出的 (H,W)、拟合分数、像素级信息。"""
    raw = png_path.read_bytes()
    img = Image.open(io.BytesIO(raw))
    img.load()
    if img.size != (RES, RES):
        raise ValueError(f"不是 {RES}×{RES} 的渲染图：{img.size}")
    arr = np.asarray(img.convert("RGB"))
    key = _color_key(arr)

    W, score_w = fit_dimension(_edge_positions(key, axis=1))
    H, score_h = fit_dimension(_edge_positions(key, axis=0))

    # 每个单元格取块中心像素：src_col = floor(x * W / 448)
    cx = ((np.arange(W) + 0.5) * RES / W).astype(int).clip(0, RES - 1)
    ry = ((np.arange(H) + 0.5) * RES / H).astype(int).clip(0, RES - 1)
    sampled = arr[np.ix_(ry, cx)]
    matrix = np.zeros((H, W), dtype=np.uint8)
    unknown = set()
    for pix, val in COLOR_TO_VALUE.items():
        matrix[(sampled[:, :, 0] == pix[0]) & (sampled[:, :, 1] == pix[1]) & (sampled[:, :, 2] == pix[2])] = val
    for p in np.unique(sampled.reshape(-1, 3), axis=0):
        t = tuple(int(v) for v in p)
        if t not in COLOR_TO_VALUE:
            unknown.add(t)

    return {
        "matrix": matrix,
        "H": H, "W": W,
        "edge_fit_score_W": round(score_w, 4),
        "edge_fit_score_H": round(score_h, 4),
        "png_sha256": hashlib.sha256(raw).hexdigest(),
        "unexpected_colors": sorted(unknown),
    }


def roundtrip(matrix: np.ndarray, original_png: Path) -> dict:
    """用历史渲染函数重渲染，与磁盘上的原 PNG 逐**像素**比对。

    只比像素、不比字节：PNG 字节由编码器版本决定，不是内容的函数。
    实测（PIL 12.0.0）用 optimize/默认/compress_level=6/9 都得不到原字节，
    但像素差为 0 —— 说明原图是服务器上另一个 PIL/zlib 写的。
    这里要证的是「反解无损」，那是像素级的事实，所以像素相等才算通过。
    """
    buf = io.BytesIO()
    matrix_to_image(matrix, RES).save(buf, format="PNG", optimize=True)
    mine_bytes = buf.getvalue()
    buf.seek(0)
    mine = np.asarray(Image.open(buf).convert("RGB"))
    orig_bytes = original_png.read_bytes()
    orig = np.asarray(Image.open(io.BytesIO(orig_bytes)).convert("RGB"))
    differing = int((mine != orig).any(axis=2).sum())
    return {
        "roundtrip_pixels_differing": differing,
        "roundtrip_pixels_identical": differing == 0,
        "roundtrip_bytes_identical": hashlib.sha256(mine_bytes).hexdigest()
        == hashlib.sha256(orig_bytes).hexdigest(),
        "original_bytes": len(orig_bytes),
        "rerender_bytes": len(mine_bytes),
    }


def main() -> int:
    ids = sys.argv[1:]
    if not ids:
        print("用法: reconstruct.py <sample_id> [...]")
        return 2
    print(f"{'sample_id':22s} {'反解 HxW':11s} {'格点分 H/W':16s} {'往返像素差':11s} 异常色")
    for sid in ids:
        p = REPO / "data" / "images" / f"{sid}.png"
        r = recover_matrix(p)
        rt = roundtrip(r["matrix"], p)
        hw = f"{r['H']}x{r['W']}"
        mark = "✓ 0" if rt["roundtrip_pixels_identical"] else f"✗ {rt['roundtrip_pixels_differing']}"
        print(f"{sid:22s} {hw:11s} "
              f"{r['edge_fit_score_H']:.3f}/{r['edge_fit_score_W']:.3f}        "
              f"{mark:11s} {r['unexpected_colors'] or '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
