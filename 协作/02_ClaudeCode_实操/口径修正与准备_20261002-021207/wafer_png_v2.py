#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 448×448 渲染 PNG 逐格反解 BIN 矩阵 —— v2（修正说明与坐标映射）。

**这是 `核查_20261002_四张分歧/wafer_png.py` 的另存版本，原文件不覆盖。**

相对 v1 改了什么（逐条对应 Codex 验收裁决 §1 与任务书 §2.6）：

1. **坐标映射写对了。** v1 文档里写 `src = floor(x*W/448)`；**实测 Pillow 不是这个**。
   本机 PIL 12.0.0 实测 8 个宽度（29/31/33/41/42/63/72/150）全部符合

       src = floor((x + 0.5) * W / 448)        （像素中心映射）

   v1 的**反解代码**用的是 `(i+0.5)*448/W` 取格中心，恰好落在正确的目标区间内，
   所以 v1 的**结果是对的**，错的只是文档。本版把两边都写清并加了实测断言。

2. **并列取值写对了。** v1 文档说"并列时取最大的 N"，代码实际取**首个（最小）**。
   本版说明：满分候选**不唯一**，真值的整数倍都会满分（因为 N 的格点集 ⊂ 2N 的格点集），
   所以取**最小**满分候选才是真值；代码行为原本就对，文档是错的。这一点有实测证据
   （见 `candidates_at_full_score`）。

3. **限制写全了**：候选分辨率多解、稀少颜色边界导致的退化、只适用于放大（n_src ≤ 448）。

4. **格点容差从 ±1 改成 0（本轮自查发现的、比 v1 更严重的错）。**
   v1 用"边界落在格点 ±1 px 内"打分。实测发现跃变位置**恰好等于** `round(k·448/N)`，一点不差，
   所以容差本可以是 0；加了 ±1 之后，颜色边界密集时**偏小的候选也会满分** ——
   合成随机矩阵边长 ≥150 时 v1 全部反解错（150→148、200→144、300→180）。
   改零容差后合成 9/9、真实 9/9 全对，且候选集恰好是**真值的整数倍**。
   另外本版按候选**升序取第一个能像素级往返还原的**，让"无损"自己当判据。

**只读；不联网；不调用模型。**
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
COLOR_TO_VALUE = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2}

# 候选分辨率的搜索范围。**实现选择，不是数据事实。**
#   FIT_HI=200 是**已验证范围**：合成测试（`test_wafer_png.py` B 组）证明边长 ≤200 时可无损反解；
#   边长 ≥255 时本方法会失败 —— 原因是我的区间公式与 Pillow 的实际块边界在浮点层面错开
#   （与实测 W=256 的边界效应同源），格心会取到相邻格。所以**不声称对任意分辨率都可用**。
#   真实数据实测边长在 25..72，远在范围内。
#   取 200 还顺带避开了"平凡解"：n=448 的格点恰好是所有整数，任何图的边界位置都是整数，
#   于是 n=448 恒满分。这类"密到无法证伪"的候选不应进入候选集。
FIT_LO, FIT_HI = 2, 200
FIT_VALIDATED_MAX = 200

# 格点判据的容差。**必须为 0**：实测（`--self-test`）NEAREST 放大后的跃变位置
# 恰好等于 `round(k*448/N)`，一点不差。v1 用了 ±1，看似稳健，实际会在颜色边界密集时
# 让**偏小的候选也满分** —— 合成随机矩阵边长 ≥150 时全部反解错，取零容差后正确。
FIT_TOL = 0


def source_index_for(x: int, n_src: int, dst: int = RES) -> int:
    """Pillow NEAREST 放大时，目标像素 x 取自哪个源格。

    实测（PIL 12.0.0，8 个宽度全通过）：
        src = floor((x + 0.5) * n_src / dst)   ，再截断到 [0, n_src-1]

    注意分母是**目标**尺寸、分子是**源**格数 —— 这就是"放大"的方向。
    仅当 n_src <= dst 时每个源格至少覆盖一个目标像素，反解才无损；
    n_src > dst 是缩小，会丢格，本模块不适用（会显式报错）。
    """
    return int(min(max(np.floor((x + 0.5) * n_src / dst), 0), n_src - 1))


def source_index_for_vector(arange: np.ndarray, n_src: int, dst: int = RES) -> np.ndarray:
    return np.floor((arange + 0.5) * n_src / dst).astype(int).clip(0, n_src - 1)


def _color_key(arr: np.ndarray) -> np.ndarray:
    return arr[:, :, 0].astype(np.int32) * 1000 + arr[:, :, 1].astype(np.int32)


def _edge_positions(key: np.ndarray, axis: int) -> list[int]:
    a = key if axis == 1 else key.T
    pos = set()
    for line in a:
        diff = np.nonzero(line[1:] != line[:-1])[0]
        pos.update((diff + 1).tolist())
    return sorted(pos)


def _full_score_candidates(edges: list[int], lo: int = FIT_LO, hi: int = FIT_HI,
                           tol: int = FIT_TOL) -> list[int]:
    """所有"能解释全部颜色边界"的候选格数，升序。

    **多解是常态，不是异常**：若 N 能解释全部边界，则 kN（k=2,3,…）也能 ——
    因为 N 的格点集是 kN 格点集的子集，每条边界仍落在格点上。
    实测（零容差）：真值 33 的候选集恰好是 [33, 66, 99]，真值 29 的是 [29, 58, 87]，
    就是整数倍本身。所以真值是**最小**的满分候选。
    """
    if not edges:
        return []
    e = np.asarray(edges)
    out = []
    for n in range(lo, hi + 1):
        lat = np.array([round(k * RES / n) for k in range(1, n)])
        d = np.abs(e[:, None] - lat[None, :]).min(axis=1)
        if (d <= tol).all():
            out.append(n)
    return out


def _recover_at(arr: np.ndarray, H: int, W: int) -> tuple[np.ndarray, set]:
    """按给定 (H,W) 逐格取样。返回 (矩阵, 未知颜色集)。"""
    def _cell_center(n_src: int) -> np.ndarray:
        i = np.arange(n_src)
        lo = i * RES / n_src - 0.5
        hi = (i + 1) * RES / n_src - 0.5
        return np.floor((lo + hi) / 2).astype(int).clip(0, RES - 1)

    sampled = arr[np.ix_(_cell_center(H), _cell_center(W))]     # (H, W, 3)
    matrix = np.zeros((H, W), dtype=np.uint8)
    for pix, val in COLOR_TO_VALUE.items():
        matrix[(sampled[:, :, 0] == pix[0]) & (sampled[:, :, 1] == pix[1])
               & (sampled[:, :, 2] == pix[2])] = val
    unknown = set()
    for p in np.unique(sampled.reshape(-1, 3), axis=0):
        t = tuple(int(v) for v in p)
        if t not in COLOR_TO_VALUE:
            unknown.add(t)
    return matrix, unknown


def fit_dimension(edges: list[int], lo: int = FIT_LO, hi: int = FIT_HI) -> tuple[int, float, list[int]]:
    """返回 (选中格数, 满分率, 全部满分候选)。选中规则 = **最小的满分候选**。"""
    cands = _full_score_candidates(edges, lo, hi)
    if not cands:
        e = np.asarray(edges)
        best, best_score = 0, -1.0
        for n in range(lo, hi + 1):
            lat = np.array([round(k * RES / n) for k in range(1, n)])
            d = np.abs(e[:, None] - lat[None, :]).min(axis=1)
            s = float((d <= FIT_TOL).mean())
            if s > best_score + 1e-9:
                best, best_score = n, s
        return best, best_score, []
    return cands[0], 1.0, cands


def recover_matrix(png_path: Path, validate_roundtrip: bool = True) -> dict:
    """从 PNG 反解矩阵。

    `validate_roundtrip=True`（默认）时，按候选**升序**取第一个能像素级往返还原的候选 ——
    也就是让"反解无损"这件事自己当判据，而不是只信格点评分。
    """
    raw = png_path.read_bytes()
    img = Image.open(io.BytesIO(raw))
    img.load()
    if img.size != (RES, RES):
        raise ValueError(f"不是 {RES}×{RES} 的渲染图：{img.size}")
    arr = np.asarray(img.convert("RGB"))
    key = _color_key(arr)

    H, score_h, cands_h = fit_dimension(_edge_positions(key, axis=0))
    W, score_w, cands_w = fit_dimension(_edge_positions(key, axis=1))
    if H > RES or W > RES:
        raise ValueError(f"反解出 {H}×{W} 大于 {RES}，这是缩小渲染，本模块不适用")

    matrix, unknown = _recover_at(arr, H, W)

    # 往返验证：按候选升序取第一个能像素级还原的。
    # **没有候选能还原时不做静默回退** —— 明确标记失败，由调用方决定。
    # （v2 初版会让循环跑完、然后悄悄用 448 那个平凡解，本轮自查时发现。）
    validated: bool | None = None
    switched_from: list[int] = []
    if validate_roundtrip:
        validated = False
        hs = cands_h or [H]
        ws = cands_w or [W]
        done = False
        for h in hs:
            if done:
                break
            for w in ws:
                cand, _ = _recover_at(arr, h, w)
                buf = io.BytesIO()
                matrix_to_image(cand, RES).save(buf, format="PNG", optimize=True)
                buf.seek(0)
                if (np.asarray(Image.open(buf).convert("RGB")) == arr).all():
                    if (h, w) != (H, W):
                        switched_from = [H, W]
                    H, W, matrix = h, w, cand
                    validated, done = True, True
                    break

    return {
        "matrix": matrix,
        "H": H, "W": W,
        "edge_fit_score_H": round(score_h, 4),
        "edge_fit_score_W": round(score_w, 4),
        "full_score_candidates_H": cands_h,
        "full_score_candidates_W": cands_w,
        "resolution_unique_H": len(cands_h) == 1,
        "resolution_unique_W": len(cands_w) == 1,
        "roundtrip_validated": validated,
        "roundtrip_switched_away_from": switched_from,
        "fit_tolerance_px": FIT_TOL,
        "fit_search_range": [FIT_LO, FIT_HI],
        "png_sha256": hashlib.sha256(raw).hexdigest(),
        "unexpected_colors": sorted(unknown),
    }


def roundtrip(matrix: np.ndarray, original_png: Path) -> dict:
    """用历史同一个渲染函数重渲染，与原 PNG 逐**像素**比对。

    只比像素不比字节：PNG 字节由编码器版本决定。实测 PIL 12.0.0 用
    optimize=True / 默认 / compress_level=6/9 都得不到原字节（IDAT 长度不同），
    但像素差为 0 —— 原图是服务器上另一个 PIL/zlib 写的。
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


def assert_forward_mapping(verbose: bool = False) -> list[int]:
    """自检：实测 Pillow NEAREST 的源格映射是否符合 `source_index_for`。

    返回不合规的宽度列表；空列表 = 全部通过。

    源格号用**两个通道**编码（R=低字节, G=高字节）。这一点是踩过坑的：
    第一版把源格号整个塞进 uint8，W>256 时 `np.arange(W, dtype=uint8)` 回绕，
    于是"测试失败"其实是**测试自己溢出**，不是 Pillow 不符。
    """
    bad = []
    for W in (29, 31, 33, 41, 42, 63, 72, 150, 200, 255, 300, 400, 447, 448):
        idx = np.arange(W)
        rgb = np.zeros((1, W, 3), dtype=np.uint8)
        rgb[0, :, 0] = (idx % 256).astype(np.uint8)
        rgb[0, :, 1] = (idx // 256).astype(np.uint8)
        out = np.asarray(Image.fromarray(rgb, "RGB").resize((RES, RES),
                                                            Image.Resampling.NEAREST))
        got = out[0, :, 1].astype(int) * 256 + out[0, :, 0].astype(int)
        want = source_index_for_vector(np.arange(RES), W, RES)
        if not (got == want).all():
            bad.append(W)
        elif verbose:
            print(f"  W={W:3d} ✓")
    return bad


def main() -> int:
    ids = sys.argv[1:]
    if not ids:
        print("用法: wafer_png_v2.py <sample_id> [...]")
        print("      wafer_png_v2.py --self-test")
        return 2
    if ids == ["--self-test"]:
        print("实测 Pillow NEAREST 源格映射 vs source_index_for：")
        bad = assert_forward_mapping(verbose=True)
        print(f"\n{'全部通过' if not bad else '不符的宽度: ' + str(bad)}")
        return 0 if not bad else 1

    print(f"{'sample_id':22s} {'反解 HxW':11s} {'满分候选 H':26s} {'满分候选 W':26s} {'往返像素差':9s}")
    for sid in ids:
        p = REPO / "data" / "images" / f"{sid}.png"
        r = recover_matrix(p)
        rt = roundtrip(r["matrix"], p)
        hw = f"{r['H']}x{r['W']}"
        ch = str(r["full_score_candidates_H"][:4])
        cw = str(r["full_score_candidates_W"][:4])
        print(f"{sid:22s} {hw:11s} {ch:26s} {cw:26s} {rt['roundtrip_pixels_differing']:9d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
