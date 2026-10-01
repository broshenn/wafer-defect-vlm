#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`wafer_png_v2.py` 的自测与回归。

三组，全部可重跑、无需网络、无需 GPU：

  A 前向映射：实测 Pillow NEAREST 的源格映射是否等于 `source_index_for`
  B 合成往返：随机矩阵 → 渲染 → 反解 → 逐格比对（含边长 ≥150 的大尺寸）
  C 真实九张：反解形状 vs manifest `matrix_shape`，以及往返像素差

用法：  python test_wafer_png.py          （全部）
        python test_wafer_png.py A B C    （选做）
退出码 0 = 全过。
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wafer_png_v2 import (REPO, RES, assert_forward_mapping, recover_matrix,  # noqa: E402
                          roundtrip)

SYNTH_SHAPES_IN_RANGE = [(22, 22), (25, 27), (29, 26), (33, 29), (41, 42), (56, 41),
                         (64, 63), (72, 72), (100, 97), (128, 127), (150, 150),
                         (180, 179), (200, 199)]
# 超出已验证范围：**期望失败**，用来固定"本方法不是任意分辨率都可用"这条限制。
SYNTH_SHAPES_OUT_OF_RANGE = [(255, 255), (300, 299)]
REAL_IDS = ["wafer_00044523_017", "wafer_00025336_019", "wafer_00018489_020",
            "wafer_00017102_001", "wafer_00043922_015", "wafer_00015139_006",
            "wafer_00047178_011", "wafer_00013704_014", "wafer_00011911_008"]


def group_a() -> bool:
    print("── A 前向映射 ──")
    bad = assert_forward_mapping(verbose=True)
    print(f"   {'全部通过' if not bad else '✗ 不符宽度 ' + str(bad)}")
    return not bad


def _run_shapes(shapes, expect_lossless: bool) -> bool:
    from wafer_vlm.utils import matrix_to_image
    rng = np.random.default_rng(3407)
    tmp = HERE / "_synth_tmp.png"
    ok = True
    print(f"   {'真值 HxW':11s} {'反解':11s} {'往返验证':8s} {'逐格一致':8s} {'满分候选(前3)':24s}")
    for H, W in shapes:
        m = rng.integers(0, 3, size=(H, W)).astype(np.uint8)
        buf = io.BytesIO()
        matrix_to_image(m, RES).save(buf, format="PNG", optimize=True)
        tmp.write_bytes(buf.getvalue())
        r = recover_matrix(tmp)
        same = (r["H"] == H and r["W"] == W and r["matrix"].shape == m.shape
                and (r["matrix"] == m).all())
        if same != expect_lossless:
            ok = False
        cands = f"{r['full_score_candidates_H'][:3]}/{r['full_score_candidates_W'][:3]}"
        print(f"   {str((H, W)):11s} {str((r['H'], r['W'])):11s} "
              f"{str(r['roundtrip_validated']):8s} {str(same):8s} {cands:24s}")
    tmp.unlink(missing_ok=True)
    return ok


def group_b() -> bool:
    print("── B1 合成往返（范围内，期望无损）──")
    ok1 = _run_shapes(SYNTH_SHAPES_IN_RANGE, expect_lossless=True)
    print(f"   {'全部通过' if ok1 else '✗ 有失败'}")
    print("── B2 合成往返（超出已验证范围，**期望失败**，固定限制）──")
    ok2 = _run_shapes(SYNTH_SHAPES_OUT_OF_RANGE, expect_lossless=False)
    print(f"   {'符合预期（都失败）' if ok2 else '✗ 与预期不符'}")
    return ok1 and ok2


def group_c() -> bool:
    print("── C 真实九张 ──")
    man = {}
    with open(REPO / "data" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                man[r["sample_id"]] = r
    ok = True
    for sid in REAL_IDS:
        p = REPO / "data" / "images" / f"{sid}.png"
        r = recover_matrix(p)
        rt = roundtrip(r["matrix"], p)
        want = list(man[sid]["matrix_shape"])
        shape_ok = [r["H"], r["W"]] == want
        px_ok = rt["roundtrip_pixels_identical"]
        ok &= shape_ok and px_ok
        print(f"   {sid:22s} 反解 {r['H']:3d}x{r['W']:3d}  manifest {want}  "
              f"形状 {'✓' if shape_ok else '✗'}  往返像素差 {rt['roundtrip_pixels_differing']}")
    print(f"   {'全部通过' if ok else '✗ 有失败'}")
    return ok


def main() -> int:
    which = set(a.upper() for a in sys.argv[1:]) or {"A", "B", "C"}
    results = {}
    if "A" in which:
        results["A"] = group_a()
    if "B" in which:
        results["B"] = group_b()
    if "C" in which:
        results["C"] = group_c()
    print()
    for k, v in results.items():
        print(f"   {k}: {'通过' if v else '未通过'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
