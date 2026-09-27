# -*- coding: utf-8 -*-
"""把「那个 bug」画出来：为什么边缘缺陷会被判成 center

运行：
    python 质心实验.py

不重新算任何东西 —— 所有数字都直接读 data/manifest.jsonl 里项目自己算好的值，
这里只负责把它们画成图，让你亲眼看到。
"""

import collections
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST = ROOT / "data" / "manifest.jsonl"
IMAGES = ROOT / "data" / "images"
OUTPUT = HERE / "质心实验.png"

PANEL = 300          # 每个小图的边长
PAD = 14
LABEL_H = 74         # 每个小图下面留的说明高度
S = 118              # 半径在画布上占多少像素


def load_font(size):
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = Path("C:/Windows/Fonts") / name
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


def to_canvas(nx, ny):
    """归一化坐标（单位：半径，y 向上）-> 画布像素（y 向下）。"""
    return PANEL / 2 + nx * S, PANEL / 2 - ny * S


def draw_panel(row, font_big, font_small):
    """画一片晶圆：缺陷 die 用红点，质心用大叉。"""
    img = Image.new("RGB", (PANEL, PANEL + LABEL_H), (22, 24, 28))
    d = ImageDraw.Draw(img)

    shape = tuple(row["matrix_shape"])
    f = row["features"]
    cx_m, cy_m = f["wafer_center_xy"]
    radius = f["wafer_radius"]

    # 从 448 的图还原出矩阵
    p = IMAGES / Path(row["image_path"]).name
    matrix = np.array(Image.open(p).convert("RGB").resize(shape, Image.Resampling.NEAREST))

    # 晶圆外圈（用项目算好的圆心半径画）
    d.ellipse(
        [to_canvas(-1, 1)[0], to_canvas(0, 1)[1], to_canvas(1, -1)[0], to_canvas(0, -1)[1]],
        outline=(70, 78, 90), width=2,
    )

    # 每个缺陷 die 画一个红点
    red = (matrix[:, :, 0] > 150) & (matrix[:, :, 1] < 100)
    ys, xs = np.nonzero(red)
    norm = []
    for r_, c_ in zip(ys, xs):
        nx = (c_ - cx_m) / radius
        ny = (cy_m - r_) / radius
        norm.append((nx, ny))
        px, py = to_canvas(nx, ny)
        dot = 5 if len(ys) < 200 else 3
        d.ellipse([px - dot, py - dot, px + dot, py + dot], fill=(235, 70, 70))

    # 质心：用项目算好的 centroid_xy_r
    gx, gy = f["centroid_xy_r"]
    px, py = to_canvas(gx, gy)
    r_ = 11
    d.line([px - r_, py - r_, px + r_, py + r_], fill=(90, 220, 255), width=4)
    d.line([px - r_, py + r_, px + r_, py - r_], fill=(90, 220, 255), width=4)

    # 文字
    y0 = PANEL + 6
    d.text((8, y0), f"真值：{row['failure_type']}", font=font_big, fill=(240, 240, 240))
    d.text((8, y0 + 24), f"质心离圆心：{f['centroid_radius_r']:.3f} 个半径",
           font=font_small, fill=(150, 200, 235))
    verdict = f["radial_zone"]
    color = (255, 110, 110) if verdict != "center" else (255, 200, 110)
    d.text((8, y0 + 44), f"→ 判定 radial_zone = «{verdict}»", font=font_small, fill=color)

    return img


def pick_typical(rows, label):
    """挑这一类里【最典型】的一片：质心半径最接近该类中位数。

    为什么不随便取第一个？因为单张样本可能很特殊，看一张就下结论是错的。
    """
    cand = [r for r in rows
            if r["failure_type"] == label
            and r["features"]["status"] == "ok"
            and r["features"]["centroid_radius_r"] is not None
            and (IMAGES / Path(r["image_path"]).name).is_file()]
    if not cand:
        return None
    med = statistics.median(r["features"]["centroid_radius_r"] for r in cand)
    return min(cand, key=lambda r: abs(r["features"]["centroid_radius_r"] - med))


def main():
    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8")]
    picks = ["Edge_Ring", "Random", "Donut", "Center"]

    chosen = [r for r in (pick_typical(rows, lab) for lab in picks) if r]

    print("=" * 76)
    print("先看全库统计：每一类的缺陷到底在哪，以及被判定成什么")
    print("=" * 76)
    print(f"{'真值类别':<11}{'n':>5}{'质心半径中位':>13}{'平均半径中位':>13}"
          f"{'判center':>10}{'判middle':>10}{'判edge':>9}")
    print("-" * 76)
    for label in ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
                  "Near_full", "Random", "Scratch", "none"]:
        fs = [r["features"] for r in rows if r["failure_type"] == label]
        n = len(fs)
        cm = statistics.median([f["centroid_radius_r"] for f in fs
                                if f["centroid_radius_r"] is not None] or [0])
        mm = statistics.median([f["mean_radius_r"] for f in fs
                                if f["mean_radius_r"] is not None] or [0])
        z = collections.Counter(f["radial_zone"] for f in fs)
        print(f"{label:<11}{n:>5}{cm:>13.3f}{mm:>13.3f}"
              f"{z['center'] / n:>10.0%}{z['middle'] / n:>10.0%}{z['edge'] / n:>9.0%}")

    print()
    print("=" * 76)
    print("再看四张典型样本：红点 = 缺陷 die，蓝叉 = 它们的平均位置，灰圈 = 晶圆边界")
    print("=" * 76)
    print(f"{'真值类别':<12}{'质心半径':>10}{'判定 zone':>12}{'平均半径':>10}"
          f"{'缺陷数':>8}")
    print("-" * 54)
    for r in chosen:
        f = r["features"]
        print(f"{r['failure_type']:<12}{f['centroid_radius_r']:>10.3f}"
              f"{f['radial_zone']:>12}{f['mean_radius_r']:>10.3f}"
              f"{f['defect_count']:>8}")

    font_big = load_font(17)
    font_small = load_font(14)
    cols, rows_n = 2, 2
    W = cols * PANEL + (cols + 1) * PAD
    H = rows_n * (PANEL + LABEL_H) + (rows_n + 1) * PAD
    canvas = Image.new("RGB", (W, H), (16, 18, 21))
    for i, r in enumerate(chosen):
        panel = draw_panel(r, font_big, font_small)
        cx = PAD + (i % cols) * (PANEL + PAD)
        cy = PAD + (i // cols) * (PANEL + LABEL_H + PAD)
        canvas.paste(panel, (cx, cy))

    canvas.save(OUTPUT)
    print()
    print("图已写出：", OUTPUT)
    print("红点 = 缺陷 die；蓝叉 = 这些点的【平均位置】（质心）；灰圈 = 晶圆边界")
    print("看清楚：边缘绕一圈的缺陷，它的平均位置落在【圆心】上。")


if __name__ == "__main__":
    main()
