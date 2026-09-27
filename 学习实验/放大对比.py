# -*- coding: utf-8 -*-
"""放大对比实验：26×26 的矩阵 -> 448×448，两种放大法有什么不同

运行：
    python 放大对比.py

会用项目里真实的那张 wafer_00000096_002.png 做实验，
输出一张对比图（nearest_vs_bilinear.png）和几行统计。
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "data" / "images" / "wafer_00000096_002.png"
OUTPUT = HERE / "nearest_vs_bilinear.png"

TARGET = 448          # 项目用的目标尺寸
GRID = 26             # 这张晶圆的矩阵大小


def count_colors(arr):
    """一张图里到底有多少种颜色。"""
    return len(np.unique(arr.reshape(-1, 3), axis=0))


def find_boundary(arr):
    """随便找一个「红绿相邻」的位置，用来做局部放大。"""
    red = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 60)
    green = (arr[:, :, 1] > 200) & (arr[:, :, 0] < 60)
    both = red[:, :-1] & green[:, 1:]        # 左边红、右边绿
    both |= green[:, :-1] & red[:, 1:]
    ys, xs = np.nonzero(both)
    if len(ys) == 0:
        h, w = arr.shape[:2]
        return w // 2, h // 2
    k = len(ys) // 2                          # 取中间那一个，别取边角
    return int(xs[k]), int(ys[k])


def zoom(arr, cx, cy, box=30, scale=5):
    """以 (cx, cy) 为中心抠一小块，再放大 scale 倍（最近邻，看清像素）。"""
    h, w = arr.shape[:2]
    x0, y0 = max(cx - box, 0), max(cy - box, 0)
    x1, y1 = min(cx + box, w), min(cy + box, h)
    crop = Image.fromarray(arr[y0:y1, x0:x1])
    return crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)


def label(img, text, size=22):
    """在图上写个英文标签（默认字体只支持英文）。"""
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, len(text) * size // 2 + 16, size + 10], fill=(255, 255, 255))
    draw.text((8, 4), text, fill=(0, 0, 0))
    return img


def main():
    print("=" * 60)
    print("实验：从 26x26 的矩阵放大到 448x448")
    print("=" * 60)

    if not SOURCE.is_file():
        print(f"找不到 {SOURCE}，跳过")
        return

    # 这张 448 的图本来就是放大的产物，先缩回 26x26 拿回矩阵
    original = Image.open(SOURCE).convert("RGB")
    matrix = np.array(original.resize((GRID, GRID), Image.Resampling.NEAREST))
    print(f"\n从图里还原出矩阵：{matrix.shape[0]}x{matrix.shape[1]}")
    print(f"矩阵里的颜色数：{count_colors(matrix)}")

    print(f"\n两种放大法，都放大到 {TARGET}x{TARGET}：")
    print(f"  放大倍数 = {TARGET} / {GRID} = {TARGET / GRID:.2f} 倍")
    print(f"  也就是【一个 die 的 1 个像素】->【大约 {TARGET // GRID} x {TARGET // GRID} 个像素】")

    # 两种放大
    src = Image.fromarray(matrix)
    near = np.array(src.resize((TARGET, TARGET), Image.Resampling.NEAREST))
    bilin = np.array(src.resize((TARGET, TARGET), Image.Resampling.BILINEAR))

    n_near = count_colors(near)
    n_bilin = count_colors(bilin)
    print(f"\n{'放大法':<14}{'放大后一共几种颜色':>18}")
    print(f"{'NEAREST':<14}{n_near:>18}   <- 和原矩阵一样，没造出新颜色")
    print(f"{'BILINEAR':<14}{n_bilin:>18}   <- 凭空多出来这么多颜色")

    # 拼一张对比图
    cx, cy = find_boundary(near)
    print(f"\n局部放大位置：({cx}, {cy})   —— 一个红绿交界处")

    panel = TARGET // 2
    top_left = label(Image.fromarray(near).resize((panel, panel), Image.Resampling.NEAREST), "NEAREST")
    top_right = label(Image.fromarray(bilin).resize((panel, panel), Image.Resampling.NEAREST), "BILINEAR")
    bot_left = label(zoom(near, cx, cy, box=panel // 10, scale=6), "NEAREST zoom")
    bot_right = label(zoom(bilin, cx, cy, box=panel // 10, scale=6), "BILINEAR zoom")

    w = max(bot_left.width, bot_right.width)
    h = max(bot_left.height, bot_right.height)
    canvas = Image.new("RGB", (panel * 2 + 30, panel * 2 + h + 30), (235, 235, 235))
    canvas.paste(top_left, (0, 0))
    canvas.paste(top_right, (panel + 30, 0))
    canvas.paste(bot_left, (0, panel + 30))
    canvas.paste(bot_right, (bot_left.width + 30, panel + 30))
    canvas.save(OUTPUT)
    print(f"\n对比图已写出：{OUTPUT}")
    print("上半：整张图（看不出差别）；下半：局部放大（差别一眼可见）")


if __name__ == "__main__":
    main()
