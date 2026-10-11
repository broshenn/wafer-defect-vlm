#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 7 步：S3 与 _x2 评测用的 896 图。
输入：sft_S3.jsonl 与全部 *_x2.jsonl 里引用的 `__IMG__/<名>.png`
输出：img896/<名>（同一批名字，从 img448/ 对应源图最近邻放大 896×896）
验收：每个被引用的 _x2 文件都存在；抽 3 张检查尺寸 896×896、颜色只有黑绿红三种。
用法：python step7_img896.py [--workers N]
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def collect_names():
    names = set()
    files = [HERE / "sft_S3.jsonl"] + sorted(HERE.glob("*_x2.jsonl"))
    for p in files:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                img = json.loads(line)["images"][0]
                assert img.startswith("__IMG__/"), img
                names.add(img[len("__IMG__/"):])
    return sorted(names)


def upscale(name):
    dst = HERE / "img896" / name
    if dst.exists():
        return ("skip", name)
    src_name = name[:-4] if name.endswith(".png") else name
    if src_name.endswith("_x2"):
        src_name = src_name[:-3]
    src = HERE / "img448" / f"{src_name}.png"
    if not src.exists():
        return ("missing_src", name)
    img = Image.open(src).convert("RGB")
    img.resize((896, 896), Image.NEAREST).save(dst, format="PNG", optimize=True)
    return ("ok", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    (HERE / "img896").mkdir(exist_ok=True)
    names = collect_names()
    print(f"被引用图片 {len(names)} 个（sft_S3 + 全部 *_x2）")
    t0 = time.time()
    ok = missing = 0
    with Pool(args.workers) as p:
        for kind, name in p.imap_unordered(upscale, names, chunksize=32):
            if kind == "missing_src":
                missing += 1
                print("  缺源:", name)
            else:
                ok += 1
    print(f"完成 {ok}，缺源 {missing}，耗时 {time.time() - t0:.0f}s")
    # 抽 3 张检查
    checked = 0
    for name in names:
        dst = HERE / "img896" / name
        if not dst.exists():
            print("  FAIL 缺文件:", name)
            continue
        arr = np.array(Image.open(dst).convert("RGB"))
        flat = arr.reshape(-1, 3)
        sample = flat[:: max(1, len(flat) // 100000)]
        colors = {tuple(int(x) for x in c) for c in sample}
        okc = arr.shape == (896, 896, 3) and colors <= {(0, 0, 0), (0, 255, 0), (255, 0, 0)}
        print(f"  抽检 {name}: shape={arr.shape} 颜色集={sorted(colors)[:4]} {'PASS' if okc else 'FAIL'}")
        checked += 1
        if checked >= 3:
            break
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
