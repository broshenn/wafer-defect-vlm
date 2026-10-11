#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 8 步（P2）：叠加核验图。
输入：TEST 分层 120 张（seed 3407）+ 合成图 20 张；img448/、matrices/、geom_*.jsonl
输出：overlays/<名>.png（672×672 放大图 + 白椭圆轮廓、黄色 1/3、2/3 分界、灰色 12 扇区分界与编号、
      single/opposed 扇区半透明蓝、底部一行程序真值文字）+ overlays/index.jsonl
验收：140 张全部生成；打印 3 张路径与尺寸。本脚本不能看图，位置自查请用户进行。
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import syn_lib  # noqa: E402

SCALE = 672
SIZE = 448


def strat_sample_ids(meta, ids, k, seed):
    groups = {}
    for s in ids:
        groups.setdefault(meta[s]["failure_type"], []).append(s)
    counts = {c: len(v) for c, v in groups.items()}
    total = sum(counts.values())
    alloc = {c: max(min(5, counts[c]), min(counts[c], round(k * counts[c] / total))) for c in counts}
    while sum(alloc.values()) > k:
        c = max(sorted(counts), key=lambda c: (alloc[c] - min(5, counts[c]), c))
        if alloc[c] > min(5, counts[c]):
            alloc[c] -= 1
        else:
            break
    while sum(alloc.values()) < k:
        c = max(sorted(counts), key=lambda c: (counts[c] - alloc[c], c))
        alloc[c] += 1
    rng = random.Random(seed)
    out = []
    for c in sorted(counts):
        out.extend(rng.sample(sorted(groups[c]), alloc[c]))
    return sorted(out)


def draw_one(name, matrix, g, out_path, label):
    img = Image.open(HERE / "img448" / f"{name}.png").convert("RGB").resize((SCALE, SCALE), Image.NEAREST)
    base = img.convert("RGBA")
    ov = Image.new("RGBA", (SCALE, SCALE), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    geom = syn_lib.load_core().load_geometry()
    valid = (matrix == 1) | (matrix == 2)
    cy, cx, a, b = geom._ellipse_of(valid)
    H, W = matrix.shape

    def to_px(u, v):
        return ((cx + v * b) * SCALE / W, (cy + u * a) * SCALE / H)

    # 白椭圆轮廓 + 1/3、2/3 黄色分界
    for frac, color, width in ((1.0, (255, 255, 255, 235), 2), (0.6667, (255, 255, 0, 220), 1),
                               (0.3333, (255, 255, 0, 220), 1)):
        pts = []
        for k in range(72):
            deg = k * 5
            u, v = -np.cos(np.radians(deg)) * frac, np.sin(np.radians(deg)) * frac
            pts.append(to_px(u, v))
        d.line(pts + [pts[0]], fill=color, width=width)
    # 12 扇区分界（边界在 12 点两侧 ±15°，即 -15°,15°,...）
    for k in range(12):
        deg = -15 + k * 30
        p1 = to_px(0, 0)
        p2 = to_px(-np.cos(np.radians(deg)), np.sin(np.radians(deg)))
        d.line([p1, p2], fill=(160, 160, 160, 180), width=1)
    # 扇区编号（扇区中心角 0,30,...330；编号 1..12）
    for k in range(12):
        deg = k * 30
        u, v = -np.cos(np.radians(deg)) * 1.09, np.sin(np.radians(deg)) * 1.09
        x, y = to_px(u, v)
        d.text((min(max(x - 4, 0), SCALE - 10), min(max(y - 6, 0), SCALE - 12)), str(k + 1),
               fill=(255, 255, 255, 255))
    # single/opposed 扇区涂半透明蓝
    if g["angular_type"] in ("single", "opposed"):
        for sec in g["clock_sectors"]:
            center = (sec - 1) * 30          # 扇区中心角（deg）
            pts = [to_px(0, 0)]
            for t in np.linspace(center - 15, center + 15, 13):
                pts.append(to_px(-np.cos(np.radians(t)), np.sin(np.radians(t))))
            d.polygon(pts, fill=(60, 120, 255, 90))
    out = Image.alpha_composite(base, ov).convert("RGB")
    strip = Image.new("RGB", (SCALE, 40), (255, 255, 255))
    ds = ImageDraw.Draw(strip)
    ds.text((6, 12), label, fill=(0, 0, 0))
    canvas = Image.new("RGB", (SCALE, SCALE + 40), (255, 255, 255))
    canvas.paste(out, (0, 0))
    canvas.paste(strip, (0, SCALE))
    canvas.save(out_path)


def main():
    (HERE / "overlays").mkdir(exist_ok=True)
    meta = {r["sample_id"]: r for r in
            (json.loads(l) for l in (HERE / "geom_all.jsonl").read_text(encoding="utf-8").splitlines())}
    test_ids = (HERE / "TEST.txt").read_text(encoding="utf-8").split()
    pick120 = strat_sample_ids(meta, test_ids, 120, 3407)
    syn = [json.loads(l) for l in (HERE / "geom_syn.jsonl").read_text(encoding="utf-8").splitlines()]
    index = []
    for sid in pick120:
        g = meta[sid]
        m = np.load(HERE / "matrices" / f"{sid}.npy")
        label = (f"{sid} coverage={g['coverage']:.2f} ({g['coverage_band']}) "
                 f"zones={g['red_mass_zones']} angular={g['angular_type']} sectors={g['clock_sectors']}")
        draw_one(sid, m, g, HERE / "overlays" / f"{sid}.png", label)
        index.append({"file": f"{sid}.png", "sample_id": sid, "coverage": g["coverage"],
                      "coverage_band": g["coverage_band"], "red_mass_zones": g["red_mass_zones"],
                      "angular_type": g["angular_type"], "clock_sectors": g["clock_sectors"]})
    for g in syn[:20]:
        sid = g["sample_id"]
        # 合成图没有存矩阵：用几何反解器从 PNG 还原
        geom = syn_lib.load_core().load_geometry()
        rec = geom.recover_matrix(HERE / "img448" / f"{sid}.png", hint_hw=tuple(g["matrix_hw"]))
        assert rec["roundtrip_validated"], sid
        m = rec["matrix"]
        label = (f"{sid} [syn {g['pattern']}] coverage={g['coverage']:.2f} ({g['coverage_band']}) "
                 f"zones={g['red_mass_zones']} angular={g['angular_type']} sectors={g['clock_sectors']}")
        draw_one(sid, m, g, HERE / "overlays" / f"{sid}.png", label)
        index.append({"file": f"{sid}.png", "sample_id": sid, "pattern": g["pattern"],
                      "coverage": g["coverage"], "coverage_band": g["coverage_band"],
                      "red_mass_zones": g["red_mass_zones"], "angular_type": g["angular_type"],
                      "clock_sectors": g["clock_sectors"]})
    with open(HERE / "overlays" / "index.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in index:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    n = len(list((HERE / "overlays").glob("*.png")))
    print(f"overlays 生成 {n} 张（应 140）; index.jsonl {len(index)} 行")
    for r in index[:3]:
        p = HERE / "overlays" / r["file"]
        print(f"  抽检 {r['file']}: 尺寸={Image.open(p).size} 路径={p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
