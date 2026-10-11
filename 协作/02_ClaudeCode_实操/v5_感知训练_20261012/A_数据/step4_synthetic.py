#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 4 步：合成图（合同 5.2，只用于原子问答，不用于类别）。
输入：TRAIN_POOL.txt、geom_all.jsonl、matrices/*.npy
输出：img448/syn_000001.png…（最终 1500 张）、geom_syn.jsonl、syn_contact_sheet.png
做法：none 类且 coverage<0.02 的底图（前 300，按 id 排序）红改绿；单个 rng(3407) 生成
      3000 张（七图案随机 + 背景噪声 q∈[0,0.01]）；measure_v3 重算；按 angular_type
      挑 1500（single 尽量 500、opposed 尽量 250、其余随机补足）；联系表 20 张 4×5。
前置：test_synthetic.py 5/5 PASS。
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import syn_lib  # noqa: E402

N_GEN = 3000
N_KEEP = 1500


def main():
    core = syn_lib.load_core()
    pool = (HERE / "TRAIN_POOL.txt").read_text(encoding="utf-8").split()
    geo = {json.loads(l)["sample_id"]: json.loads(l)
           for l in (HERE / "geom_all.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    bases = sorted(s for s in pool if geo[s]["failure_type"] == "none" and geo[s]["coverage"] < 0.02)[:300]
    print(f"合成底图 {len(bases)} 张（none 且 coverage<0.02 前 300）")
    assert len(bases) > 0

    rng = np.random.default_rng(3407)
    t0 = time.time()
    records, kept_matrices = [], []
    n_ineligible = 0
    for i in range(N_GEN):
        base_id = bases[i % len(bases)]
        m0 = np.load(HERE / "matrices" / f"{base_id}.npy")
        m0 = m0.copy()
        m0[m0 == 2] = 1                                  # none 底图：红→绿
        name = syn_lib.PATTERN_NAMES[int(rng.integers(0, len(syn_lib.PATTERN_NAMES)))]
        m, pr = syn_lib.draw_pattern(m0, name, rng)
        q = float(rng.uniform(0, 0.01))
        valid = (m == 1) | (m == 2)
        noise = valid & (rng.random(m.shape) < q)
        m[noise] = 2
        g = core.measure_v3(m)
        if g.get("status") not in ("ok", "no_red") or g.get("outline", {}).get("anomalous") is not False:
            n_ineligible += 1
            continue
        n_red = int(g["n_red"])
        if n_red == 0:
            r1 = r2 = 0.0
            zf = [0.0, 0.0, 0.0]
        else:
            r1 = float(g["direction"]["R1_concentration"])
            r2 = float(g["direction"]["R2_axial"])
            zf = [float(g["radial"]["zone_frac"][x]) for x in ("center", "middle", "edge")]
        records.append({
            "gen_index": i, "base_id": base_id, "pattern": name,
            "params": {**pr, "noise_q": round(q, 6)},
            "matrix_hw": [int(m.shape[0]), int(m.shape[1])], "status": g["status"],
            "coverage": float(g["coverage"]),
            "coverage_band": core.COV_MAP[g["coverage_level"]],
            "red_mass_zones": g["red_mass_zones"],
            "angular_type": core.ANG_MAP[g["direction_type"]],
            "clock_sectors": list(g["clock_sectors"]),
            "zone_frac": zf, "R1": r1, "R2": r2, "n_red": n_red,
        })
        kept_matrices.append(m)
        if (i + 1) % 1000 == 0:
            print(f"  生成 {i + 1}/{N_GEN}，用时 {time.time() - t0:.0f}s", flush=True)
    print(f"生成完成：可用 {len(records)}，不可用 {n_ineligible}，总耗时 {time.time() - t0:.0f}s")

    # ── 挑选 1500 ──
    idx_single = [k for k, r in enumerate(records) if r["angular_type"] == "single"]
    idx_opposed = [k for k, r in enumerate(records) if r["angular_type"] == "opposed"]
    n_single_want = min(500, len(idx_single))
    n_opposed_want = min(250, len(idx_opposed))
    # 在各自集合里随机取（rng.permutation 保证确定性）
    pick_single = [idx_single[j] for j in np.sort(rng.permutation(len(idx_single))[:n_single_want])]
    pick_opposed = [idx_opposed[j] for j in np.sort(rng.permutation(len(idx_opposed))[:n_opposed_want])]
    used = set(pick_single) | set(pick_opposed)
    remaining = [k for k in range(len(records)) if k not in used]
    need = N_KEEP - len(used)
    pick_rest = [remaining[j] for j in np.sort(rng.permutation(len(remaining))[:need])] if need > 0 else []
    picked = pick_single + pick_opposed + pick_rest
    assert len(picked) == min(N_KEEP, len(records)), (len(picked), len(records))

    # ── 渲染 + 命名 ──
    wpng_spec = syn_lib._CORE.load_geometry()
    import importlib.util
    spec = importlib.util.spec_from_file_location("_wafer_png_v2", syn_lib.REPO / "协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/wafer_png_v2.py")
    wpng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wpng)
    out_records = []
    from collections import Counter
    for n, k in enumerate(picked, start=1):
        sid = f"syn_{n:06d}"
        r = records[k]
        wpng.matrix_to_image(kept_matrices[k], 448).save(HERE / "img448" / f"{sid}.png",
                                                         format="PNG", optimize=True)
        out_records.append({"sample_id": sid, **r})
    with open(HERE / "geom_syn.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    sha = hashlib.sha256((HERE / "geom_syn.jsonl").read_bytes()).hexdigest()
    print(f"geom_syn.jsonl {len(out_records)} 行 SHA={sha}")
    print("最终 angular_type:", dict(Counter(r["angular_type"] for r in out_records)))
    print("最终 pattern:", dict(Counter(r["pattern"] for r in out_records)))
    print("最终 coverage_band:", dict(Counter(r["coverage_band"] for r in out_records)))

    # ── 联系表 20 张（4×5）──
    cell, strip = 224, 42
    sheet = Image.new("RGB", (cell * 5, (cell + strip) * 4), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for n, r in enumerate(out_records[:20]):
        img = Image.open(HERE / "img448" / f"{r['sample_id']}.png").resize((cell, cell), Image.NEAREST)
        row, col = divmod(n, 5)
        y = row * (cell + strip)
        sheet.paste(img, (col * cell, y))
        label = (f"{r['sample_id']} {r['pattern']}\n{r['coverage_band']} "
                 f"{','.join(r['red_mass_zones']) if isinstance(r['red_mass_zones'], list) else r['red_mass_zones']} "
                 f"{r['angular_type']}")
        draw.text((col * cell + 3, y + cell + 2), label, fill=(0, 0, 0))
    sheet.save(HERE / "syn_contact_sheet.png")
    print("联系表 syn_contact_sheet.png 已生成（4×5，前 20 张）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
