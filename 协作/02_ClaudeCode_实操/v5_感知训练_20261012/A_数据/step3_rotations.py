#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 3 步：旋转增广 + 程序等变性自检（合同 5.1）。
输入：TRAIN_POOL.txt、SELECT200.txt、TEST.txt、matrices/*.npy、geom_all.jsonl、data/images/*.png
输出：img448/（原图复制 + TRAIN_POOL 三个旋转版）、geom_rot.jsonl、rotation_selfcheck.json
验收：扇区一致率 ≥95%（排除变 unknown 的）；coverage_band 与 zones 不变率 ≥99%。
用法：python step3_rotations.py [--workers N]
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CORE_V3 = REPO / "协作/01_Codex_指挥/无卡CPU整备_20261010/core_v3.py"
WAFER_PNG_V2 = REPO / "协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/wafer_png_v2.py"
IMAGES = REPO / "data/images"
_core = None
_wpng = None


def _load():
    global _core, _wpng
    if _core is None:
        spec = importlib.util.spec_from_file_location("_core_v3", CORE_V3)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _core = mod
        spec2 = importlib.util.spec_from_file_location("_wafer_png_v2", WAFER_PNG_V2)
        m2 = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(m2)
        assert hashlib.sha256(WAFER_PNG_V2.read_bytes()).hexdigest() == _core.load_geometry().RECOVER_SRC_SHA256, \
            "wafer_png_v2 SHA 与几何 v2 钉住的值不一致"
        _wpng = m2
    return _core, _wpng


def rot_task(task):
    import numpy as np
    sid, k, deg = task
    core, wpng = _load()
    m = np.load(HERE / "matrices" / f"{sid}.npy")
    m_rot = np.rot90(m, k=k)
    img = wpng.matrix_to_image(m_rot, 448)
    img.save(HERE / "img448" / f"{sid}_r{deg:03d}.png", format="PNG", optimize=True)
    g = core.measure_v3(m_rot)
    if g.get("status") not in ("ok", "no_red") or g.get("outline", {}).get("anomalous") is not False:
        return ("bad", {"src_id": sid, "rot_deg": deg, "status": g.get("status"),
                        "reason": "ineligible_rotation"})
    n_red = int(g["n_red"])
    if n_red == 0:
        r1 = r2 = 0.0
        zf = [0.0, 0.0, 0.0]
        tie = False
    else:
        r1 = float(g["direction"]["R1_concentration"])
        r2 = float(g["direction"]["R2_axial"])
        zf = [float(g["radial"]["zone_frac"][x]) for x in ("center", "middle", "edge")]
        tie = bool(g["direction"].get("sector_tie", False))
    rec = {
        "sample_id": f"{sid}_r{deg:03d}", "src_id": sid, "rot_deg": deg,
        "matrix_hw": [int(m_rot.shape[0]), int(m_rot.shape[1])],
        "status": g["status"],
        "coverage": float(g["coverage"]),
        "coverage_band": core.COV_MAP[g["coverage_level"]],
        "red_mass_zones": g["red_mass_zones"],
        "angular_type": core.ANG_MAP[g["direction_type"]],
        "clock_sectors": list(g["clock_sectors"]),
        "zone_frac": zf, "R1": r1, "R2": r2, "n_red": n_red, "sector_tie": tie,
    }
    return ("ok", rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    (HERE / "img448").mkdir(exist_ok=True)

    pool_ids = (HERE / "TRAIN_POOL.txt").read_text(encoding="utf-8").split()
    # 1) 原图复制（TRAIN_POOL + SELECT200 + TEST；本地已有则跳过）
    copies = pool_ids + (HERE / "SELECT200.txt").read_text(encoding="utf-8").split() \
             + (HERE / "TEST.txt").read_text(encoding="utf-8").split()
    n_copied = 0
    for sid in copies:
        dst = HERE / "img448" / f"{sid}.png"
        if not dst.exists():
            shutil.copyfile(IMAGES / f"{sid}.png", dst)
            n_copied += 1
    print(f"原图复制 {n_copied}（共 {len(copies)} 张在 img448/）", flush=True)

    # 2) 三个旋转版
    tasks = [(sid, k, deg) for sid in pool_ids for k, deg in ((-1, 90), (-2, 180), (-3, 270))]
    t0 = time.time()
    ok, bad = [], []
    if args.workers > 1:
        with Pool(args.workers) as p:
            for kind, rec in p.imap_unordered(rot_task, tasks, chunksize=16):
                (ok if kind == "ok" else bad).append(rec)
    else:
        for t in tasks:
            kind, rec = rot_task(t)
            (ok if kind == "ok" else bad).append(rec)
    ok.sort(key=lambda r: r["sample_id"])
    print(f"旋转 {len(ok)} 可用 + {len(bad)} 不可用 = {len(tasks)}；耗时 {time.time() - t0:.0f}s")
    with open(HERE / "geom_rot.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in ok:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    sha = hashlib.sha256((HERE / "geom_rot.jsonl").read_bytes()).hexdigest()
    print(f"geom_rot.jsonl {len(ok)} 行 SHA={sha}")

    # 3) 等变性自检
    geo = {json.loads(l)["sample_id"]: json.loads(l)
           for l in (HERE / "geom_all.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    rot = {}
    for r in ok:
        rot[(r["src_id"], r["rot_deg"])] = r
    singles = [sid for sid in pool_ids if geo[sid]["angular_type"] == "single"]
    sector_total = sector_ok = tie_unknown = 0
    inconsistent = []
    for sid in singles:
        o = geo[sid]
        for deg, add in ((90, 3), (180, 6), (270, 9)):
            r = rot.get((sid, deg))
            if r is None:
                continue
            if r["angular_type"] == "single":
                want = (o["clock_sectors"][0] - 1 + add) % 12 + 1
                sector_total += 1
                if r["clock_sectors"] and r["clock_sectors"][0] == want:
                    sector_ok += 1
                elif deg == 90:
                    inconsistent.append({"src_id": sid, "rot": deg, "orig": o["clock_sectors"],
                                         "rot_sectors": r["clock_sectors"], "want": want})
            else:
                if r["angular_type"] == "unknown" and r.get("sector_tie"):
                    tie_unknown += 1
                elif deg == 90:
                    inconsistent.append({"src_id": sid, "rot": deg, "orig": o["clock_sectors"],
                                         "rot_type": r["angular_type"], "note": "旋转后非 single"})
    band_same = band_total = 0
    zone_same = zone_total = 0
    band_bad, zone_bad = [], []
    for sid in pool_ids:
        o = geo[sid]
        for deg in (90, 180, 270):
            r = rot.get((sid, deg))
            if r is None:
                continue
            band_total += 1
            zone_total += 1
            if r["coverage_band"] == o["coverage_band"]:
                band_same += 1
            elif len(band_bad) < 50:
                band_bad.append({"src_id": sid, "rot": deg, "orig": o["coverage_band"],
                                 "rot": r["coverage_band"]})
            oz = o["red_mass_zones"]
            rz = r["red_mass_zones"]
            same = (oz == rz) if isinstance(oz, list) else (rz == oz)
            if same:
                zone_same += 1
            elif len(zone_bad) < 50:
                zone_bad.append({"src_id": sid, "rot": deg, "orig": oz, "rot": rz})
    sector_rate = sector_ok / sector_total if sector_total else None
    band_rate = band_same / band_total if band_total else None
    zone_rate = zone_same / zone_total if zone_total else None
    report = {
        "n_train_pool": len(pool_ids), "n_single_orig": len(singles),
        "sector_total": sector_total, "sector_ok": sector_ok,
        "sector_rate": sector_rate, "tie_unknown": tie_unknown,
        "band_total": band_total, "band_same": band_same, "band_rate": band_rate,
        "zone_total": zone_total, "zone_same": zone_same, "zone_rate": zone_rate,
        "inconsistent_first50": inconsistent[:50], "band_bad_first50": band_bad,
        "zone_bad_first50": zone_bad,
        "n_rotation_ineligible": len(bad),
        "rotation_ineligible": bad[:50],
    }
    (HERE / "rotation_selfcheck.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"等变性：扇区 {sector_ok}/{sector_total} = {sector_rate}（要求 ≥0.95）"
          f"；并列变 unknown {tie_unknown}；coverage_band 不变 {band_same}/{band_total} = {band_rate}（≥0.99）"
          f"；zones 不变 {zone_same}/{zone_total} = {zone_rate}（≥0.99）")
    if sector_rate is not None and sector_rate < 0.95:
        print("**扇区一致率不达标，按 0.4 决策表处理（写 DECISIONS.md）**")
    if (band_rate or 0) < 0.99 or (zone_rate or 0) < 0.99:
        print("**band/zones 不变率不达标，按 0.4 决策表处理（写 DECISIONS.md）**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
