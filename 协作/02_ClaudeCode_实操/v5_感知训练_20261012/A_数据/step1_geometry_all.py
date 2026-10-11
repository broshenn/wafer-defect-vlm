#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 1 步：全量几何真值。
输入：data/manifest.jsonl（5904 行）、data/images/<sample_id>.png
输出：geom_all.jsonl（可用图）、ineligible.jsonl（不可用+原因）、matrices/<id>.npy
做法：core_v3.measure_v3（钉住 SHA 的几何 v2）；可用性=roundtrip_validated 且
      status∈{ok,no_red} 且 outline.anomalous==False（合同 4.1）。
验收：总数=5904；可用+不可用=5904；打印各 split 数量、coverage_band/angular_type 分布；
      抽 5 张 no_red 确认 zone_frac 全 0、R1=R2=0；打印耗时。
用法：python step1_geometry_all.py [--workers N] [--limit N]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CORE_V3 = REPO / "协作/01_Codex_指挥/无卡CPU整备_20261010/core_v3.py"
MANIFEST = REPO / "data/manifest.jsonl"
IMAGES = REPO / "data/images"

_core = None


def _load_core():
    global _core
    if _core is None:
        spec = importlib.util.spec_from_file_location("_core_v3", CORE_V3)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _core = mod
    return _core


def process_one(row):
    """返回 (kind, record)。kind ∈ {ok, ineligible}。"""
    core = _load_core()
    geom_v2 = core.load_geometry()  # 反解器接口用；measure_v3 内部另有 SHA 校验
    sid = row["sample_id"]
    rec_base = {"sample_id": sid, "split": row["split"], "lot_name": row["lot_name"],
                "failure_type": row["failure_type"]}
    png = IMAGES / f"{sid}.png"
    if not png.exists():
        return "ineligible", {**rec_base, "reason": "png_missing"}
    hint = (row["matrix_shape"][0], row["matrix_shape"][1]) if row.get("matrix_shape") else None
    try:
        rec = geom_v2.recover_matrix(png, hint_hw=hint)
    except Exception as exc:
        return "ineligible", {**rec_base, "reason": f"recover_exc:{type(exc).__name__}"}
    if not rec.get("roundtrip_validated"):
        return "ineligible", {**rec_base, "reason": "roundtrip_not_validated"}
    matrix = rec["matrix"]
    try:
        g = core.measure_v3(matrix)
    except Exception as exc:
        return "ineligible", {**rec_base, "reason": f"measure_exc:{type(exc).__name__}"}
    if g.get("status") not in ("ok", "no_red"):
        return "ineligible", {**rec_base, "reason": f"status:{g.get('status')}",
                              "png_sha256": rec["png_sha256"]}
    if g.get("outline", {}).get("anomalous") is not False:
        return "ineligible", {**rec_base, "reason": "outline_anomalous",
                              "png_sha256": rec["png_sha256"]}
    # 存矩阵（旋转/合成要复用）
    import numpy as np
    np.save(HERE / "matrices" / f"{sid}.npy", matrix)
    n_red = int(g["n_red"])
    if n_red == 0:
        r1 = r2 = 0.0
        zf = [0.0, 0.0, 0.0]
    else:
        r1 = float(g["direction"]["R1_concentration"])
        r2 = float(g["direction"]["R2_axial"])
        zf = [float(g["radial"]["zone_frac"][k]) for k in ("center", "middle", "edge")]
    out = {
        **rec_base,
        "png_sha256": rec["png_sha256"],
        "matrix_hw": [int(matrix.shape[0]), int(matrix.shape[1])],
        "status": g["status"],
        "coverage": float(g["coverage"]),
        "coverage_band": core.COV_MAP[g["coverage_level"]],
        "red_mass_zones": g["red_mass_zones"],
        "angular_type": core.ANG_MAP[g["direction_type"]],
        "clock_sectors": list(g["clock_sectors"]),
        "zone_frac": zf,
        "R1": r1, "R2": r2,
        "n_red": n_red,
    }
    return "ok", out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    (HERE / "matrices").mkdir(exist_ok=True)
    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"manifest 行数={len(rows)} workers={args.workers}", flush=True)

    t0 = time.time()
    ok_records, bad_records = [], []
    if args.workers > 1:
        env_pool = Pool(args.workers)
        for kind, rec in env_pool.imap_unordered(process_one, rows, chunksize=8):
            (ok_records if kind == "ok" else bad_records).append(rec)
        env_pool.close()
        env_pool.join()
    else:
        for i, row in enumerate(rows):
            kind, rec = process_one(row)
            (ok_records if kind == "ok" else bad_records).append(rec)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(rows)} 用时 {time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0

    ok_records.sort(key=lambda r: r["sample_id"])
    bad_records.sort(key=lambda r: r["sample_id"])

    def write_jsonl(path, records):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        print(f"写出 {path.name}: {len(records)} 行 SHA={sha}", flush=True)

    if not args.limit:
        write_jsonl(HERE / "geom_all.jsonl", ok_records)
        write_jsonl(HERE / "ineligible.jsonl", bad_records)

    # ── 验收打印 ──
    print(f"\n总耗时 {dt:.0f}s（{dt / max(1, len(rows)) * 1000:.1f} ms/张）")
    print(f"可用 {len(ok_records)} + 不可用 {len(bad_records)} = {len(ok_records) + len(bad_records)}（应={len(rows)}）")
    from collections import Counter
    print("各 split 可用:", dict(Counter(r["split"] for r in ok_records)))
    print("coverage_band 分布:", dict(Counter(r["coverage_band"] for r in ok_records)))
    print("angular_type 分布:", dict(Counter(r["angular_type"] for r in ok_records)))
    print("failure_type 分布:", dict(Counter(r["failure_type"] for r in ok_records)))
    print("不可用原因:", dict(Counter(r["reason"] for r in bad_records)))
    nored = [r for r in ok_records if r["status"] == "no_red"]
    print(f"no_red 数量={len(nored)}")
    for r in nored[:5]:
        assert all(x == 0.0 for x in r["zone_frac"]) and r["R1"] == 0.0 and r["R2"] == 0.0, r
        print(f"  no_red 抽查 {r['sample_id']}: zone_frac={r['zone_frac']} R1={r['R1']} R2={r['R2']}")


if __name__ == "__main__":
    main()
