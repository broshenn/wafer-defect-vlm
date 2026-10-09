#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为全部 5904 张公开图计算几何参考字段（CPU，只读图片与 manifest）。

产出 `几何特征表.jsonl`：一行一图，含 `几何参考_v2.py` 的全部测量字段。
**这张表只用于评分侧参考与数据分层，不是模型输入。**
模型输入永远只有 PNG + 题面。

用法:
  python 建几何特征表.py            # 全量 5904
  python 建几何特征表.py --limit 200
  python 建几何特征表.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from 几何参考_v2 import DEFINITION_VERSION, measure_matrix, recover_matrix  # noqa: E402

MANIFEST = REPO / "data" / "manifest.jsonl"
IMAGES = REPO / "data" / "images"


def one(row: dict) -> dict:
    sid = row["sample_id"]
    png = IMAGES / f"{sid}.png"
    base = {"sample_id": sid, "failure_type": row.get("failure_type"),
            "split": row.get("split"), "lot_name": row.get("lot_name"),
            "matrix_shape_manifest": row.get("matrix_shape")}
    if not png.exists():
        return {**base, "status": "png_missing"}
    t0 = time.time()
    try:
        rec = recover_matrix(png, hint_hw=tuple(row["matrix_shape"]))
    except Exception as exc:
        return {**base, "status": "recover_failed", "error": f"{type(exc).__name__}: {exc}"}
    if not rec.get("roundtrip_validated"):
        return {**base, "status": "recover_not_lossless", "recover_path": rec.get("path_used")}
    m = measure_matrix(rec["matrix"])
    m.update({"sample_id": sid, "failure_type": row.get("failure_type"),
              "split": row.get("split"), "lot_name": row.get("lot_name"),
              "matrix_shape_manifest": row.get("matrix_shape"),
              "recover_path": rec["path_used"], "png_sha256": rec["png_sha256"],
              "seconds": round(time.time() - t0, 4)})
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "几何特征表.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(line) for line in MANIFEST.open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    workers = args.workers or 0
    t0 = time.time()
    out = []
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, r) for r in rows]
            for i, f in enumerate(as_completed(futs), 1):
                out.append(f.result())
                if i % 500 == 0:
                    print(f"  {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    else:
        for i, r in enumerate(rows, 1):
            out.append(one(r))
            if i % 500 == 0:
                print(f"  {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    out.sort(key=lambda d: d["sample_id"])
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for d in out:
            fh.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")

    st = {}
    for d in out:
        st[d.get("status", "?")] = st.get(d.get("status", "?"), 0) + 1
    print(f"写出 {len(out)} 行 → {args.out}")
    print("状态计数:", st)
    print(f"用时 {time.time()-t0:.1f}s，定义版本 {DEFINITION_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
