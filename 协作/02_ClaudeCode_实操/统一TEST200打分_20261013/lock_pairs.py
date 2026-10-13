#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预先锁定配对比较（**在看任何分数之前**）。

规则是固定的、不挑模型的：
  · 类别参照 = `--cat-ref`（默认 L）：**清单里每个模型**都对它做 macro_f1 与 accuracy 配对；
  · 几何参照 = `--geo-ref`（默认 S0_rerun）：**清单里每个 main 类模型**都对它做 geo/macro_f1/accuracy 配对。
不含参照自身。之所以对所有行都点名，是为了避免"挑几对报区间"造成的选择偏差；
除此之外不按分数增删任何一对。

锁定后把 pairs.json 的 SHA 写进 source_manifest.json，打分器会核对。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo-ref", default="S0_rerun")
    ap.add_argument("--cat-ref", default="L")
    a = ap.parse_args()

    contract_path = HERE / "contract.json"
    manifest_path = HERE / "source_manifest.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = [e["name"] for e in manifest["sources"]]
    kinds = {e["name"]: e["kind"] for e in manifest["sources"]}

    for ref, label in ((a.geo_ref, "几何参照"), (a.cat_ref, "类别参照")):
        if ref not in kinds:
            raise SystemExit(f"{label} {ref} 不在清单里（清单共 {len(names)} 条）")

    pairs = []
    for n in names:
        if n == a.geo_ref:
            continue
        if kinds[n] == "main":
            pairs.append({"a": n, "b": a.geo_ref, "metrics": ["geo", "macro_f1", "accuracy"]})
    for n in names:
        if n == a.cat_ref:
            continue
        pairs.append({"a": n, "b": a.cat_ref, "metrics": ["macro_f1", "accuracy"]})

    lock = {
        "schema": "unified-test200-pairs-v1",
        "locked_before_scoring": True,
        "locked_at_note": "本文件在任何分数计算之前生成；模型名单只来自清单里存在的行，与分数无关",
        "contract_sha256": sha_file(contract_path),
        "repeats": 10000,
        "seed": 3407,
        "resampling_unit": "lot",
        "geo_reference": a.geo_ref,
        "category_reference": a.cat_ref,
        "rule": ("每个清单行 vs 类别参照（macro_f1/accuracy）；每个 main 类行 vs 几何参照（geo/macro_f1/accuracy）。"
                 "不含参照自身；不按分数增删。"),
        "n_rows_in_manifest": len(names),
        "pairs": pairs,
    }
    pairs_path = HERE / "pairs.json"
    pairs_path.write_text(json.dumps(lock, ensure_ascii=False, indent=1), encoding="utf-8")
    pairs_sha = sha_file(pairs_path)

    manifest["pairs_sha256"] = pairs_sha
    manifest["pairs_note"] = f"{len(pairs)} 对，固定规则，看分前锁定"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    print(json.dumps({"pairs": len(pairs), "geo_ref": a.geo_ref, "cat_ref": a.cat_ref,
                      "pairs_sha256": pairs_sha, "manifest_rows": len(names)},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
