"""Flatten a prepared manifest.jsonl into the canonical manifest.parquet.

One row per wafer sample. Scalar geometry features become prefixed columns,
fixed-length vectors stay as list<float> columns so nothing is lost or
silently stringified.
"""
import argparse, hashlib, json
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "status", "valid_count", "raw_defect_count", "defect_count",
    "filtered_points_removed", "defect_ratio", "wafer_radius",
    "centroid_radius_r", "radial_zone", "clock_sector", "mean_radius_r",
    "std_radius_r", "defect_extent_r", "radial_span_r", "resultant_length",
    "angle_entropy_bits", "anisotropy_min_over_max",
]
VECTOR_COLUMNS = ["wafer_center_xy", "centroid_xy_r", "radial_density", "angular_probability"]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="manifest.jsonl")
    ap.add_argument("--output", required=True, help="manifest.parquet")
    ap.add_argument("--hash-images", action="store_true")
    ap.add_argument("--jsonl-output", help="optional JSONL copy next to the parquet")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.input).open(encoding="utf-8") if line.strip()]
    flat = []
    for row in rows:
        rec = {
            "sample_id": row["sample_id"],
            "source_index": row.get("source_index"),
            "lot_name": row["lot_name"],
            "wafer_index": row.get("wafer_index"),
            "failure_type": row["failure_type"],
            "label_source": row.get("label_source"),
            "split": row["split"],
            "image_path": row["image_path"],
            "matrix_shape": list(row.get("matrix_shape") or []),
        }
        feats = row.get("features") or {}
        for key in FEATURE_COLUMNS:
            rec["feat_" + key] = feats.get(key)
        for key in VECTOR_COLUMNS:
            value = feats.get(key)
            rec["vec_" + key] = list(value) if isinstance(value, (list, tuple)) else value
        if args.hash_images:
            path = Path(row["image_path"])
            rec["image_sha256"] = sha256_file(path) if path.exists() else None
        flat.append(rec)

    frame = pd.DataFrame(flat)
    frame = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False, compression="zstd")

    if args.jsonl_output:
        with Path(args.jsonl_output).open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(json.dumps({
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "bytes": out.stat().st_size,
        "unique_sample_ids": int(frame["sample_id"].nunique()),
        "splits": frame["split"].value_counts().to_dict(),
        "classes": frame["failure_type"].value_counts().to_dict(),
        "hashed_images": bool(args.hash_images),
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
