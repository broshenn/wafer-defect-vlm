"""Prepare a conservative, lot-isolated WM811K subset for teacher annotation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .lswmd import load_lswmd_frame
from .utils import (
    LABELS,
    calculate_static_features,
    flatten_scalar,
    matrix_to_image,
    normalize_label_series,
    sample_id,
    split_for_lot,
    write_jsonl,
)


def locate_pickle(raw_path: Path) -> Path:
    if raw_path.is_file():
        return raw_path
    candidates = list(raw_path.rglob("LSWMD.pkl"))
    if not candidates:
        raise FileNotFoundError(f"LSWMD.pkl not found below {raw_path}")
    return candidates[0]


def choose_indices(labels: pd.Series, per_class: int, none_limit: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in LABELS:
        indices = labels.index[labels == label].to_numpy()
        limit = none_limit if label == "none" else per_class
        if limit >= 0 and len(indices) > limit:
            indices = rng.choice(indices, size=limit, replace=False)
        selected.extend(int(i) for i in indices)
    rng.shuffle(selected)
    return selected


def build(args: argparse.Namespace) -> None:
    args.inspect_only = args.inspect_only or getattr(args, "dry_run", False)
    source = locate_pickle(Path(args.input))
    output = Path(args.output)
    image_dir = output / "images"
    output.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {source}")
    frame = load_lswmd_frame(source)
    required = {"waferMap", "lotName", "waferIndex", "failureType"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    labels = pd.Series(normalize_label_series(frame["failureType"]), index=frame.index, dtype=object)
    counts = Counter(x for x in labels if x is not None)
    stats = {
        "source": str(source.resolve()),
        "total_rows": len(frame),
        "labeled_rows": int(labels.notna().sum()),
        "class_counts": dict(sorted(counts.items())),
        "seed": args.seed,
        "per_class_limit": args.per_class,
        "none_limit": args.none_limit,
    }
    (output / "source_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.inspect_only:
        return

    indices = choose_indices(labels, args.per_class, args.none_limit, args.seed)
    records: list[dict] = []
    for index in tqdm(indices, desc="Rendering wafer maps"):
        row = frame.loc[index]
        label = labels.loc[index]
        if label not in LABELS:
            continue
        lot = str(row["lotName"])
        sid = sample_id(lot, row["waferIndex"])
        image_path = image_dir / f"{sid}.png"
        wafer = np.asarray(row["waferMap"], dtype=np.uint8)
        if not image_path.exists() or args.overwrite_images:
            matrix_to_image(wafer, args.resolution).save(image_path, format="PNG", optimize=True)
        record = {
            "sample_id": sid,
            "source_index": int(index),
            "lot_name": lot,
            "wafer_index": int(float(flatten_scalar(row["waferIndex"]))),
            "failure_type": label,
            "label_source": "ground_truth",
            "split": split_for_lot(lot, args.seed),
            "image_path": str(image_path.resolve()),
            "matrix_shape": list(wafer.shape),
            "features": calculate_static_features(wafer, apply_filter=True),
        }
        records.append(record)

    # A stable order makes diffs, hashes and resume behavior deterministic.
    records.sort(key=lambda x: x["sample_id"])
    ids = [record["sample_id"] for record in records]
    if len(ids) != len(set(ids)):
        duplicates = [sid for sid, count in Counter(ids).items() if count > 1]
        raise RuntimeError(f"Non-unique sample_id values: {duplicates[:10]}")
    split_lots = {
        split: {record["lot_name"] for record in records if record["split"] == split}
        for split in ("train", "val", "test")
    }
    intersections = {
        "train_val": sorted(split_lots["train"] & split_lots["val"]),
        "train_test": sorted(split_lots["train"] & split_lots["test"]),
        "val_test": sorted(split_lots["val"] & split_lots["test"]),
    }
    if any(intersections.values()):
        raise RuntimeError(f"Lot leakage detected: {intersections}")
    write_jsonl(output / "manifest.jsonl", records)
    split_counts = Counter((r["split"], r["failure_type"]) for r in records)
    summary = {
        "samples": len(records),
        "splits": {
            split: {label: split_counts[(split, label)] for label in LABELS}
            for split in ("train", "val", "test")
        },
        "unique_sample_ids": len(set(ids)),
        "lot_split_intersections": intersections,
    }
    (output / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="LSWMD.pkl or a directory containing it")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=800, help="Maximum per defect class; -1 keeps all")
    parser.add_argument("--none-limit", type=int, default=400)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --inspect-only; writes no images")
    parser.add_argument("--overwrite-images", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.inspect_only = args.inspect_only or args.dry_run
    build(args)


if __name__ == "__main__":
    main()
