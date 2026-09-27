"""Prepare WM811K under a tight RAM limit by spilling embedded wafer arrays to disk."""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.compat.pickle_compat import Unpickler as PandasUnpickler
from tqdm import tqdm

from .prepare_data import choose_indices
from .utils import (
    LABELS, calculate_static_features, flatten_scalar, matrix_to_image,
    normalize_label_series, sample_id, split_for_lot, write_jsonl,
)

_SPILL_HANDLE = None
_SPILL_PATH: Path | None = None


class SpillCandidate(np.ndarray):
    """An ndarray subclass that replaces small 2-D uint arrays with an on-disk reference."""

    def __new__(cls) -> "SpillCandidate":
        return np.ndarray.__new__(cls, (0,), dtype=np.uint8)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        global _SPILL_HANDLE, _SPILL_PATH
        version, shape, dtype, fortran, payload = state
        dtype = np.dtype(dtype)
        is_wafer = (
            len(shape) == 2 and dtype.kind in "uib" and 16 <= math.prod(shape) <= 250_000
            and isinstance(payload, (bytes, bytearray, memoryview))
        )
        if not is_wafer:
            np.ndarray.__setstate__(self, state)
            return
        if _SPILL_HANDLE is None or _SPILL_PATH is None:
            raise RuntimeError("spill writer is not initialized")
        offset = _SPILL_HANDLE.tell()
        _SPILL_HANDLE.write(payload)
        self.spill_offset = offset
        self.spill_length = len(payload)
        self.spill_shape = tuple(int(x) for x in shape)
        self.spill_dtype = dtype.str
        self.spill_fortran = bool(fortran)
        self.spill_path = str(_SPILL_PATH)

    def load(self) -> np.ndarray:
        with Path(self.spill_path).open("rb") as handle:
            handle.seek(self.spill_offset)
            payload = handle.read(self.spill_length)
        order = "F" if self.spill_fortran else "C"
        return np.frombuffer(payload, dtype=np.dtype(self.spill_dtype)).reshape(self.spill_shape, order=order).copy()


def _reconstruct_spill(*_args: Any, **_kwargs: Any) -> SpillCandidate:
    return SpillCandidate()


def _frombuffer_spill(payload: Any, dtype: Any, shape: tuple[int, ...], order: str) -> np.ndarray:
    global _SPILL_HANDLE, _SPILL_PATH
    dtype = np.dtype(dtype)
    is_wafer = (
        len(shape) == 2 and dtype.kind in "uib" and 16 <= math.prod(shape) <= 250_000
        and isinstance(payload, (bytes, bytearray, memoryview))
    )
    if not is_wafer:
        return np.frombuffer(payload, dtype=dtype).reshape(shape, order=order)
    if _SPILL_HANDLE is None or _SPILL_PATH is None:
        raise RuntimeError("spill writer is not initialized")
    offset = _SPILL_HANDLE.tell()
    _SPILL_HANDLE.write(payload)
    result = SpillCandidate()
    result.spill_offset = offset
    result.spill_length = len(payload)
    result.spill_shape = tuple(int(x) for x in shape)
    result.spill_dtype = dtype.str
    result.spill_fortran = order == "F"
    result.spill_path = str(_SPILL_PATH)
    return result


class SpillUnpickler(PandasUnpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("pandas.indexes."):
            module = module.replace("pandas.indexes.", "pandas.core.indexes.", 1)
        if name == "_reconstruct" and module in {"numpy.core.multiarray", "numpy._core.multiarray"}:
            return _reconstruct_spill
        if name == "_frombuffer" and module in {"numpy.core.numeric", "numpy._core.numeric"}:
            return _frombuffer_spill
        return super().find_class(module, name)


def load_frame_with_spill(source: Path, spill_path: Path) -> Any:
    global _SPILL_HANDLE, _SPILL_PATH
    partial = spill_path.with_suffix(spill_path.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    _SPILL_PATH = spill_path.resolve()
    try:
        with source.open("rb") as source_handle, partial.open("wb") as spill_handle:
            _SPILL_HANDLE = spill_handle
            frame = SpillUnpickler(source_handle).load()
        partial.replace(spill_path)
        return frame
    finally:
        _SPILL_HANDLE = None


def materialize(value: Any) -> np.ndarray:
    if isinstance(value, SpillCandidate) and hasattr(value, "spill_offset"):
        return value.load()
    return np.asarray(value)


def build(args: argparse.Namespace) -> None:
    source = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    spill_path = output / "wafer_arrays.bin"
    print(json.dumps({"stage": "unpickle_and_spill", "source": str(source), "spill": str(spill_path)}))
    frame = load_frame_with_spill(source, spill_path)
    required = {"waferMap", "lotName", "waferIndex", "failureType"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    labels = pd.Series(normalize_label_series(frame["failureType"]), index=frame.index, dtype=object)
    indices = choose_indices(labels, args.per_class, args.none_limit, args.seed)
    selected = frame.loc[indices, list(required)].copy()
    selected["_label"] = labels.loc[indices]
    total_rows = len(frame)
    class_counts = Counter(x for x in labels if x is not None)
    del frame, labels
    gc.collect()

    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, row in tqdm(selected.iterrows(), total=len(selected), desc="Rendering wafer maps"):
        label = row["_label"]
        if label not in LABELS:
            continue
        lot = str(row["lotName"])
        sid = sample_id(lot, row["waferIndex"])
        wafer = materialize(row["waferMap"]).astype(np.uint8, copy=False)
        image_path = image_dir / f"{sid}.png"
        if not image_path.exists() or args.overwrite_images:
            matrix_to_image(wafer, args.resolution).save(image_path, "PNG", optimize=True)
        records.append({
            "sample_id": sid, "source_index": int(index), "lot_name": lot,
            "wafer_index": int(float(flatten_scalar(row["waferIndex"]))),
            "failure_type": label, "label_source": "ground_truth",
            "split": split_for_lot(lot, args.seed), "image_path": str(image_path.resolve()),
            "matrix_shape": list(wafer.shape), "features": calculate_static_features(wafer),
        })
    records.sort(key=lambda row: row["sample_id"])
    ids = [row["sample_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate sample_id detected")
    split_lots = {name: {r["lot_name"] for r in records if r["split"] == name} for name in ("train", "val", "test")}
    intersections = {
        "train_val": sorted(split_lots["train"] & split_lots["val"]),
        "train_test": sorted(split_lots["train"] & split_lots["test"]),
        "val_test": sorted(split_lots["val"] & split_lots["test"]),
    }
    if any(intersections.values()):
        raise RuntimeError(f"lot leakage detected: {intersections}")
    write_jsonl(output / "manifest.jsonl", records)
    summary = {
        "source": str(source.resolve()), "total_rows": total_rows, "selected_samples": len(records),
        "source_class_counts": dict(sorted(class_counts.items())),
        "selected_class_counts": dict(sorted(Counter(r["failure_type"] for r in records).items())),
        "split_counts": dict(sorted(Counter(r["split"] for r in records).items())),
        "lot_split_intersections": intersections, "seed": args.seed,
    }
    (output / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=800)
    parser.add_argument("--none-limit", type=int, default=400)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--overwrite-images", action="store_true")
    return parser


def main() -> None:
    build(make_parser().parse_args())


if __name__ == "__main__":
    main()
