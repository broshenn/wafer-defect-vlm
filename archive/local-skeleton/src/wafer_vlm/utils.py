"""Core wafer-map utilities shared by data, annotation and benchmark stages."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy import ndimage


LABELS = (
    "Center",
    "Donut",
    "Edge_Loc",
    "Edge_Ring",
    "Loc",
    "Near_full",
    "Random",
    "Scratch",
    "none",
)

LABEL_ALIASES = {
    "edge-loc": "Edge_Loc",
    "edge_loc": "Edge_Loc",
    "edgeloc": "Edge_Loc",
    "edge-ring": "Edge_Ring",
    "edge_ring": "Edge_Ring",
    "edgering": "Edge_Ring",
    "near-full": "Near_full",
    "near_full": "Near_full",
    "nearfull": "Near_full",
    "center": "Center",
    "donut": "Donut",
    "loc": "Loc",
    "random": "Random",
    "scratch": "Scratch",
    "none": "none",
    "normal": "none",
}


def flatten_scalar(value: Any) -> Any:
    """Return the first scalar contained in nested list/array containers."""
    while isinstance(value, (list, tuple, np.ndarray)):
        if np.asarray(value, dtype=object).size == 0:
            return None
        value = np.asarray(value, dtype=object).reshape(-1)[0]
    return value


def normalize_label(value: Any) -> str | None:
    value = flatten_scalar(value)
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "unknown", "[]", "none"}:
        return "none" if text.lower() == "none" else None
    return LABEL_ALIASES.get(text.lower().replace(" ", ""), text.replace("-", "_"))


def normalize_label_series(values: Iterable[Any]) -> list[str | None]:
    """Apply :func:`normalize_label` to a column, preserving ``None``.

    ``pandas.Series.map`` infers a string dtype on pandas 3.x and rewrites the
    ``None`` it gets for unlabelled rows as ``NaN``, so both ``is None`` and
    ``==`` checks downstream see a float where the caller expects a missing
    label. Building a plain list keeps the distinction between "labelled",
    "explicitly none" and "unlabelled" that the rest of the pipeline relies on.
    """
    return [normalize_label(value) for value in values]


def stable_bucket(group: str, seed: int = 3407, modulo: int = 100) -> int:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def split_for_lot(lot_name: str, seed: int = 3407) -> str:
    bucket = stable_bucket(lot_name, seed)
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def sample_id(lot_name: str, wafer_index: Any) -> str:
    lot_match = re.search(r"(\d+)", str(lot_name))
    lot_part = int(lot_match.group(1)) if lot_match else stable_bucket(str(lot_name), 0, 10**8)
    try:
        wafer_part = int(float(flatten_scalar(wafer_index)))
    except (TypeError, ValueError):
        wafer_part = 0
    return f"wafer_{lot_part:08d}_{wafer_part:03d}"


def isolated_point_filter(wafer: np.ndarray) -> np.ndarray:
    """Replace defect pixels without any 8-neighbour defect with good-die value 1."""
    arr = np.asarray(wafer, dtype=np.uint8).copy()
    defects = arr == 2
    neighbours = ndimage.convolve(defects.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
    arr[defects & (neighbours <= 1)] = 1
    return arr


def fit_wafer_circle(wafer: np.ndarray) -> tuple[float, float, float]:
    """Least-squares circle fit on the boundary of the valid wafer mask."""
    mask = np.asarray(wafer) > 0
    if not mask.any():
        h, w = mask.shape
        return (w - 1) / 2, (h - 1) / 2, max(min(h, w) / 2, 1.0)
    boundary = mask & ~ndimage.binary_erosion(mask)
    rows, cols = np.nonzero(boundary)
    if len(rows) < 3:
        rows, cols = np.nonzero(mask)
    a = np.column_stack([cols, rows, np.ones_like(cols)])
    b = cols.astype(float) ** 2 + rows.astype(float) ** 2
    try:
        coef, *_ = np.linalg.lstsq(a, b, rcond=None)
        cx, cy = coef[0] / 2, coef[1] / 2
        radius = math.sqrt(max(coef[2] + cx * cx + cy * cy, 1.0))
    except (ValueError, np.linalg.LinAlgError):
        cx, cy = cols.mean(), rows.mean()
        radius = float(np.max(np.hypot(cols - cx, rows - cy)))
    return float(cx), float(cy), max(float(radius), 1.0)


def _zone(radius_ratio: float) -> str:
    if radius_ratio < 0.35:
        return "center"
    if radius_ratio < 0.72:
        return "middle"
    return "edge"


def calculate_static_features(wafer: np.ndarray, apply_filter: bool = True) -> dict[str, Any]:
    """Calculate size-independent facts used for curation and deterministic evaluation."""
    raw = np.asarray(wafer, dtype=np.uint8)
    arr = isolated_point_filter(raw) if apply_filter else raw
    valid = arr > 0
    defects = arr == 2
    raw_defect_count = int(np.count_nonzero(raw == 2))
    valid_count = int(valid.sum())
    defect_count = int(defects.sum())
    cx, cy, radius = fit_wafer_circle(raw)
    result: dict[str, Any] = {
        "status": "ok" if defect_count else ("filtered_empty" if np.any(raw == 2) else "no_defect"),
        "valid_count": valid_count,
        "raw_defect_count": raw_defect_count,
        "defect_count": defect_count,
        "filtered_points_removed": raw_defect_count - defect_count,
        "defect_ratio": defect_count / max(valid_count, 1),
        "wafer_center_xy": [cx, cy],
        "wafer_radius": radius,
    }
    if not defect_count:
        result.update({
            "centroid_xy_r": None,
            "centroid_radius_r": None,
            "radial_zone": "none",
            "clock_sector": None,
            "mean_radius_r": None,
            "std_radius_r": None,
            "defect_extent_r": None,
            "radial_span_r": None,
            "radial_density": [0.0] * 10,
            "angular_probability": [0.0] * 12,
            "resultant_length": 0.0,
            "angle_entropy_bits": 0.0,
            "anisotropy_min_over_max": None,
        })
        return result

    rows, cols = np.nonzero(defects)
    x = (cols.astype(float) - cx) / radius
    y = (cy - rows.astype(float)) / radius  # positive y is 12 o'clock
    radial = np.hypot(x, y)
    centroid_x, centroid_y = float(x.mean()), float(y.mean())
    centroid_r = float(math.hypot(centroid_x, centroid_y))
    angle = np.mod(np.arctan2(x, y), 2 * np.pi)  # clockwise from 12 o'clock

    radial_counts, radial_edges = np.histogram(radial, bins=10, range=(0.0, 1.0))
    annulus_areas = np.pi * (radial_edges[1:] ** 2 - radial_edges[:-1] ** 2)
    radial_density = radial_counts / np.maximum(annulus_areas, 1e-12)
    if radial_density.sum() > 0:
        radial_density = radial_density / radial_density.sum()

    angular_counts, _ = np.histogram(angle, bins=12, range=(0.0, 2 * np.pi))
    angular_prob = angular_counts / max(angular_counts.sum(), 1)
    mean_sin = float(np.sin(angle).mean())
    mean_cos = float(np.cos(angle).mean())
    resultant = float(math.hypot(mean_sin, mean_cos))
    nonzero = angular_prob[angular_prob > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    clock_sector = int(np.argmax(angular_counts)) + 1

    points = np.column_stack([x, y])
    if len(points) >= 2:
        eig = np.linalg.eigvalsh(np.cov(points.T))
        anisotropy = float(max(eig[0], 0.0) / max(eig[-1], 1e-12))
    else:
        anisotropy = 0.0

    result.update({
        "centroid_xy_r": [centroid_x, centroid_y],
        "centroid_radius_r": centroid_r,
        "radial_zone": _zone(centroid_r),
        "clock_sector": clock_sector,
        "mean_radius_r": float(radial.mean()),
        "std_radius_r": float(radial.std()),
        "defect_extent_r": float(max(np.ptp(x), np.ptp(y))) if len(x) > 1 else 0.0,
        "radial_span_r": float(np.ptp(radial)) if len(radial) > 1 else 0.0,
        "radial_density": radial_density.round(8).tolist(),
        "angular_probability": angular_prob.round(8).tolist(),
        "resultant_length": resultant,
        "angle_entropy_bits": entropy,
        "anisotropy_min_over_max": anisotropy,
    })
    return result


def matrix_to_image(wafer: np.ndarray, resolution: int = 448) -> Image.Image:
    arr = np.asarray(wafer, dtype=np.uint8)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    rgb[arr == 1] = (0, 255, 0)
    rgb[arr == 2] = (255, 0, 0)
    return Image.fromarray(rgb, "RGB").resize((resolution, resolution), Image.Resampling.NEAREST)


def image_data_url(path: str | Path) -> str:
    path = Path(path)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON object found in model output")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(target)
