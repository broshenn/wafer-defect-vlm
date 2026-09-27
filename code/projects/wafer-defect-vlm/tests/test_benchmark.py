import argparse
import json

from PIL import Image

import pytest

from wafer_vlm.benchmark import build, freeze, rotate_clock
from wafer_vlm.utils import LABELS, read_jsonl, write_jsonl


def test_rotation_updates_clock_direction() -> None:
    assert rotate_clock(3, 1) == 12
    assert rotate_clock(12, 1) == 9
    assert rotate_clock(None, 1) is None


def test_benchmark_excludes_same_lot_retrieval_pairs(tmp_path) -> None:
    image = tmp_path / "wafer.png"
    Image.new("RGB", (16, 16), (0, 255, 0)).save(image)
    records = []
    for label_index, label in enumerate(LABELS):
        for item in range(2):
            records.append({
                "sample_id": f"s{label_index}_{item}", "split": "test", "failure_type": label,
                "lot_name": f"lot_{label_index}_{item}", "image_path": str(image),
                "features": {
                    "status": "no_defect" if label == "none" else "ok", "defect_count": 20,
                    "filtered_points_removed": 0, "centroid_radius_r": 0.2,
                    "radial_zone": "none" if label == "none" else "center",
                    "clock_sector": None if label == "none" else 1, "resultant_length": 0.5,
                    "defect_ratio": 0.1, "defect_extent_r": 0.3,
                },
            })
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, records)
    output = tmp_path / "bench"
    build(argparse.Namespace(
        manifest=str(manifest), output=str(output), per_class=2, queries_per_class=1,
        hard_fraction=0.5, seed=3407, overwrite=False,
    ))
    queries = {r["query_id"]: r for r in read_jsonl(output / "retrieval_queries.jsonl")}
    gallery = {r["gallery_id"]: r for r in read_jsonl(output / "retrieval_gallery.jsonl")}
    qrels = read_jsonl(output / "retrieval_qrels.jsonl")
    assert len(read_jsonl(output / "core.jsonl")) == 2 * len(LABELS)
    assert all(queries[r["query_id"]]["lot_name"] != gallery[r["gallery_id"]]["lot_name"] for r in qrels)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "draft_pending_human_review"


def _built_benchmark(tmp_path):
    image = tmp_path / "wafer.png"
    Image.new("RGB", (16, 16), (0, 255, 0)).save(image)
    records = [{
        "sample_id": f"s{i}", "split": "test", "failure_type": LABELS[i % len(LABELS)],
        "lot_name": f"lot_{i}", "image_path": str(image),
        "features": {"status": "ok", "defect_count": 20, "filtered_points_removed": 0,
                     "centroid_radius_r": 0.2, "radial_zone": "center", "clock_sector": 1,
                     "resultant_length": 0.5, "defect_ratio": 0.1, "defect_extent_r": 0.3},
    } for i in range(len(LABELS))]
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, records)
    output = tmp_path / "bench"
    build(argparse.Namespace(
        manifest=str(manifest), output=str(output), per_class=1, queries_per_class=1,
        hard_fraction=0.0, seed=3407, overwrite=False,
    ))
    return output


def test_freeze_refuses_while_review_is_outstanding(tmp_path) -> None:
    output = _built_benchmark(tmp_path)
    with pytest.raises(RuntimeError, match="need two-person approval"):
        freeze(argparse.Namespace(output=str(output), allow_pending=False))


def test_freeze_with_allow_pending_still_reports_a_draft(tmp_path) -> None:
    # Pinning hashes is legitimate; claiming the review happened is not. The
    # status must keep saying draft and name the bypass.
    output = _built_benchmark(tmp_path)
    freeze(argparse.Namespace(output=str(output), allow_pending=True))
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "draft_pending_human_review"
    assert metadata["status"] != "frozen"
    assert "bypassed_by_allow_pending" in metadata["review_gate"]
    assert metadata["pending_review_count"] == len(LABELS)
    assert metadata["hashes_pinned"] is True
    assert (output / "SHA256SUMS").is_file()
