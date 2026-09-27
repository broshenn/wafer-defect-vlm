"""Run the section-seven acceptance checks and write one consolidated report.

Every check is recomputed from the artefacts on disk rather than trusted from an
earlier log, so the report is evidence about the current state of the run. A
failing check is recorded as a failure and the script still writes the report;
it never upgrades a failure into a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_pytest(project: Path) -> dict:
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                          cwd=project, capture_output=True, text=True)
    tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    return {"passed": proc.returncode == 0, "summary": tail[0][:200], "returncode": proc.returncode}


def renderer_palette() -> set[tuple[int, int, int]]:
    """Read the palette off the renderer rather than restating it.

    A hardcoded copy here would silently stop matching the moment the renderer
    changes, and the check would then fail every image instead of the ones that
    are actually wrong.
    """
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects" / "wafer-defect-vlm" / "src"))
    from wafer_vlm.utils import matrix_to_image

    probe = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    arr = np.asarray(matrix_to_image(probe, resolution=2).convert("RGB"))
    return {(int(r), int(g), int(b)) for r, g, b in arr.reshape(-1, 3)}


def check_manifest(manifest: Path, palette: set[tuple[int, int, int]], sample: int) -> dict:
    import numpy as np
    from PIL import Image

    rows = read_jsonl(manifest)
    ids = [r["sample_id"] for r in rows]
    splits = Counter(r["split"] for r in rows)
    missing = [r["sample_id"] for r in rows if not Path(r["image_path"]).is_file()]
    bad_palette, bad_size = [], []
    for row in rows[:sample]:
        with Image.open(row["image_path"]) as handle:
            arr = np.asarray(handle.convert("RGB"))
        if arr.shape[:2] != (448, 448):
            bad_size.append(row["sample_id"])
        packed = ((arr[:, :, 0].astype(np.uint32) << 16)
                  | (arr[:, :, 1].astype(np.uint32) << 8)
                  | arr[:, :, 2].astype(np.uint32))
        if not set(np.unique(packed).tolist()) <= {p[0] << 16 | p[1] << 8 | p[2] for p in palette}:
            bad_palette.append(row["sample_id"])
    return {
        "records": len(rows),
        "unique_sample_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "splits": dict(splits),
        "class_counts": dict(Counter(r["failure_type"] for r in rows)),
        "missing_images": missing[:10],
        "missing_image_count": len(missing),
        "images_checked": min(sample, len(rows)),
        "wrong_size": bad_size[:10],
        "off_palette": bad_palette[:10],
        "passed": not missing and not bad_size and not bad_palette and len(ids) == len(set(ids)),
    }


def check_lot_isolation(manifest: Path, seed: int) -> dict:
    """Confirm the split labels obey the lot rule and that lots stay whole.

    Two separate things are checked, because they fail differently: a row whose
    split disagrees with the rule means the file was edited or built by other
    code, while a lot spanning two splits means near-duplicate wafers straddle
    the boundary and every split-derived number is inflated.

    The rule itself is imported from the project so this check cannot drift from
    the code that assigned the splits.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects" / "wafer-defect-vlm" / "src"))
    from wafer_vlm.utils import split_for_lot

    rows = read_jsonl(manifest)
    lots: dict[str, set[str]] = {}
    rule_mismatch = []
    for row in rows:
        lots.setdefault(row["lot_name"], set()).add(row["split"])
        if split_for_lot(row["lot_name"], seed) != row["split"]:
            rule_mismatch.append(row["sample_id"])
    straddling = {lot: sorted(v) for lot, v in lots.items() if len(v) > 1}
    return {
        "lots": len(lots),
        "rows_disagreeing_with_the_rule": rule_mismatch[:10],
        "rule_mismatch_count": len(rule_mismatch),
        "lots_straddling_splits": dict(list(straddling.items())[:10]),
        "straddling_count": len(straddling),
        "passed": not straddling and not rule_mismatch,
        "note": "a lot appearing in two splits would leak near-duplicate wafers across the split boundary",
    }


def check_benchmark(bench: Path) -> dict:
    sums = bench / "SHA256SUMS"
    if not sums.is_file():
        return {"passed": False, "reason": "SHA256SUMS missing"}
    mismatches, checked = [], 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        path = bench / name
        checked += 1
        if not path.is_file() or sha256(path) != digest:
            mismatches.append(name)
    metadata = json.loads((bench / "metadata.json").read_text(encoding="utf-8"))
    review_status = json.loads((bench / "review" / "status.json").read_text(encoding="utf-8")) \
        if (bench / "review" / "status.json").is_file() else {}
    return {
        "files_checked": checked,
        "mismatches": mismatches[:10],
        "metadata_status": metadata.get("status"),
        "review_gate": metadata.get("review_gate"),
        "core_samples": metadata.get("core_samples"),
        "human_review_status": review_status.get("status", "not_started"),
        "passed": not mismatches,
    }


def check_model(model: Path) -> dict:
    index = model / "model.safetensors.index.json"
    shards = sorted(model.glob("*.safetensors"))
    incomplete = [p.name for p in model.rglob("*.incomplete")]
    result = {
        "shards": len(shards),
        "incomplete_files": incomplete,
        "has_index": index.is_file(),
        "has_config": (model / "config.json").is_file(),
    }
    if index.is_file():
        wanted = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        present = {p.name for p in shards}
        result["missing_shards"] = sorted(wanted - present)
        result["passed"] = not result["missing_shards"] and not incomplete
    else:
        result["passed"] = False
    return result


def check_curated(curated: Path) -> dict:
    train = read_jsonl(curated / "splits" / "train.jsonl")
    val = read_jsonl(curated / "splits" / "val.jsonl")
    report = json.loads((curated / "curation_report.json").read_text(encoding="utf-8"))
    train_ids = {r["sample_id"] for r in train}
    val_ids = {r["sample_id"] for r in val}
    return {
        "train_examples": len(train),
        "val_examples": len(val),
        "overlap_train_val": sorted(train_ids & val_ids)[:10],
        "tasks_per_example": dict(Counter(r["task"] for r in train)),
        "accepted_samples": report.get("accepted"),
        "quarantine_samples": report.get("quarantine"),
        "decisions": report.get("decisions"),
        "passed": not (train_ids & val_ids) and bool(train) and bool(val),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--palette-sample", type=int, default=600)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    root, project = Path(args.root), Path(args.project)
    palette = renderer_palette()
    checks = {
        "pytest": check_pytest(project),
        "manifest": check_manifest(root / "data/prepared_v1/manifest.jsonl", palette, args.palette_sample),
        "lot_isolation": check_lot_isolation(root / "data/prepared_v1/manifest.jsonl", args.seed),
        "benchmark": check_benchmark(root / "benchmarks/wafer_bench_v1"),
        "model_shards": check_model(root / "models/Qwen3.5-9B"),
        "curated": check_curated(root / "data/curated_v2"),
    }
    artifacts = {}
    for rel in ("data/raw/wm811k-wafer-map.zip",
                "data/raw/wm811k/LSWMD.pkl",
                "data/prepared_v1/manifest.jsonl",
                "data/curated_v2/splits/train.jsonl",
                "data/curated_v2/splits/val.jsonl",
                "benchmarks/wafer_bench_v1/all_requests.jsonl"):
        path = root / rel
        artifacts[rel] = {"exists": path.is_file(),
                          "sha256": sha256(path) if path.is_file() else None,
                          "bytes": path.stat().st_size if path.is_file() else None}

    report = {
        "checks": checks,
        "all_passed": all(c.get("passed") for c in checks.values()),
        "artifacts": artifacts,
        "note": "benchmark stays draft until two humans complete benchmarks/wafer_bench_v1/review; "
                "that step is not simulated here",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v.get("passed") for k, v in checks.items()}, ensure_ascii=False))
    print("ALL PASSED" if report["all_passed"] else "SOME CHECKS FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
