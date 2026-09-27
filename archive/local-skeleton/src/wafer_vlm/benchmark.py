"""Build a leakage-safe, reviewable wafer benchmark and retrieval qrels."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .utils import LABELS, read_jsonl, write_jsonl


SHAPE_TERMS = {
    "Center": ["中心", "团簇"], "Donut": ["环", "圆环"],
    "Edge_Loc": ["边缘", "局部"], "Edge_Ring": ["边缘", "环"],
    "Loc": ["局部", "团簇"], "Near_full": ["满圆", "大面积"],
    "Random": ["随机", "散点"], "Scratch": ["线", "划痕"], "none": ["无缺陷", "正常"],
}
FORBIDDEN_TERMS = ["污染", "光刻", "蚀刻", "设备故障", "工艺原因", "contamination", "lithography"]


def stable_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def difficulty(record: dict[str, Any]) -> float:
    f = record.get("features", {})
    score = 0.0
    if f.get("status") != "ok":
        score += 3
    if int(f.get("defect_count", 0)) < 10:
        score += 2
    score += min(int(f.get("filtered_points_removed", 0)), 5) / 5
    radius = f.get("centroid_radius_r")
    if radius is not None and min(abs(radius - 0.35), abs(radius - 0.72)) <= 0.05:
        score += 2
    resultant = float(f.get("resultant_length", 0))
    if 0.18 <= resultant <= 0.32:
        score += 1
    if record["failure_type"] in {"Edge_Loc", "Edge_Ring", "Loc", "Random"}:
        score += 0.5
    return score


def balanced_core(records: list[dict[str, Any]], per_class: int, hard_fraction: float, seed: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for label in LABELS:
        group = [r for r in records if r["failure_type"] == label]
        group.sort(key=lambda r: (difficulty(r), stable_key(seed, r["sample_id"])))
        limit = min(per_class, len(group))
        hard_n = min(round(limit * hard_fraction), limit)
        typical_n = limit - hard_n
        typical = group[:typical_n]
        hard = group[-hard_n:] if hard_n else []
        seen = {r["sample_id"] for r in typical}
        selected = typical + [r for r in hard if r["sample_id"] not in seen]
        if len(selected) < limit:
            selected.extend(r for r in group if r["sample_id"] not in {x["sample_id"] for x in selected})
        chosen.extend(selected[:limit])
    return sorted(chosen, key=lambda r: r["sample_id"])


def clock_close(a: Any, b: Any) -> bool:
    if not isinstance(a, int) or not isinstance(b, int):
        return a == b
    return min((a - b) % 12, (b - a) % 12) <= 1


def graded_relevance(query: dict[str, Any], candidate: dict[str, Any]) -> int:
    if query["failure_type"] != candidate["failure_type"]:
        return 0
    qf, cf = query.get("features", {}), candidate.get("features", {})
    fine = qf.get("radial_zone") == cf.get("radial_zone")
    if qf.get("clock_sector") is not None and cf.get("clock_sector") is not None:
        fine = fine and clock_close(qf["clock_sector"], cf["clock_sector"])
    return 2 if fine else 1


def recolor(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).copy()
    green = (arr[:, :, 1] > arr[:, :, 0]) & (arr[:, :, 1] > arr[:, :, 2])
    red = (arr[:, :, 0] > arr[:, :, 1]) & (arr[:, :, 0] > arr[:, :, 2])
    arr[green] = (0, 180, 30)
    arr[red] = (255, 80, 80)
    return Image.fromarray(arr)


def rotate_clock(clock: int | None, quarter_turns_ccw: int) -> int | None:
    if clock is None:
        return None
    return ((int(clock) - 1 - 3 * quarter_turns_ccw) % 12) + 1


def make_robustness(core: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    image_dir = output / "robustness_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in core:
        source = Image.open(record["image_path"]).convert("RGB")
        variants = [
            ("resize_224", source.resize((224, 224), Image.Resampling.NEAREST), 0),
            ("recolor_v1", recolor(source), 0),
            ("rotate_90_ccw", source.rotate(90, resample=Image.Resampling.NEAREST, expand=False), 1),
        ]
        for name, image, turns in variants:
            path = image_dir / f"{record['sample_id']}__{name}.png"
            image.save(path, "PNG", optimize=True)
            features = dict(record.get("features", {}))
            features["clock_sector"] = rotate_clock(features.get("clock_sector"), turns)
            rows.append({
                "variant_id": f"{record['sample_id']}__{name}", "sample_id": record["sample_id"],
                "perturbation": name, "image_path": str(path.resolve()),
                "failure_type": record["failure_type"], "features": features,
                "gold_transform": {"class_unchanged": True, "clock_quarter_turns_ccw": turns},
            })
    return rows


def build(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} is not empty; use --overwrite for a draft rebuild")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    records = [r for r in read_jsonl(args.manifest) if r.get("split") == "test"]
    core = balanced_core(records, args.per_class, args.hard_fraction, args.seed)
    benchmark_rows: list[dict[str, Any]] = []
    for row in core:
        benchmark_rows.append({
            **row, "difficulty_score": difficulty(row),
            "gold_source": "ground_truth_label+deterministic_geometry",
            "annotator_count": 0, "agreement": "pending_human_review",
            "caption_rubric": {
                "must_hit_any": SHAPE_TERMS[row["failure_type"]],
                "must_avoid": FORBIDDEN_TERMS,
                "check_radial_zone": row.get("features", {}).get("radial_zone"),
                "check_clock_sector": row.get("features", {}).get("clock_sector"),
            },
        })
    write_jsonl(output / "core.jsonl", benchmark_rows)
    write_jsonl(output / "classification.jsonl", [{
        "sample_id": r["sample_id"], "image_path": r["image_path"], "gold_label": r["failure_type"],
    } for r in benchmark_rows])
    write_jsonl(output / "structured.jsonl", [{
        "sample_id": r["sample_id"], "image_path": r["image_path"], "gold": {
            "defect_type": r["failure_type"], "radial_zone": r["features"].get("radial_zone"),
            "clock_sector": r["features"].get("clock_sector"),
            "defect_ratio": r["features"].get("defect_ratio"),
            "centroid_radius_r": r["features"].get("centroid_radius_r"),
            "size_r": r["features"].get("defect_extent_r"),
        },
    } for r in benchmark_rows])
    write_jsonl(output / "caption.jsonl", [{
        "sample_id": r["sample_id"], "image_path": r["image_path"], "rubric": r["caption_rubric"],
    } for r in benchmark_rows])
    requests: list[dict[str, Any]] = []
    labels = "|".join(LABELS)
    structured_prompt = (
        "请只输出一个JSON对象，不要输出Markdown代码块或任何其他文字。字段与取值："
        f"defect_type取{labels}之一；"
        "shape取环形|局部环形|团簇状|线状|弧形|射线状|满圆|边缘|划痕状|扇形|随机点之一；"
        "radial_zone取center|middle|edge|full|none之一；"
        "clock_direction取1-12的整数、all或none；"
        "density取致密|稀疏之一；"
        "continuity取连续|断续|不适用之一；"
        "size_r为以晶圆半径R为单位的数字或null。"
    )
    request_prompts = {
        "classification": (
            "请判断这张晶圆图的缺陷类别。只能从以下类别中选择一个，只回答类别名称，"
            f"不要输出其他内容：{labels}"
        ),
        "caption": "请客观描述主要缺陷的形态、位置和相对晶圆半径R的尺寸。",
        "structured": structured_prompt,
    }
    for row in benchmark_rows:
        for task, prompt in request_prompts.items():
            requests.append({
                "messages": [{"role": "user", "content": f"<image>\n{prompt}"}],
                "images": [row["image_path"]], "sample_id": row["sample_id"], "task": task,
            })
    write_jsonl(output / "inference_requests.jsonl", requests)
    robustness_rows = make_robustness(core, output)
    write_jsonl(output / "robustness.jsonl", robustness_rows)
    write_jsonl(output / "robustness_inference_requests.jsonl", [{
        "messages": [{"role": "user", "content": f"<image>\n{request_prompts['classification']}"}],
        "images": [r["image_path"]], "sample_id": r["variant_id"], "task": "robustness_classification",
    } for r in robustness_rows])

    query_rows: list[dict[str, Any]] = []
    for label in LABELS:
        group = [r for r in core if r["failure_type"] == label]
        group.sort(key=lambda r: stable_key(args.seed + 1, r["sample_id"]))
        query_rows.extend(group[:args.queries_per_class])
    write_jsonl(output / "retrieval_queries.jsonl", [{
        "query_id": r["sample_id"], "image_path": r["image_path"], "failure_type": r["failure_type"],
        "lot_name": r["lot_name"], "features": r["features"],
    } for r in query_rows])
    write_jsonl(output / "retrieval_gallery.jsonl", [{
        "gallery_id": r["sample_id"], "image_path": r["image_path"], "failure_type": r["failure_type"],
        "lot_name": r["lot_name"], "features": r["features"],
    } for r in core])
    qrels: list[dict[str, Any]] = []
    for query in query_rows:
        for candidate in core:
            if candidate["sample_id"] == query["sample_id"] or candidate["lot_name"] == query["lot_name"]:
                continue
            qrels.append({
                "query_id": query["sample_id"], "gallery_id": candidate["sample_id"],
                "relevance": graded_relevance(query, candidate),
            })
    write_jsonl(output / "retrieval_qrels.jsonl", qrels)

    counts = Counter(r["failure_type"] for r in core)
    metadata = {
        "name": "wafer_bench_v1", "status": "draft_pending_human_review", "seed": args.seed,
        "source_manifest": str(Path(args.manifest).resolve()), "test_candidates": len(records),
        "core_samples": len(core), "class_counts": {label: counts[label] for label in LABELS},
        "queries": len(query_rows), "qrels": len(qrels),
        "leakage_rule": "benchmark uses test split only; retrieval relevance excludes same-lot pairs",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def freeze(args: argparse.Namespace) -> None:
    output = Path(args.output)
    core = read_jsonl(output / "core.jsonl")
    pending = [r["sample_id"] for r in core if r.get("annotator_count", 0) < 2 or r.get("agreement") != "approved"]
    if pending and not args.allow_pending:
        raise RuntimeError(f"Cannot freeze: {len(pending)} samples still need two-person approval")
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # The checksums are worth pinning either way, so the two outcomes are
    # recorded separately: only a fully reviewed benchmark may call itself
    # frozen. When the gate is bypassed the status keeps saying "draft" and the
    # bypass is named, so the string cannot be quoted out of context as evidence
    # that a human review happened.
    if pending:
        metadata["status"] = "draft_pending_human_review"
        metadata["hashes_pinned"] = True
        metadata["review_gate"] = f"bypassed_by_allow_pending ({len(pending)} samples unapproved)"
    else:
        metadata["status"] = "frozen"
        metadata["hashes_pinned"] = True
        metadata["review_gate"] = "satisfied"
    metadata["pending_review_count"] = len(pending)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    files = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    checksums = []
    for path in files:
        checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}")
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps({"files_hashed": len(files), "pending": len(pending)}, ensure_ascii=False))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--manifest", required=True)
    builder.add_argument("--output", required=True)
    builder.add_argument("--per-class", type=int, default=30)
    builder.add_argument("--queries-per-class", type=int, default=10)
    builder.add_argument("--hard-fraction", type=float, default=0.35)
    builder.add_argument("--seed", type=int, default=3407)
    builder.add_argument("--overwrite", action="store_true")
    builder.set_defaults(func=build)
    freezer = sub.add_parser("freeze")
    freezer.add_argument("--output", required=True)
    freezer.add_argument("--allow-pending", action="store_true")
    freezer.set_defaults(func=freeze)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
