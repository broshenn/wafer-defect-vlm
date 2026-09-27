"""Validate teacher annotations, plan Qwen audits, and build ms-swift datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .utils import LABELS, normalize_label, read_jsonl, write_jsonl


REQUIRED_FIELDS = {
    "defect_type", "shape", "radial_zone", "clock_direction", "density",
    "continuity", "size_r", "caption_zh", "caption_en", "uncertain",
}
ROOT_CAUSE_WORDS = (
    "工艺", "设备", "污染", "光刻", "蚀刻", "沉积", "温度", "压力", "process",
    "equipment", "contamination", "lithography", "etch", "deposition",
)
SHAPES_BY_LABEL = {
    "Center": {"团簇状", "满圆"},
    "Donut": {"环形", "局部环形"},
    "Edge_Loc": {"团簇状", "弧形", "边缘"},
    "Edge_Ring": {"环形", "局部环形", "弧形", "边缘"},
    "Loc": {"团簇状"},
    "Near_full": {"满圆", "随机点"},
    "Random": {"随机点"},
    "Scratch": {"线状", "划痕状", "射线状"},
    "none": {"无缺陷"},
}
SYSTEM = "你是半导体晶圆 BIN 图缺陷分析专家，只依据图像回答，不推测工艺根因。"


def annotation_map(path: str | Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") == "ok" and isinstance(row.get("result"), dict):
            result[row["sample_id"]] = row
    return result


def circular_distance(a: int, b: int) -> int:
    return min((a - b) % 12, (b - a) % 12)


def expected_clock(features: dict[str, Any]) -> int | None:
    value = features.get("clock_sector")
    if value is None or float(features.get("resultant_length", 0)) < 0.25:
        return None
    return int(value)


def validate(record: dict[str, Any], annotation: dict[str, Any] | None) -> dict[str, Any]:
    """Return a deterministic confidence score and machine-readable issues."""
    issues: list[str] = []
    score = 100
    if not annotation:
        return {"confidence": 0, "passed": False, "issues": ["missing_annotation"]}
    result = annotation.get("result")
    if not isinstance(result, dict):
        return {"confidence": 0, "passed": False, "issues": ["invalid_result"]}
    missing = sorted(REQUIRED_FIELDS.difference(result))
    if missing:
        issues.append("missing_fields:" + ",".join(missing))
        score -= min(40, 5 * len(missing))

    gold_label = record["failure_type"]
    predicted = normalize_label(result.get("defect_type"))
    if predicted != gold_label:
        issues.append(f"type_mismatch:{predicted}!={gold_label}")
        score -= 50

    features = record.get("features", {})
    expected_zone = "none" if gold_label == "none" else features.get("radial_zone")
    zone = str(result.get("radial_zone", ""))
    if zone not in {"center", "middle", "edge", "full", "none"}:
        issues.append("invalid_radial_zone")
        score -= 20
    elif expected_zone and zone not in {expected_zone, "full" if gold_label == "Near_full" else expected_zone}:
        issues.append(f"radial_mismatch:{zone}!={expected_zone}")
        score -= 15

    shape = result.get("shape")
    if shape not in SHAPES_BY_LABEL.get(gold_label, set()):
        issues.append(f"shape_suspicious:{shape}")
        score -= 15

    clock = result.get("clock_direction")
    clock_gold = expected_clock(features)
    if clock_gold is not None and clock not in {"all", "none", None}:
        try:
            if circular_distance(int(clock), clock_gold) > 1:
                issues.append(f"clock_mismatch:{clock}!={clock_gold}")
                score -= 15
        except (TypeError, ValueError):
            issues.append("invalid_clock")
            score -= 15

    size = result.get("size_r")
    if size is not None:
        try:
            if not 0 <= float(size) <= 2.5:
                raise ValueError
        except (TypeError, ValueError):
            issues.append("invalid_size_r")
            score -= 15

    captions = f"{result.get('caption_zh', '')} {result.get('caption_en', '')}".strip()
    if not captions:
        issues.append("empty_caption")
        score -= 35
    if any(term.lower() in captions.lower() for term in ROOT_CAUSE_WORDS):
        issues.append("root_cause_hallucination")
        score -= 25
    if bool(result.get("uncertain")):
        issues.append("teacher_uncertain")
        score -= 20

    score = max(0, min(100, score))
    hard_failure = any(x.startswith(("type_mismatch", "missing_fields", "invalid_result")) for x in issues)
    return {"confidence": score, "passed": score >= 60 and not hard_failure, "issues": issues}


def is_boundary(record: dict[str, Any]) -> bool:
    f = record.get("features", {})
    radius = f.get("centroid_radius_r")
    near_zone_boundary = radius is not None and min(abs(radius - 0.35), abs(radius - 0.72)) <= 0.05
    return bool(
        f.get("status") != "ok"
        or int(f.get("defect_count", 0)) < 8
        or int(f.get("filtered_points_removed", 0)) > 0
        or near_zone_boundary
        or 0.18 <= float(f.get("resultant_length", 0)) <= 0.32
    )


def stable_audit(sample_id: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def plan(args: argparse.Namespace) -> None:
    manifest = read_jsonl(args.manifest)
    primary = annotation_map(args.primary)
    rows: list[dict[str, Any]] = []
    audit_ids: list[str] = []
    for record in manifest:
        check = validate(record, primary.get(record["sample_id"]))
        needs_qwen = (
            not check["passed"] or is_boundary(record)
            or stable_audit(record["sample_id"], args.seed, args.audit_fraction)
        )
        rows.append({
            "sample_id": record["sample_id"], "split": record["split"],
            "failure_type": record["failure_type"], "primary_validation": check,
            "boundary": is_boundary(record), "needs_qwen": needs_qwen,
        })
        if needs_qwen:
            audit_ids.append(record["sample_id"])
    write_jsonl(args.output, rows)
    ids_path = Path(args.ids_output)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("".join(f"{sid}\n" for sid in sorted(audit_ids)), encoding="utf-8")
    counts = Counter("qwen" if r["needs_qwen"] else "primary_only" for r in rows)
    print(json.dumps({"samples": len(rows), **counts}, ensure_ascii=False, indent=2))


def core_disagreement(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for key in ("defect_type", "shape", "radial_zone"):
        if a.get(key) != b.get(key):
            mismatches.append(key)
    a_clock, b_clock = a.get("clock_direction"), b.get("clock_direction")
    if isinstance(a_clock, int) and isinstance(b_clock, int) and circular_distance(a_clock, b_clock) > 1:
        mismatches.append("clock_direction")
    return mismatches


def geometry_verdict(record: dict[str, Any], result: dict[str, Any], field: str) -> bool:
    """Say whether deterministic geometry/label facts support ``result[field]``.

    This is the independent third opinion used to break teacher ties. It never
    consults either teacher, so it cannot be biased toward the one that happened
    to be called primary.
    """
    features = record.get("features", {})
    gold_label = record["failure_type"]
    if field == "defect_type":
        return normalize_label(result.get("defect_type")) == gold_label
    if field == "radial_zone":
        if gold_label == "none":
            return result.get("radial_zone") == "none"
        expected = features.get("radial_zone")
        if gold_label == "Near_full":
            return result.get("radial_zone") in {"full", expected}
        return expected is not None and result.get("radial_zone") == expected
    if field == "shape":
        return result.get("shape") in SHAPES_BY_LABEL.get(gold_label, set())
    if field == "clock_direction":
        gold_clock = expected_clock(features)
        value = result.get("clock_direction")
        if gold_clock is None:
            return value in {"all", "none", None}
        if value is None or value in {"all", "none"}:
            return False
        try:
            return circular_distance(int(value), gold_clock) <= 1
        except (TypeError, ValueError):
            return False
    return False


def arbitrate(record: dict[str, Any], a: dict[str, Any], b: dict[str, Any], fields: list[str]) -> str | None:
    """Pick the teacher that geometry supports on *every* disputed field.

    Returns ``None`` when the disputed fields do not all point at the same
    teacher, which is the honest answer for a genuine disagreement.
    """
    a_ok = all(geometry_verdict(record, a, field) for field in fields)
    b_ok = all(geometry_verdict(record, b, field) for field in fields)
    if a_ok and not b_ok:
        return "primary"
    if b_ok and not a_ok:
        return "secondary"
    return None


STRUCTURED_FIELDS = (
    "defect_type", "shape", "radial_zone", "clock_direction", "density",
    "continuity", "size_r", "caption_zh", "caption_en",
)


def structured_target(record: dict[str, Any], result: dict[str, Any], geometry_gold: bool) -> dict[str, Any]:
    """Build the structured answer for one sample.

    ``defect_type``, ``radial_zone``, ``clock_direction`` and ``size_r`` are
    scored by the benchmark against the trusted class label and deterministic
    geometry. When ``geometry_gold`` is set those four are written from that
    gold rather than from the teacher, because a teacher guess contradicting
    geometry would train the student to miss the metric it is graded on. The
    remaining fields have no geometric gold and stay distilled from the teacher.
    """
    target = {key: result.get(key) for key in STRUCTURED_FIELDS}
    if geometry_gold:
        features = record.get("features", {})
        target["defect_type"] = record["failure_type"]
        target["radial_zone"] = features.get("radial_zone")
        target["clock_direction"] = features.get("clock_sector")
        target["size_r"] = features.get("defect_extent_r")
    return target


def swift_rows(record: dict[str, Any], result: dict[str, Any], geometry_gold: bool = False) -> list[dict[str, Any]]:
    image = record["image_path"]
    structured = structured_target(record, result, geometry_gold)
    tasks = [
        ("请描述晶圆图中的主要缺陷形态、位置和尺寸。", str(result["caption_zh"])),
        ("请判断这张晶圆图的缺陷类别，只回答类别名称。", str(record["failure_type"])),
        ("请以 JSON 输出缺陷类别、形态、径向区域、钟点方向、密度、连续性、尺寸及中英文描述。",
         json.dumps(structured, ensure_ascii=False, separators=(",", ":"))),
    ]
    return [{
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<image>\n{question}"},
            {"role": "assistant", "content": answer},
        ],
        "images": [image],
        "sample_id": record["sample_id"],
        "task": task,
    } for task, (question, answer) in zip(("caption", "classification", "structured"), tasks)]


def finalize(args: argparse.Namespace) -> None:
    manifest = read_jsonl(args.manifest)
    primary = annotation_map(args.primary)
    secondary = annotation_map(args.secondary) if args.secondary else {}
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    decisions = Counter()

    for record in manifest:
        sid = record["sample_id"]
        p, s = primary.get(sid), secondary.get(sid)
        pv, sv = validate(record, p), validate(record, s)
        chosen: dict[str, Any] | None = None
        source = None
        resolved_by = None
        disagreement: list[str] = []
        if pv["passed"]:
            if s and sv["passed"]:
                disagreement = core_disagreement(p["result"], s["result"])
            if disagreement and args.arbitrate_disagreement:
                verdict = arbitrate(record, p["result"], s["result"], disagreement)
                if verdict == "primary":
                    chosen, source, resolved_by = p, "primary", "geometry"
                elif verdict == "secondary":
                    chosen, source, resolved_by = s, "secondary", "geometry"
            if chosen is None and not disagreement:
                chosen, source = p, "primary"
        elif sv["confidence"] > 60 and sv["passed"]:
            chosen, source = s, "secondary"

        if chosen is None:
            reason = "teacher_disagreement" if disagreement else "no_valid_teacher"
            quarantine.append({
                **record, "curation_status": "quarantine", "reason": reason,
                "primary_validation": pv, "secondary_validation": sv,
                "teacher_disagreement_fields": disagreement,
            })
            decisions[reason] += 1
            continue

        curated = {
            **record, "curation_status": "accepted", "teacher_source": source,
            "teacher_model": chosen.get("model"), "teacher_prompt_version": chosen.get("prompt_version"),
            "teacher_result": chosen["result"],
            "teacher_confidence": pv["confidence"] if source == "primary" else sv["confidence"],
            "primary_validation": pv, "secondary_validation": sv if s else None,
            "disagreement_resolved_by": resolved_by,
            "teacher_disagreement_fields": disagreement,
        }
        accepted.append(curated)
        decisions[source + "+geometry" if resolved_by else source] += 1
        splits[record["split"]].extend(swift_rows(record, chosen["result"], args.geometry_gold_structure))

    output = Path(args.output_dir)
    write_jsonl(output / "accepted.jsonl", accepted)
    write_jsonl(output / "quarantine.jsonl", quarantine)
    for split, rows in splits.items():
        write_jsonl(output / "splits" / f"{split}.jsonl", rows)
    report = {
        "manifest_samples": len(manifest), "accepted": len(accepted), "quarantine": len(quarantine),
        "decisions": dict(decisions), "swift_examples": {k: len(v) for k, v in splits.items()},
    }
    (output / "curation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="score DeepSeek output and select Qwen audit IDs")
    plan_parser.add_argument("--manifest", required=True)
    plan_parser.add_argument("--primary", required=True)
    plan_parser.add_argument("--output", required=True)
    plan_parser.add_argument("--ids-output", required=True)
    plan_parser.add_argument("--audit-fraction", type=float, default=0.15)
    plan_parser.add_argument("--seed", type=int, default=3407)
    plan_parser.set_defaults(func=plan)
    final_parser = sub.add_parser("finalize", help="merge validated teachers and export ms-swift JSONL")
    final_parser.add_argument("--manifest", required=True)
    final_parser.add_argument("--primary", required=True)
    final_parser.add_argument("--secondary")
    final_parser.add_argument("--output-dir", required=True)
    final_parser.add_argument(
        "--arbitrate-disagreement", action="store_true",
        help="when both teachers pass but disagree, let deterministic geometry pick the "
             "better field values instead of quarantining the sample",
    )
    final_parser.add_argument(
        "--geometry-gold-structure", action="store_true",
        help="write defect_type/radial_zone/clock_direction/size_r in the structured "
             "target from the trusted label and deterministic geometry, matching the "
             "benchmark's own gold, instead of the teacher's guess",
    )
    final_parser.set_defaults(func=finalize)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
