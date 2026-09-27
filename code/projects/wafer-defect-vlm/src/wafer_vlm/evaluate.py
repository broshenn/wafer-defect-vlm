"""Score ms-swift generations against wafer_bench_v1 deterministic gold fields."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .utils import LABELS, extract_json_object, normalize_label, read_jsonl


_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)


def strip_thinking(text: str) -> str:
    """Drop the reasoning block ms-swift leaves in front of the answer.

    The Qwen3.5 chat template opens a ``<think>`` block even when the answer is
    non-thinking; leaving it in makes ``normalize_label`` see "<think></think>
    Scratch" and score every classification as invalid.
    """
    return _THINK_BLOCK.sub("", text).strip()


def response_text(row: dict[str, Any]) -> str:
    return strip_thinking(_raw_response_text(row))


def _raw_response_text(row: dict[str, Any]) -> str:
    for key in ("response", "prediction", "output", "generated_text"):
        if isinstance(row.get(key), str):
            return row[key]
    choices = row.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        if isinstance(message.get("content"), str):
            return message["content"]
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                return message["content"]
    return ""


def align_predictions(requests: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(row.get("sample_id")), str(row.get("task"))): row
        for row in predictions if row.get("sample_id") is not None and row.get("task") is not None
    }
    aligned: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        key = (str(request.get("sample_id")), str(request.get("task")))
        prediction = by_key.get(key)
        if prediction is None and index < len(predictions):
            prediction = predictions[index]
        aligned.append({**request, "response": response_text(prediction or {})})
    return aligned


def bootstrap_ci(
    gold: list[Any], predicted: list[Any], metric: Callable[[list[Any], list[Any]], float],
    seed: int = 3407, samples: int = 500,
) -> list[float]:
    if not gold:
        return [math.nan, math.nan]
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        indices = [rng.randrange(len(gold)) for _ in gold]
        values.append(metric([gold[i] for i in indices], [predicted[i] for i in indices]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def predicted_label(response: str) -> str:
    """Map a free-text answer onto a benchmark label, or INVALID if off-vocabulary.

    ``normalize_label`` passes unrecognised text straight through, so a reply
    like "Particle" would otherwise be scored as an ordinary wrong answer and
    the invalid-rate would read as zero even when the model ignores the
    requested vocabulary.
    """
    normalized = normalize_label(response)
    return normalized if normalized in LABELS else "INVALID"


def classification_metrics(rows: list[dict[str, Any]], gold_map: dict[str, str]) -> dict[str, Any]:
    gold, predicted = [], []
    for row in rows:
        if row["task"] != "classification":
            continue
        gold.append(gold_map[row["sample_id"]])
        predicted.append(predicted_label(row["response"]))
    macro = lambda y, p: float(f1_score(y, p, labels=list(LABELS), average="macro", zero_division=0))
    report = classification_report(gold, predicted, labels=list(LABELS), output_dict=True, zero_division=0)
    return {
        "samples": len(gold), "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": macro(gold, predicted), "macro_f1_95ci": bootstrap_ci(gold, predicted, macro),
        "per_class": {label: report[label] for label in LABELS},
        "invalid_predictions": sum(p == "INVALID" for p in predicted),
    }


def clock_error(predicted: Any, gold: Any) -> float | None:
    try:
        p, g = int(predicted), int(gold)
    except (TypeError, ValueError):
        return None
    return float(min((p - g) % 12, (g - p) % 12))


def structured_metrics(rows: list[dict[str, Any]], gold_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    totals = defaultdict(int)
    correct = defaultdict(int)
    clock_errors: list[float] = []
    size_errors: list[float] = []
    parse_errors = 0
    for row in rows:
        if row["task"] != "structured":
            continue
        try:
            predicted = extract_json_object(row["response"])
        except ValueError:
            parse_errors += 1
            continue
        gold = gold_map[row["sample_id"]]
        for field in ("defect_type", "radial_zone"):
            totals[field] += 1
            pvalue = normalize_label(predicted.get(field)) if field == "defect_type" else predicted.get(field)
            if pvalue == gold.get(field):
                correct[field] += 1
        error = clock_error(predicted.get("clock_direction"), gold.get("clock_sector"))
        if error is not None:
            clock_errors.append(error)
        if predicted.get("size_r") is not None and gold.get("size_r") is not None:
            try:
                size_errors.append(abs(float(predicted["size_r"]) - float(gold["size_r"])))
            except (TypeError, ValueError):
                pass
    return {
        "parse_errors": parse_errors,
        "field_accuracy": {field: correct[field] / max(totals[field], 1) for field in totals},
        "clock_circular_mae_sectors": float(np.mean(clock_errors)) if clock_errors else None,
        "size_mae_r": float(np.mean(size_errors)) if size_errors else None,
        "size_note": "automatic gold is the defect bounding extent/R; ring width and multi-cluster cases require human review",
    }


def caption_metrics(rows: list[dict[str, Any]], rubrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    count = hits = empty = hallucinated = 0
    per_class = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["task"] != "caption":
            continue
        count += 1
        text = row["response"].strip()
        rubric = rubrics[row["sample_id"]]
        label = rubric["failure_type"]
        per_class[label][1] += 1
        if not text:
            empty += 1
        if any(term.lower() in text.lower() for term in rubric["must_avoid"]):
            hallucinated += 1
        if any(term.lower() in text.lower() for term in rubric["must_hit_any"]):
            hits += 1
            per_class[label][0] += 1
    return {
        "samples": count, "must_hit_rate": hits / max(count, 1),
        "empty_rate": empty / max(count, 1), "root_cause_hallucination_rate": hallucinated / max(count, 1),
        "must_hit_rate_per_class": {k: v[0] / max(v[1], 1) for k, v in per_class.items()},
        "note": "shape/location/size boundary cases still require the frozen human rubric review",
    }


def robustness_metrics(rows: list[dict[str, Any]], variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Score the perturbed copies of the core samples.

    Resizing, recolouring and rotating a wafer map must not change its defect
    class, so every mismatch here is a real instability rather than a labelling
    difference. ``flip_rate_vs_clean`` compares against the same sample's
    unperturbed prediction, which isolates the perturbation from plain error.
    """
    clean = {
        row["sample_id"]: predicted_label(row["response"])
        for row in rows if row["task"] == "classification"
    }
    per_perturbation: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    scored = invalid = flips = 0
    for row in rows:
        if row["task"] != "robustness_classification":
            continue
        variant = variants.get(row["sample_id"])
        if variant is None:
            continue
        predicted = predicted_label(row["response"])
        if predicted == "INVALID":
            invalid += 1
        gold = variant["failure_type"]
        scored += 1
        bucket = per_perturbation[variant["perturbation"]]
        bucket[1] += 1
        if predicted == gold:
            bucket[0] += 1
        baseline = clean.get(variant["sample_id"])
        if baseline is not None and predicted != baseline:
            flips += 1
    hits = sum(v[0] for v in per_perturbation.values())
    return {
        "samples": scored,
        "accuracy": (hits / scored) if scored else None,
        "accuracy_per_perturbation": {
            name: v[0] / max(v[1], 1) for name, v in sorted(per_perturbation.items())
        },
        "invalid_predictions": invalid,
        "flip_rate_vs_clean": (flips / scored) if scored else None,
        "note": "class is invariant under resize/recolor/rotate, so a flip is genuine instability; "
                "clock-direction gold under rotation is recorded in robustness.jsonl gold_transform",
    }


def dcg(relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(index + 2) for index, rel in enumerate(relevances))


def retrieval_metrics(rankings_path: str | None, qrels_path: Path) -> dict[str, Any] | None:
    if not rankings_path:
        return None
    rankings = {row["query_id"]: row["gallery_ids"] for row in read_jsonl(rankings_path)}
    qrels_by_query: dict[str, dict[str, int]] = defaultdict(dict)
    for row in read_jsonl(qrels_path):
        qrels_by_query[row["query_id"]][row["gallery_id"]] = int(row["relevance"])
    ks = (1, 5, 10, 25, 50)
    ap_values = {k: [] for k in ks}
    recall_values = {k: [] for k in ks}
    ndcg_values = {10: [], 50: []}
    for query, qrels in qrels_by_query.items():
        ranking = rankings.get(query, [])
        relevant = {gid for gid, rel in qrels.items() if rel >= 1}
        if not relevant:
            continue
        for k in ks:
            hits = 0
            precision_sum = 0.0
            for rank, gid in enumerate(ranking[:k], 1):
                if gid in relevant:
                    hits += 1
                    precision_sum += hits / rank
            ap_values[k].append(precision_sum / min(len(relevant), k))
            recall_values[k].append(hits / len(relevant))
        for k in ndcg_values:
            rels = [qrels.get(gid, 0) for gid in ranking[:k]]
            ideal = sorted(qrels.values(), reverse=True)[:k]
            ndcg_values[k].append(dcg(rels) / max(dcg(ideal), 1e-12))
    return {
        **{f"mAP@{k}": float(np.mean(v)) if v else None for k, v in ap_values.items()},
        **{f"Recall@{k}": float(np.mean(v)) if v else None for k, v in recall_values.items()},
        **{f"nDCG@{k}": float(np.mean(v)) if v else None for k, v in ndcg_values.items()},
    }


def load_requests(benchmark: Path) -> tuple[list[dict[str, Any]], str]:
    """Prefer the combined request file so robustness rows are scored too."""
    combined = benchmark / "all_requests.jsonl"
    if combined.exists():
        return read_jsonl(combined), combined.name
    return read_jsonl(benchmark / "inference_requests.jsonl"), "inference_requests.jsonl"


def evaluate(args: argparse.Namespace) -> None:
    benchmark = Path(args.benchmark)
    requests, request_file = load_requests(benchmark)
    aligned = align_predictions(requests, read_jsonl(args.predictions))
    classification_gold = {r["sample_id"]: r["gold_label"] for r in read_jsonl(benchmark / "classification.jsonl")}
    structured_gold = {r["sample_id"]: r["gold"] for r in read_jsonl(benchmark / "structured.jsonl")}
    rubrics = {}
    for row in read_jsonl(benchmark / "core.jsonl"):
        rubrics[row["sample_id"]] = {"failure_type": row["failure_type"], **row["caption_rubric"]}
    variants = {r["variant_id"]: r for r in read_jsonl(benchmark / "robustness.jsonl")}
    report = {
        "requests_scored_from": request_file,
        "requests_expected": len(requests),
        "classification": classification_metrics(aligned, classification_gold),
        "structured": structured_metrics(aligned, structured_gold),
        "caption": caption_metrics(aligned, rubrics),
        "robustness": robustness_metrics(aligned, variants),
        "retrieval": retrieval_metrics(args.rankings, benchmark / "retrieval_qrels.jsonl"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--rankings")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    evaluate(make_parser().parse_args())


if __name__ == "__main__":
    main()
