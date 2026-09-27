"""The floors a real model has to beat: always-majority, uniform-random, and random ranking.

These reuse ``wafer_vlm.evaluate``'s own metric functions rather than
reimplementing them, so the floors are computed with the same accuracy, macro-F1
and retrieval definitions as the model scores they are compared against. A
hand-rolled metric here would silently make the comparison meaningless.

The random baseline is repeated over several seeds because a single draw of 252
uniform labels is itself noisy; the spread is reported, not just one sample.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects" / "wafer-defect-vlm" / "src"))

import numpy as np  # noqa: E402

from wafer_vlm.evaluate import bootstrap_ci, classification_metrics, retrieval_metrics  # noqa: E402
from wafer_vlm.utils import LABELS, read_jsonl  # noqa: E402

RANDOM_SEEDS = (3407, 1, 2, 3, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    bench = Path(args.benchmark)
    core = read_jsonl(bench / "core.jsonl")
    gold = {row["sample_id"]: row["failure_type"] for row in core}
    counts = Counter(gold.values())
    majority = counts.most_common(1)[0][0]

    majority_rows = [{"task": "classification", "sample_id": sid, "response": majority} for sid in gold]
    majority_metrics = classification_metrics(majority_rows, gold)

    random_accuracies, random_f1s = [], []
    for seed in RANDOM_SEEDS:
        rng = random.Random(seed)
        rows = [{"task": "classification", "sample_id": sid,
                 "response": rng.choice(LABELS)} for sid in gold]
        metrics = classification_metrics(rows, gold)
        random_accuracies.append(metrics["accuracy"])
        random_f1s.append(metrics["macro_f1"])

    # A random ranking, scored with the same nDCG/mAP code as the model's.
    queries = read_jsonl(bench / "retrieval_queries.jsonl")
    gallery = read_jsonl(bench / "retrieval_gallery.jsonl")
    rng = random.Random(args.seed)
    rankings = []
    for query in queries:
        pool = [g["gallery_id"] for g in gallery
                if g["gallery_id"] != query["query_id"] and g["lot_name"] != query["lot_name"]]
        rng.shuffle(pool)
        rankings.append({"query_id": query["query_id"], "gallery_ids": pool[:50]})
    scratch = Path(args.output).with_suffix(".rankings.jsonl")
    scratch.parent.mkdir(parents=True, exist_ok=True)
    with scratch.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rankings:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    retrieval = retrieval_metrics(str(scratch), bench / "retrieval_qrels.jsonl")

    report = {
        "core_samples": len(core),
        "class_counts": {label: counts[label] for label in LABELS},
        "majority_class": majority,
        "classification": {
            "majority": majority_metrics,
            "uniform_random": {
                "seeds": list(RANDOM_SEEDS),
                "accuracy_mean": float(np.mean(random_accuracies)),
                "accuracy_std": float(np.std(random_accuracies)),
                "macro_f1_mean": float(np.mean(random_f1s)),
                "accuracy_95ci_across_seeds": [float(np.percentile(random_accuracies, 2.5)),
                                               float(np.percentile(random_accuracies, 97.5))],
            },
        },
        "retrieval": retrieval,
        "retrieval_note": "random ranking, scored with the benchmark's own qrels and metric code",
        "note": "floors for reading the model scores, not model results",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Trivial baselines (floors)",
        "",
        f"Benchmark core: {len(core)} samples. Majority class: `{majority}` "
        f"({counts[majority]}/{len(core)} = {counts[majority] / len(core):.4f}).",
        "",
        f"- always-`{majority}`: accuracy {majority_metrics['accuracy']:.4f}, "
        f"macro-F1 {majority_metrics['macro_f1']:.4f} "
        f"(95% CI {majority_metrics['macro_f1_95ci'][0]:.4f}–{majority_metrics['macro_f1_95ci'][1]:.4f})",
        f"- uniform random over 9 labels: accuracy {np.mean(random_accuracies):.4f} "
        f"± {np.std(random_accuracies):.4f}, macro-F1 {np.mean(random_f1s):.4f} "
        f"(over seeds {list(RANDOM_SEEDS)})",
    ]
    if retrieval:
        lines.append(f"- random ranking: mAP@10 {retrieval['mAP@10']:.4f}, "
                     f"nDCG@10 {retrieval['nDCG@10']:.4f}, Recall@10 {retrieval['Recall@10']:.4f}")
    lines += ["", "Bootstrap CIs elsewhere in this project use the same resampling code as the model scores."]
    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
