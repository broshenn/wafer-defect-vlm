"""Merge the two blind reviewer sheets and score the benchmark's own gold.

Three outcomes matter and they are kept distinct:

- the reviewers agree with each other and with the automatic gold -> the sample
  is confirmed;
- the reviewers agree with each other but not with gold -> the automatic label
  is wrong, which is a benchmark defect rather than a reviewer problem;
- the reviewers disagree with each other -> genuinely ambiguous sample.

Only confirmed samples are marked approved, which is what ``curate``/``freeze``
requires before the benchmark can stop being a draft.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_sheet(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle) if row.get("sample_id")}


def normalise(value: str | None) -> str:
    return (value or "").strip()


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Agreement corrected for what two raters would agree on by chance."""
    if not pairs:
        return None
    labels = sorted({value for pair in pairs for value in pair})
    observed = sum(a == b for a, b in pairs) / len(pairs)
    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)
    expected = sum((a_counts[label] / len(pairs)) * (b_counts[label] / len(pairs)) for label in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--core", required=True)
    parser.add_argument("--output")
    parser.add_argument("--update-core", action="store_true",
                        help="write annotator_count/agreement back into core.jsonl")
    args = parser.parse_args()

    sheet_a, sheet_b = read_sheet(args.a), read_sheet(args.b)
    core_path = Path(args.core)
    core = [json.loads(line) for line in core_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    pairs: list[tuple[str, str]] = []
    outcomes = Counter()
    confirmed, gold_mismatch, reviewer_disagreement, unlabelled = [], [], [], []

    for row in core:
        sid = row["sample_id"]
        a, b = sheet_a.get(sid, {}), sheet_b.get(sid, {})
        va, vb = normalise(a.get("defect_class")), normalise(b.get("defect_class"))
        if not va or not vb:
            unlabelled.append(sid)
            outcomes["missing_review"] += 1
            continue
        pairs.append((va, vb))
        if va != vb:
            reviewer_disagreement.append(sid)
            outcomes["reviewer_disagreement"] += 1
        elif va == row["failure_type"]:
            confirmed.append(sid)
            outcomes["confirmed"] += 1
        else:
            gold_mismatch.append(sid)
            outcomes["gold_mismatch"] += 1

    report = {
        "core_samples": len(core),
        "reviewed_by_both": len(pairs),
        "outcomes": dict(outcomes),
        "class_agreement_between_reviewers": (sum(a == b for a, b in pairs) / len(pairs)) if pairs else None,
        "class_cohen_kappa": cohen_kappa(pairs),
        "gold_match_given_agreement": (len(confirmed) / (len(confirmed) + len(gold_mismatch)))
                                     if (confirmed or gold_mismatch) else None,
        "gold_mismatch_samples": gold_mismatch[:50],
        "reviewer_disagreement_samples": reviewer_disagreement[:50],
        "unlabelled_samples": unlabelled[:50],
        "note": "gold_match is the benchmark's own accuracy, not a model score",
    }

    if args.update_core:
        by_id = {row["sample_id"]: row for row in core}
        for sid in confirmed:
            by_id[sid]["annotator_count"] = 2
            by_id[sid]["agreement"] = "approved"
        with core_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in core:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["core_updated"] = True

    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
