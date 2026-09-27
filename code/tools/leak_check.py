"""Check that no wafer or lot is shared across splits or with the benchmark.

Two different leaks matter here and they fail independently:

* **Sample leakage** — the same wafer appearing in both training data and the
  benchmark. The benchmark score would then partly measure memorisation.
* **Lot leakage** — two wafers from one lot landing in different splits. Wafers
  from a lot share process conditions, so a lot-split model looks better than it
  is on genuinely unseen lots.

A sample-level intersection of zero is necessary but not sufficient: lots must
also be disjoint, which is why both are reported separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects" / "wafer-defect-vlm" / "src"))

from wafer_vlm.utils import read_jsonl, split_for_lot  # noqa: E402

# Pairs whose overlap is intended rather than a defect. Only GRPO appears here:
# it is an RL pass over the SFT training split, so it must reuse those wafers.
# Nothing is ever allowed to overlap the benchmark or the validation split.
ALLOWED_OVERLAPS = {("grpo", "train")}


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return read_jsonl(path)


def manifest_index(path: Path) -> dict[str, dict]:
    """sample_id -> manifest row, which is where lot and split actually live.

    The curated splits are in ms-swift's training format and carry only
    images/messages/sample_id/task, so the lot has to be joined in from the
    manifest. Doing the join also cross-checks that every training row refers to
    a sample the manifest knows about.
    """
    return {row["sample_id"]: row for row in load(path)}


def describe(name: str, rows: list[dict], index: dict[str, dict]) -> dict:
    lots, unresolved = set(), []
    for row in rows:
        entry = index.get(row["sample_id"])
        if entry is None:
            unresolved.append(row["sample_id"])
        elif entry.get("lot_name"):
            lots.add(entry["lot_name"])
    return {
        "name": name,
        "rows": len(rows),
        "samples": {r["sample_id"] for r in rows},
        "lots": lots,
        "unresolved": unresolved,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--curated", required=True, help="directory holding splits/*.jsonl")
    ap.add_argument("--manifest", required=True, help="manifest.jsonl with lot_name and split")
    ap.add_argument("--grpo", default=None, help="optional GRPO dataset jsonl")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    bench_dir = Path(args.benchmark)
    curated = Path(args.curated)
    index = manifest_index(Path(args.manifest))

    bench_rows = load(bench_dir / "core.jsonl")
    bench = describe("benchmark_core", bench_rows, index)
    groups = {
        "train": load(curated / "splits" / "train.jsonl"),
        "val": load(curated / "splits" / "val.jsonl"),
    }
    if args.grpo:
        groups["grpo"] = load(Path(args.grpo))

    described = {name: describe(name, rows, index) for name, rows in groups.items()}

    failures: list[str] = []
    sample_overlaps, lot_overlaps = {}, {}

    for name, other in described.items():
        shared_samples = sorted(bench["samples"] & other["samples"])
        shared_lots = sorted(bench["lots"] & other["lots"])
        sample_overlaps[name] = shared_samples[:20]
        lot_overlaps[name] = shared_lots[:20]
        if shared_samples:
            failures.append(f"{name} shares {len(shared_samples)} samples with the benchmark")
        if shared_lots:
            failures.append(f"{name} shares {len(shared_lots)} lots with the benchmark")
        if other["unresolved"]:
            failures.append(
                f"{name} has {len(other['unresolved'])} sample_ids missing from the manifest")

    # Splits must be disjoint from each other too, not just from the benchmark —
    # except for the one pair where overlap is the point. GRPO is a
    # reinforcement-learning pass over the training split starting from the SFT
    # checkpoint, so it is supposed to reuse those wafers; treating that as a
    # leak would flag the design as a defect.
    names = sorted(described)
    expected_overlaps = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = described[left], described[right]
            shared_samples, shared_lots = a["samples"] & b["samples"], a["lots"] & b["lots"]
            if tuple(sorted((left, right))) in ALLOWED_OVERLAPS:
                expected_overlaps[f"{left}/{right}"] = {
                    "samples": len(shared_samples), "lots": len(shared_lots),
                    "why": "GRPO trains on the SFT training split by design",
                }
                continue
            if shared_samples:
                failures.append(f"{left} and {right} share samples")
            if shared_lots:
                failures.append(f"{left} and {right} share lots")

    # Even where overlap is allowed, GRPO should not be reaching wafers that the
    # SFT stage never saw: that would be RL on data outside the training split.
    grpo_outside_train = []
    if "grpo" in described and "train" in described:
        grpo_outside_train = sorted(described["grpo"]["samples"] - described["train"]["samples"])

    # The manifest must place every sample in the split its lot hashes to;
    # otherwise the isolation above holds but is not reproducible from the rule.
    rule_mismatch = [
        row["sample_id"] for row in bench_rows
        if row.get("lot_name") and split_for_lot(row["lot_name"], args.seed) != row.get("split")
    ]
    bench_split = {row.get("split") for row in bench_rows}
    if rule_mismatch:
        failures.append(f"{len(rule_mismatch)} benchmark rows disagree with the lot split rule")
    if bench_split and bench_split != {"test"}:
        failures.append(f"benchmark core is not drawn only from the test split: {sorted(bench_split)}")

    lot_splits: dict[str, set[str]] = defaultdict(set)
    for row in index.values():
        if row.get("lot_name") and row.get("split"):
            lot_splits[row["lot_name"]].add(row["split"])
    straddling = {lot: sorted(v) for lot, v in lot_splits.items() if len(v) > 1}
    if straddling:
        failures.append(f"{len(straddling)} lots straddle splits in the manifest")

    report = {
        "seed": args.seed,
        "manifest_samples": len(index),
        "groups": {name: {"rows": d["rows"], "samples": len(d["samples"]),
                          "lots": len(d["lots"]), "unresolved": len(d["unresolved"])}
                   for name, d in described.items()},
        "benchmark": {"rows": len(bench_rows), "samples": len(bench["samples"]),
                      "lots": len(bench["lots"]), "splits": sorted(bench_split)},
        "sample_overlap_with_benchmark": sample_overlaps,
        "lot_overlap_with_benchmark": lot_overlaps,
        "expected_overlaps": expected_overlaps,
        "grpo_samples_outside_train_count": len(grpo_outside_train),
        "grpo_samples_outside_train_examples": grpo_outside_train[:10],
        "rule_mismatch_count": len(rule_mismatch),
        "rule_mismatch_examples": rule_mismatch[:10],
        "lots_straddling_splits": dict(list(straddling.items())[:10]),
        "failures": failures,
        "passed": not failures,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("passed", "failures", "rule_mismatch_count")}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
