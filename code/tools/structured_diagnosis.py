"""Diagnose the two structured fields the model appears not to have learned.

The comparison table shows `radial_zone` at 0.264 and `size_r` at MAE 0.50, both
*worse than answering a constant*. Two explanations fit that pattern and they
call for very different conclusions:

  * the model never learned the field -- a real negative result; or
  * the model learned the field but the scorer discards its answer.

The second is not hypothetical. `structured_metrics` compares `radial_zone` with
a bare `==` and no normalization, so a reply of "Center" is scored wrong against
a gold of "center"; and it *skips* any row whose `clock_direction` or `size_r`
will not parse, so the surviving errors are a biased subset rather than the
whole population. An unnormalized string compare on a free-text JSON field is
exactly the kind of defect that produces a plausible-looking wrong number.

This prints the evidence either way: parse-failure rate, the confusion matrix
with and without case folding, and the predicted values beside the gold
distribution. It reports; it does not decide.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "projects" / "wafer-defect-vlm" / "src"))

from wafer_vlm.evaluate import align_predictions, clock_error, response_text  # noqa: E402
from wafer_vlm.utils import extract_json_object, read_jsonl  # noqa: E402


def circular_mae(prediction: int, truth: int, period: int = 12) -> int:
    delta = abs(int(prediction) - int(truth)) % period
    return min(delta, period - delta)


def describe(values: list[float]) -> str:
    if not values:
        return "none"
    return (f"n={len(values)} min={min(values):.3f} median={statistics.median(values):.3f} "
            f"mean={statistics.fmean(values):.3f} max={max(values):.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run", action="append", required=True,
                        metavar="NAME=PATH", help="e.g. sft=outputs/baselines/x.jsonl")
    parser.add_argument("--output", default=None, help="write the raw dump here as JSON")
    args = parser.parse_args()

    root = Path(args.root)
    requests = read_jsonl(root / "benchmarks" / "wafer_bench_v1" / "all_requests.jsonl")
    gold_rows = {row["sample_id"]: row["gold"] for row in
                 read_jsonl(root / "benchmarks" / "wafer_bench_v1" / "structured.jsonl")}
    structured_requests = [r for r in requests if r.get("task") == "structured"]
    print(f"structured requests: {len(structured_requests)}\n")

    dump: dict[str, object] = {}

    for spec in args.run:
        name, _, path = spec.partition("=")
        predictions = read_jsonl(root / path)
        aligned = align_predictions(requests, predictions)

        parsed: list[tuple[str, dict, dict]] = []
        parse_errors = 0
        for row in aligned:
            if row.get("task") != "structured":
                continue
            try:
                predicted = extract_json_object(response_text(row))
            except ValueError:
                parse_errors += 1
                continue
            parsed.append((row["sample_id"], predicted, gold_rows[row["sample_id"]]))

        print("=" * 78)
        print(f"RUN {name}   parsed={len(parsed)}  parse_errors={parse_errors}  "
              f"of {len(structured_requests)} structured")
        print("=" * 78)

        # ------------------------------------------------------- radial_zone
        gold_zones = collections.Counter(g["radial_zone"] for _, _, g in parsed)
        pred_zones = collections.Counter(str(p.get("radial_zone")) for _, p, _ in parsed)
        exact = sum(1 for _, p, g in parsed if p.get("radial_zone") == g["radial_zone"])
        folded = sum(1 for _, p, g in parsed
                     if str(p.get("radial_zone", "")).strip().lower()
                     == str(g["radial_zone"]).strip().lower())
        print("\nradial_zone")
        print(f"  gold      {dict(gold_zones)}")
        print(f"  predicted {dict(pred_zones)}")
        print(f"  accuracy exact          {exact}/{len(parsed)} = {exact / max(len(parsed),1):.4f}")
        print(f"  accuracy case-folded    {folded}/{len(parsed)} = {folded / max(len(parsed),1):.4f}")
        if folded != exact:
            print(f"  >>> case folding alone recovers {folded - exact} rows "
                  f"({(folded-exact)/max(len(parsed),1):.4f} accuracy)")

        print("  confusion (rows=gold, cols=predicted), off-diagonal only:")
        matrix = collections.Counter(
            (str(g["radial_zone"]), str(p.get("radial_zone"))) for _, p, g in parsed)
        for (gz, pz), count in sorted(matrix.items(), key=lambda kv: -kv[1]):
            if gz != pz:
                print(f"    gold={gz:<8} predicted={pz:<12} n={count}")

        # ----------------------------------------------------------- size_r
        pairs = [(float(p["size_r"]), float(g["size_r"]))
                 for _, p, g in parsed
                 if p.get("size_r") is not None and g.get("size_r") is not None]
        missing = len(parsed) - len(pairs)
        print(f"\nsize_r   usable pairs={len(pairs)}  dropped(no size_r)={missing}")
        if pairs:
            preds = [p for p, _ in pairs]
            golds = [g for _, g in pairs]
            mae = statistics.fmean(abs(p - g) for p, g in pairs)
            print(f"  predicted {describe(preds)}")
            print(f"  gold      {describe(golds)}")
            print(f"  MAE {mae:.4f}")
            print(f"  correlation {statistics.correlation(preds, golds) if len(pairs) > 2 else float('nan'):.4f}"
                  if len(set(preds)) > 1 else "  correlation undefined (constant predictions)")
            # If the model simply copied a plausible constant, its predictions
            # collapse to one value; the count of distinct values says which.
            print(f"  distinct predicted values: {len(set(round(p, 6) for p in preds))}")

        # ----------------------------------------------------- clock_sector
        clock_pairs, unparseable = [], 0
        for _, p, g in parsed:
            raw = p.get("clock_direction")
            error = clock_error(raw, g.get("clock_sector"))
            if error is None:
                unparseable += 1
            else:
                clock_pairs.append((raw, g["clock_sector"], error))
        print(f"\nclock_sector   usable={len(clock_pairs)}  unparseable={unparseable}")
        if clock_pairs:
            print(f"  raw predicted values {dict(collections.Counter(str(r) for r, _, _ in clock_pairs))}")
            print(f"  circular MAE {statistics.fmean(e for _, _, e in clock_pairs):.4f}")

        dump[name] = {
            "parsed": len(parsed), "parse_errors": parse_errors,
            "radial_zone_accuracy": exact / max(len(parsed), 1),
            "radial_zone_accuracy_casefolded": folded / max(len(parsed), 1),
            "radial_zone_predicted": dict(pred_zones),
            "radial_zone_confusion": {f"{gz}->{pz}": c for (gz, pz), c in matrix.items()},
            "size_r": {
                "pairs": len(pairs), "missing": missing,
                "mae": statistics.fmean(abs(p - g) for p, g in pairs) if pairs else None,
                "predicted": describe([p for p, _ in pairs]),
                "gold": describe([g for _, g in pairs]),
                "distinct_predictions": len(set(round(p, 6) for p, _ in pairs)) if pairs else 0,
            },
            "clock_sector": {
                "usable": len(clock_pairs), "unparseable": unparseable,
                "circular_mae": statistics.fmean(e for _, _, e in clock_pairs) if clock_pairs else None,
            },
        }
        print()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
