"""Show what the reported clock MAE is actually measured over.

`clock_error` does `int(predicted)` and returns None on failure, and
`structured_metrics` then *skips* that row. So the published MAE is computed over
the replies that parsed, not over all replies -- and if the unparsed replies are
systematically the hard ones, the number is optimistically biased rather than
merely noisier.

The unparsed replies turn out to be `"all"` and `"none"`: a model saying the
defect wraps the whole wafer, or that there is no direction to give. Those are
sensible answers to the question the prompt asks, and they are unscoreable
against a single-sector gold. This prints their defect-type composition and the
gold they were dropped against, so the bias is visible instead of implied.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "projects" / "wafer-defect-vlm" / "src"))

from wafer_vlm.evaluate import align_predictions, clock_error, response_text  # noqa: E402
from wafer_vlm.utils import extract_json_object, read_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    requests = read_jsonl(root / "benchmarks" / "wafer_bench_v1" / "all_requests.jsonl")
    gold = {r["sample_id"]: r["gold"]
            for r in read_jsonl(root / "benchmarks" / "wafer_bench_v1" / "structured.jsonl")}
    ftype = {row["sample_id"]: row["failure_type"] for row in
             read_jsonl(root / "data" / "prepared_v1" / "manifest.jsonl")}

    report = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        parsed = []
        for row in align_predictions(requests, read_jsonl(root / path)):
            if row.get("task") != "structured":
                continue
            try:
                parsed.append((row["sample_id"], extract_json_object(response_text(row))))
            except ValueError:
                continue

        good = [(sid, p) for sid, p in parsed
                if clock_error(p.get("clock_direction"), gold[sid].get("clock_sector")) is not None]
        bad = [(sid, p) for sid, p in parsed
               if clock_error(p.get("clock_direction"), gold[sid].get("clock_sector")) is None]

        print("=" * 78)
        print(f"RUN {name}   parsed={len(parsed)}  scored={len(good)}  dropped={len(bad)} "
              f"({len(bad)/max(len(parsed),1):.1%})")
        print("=" * 78)
        dropped_values = collections.Counter(str(p.get("clock_direction")) for _, p in bad)
        print(f"  dropped because clock_direction did not cast to int: {dict(dropped_values)}")
        if bad:
            print(f"  their defect_type: {dict(collections.Counter(ftype[s] for s, _ in bad))}")
            gs = [gold[s]["clock_sector"] for s, _ in bad if gold[s].get("clock_sector") is not None]
            if gs:
                print(f"  the gold they were dropped against: {dict(collections.Counter(gs))}")

        # Is the dropped set disproportionately one class? If so the surviving
        # MAE is conditioned on a stratum, not representative of the benchmark.
        all_counts = collections.Counter(ftype[s] for s, _ in parsed)
        drop_counts = collections.Counter(ftype[s] for s, _ in bad)
        print("  per-class drop rate (a class at the overall rate is unbiased):")
        for label in sorted(all_counts):
            rate = drop_counts.get(label, 0) / all_counts[label]
            flag = "  <-- over-represented" if rate > 2 * len(bad) / max(len(parsed), 1) else ""
            print(f"    {label:10s} {drop_counts.get(label,0):3d}/{all_counts[label]:3d} "
                  f"= {rate:.2f}{flag}")
        report[name] = {"parsed": len(parsed), "scored": len(good), "dropped": len(bad),
                        "dropped_values": dict(dropped_values),
                        "dropped_defect_types": dict(collections.Counter(ftype[s] for s, _ in bad))}
        print()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
