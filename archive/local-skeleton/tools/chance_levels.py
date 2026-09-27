"""Report the cardinality of the structured gold fields.

The point is to establish what "chance" means for each field, so an accuracy can
be judged rather than merely quoted. An accuracy of 0.264 sounds respectable in
a table and is in fact at-or-below chance if the field has four values; that
distinction is invisible unless the cardinality is stated next to it.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in
            (Path(args.benchmark) / "structured.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    golds = [row.get("gold") or {} for row in rows]
    print(f"structured rows: {len(rows)}")
    print(f"gold fields: {sorted({k for g in golds for k in g})}\n")

    for field in ("defect_type", "radial_zone", "clock_sector", "size_r", "centroid_angle"):
        values = [g[field] for g in golds if g.get(field) is not None]
        if not values:
            continue
        distinct = sorted(set(values))
        if isinstance(values[0], (int, float)):
            print(f"{field:14s} numeric, n={len(values)}, range=({min(values)}, {max(values)})")
            continue
        counts = collections.Counter(values)
        # The largest class is the floor a constant answer achieves; uniform
        # chance is 1/k. Both are printed because they differ a lot here.
        top = max(counts.values()) / len(values)
        print(f"{field:14s} n={len(values):4d}  k={len(distinct):2d}  "
              f"uniform chance={1/len(distinct):.4f}  majority={top:.4f}")
        print(f"{'':14s} {dict(counts)}")

    if args.report and Path(args.report).is_file():
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        structured = report.get("structured") or {}
        print("\nscored, for comparison:")
        for key, value in structured.items():
            if not isinstance(value, (dict, list)) and value is not None:
                print(f"  {key} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
