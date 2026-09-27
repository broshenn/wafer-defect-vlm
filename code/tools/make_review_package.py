"""Build a blind two-reviewer sheet for the benchmark core samples.

Gold labels are deliberately withheld: a reviewer who can see the answer is not
an independent reviewer, and the point of the exercise is to find out whether
the automatic gold survives human scrutiny. Each reviewer gets their own file
so neither can anchor on the other's row.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABELS = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_full", "Random", "Scratch", "none"]
ZONES = ["center", "middle", "edge", "full", "none"]

INSTRUCTIONS = """# Wafer benchmark human dual review

You are one of two independent reviewers. Do not discuss samples with the other
reviewer until both sheets are complete, and do not look for the automatic
labels — they are intentionally not in this file.

For every row, open `image_path` (a 448x448 PNG; green = good die, red = defect
die, black = background/outside the wafer) and fill in:

- `defect_class`  one of: {labels}
  (the dominant spatial pattern of the red die, or `none` if there is no defect)
- `radial_zone`   one of: {zones}
  (where the defect sits relative to the wafer radius R: centre / middle / edge;
   `full` if it covers the whole wafer, `none` if there is no defect)
- `clock_direction`  1-12, meaning the clock position of the defect's centre of
  mass measured clockwise from 12 o'clock; `all` if it wraps around; `none` if
  there is no defect or the position is meaningless
- `confidence_0_100`  how sure you are of `defect_class`
- `notes`  anything that made the call hard (free text)

Leave a cell empty rather than guessing, and say so in `notes`. An honest blank
is more useful than a fabricated label.

When both sheets are done, run:

    python tools/merge_review.py --a review/reviewer_A.csv --b review/reviewer_B.csv \\
        --core benchmarks/wafer_bench_v1/core.jsonl

That reports agreement and, if the two of you agree with each other, whether
the automatic gold agrees with you. Only then can the benchmark be frozen.
""".format(labels=", ".join(LABELS), zones=", ".join(ZONES))

FIELDS = ["sample_id", "image_path", "defect_class", "radial_zone",
          "clock_direction", "confidence_0_100", "notes"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reviewers", default="A,B")
    args = parser.parse_args()

    core = [json.loads(line) for line in Path(args.core).read_text(encoding="utf-8").splitlines() if line.strip()]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(INSTRUCTIONS, encoding="utf-8")

    reviewers = [name.strip() for name in args.reviewers.split(",") if name.strip()]
    for reviewer in reviewers:
        path = output / f"reviewer_{reviewer}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in core:
                writer.writerow({
                    "sample_id": row["sample_id"],
                    "image_path": row["image_path"],
                    "defect_class": "", "radial_zone": "", "clock_direction": "",
                    "confidence_0_100": "", "notes": "",
                })
    summary = {
        "core_samples": len(core),
        "reviewers": reviewers,
        "sheets": [f"reviewer_{name}.csv" for name in reviewers],
        "gold_withheld": True,
        "status": "pending_human_review",
        "note": "The benchmark stays draft until two independent reviewers fill these in "
                "and merge_review.py reports agreement. This is a real human step and is "
                "deliberately not simulated.",
    }
    (output / "status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
