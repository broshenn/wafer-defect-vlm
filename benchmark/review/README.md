# Wafer benchmark human dual review

You are one of two independent reviewers. Do not discuss samples with the other
reviewer until both sheets are complete, and do not look for the automatic
labels — they are intentionally not in this file.

For every row, open `image_path` (a 448x448 PNG; green = good die, red = defect
die, black = background/outside the wafer) and fill in:

- `defect_class`  one of: Center, Donut, Edge_Loc, Edge_Ring, Loc, Near_full, Random, Scratch, none
  (the dominant spatial pattern of the red die, or `none` if there is no defect)
- `radial_zone`   one of: center, middle, edge, full, none
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

    python tools/merge_review.py --a review/reviewer_A.csv --b review/reviewer_B.csv \
        --core benchmarks/wafer_bench_v1/core.jsonl

That reports agreement and, if the two of you agree with each other, whether
the automatic gold agrees with you. Only then can the benchmark be frozen.
