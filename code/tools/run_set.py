"""Enumerate the runs for the final comparison from the reports on disk.

The queue scripts pass runs to tools/make_report.py from a hand-written list, and
each queue's list is a snapshot of the runs that existed when it was written. That
is how a run loses its column without anyone deciding to drop it: the list is
correct when written and silently incomplete later, and nothing in the produced
table can show it.

This inverts the default. Every `*__report.json` on disk must be either named in
NAMES or explicitly listed in EXCLUDE with a reason; anything else is an error. A
new run therefore forces a decision instead of defaulting to omission.

Emits, on stdout:
  --run name=path   lines, one per available run, in table order
and writes an audit JSON with --audit PATH recording what was included, what was
expected but absent, and what was excluded.

Exit codes: 0 normally; 1 if a report on disk is unaccounted for.
"""
import argparse
import json
import sys
from pathlib import Path

# Table order, and the names already used in earlier comparison tables -- changing a
# name here would silently relabel a column that documents and records refer to.
NAMES = [
    ("Base", "qwen35_9b_zero_shot"),
    ("SFT", "qwen35_9b_adapter"),
    ("GRPO_lr1e5", "qwen35_9b_grpo"),
    ("GRPO_lr5e5", "qwen35_9b_grpo_lr5e5"),
    ("GRPO_lr1e5_s2", "qwen35_9b_grpo_lr1e5_seed3408"),
    # A seed of an existing cell is a comparison ROW and not a column: it measures the
    # same (algorithm, group size, learning rate) as the run it repeats, so a column in
    # LIMITATIONS.md's tables would count one configuration twice. It is listed here so
    # that the run set on disk stays accounted for -- an unlisted report is the error
    # this list exists to raise.
    ("GRPO_lr1e5_s3", "qwen35_9b_grpo_lr1e5_seed3409"),
    ("GSPO_lr5e5", "qwen35_9b_gspo_v1"),
    ("GSPO_lr1e5", "gspo_lr1e5"),
    ("GSPO_G32", "qwen35_9b_gspo_g32"),
    ("GSPO_G32_lr1e5", "qwen35_9b_gspo_g32_lr1e5"),
    ("GSPO_G4_lr5e5", "qwen35_9b_gspo_g4_lr5e5"),
    ("GSPO_G4_lr1e5", "qwen35_9b_gspo_g4_lr1e5"),
]

# Reports that exist on disk and are deliberately not a column, each with why. An
# entry here is a decision that gets printed; absence from both lists is an error.
EXCLUDE = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-fs/wafer-vlm")
    ap.add_argument("--audit", default=None)
    ap.add_argument("--emit", choices=["runs", "json"], default="runs")
    a = ap.parse_args()

    rep = Path(a.root) / "outputs/reports"
    on_disk = {p.name[:-len("__report.json")]
               for p in rep.glob("*__report.json")}

    mapped = {tag for _, tag in NAMES}
    unaccounted = sorted(on_disk - mapped - set(EXCLUDE))

    available, absent = [], []
    for name, tag in NAMES:
        f = rep / f"{tag}__report.json"
        (available if f.is_file() and f.stat().st_size else absent).append(
            (name, tag, f))

    audit = {
        "available": [{"name": n, "tag": t} for n, t, _ in available],
        "expected_but_absent": [{"name": n, "tag": t} for n, t, _ in absent],
        "excluded_on_disk": EXCLUDE,
        "reports_on_disk": sorted(on_disk),
        "unaccounted_reports": unaccounted,
        "note": "A report present on disk but named in neither NAMES nor EXCLUDE is "
                "listed in unaccounted_reports and fails this script: a run must be "
                "dropped by a decision that is printed, never by a list that went "
                "stale.",
    }
    if a.audit:
        Path(a.audit).write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

    if a.emit == "json":
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    else:
        for _, _, f in available:
            pass
        for name, _, f in available:
            print(f"--run {name}={f}")

    if unaccounted:
        print(f"ERROR: {len(unaccounted)} report(s) on disk are in neither NAMES "
              f"nor EXCLUDE: {unaccounted}", file=sys.stderr)
        return 1
    print(f"run set: {len(available)} included, {len(absent)} expected but absent "
          f"{[t for _, t, _ in absent] if absent else ''}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
