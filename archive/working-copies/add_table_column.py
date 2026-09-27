"""Add one run's column to the 5.2.2 metrics table, with every cell read from a record.

Why this exists. That table is 9 rows x 6 columns of hand-typed numbers, and four more
runs land today. Each hand-edit is a chance to put a wrong digit in a cell that nothing
downstream can check -- the failure mode this project has already hit more than once.
Here the cell values are *read* from the run's report and train_result, so the only
thing that can be wrong is the mapping from a row label to a metric, and that mapping
is written once, in one place, and asserted on every run.

The two ways a column can lie are both refused rather than papered over:
  * a metric that exists but is null (run 43's `retrieval` was null while its eval was
    still mid-pipeline) is NOT written as a blank or a zero -- the cell would then read
    as "measured zero". Without --allow-unmeasured the tool stops.
  * a run whose idle-step rate was never recorded stops the tool too, because the row
    it feeds has a "%" in it and an empty cell there is not a small number, it is an
    absent measurement.

--after names an existing column; the new one goes immediately to its right, so the
G=4 runs end up adjacent instead of appended in landing order.
"""
import argparse
import json
import sys
from pathlib import Path


def per_run(report, train_result):
    """Row label -> formatted cell, read from the record. Raises on a missing metric."""
    cls, st = report["classification"], report["structured"]
    ci = cls.get("macro_f1_95ci")
    if ci is None:
        raise KeyError("classification.macro_f1_95ci")
    return {
        "分类准确率": cls["accuracy"],
        "macro-F1": cls["macro_f1"],
        "macro-F1 95% CI": list(ci),
        "structured defect_type": st["field_accuracy"]["defect_type"],
        "structured radial_zone": st["field_accuracy"]["radial_zone"],
        "clock circular MAE": st["clock_circular_mae_sectors"],
        "size MAE (R)": st["size_mae_r"],
        "caption must-hit": report["caption"]["must_hit_rate"],
        "robustness accuracy": report["robustness"]["accuracy"],
        "flip rate": report["robustness"]["flip_rate_vs_clean"],
        # The one row that can legitimately have no value: retrieval needs the merged
        # model, and a failed merge leaves it null rather than zero.
        "retrieval mAP@10": (report.get("retrieval") or {}).get("mAP@10"),
        "**零优势步占比（实测）**": train_result.get("mean_frac_reward_zero_std"),
    }


FORMAT = {
    "分类准确率": "{:.4f}",
    "macro-F1": "{:.4f}",
    "macro-F1 95% CI": "[{:.4f}, {:.4f}]",
    "structured defect_type": "{:.4f}",
    "structured radial_zone": "{:.4f}",
    "clock circular MAE": "{:.4f}",
    "size MAE (R)": "{:.4f}",
    "caption must-hit": "{:.4f}",
    "robustness accuracy": "{:.4f}",
    "flip rate": "{:.4f}",
    "retrieval mAP@10": "{:.4f}",
    "**零优势步占比（实测）**": "{:.2%}",
}

# The order rows appear in the document; used to report what was and was not filled.
ROW_ORDER = list(FORMAT)


def cells_of(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
    ap.add_argument("--report", required=True, help="path to <tag>__report.json")
    ap.add_argument("--train-result", required=True, help="path to the run's train_result.json")
    ap.add_argument("--header", required=True, help="header text for the new column")
    ap.add_argument("--after", required=True, help="existing column header to insert after")
    ap.add_argument("--allow-unmeasured", action="store_true",
                    help="write a literal '未测' for a null metric instead of stopping")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    doc = Path(a.doc)
    lines = doc.read_text(encoding="utf-8").split("\n")
    report = json.loads(Path(a.report).read_text(encoding="utf-8"))
    tr_path = Path(a.train_result)
    train_result = json.loads(tr_path.read_text(encoding="utf-8")) if tr_path.is_file() else {}

    values = per_run(report, train_result)

    # ---------------------------------------------------------------- find the table
    # This document has several tables whose first cell is 指标. Picking the first
    # one would silently edit a different table, so the table is identified by
    # containing the column we insert after -- and ambiguity is an error, not a
    # coin flip.
    cands = [i for i, l in enumerate(lines)
             if l.startswith("| 指标") and a.after in cells_of(l)]
    if len(cands) != 1:
        detail = "\n".join(f"  line {i + 1}: {cells_of(lines[i])}" for i in cands) or "  (none)"
        sys.exit(f"expected exactly 1 table with column '{a.after}', found "
                 f"{len(cands)}:\n{detail}")
    hi = cands[0]
    header = cells_of(lines[hi])
    if a.header in header:
        sys.exit(f"column '{a.header}' is already in the table; nothing written")
    col = header.index(a.after) + 1

    # data rows run until the first line that is not a table row
    rows, i = [], hi + 2
    while i < len(lines) and lines[i].startswith("|"):
        rows.append(i)
        i += 1
    if len(rows) != len(ROW_ORDER):
        sys.exit(f"table has {len(rows)} data rows, the mapping knows {len(ROW_ORDER)}; "
                 f"a row was added or removed without updating this tool, so a cell "
                 f"could go to the wrong row")

    # ------------------------------------------------------- fill each row or refuse
    filled, missing = {}, []
    for idx, label in zip(rows, ROW_ORDER):
        row = cells_of(lines[idx])
        if row[0] != label:
            sys.exit(f"row {idx + 1} is '{row[0]}', the mapping expects '{label}'; "
                     f"refusing to write cells against the wrong labels")
        v = values[label]
        if v is None:
            if not a.allow_unmeasured:
                missing.append(label)
                continue
            filled[idx] = "未测"
        else:
            # the CI row formats two numbers, every other row one
            filled[idx] = FORMAT[label].format(*(v if isinstance(v, list) else (v,)))

    if missing:
        sys.exit("these rows have no value in the record, and a blank cell in this "
                 f"table reads as a measured zero: {', '.join(missing)}\n"
                 "  Fix the measurement, or re-run with --allow-unmeasured to write "
                 "'未测' explicitly.")
    if len(filled) != len(rows):
        sys.exit("internal: not every row got a cell")

    # --------------------------------------------------------------- rewrite in place
    out = list(lines)
    for idx, val in filled.items():
        row = cells_of(lines[idx])
        new = row[:col] + [val] + row[col:]
        if len(new) != len(row) + 1:
            sys.exit(f"row {idx + 1}: cell count went {len(row)} -> {len(new)}")
        out[idx] = "| " + " | ".join(new) + " |"

    hnew = cells_of(lines[hi])
    if len(hnew) != len(header):
        sys.exit("internal: header changed while editing")
    out[hi] = "| " + " | ".join(hnew[:col] + [a.header] + hnew[col:]) + " |"
    sep = cells_of(lines[hi + 1])
    out[hi + 1] = "| " + " | ".join(sep[:col] + ["---"] + sep[col:]) + " |"

    print(f"inserting '{a.header}' at column {col + 1} (after '{a.after}'), "
          f"{len(filled)} rows:")
    for idx, val in sorted(filled.items()):
        print(f"  {cells_of(lines[idx])[0]:<32} {val}")
    if a.dry_run:
        print("dry run; nothing written")
        return 0

    doc.write_text("\n".join(out), encoding="utf-8")
    print(f"written: {doc} ({len(out)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
