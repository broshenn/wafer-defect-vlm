"""Add one run's column to all four metrics tables in LIMITATIONS.md.

Four hand-written tables carry one column per run:

  5.2.2  基准成绩（指标）        no bold in the data rows
  5.2.3  奖励的组内标准差        bold = lowest (most saturated signal)
  5.2.5  结构化画像              bold per row: higher-is-better vs lower-is-better
  5.2.5  类别 F1                 bold = highest

Three runs landed today, so this is done three times: the values are read from each
run's report and train_result, and the bold is recomputed from the finished row. What
is *not* automated is the decision of where the column goes -- that is --after, because
landing order and logical order differ (G=4 runs belong next to each other).

The new header mirrors the table it lands in: 5.2.2 writes "GRPO (G=4, lr5e-5)" and
the other three write "GRPO(G=4) lr5e-5". Copying the local idiom keeps each table
internally consistent; --after is matched with spaces stripped so the caller can name
a column the same way regardless of which table it is in.

Refuses rather than guesses:
  * a metric that is null (a run whose retrieval never ran) is not written as a blank
    or a zero -- without --allow-unmeasured the tool stops, because an empty cell in
    this document reads as a measurement that came out small;
  * a table whose rows are not exactly the expected labels stops the tool, so a cell
    can never land against the wrong row.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")


def per_run(rep, tr):
    """row label -> (raw value(s), format string). Raises if a metric is missing."""
    cls, st = rep["classification"], rep["structured"]
    pc = {k: v["f1-score"] for k, v in cls["per_class"].items()}
    sig = {k: v["mean_std_across_steps"] for k, v in tr["reward_signal"].items()}
    return {
        "522": {
            "分类准确率": (cls["accuracy"], "{:.4f}"),
            "macro-F1": (cls["macro_f1"], "{:.4f}"),
            "macro-F1 95% CI": (list(cls["macro_f1_95ci"]), "[{:.4f}, {:.4f}]"),
            "structured defect_type": (st["field_accuracy"]["defect_type"], "{:.4f}"),
            "structured radial_zone": (st["field_accuracy"]["radial_zone"], "{:.4f}"),
            "clock circular MAE": (st["clock_circular_mae_sectors"], "{:.4f}"),
            "size MAE (R)": (st["size_mae_r"], "{:.4f}"),
            "caption must-hit": (rep["caption"]["must_hit_rate"], "{:.4f}"),
            "robustness accuracy": (rep["robustness"]["accuracy"], "{:.4f}"),
            "flip rate": (rep["robustness"]["flip_rate_vs_clean"], "{:.4f}"),
            "retrieval mAP@10": ((rep.get("retrieval") or {}).get("mAP@10"), "{:.4f}"),
            "**零优势步占比（实测）**":
                (tr.get("mean_frac_reward_zero_std"), "{:.2%}"),
        },
        "reward": {
            **{f"`{k}`": (sig[k], "{:.4f}") for k in
               ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock")},
            "**空转步占比**": (tr.get("mean_frac_reward_zero_std"), "{:.2%}"),
        },
        "profile": {
            "`radial_zone`（越高越好）":
                (st["field_accuracy"]["radial_zone"], "{:.4f}"),
            "时钟 MAE（越低越好）": (st["clock_circular_mae_sectors"], "{:.4f}"),
            "尺寸 MAE（越低越好）": (st["size_mae_r"], "{:.4f}"),
            "caption must-hit（越高越好）":
                (rep["caption"]["must_hit_rate"], "{:.4f}"),
        },
        "classes": {c: (pc[c], "{:.3f}") for c in
                    ("Donut", "none", "Edge_Ring", "Scratch")},
    }


TABLES = [
    {"key": "522", "header": "指标", "sentinel": "分类准确率", "bold": False,
     "dirs": {}},
    {"key": "reward", "header": "奖励的组内标准差（150 步均值）",
     "sentinel": "`WaferClass`", "bold": True, "dirs": {}},      # all rows: lowest
    {"key": "profile", "header": "指标", "sentinel": "时钟 MAE（越低越好）",
     "bold": True,
     "dirs": {"`radial_zone`（越高越好）": "max", "时钟 MAE（越低越好）": "min",
              "尺寸 MAE（越低越好）": "min", "caption must-hit（越高越好）": "max"}},
    {"key": "classes", "header": "类别（F1）", "sentinel": "Donut", "bold": True,
     "dirs": {}},                                                # all rows: highest
]


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render(cs):
    return "| " + " | ".join(cs) + " |"


def norm(s):
    return re.sub(r"[\s,]+", "", s)


# The two styles are not whitespace variants of each other: 5.2.2 writes
# "GRPO (G=4, lr5e-5)" with the lr *inside* the parentheses, the other tables write
# "GRPO(G=4) lr5e-5". Stripping spaces cannot equate those, so columns are matched on
# the thing they actually name: (algorithm, group size, learning rate).
COL = re.compile(r"^([A-Za-z]+)\s*\(\s*G=(\d+)\s*[,)]?\s*lr\s*([0-9.eE+-]+)\s*\)?$")


def colkey(s):
    """'GSPO (G=4, lr5e-5)' and 'GSPO(G=4) lr5e-5' -> the same key. None if not a run
    column (SFT, 变化, 指标, ...)."""
    m = COL.match(s.strip())
    return f"{m.group(1)}|{int(m.group(2))}|{m.group(3)}" if m else None


def number_of(c):
    try:
        return float(c.strip("*").rstrip("%"))
    except ValueError:
        return None


def find_table(lines, spec, after_key):
    """Index of the header line of the table whose header starts with spec['header'],
    contains a column matching `after`, and whose rows include spec['sentinel']."""
    hits = []
    for i, l in enumerate(lines):
        if not l.startswith("| " + spec["header"]):
            continue
        if not any(colkey(c) == after_key for c in cells(l)):
            continue
        rows, j = [], i + 2
        while j < len(lines) and lines[j].startswith("|"):
            rows.append(cells(lines[j])[0])
            j += 1
        if spec["sentinel"] in rows:
            hits.append((i, cells(l), rows))
    if len(hits) != 1:
        sys.exit(f"{spec['key']}: expected exactly 1 matching table, found {len(hits)}")
    return hits[0]


def styled(new_base, header_cells):
    """Write the new header the way this table writes its others."""
    spaced = any(re.search(r"^[A-Za-z]+ \(", c) for c in header_cells[1:])
    m = re.match(r"([A-Za-z]+)\(G=(\d+)\) lr(\S+)", new_base)
    return (f"{m.group(1)} (G={m.group(2)}, lr{m.group(3)})" if spaced else new_base)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="e.g. qwen35_9b_gspo_g4_lr1e5")
    ap.add_argument("--name", required=True, help="e.g. GSPO(G=4) lr1e-5")
    ap.add_argument("--after", required=True, help="existing column to insert after")
    ap.add_argument("--report", help="defaults to <tag>__report.json")
    ap.add_argument("--train-result", help="defaults to <tag>_train_result.json")
    ap.add_argument("--allow-unmeasured", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    doc = Path(os.environ.get("WAFER_DOC", str(ROOT / "LIMITATIONS.md")))
    rep_p = Path(a.report or ROOT / f"outputs/reports/{a.tag}__report.json")
    tr_p = Path(a.train_result or ROOT / f"outputs/reports/{a.tag}_train_result.json")
    if not rep_p.is_file():
        sys.exit(f"no report at {rep_p}")
    if not tr_p.is_file():
        sys.exit(f"no train_result at {tr_p}")
    values = per_run(json.loads(rep_p.read_text(encoding="utf-8")),
                     json.loads(tr_p.read_text(encoding="utf-8")))

    src = doc.read_text(encoding="utf-8").split("\n")
    before_bold = "\n".join(src).count("**")
    after_key = colkey(a.after)
    if after_key is None:
        sys.exit(f"--after '{a.after}' does not name a run column "
                 f"(expected something like 'GSPO(G=4) lr5e-5')")
    new_key = colkey(a.name)
    if new_key is None:
        sys.exit(f"--name '{a.name}' does not name a run column")
    if new_key == after_key:
        sys.exit("--name and --after are the same column")

    for spec in TABLES:
        hi, head, rows = find_table(src, spec, after_key)
        new_head = styled(a.name, head)
        if any(colkey(c) == new_key for c in head):
            sys.exit(f"{spec['key']}: a column for {new_key} is already present; "
                     f"nothing written")
        col = [colkey(c) for c in head].index(after_key) + 1
        vals = values[spec["key"]]
        if [r for r in rows] != [r for r in vals]:
            sys.exit(f"{spec['key']}: rows are {rows}, the mapping has {list(vals)}")
        new_lines, filled = {}, {}
        for k, lab in enumerate(rows):
            cs = cells(src[hi + 2 + k])
            raw, fmt = vals[lab]
            if raw is None:
                if not a.allow_unmeasured:
                    sys.exit(f"{spec['key']}: row '{lab}' has no value in the record "
                             f"({rep_p.name}); a blank cell here reads as a measured "
                             f"zero. Fix the measurement or pass --allow-unmeasured.")
                text = "未测"
            else:
                text = fmt.format(*(raw if isinstance(raw, list) else (raw,)))
            new = cs[:col] + [text] + cs[col:]
            if len(new) != len(cs) + 1:
                sys.exit(f"{spec['key']}: row '{lab}' cell count {len(cs)} -> {len(new)}")
            if spec["bold"]:
                numeric = [(j, number_of(c)) for j, c in enumerate(new[1:], 1)
                           if number_of(c) is not None]
                if not numeric:
                    sys.exit(f"{spec['key']}: row '{lab}' has no numeric cells")
                for j, _ in numeric:
                    new[j] = new[j].strip("*")
                d = spec["dirs"].get(lab, "min" if spec["key"] == "reward" else "max")
                best = (min if d == "min" else max)(numeric, key=lambda t: t[1])[0]
                new[best] = "**" + new[best] + "**"
            new_lines[hi + 2 + k] = render(new)
            filled[lab] = text

        for k, text in new_lines.items():
            src[k] = text
        hc = cells(src[hi])
        src[hi] = render(hc[:col] + [new_head] + hc[col:])
        sc = cells(src[hi + 1])
        src[hi + 1] = render(sc[:col] + ["---"] + sc[col:])
        print(f"{spec['key']:<8} column {col + 1} as '{new_head}' "
              f"({len(filled)} rows){' [bold recomputed]' if spec['bold'] else ''}")

    after_bold = "\n".join(src).count("**")
    if after_bold % 2:
        sys.exit(f"bold markers odd ({after_bold}); nothing written")
    if after_bold != before_bold:
        print(f"note: bold markers {before_bold} -> {after_bold} (moved, not added)")
    if a.dry_run:
        print("dry run; nothing written")
        return 0
    doc.write_text("\n".join(src), encoding="utf-8")
    print(f"written: {doc} ({len(src)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
