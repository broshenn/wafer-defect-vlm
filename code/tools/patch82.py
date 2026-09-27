"""Add run 43's column to the three remaining hand-written tables.

Run 43 (GSPO, G=4, lr 5e-5) landed. The 5.2.2 table already has its column; these
three do not, and each one carries a bold "best in row" marker that a new column can
move. So the same property is checked in all three: the cell values are read from the
run's records, and the bold is *recomputed* from the resulting row rather than left
where it was -- a stale bold is a claim that the reader cannot see is stale.

For 5.2.5's profile table this matters: run 43 has the lowest clock MAE (1.6761 vs
SFT's 1.6847) and the lowest size MAE (0.4025 vs 0.4999) of any run so far, so two
rows' "best" moves off SFT. The prose bullets that read those bolds are corrected
separately, in patch83.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
# Overridable so the whole patch can be rehearsed against a copy and diffed before it
# touches the document.
DOC = Path(os.environ.get("WAFER_DOC", str(ROOT / "LIMITATIONS.md")))
REP = json.loads((ROOT / "outputs/reports/qwen35_9b_gspo_g4_lr5e5__report.json")
                 .read_text(encoding="utf-8"))
TR = json.loads((ROOT / "outputs/reports/qwen35_9b_gspo_g4_lr5e5_train_result.json")
                .read_text(encoding="utf-8"))

NEW_HEADER = "GSPO(G=4) lr5e-5"
AFTER = "GRPO(G=4) lr5e-5"


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render(cs):
    return "| " + " | ".join(cs) + " |"


def number_of(c):
    """The float behind a cell, or None. Handles the bold markers and the '36.00%'
    cells -- the idle row in the reward table is percentages, so without the '%' the
    row would look like it had no numbers in it at all."""
    try:
        return float(c.strip("*").rstrip("%"))
    except ValueError:
        return None


def is_number(c):
    return number_of(c) is not None


def insert(lines, header_key, after_label, values, direction, label, expected_rows):
    """Insert one column into the table whose header starts with header_key.

    values: row label -> (float, cell text). direction: 'min'/'max', or a dict from
    row label to one of those -- the profile table mixes rows where higher is better
    with rows where lower is better, and using one direction for the whole table
    bolds the *worst* cell in the other half.
    """
    hi = [i for i, l in enumerate(lines)
          if l.startswith("| " + header_key) and after_label in cells(l)]
    if len(hi) != 1:
        sys.exit(f"{label}: expected 1 table with header '{header_key}' and column "
                 f"'{after_label}', found {len(hi)}")
    hi = hi[0]
    head = cells(lines[hi])
    if NEW_HEADER in head:
        sys.exit(f"{label}: column already present")
    col = head.index(after_label) + 1

    rows, i = [], hi + 2
    while i < len(lines) and lines[i].startswith("|"):
        rows.append(i)
        i += 1
    got = [cells(lines[r])[0] for r in rows]
    if got != expected_rows:
        sys.exit(f"{label}: rows are {got}, expected {expected_rows}")

    out = {}
    for r in rows:
        cs = cells(lines[r])
        lab = cs[0]
        if lab not in values:
            sys.exit(f"{label}: no value for row '{lab}'")
        v, text = values[lab]
        new = cs[:col] + [text] + cs[col:]
        if len(new) != len(cs) + 1:
            sys.exit(f"{label}: row '{lab}' cell count {len(cs)} -> {len(new)}")
        # Recompute the bold from the whole row, including the new cell.
        numeric = [(j, number_of(c)) for j, c in enumerate(new[1:], 1) if is_number(c)]
        if not numeric:
            sys.exit(f"{label}: row '{lab}' has no numeric cells to bold")
        for j, _ in numeric:
            new[j] = new[j].strip("*")
        d = direction[lab] if isinstance(direction, dict) else direction
        best = (min if d == "min" else max)(numeric, key=lambda t: t[1])[0]
        new[best] = "**" + new[best] + "**"
        out[r] = render(new)

    lines = list(lines)
    for r, text in out.items():
        lines[r] = text
    lines[hi] = render(cells(lines[hi])[:col] + [NEW_HEADER] + cells(lines[hi])[col:])
    lines[hi + 1] = render(cells(lines[hi + 1])[:col] + ["---"] + cells(lines[hi + 1])[col:])
    print(f"{label}: column {col + 1} added ({len(out)} rows); bold recomputed "
          f"({direction})")
    for r in rows:
        cs = cells(lines[r])
        bold = [c for c in cs[1:] if c.startswith("**")]
        print(f"    {cs[0]:<28} best={bold[0] if bold else '?'}")
    return lines


st, cls = REP["structured"], REP["classification"]
src = DOC.read_text(encoding="utf-8").split("\n")
before_bold = "\n".join(src).count("**")

# ---------------------------------------------------- 5.2.3 reward signal table
sig = {k: v["mean_std_across_steps"] for k, v in TR["reward_signal"].items()}
t1 = insert(
    src, "奖励的组内标准差（150 步均值）", AFTER,
    {f"`{k}`": (sig[k], f"{sig[k]:.4f}") for k in
     ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock")}
    | {"**空转步占比**": (TR["mean_frac_reward_zero_std"],
                          f"{TR['mean_frac_reward_zero_std']:.2%}")},
    "min", "5.2.3 reward-std table",
    ["`WaferClass`", "`WaferFormat`", "`WaferRadial`", "`WaferClock`",
     "**空转步占比**"])

# --------------------------------------------- 5.2.5 structured-profile table
# The full prefix, not just "指标": 5.2.2's table also starts with 指标 and also now
# has a GRPO(G=4) lr5e-5 column, so a shorter key would match two tables and stop.
t2 = insert(
    t1, "指标 | SFT | GRPO(G=4) lr1e-5", AFTER,
    {"`radial_zone`（越高越好）": (st["field_accuracy"]["radial_zone"],
                                   f"{st['field_accuracy']['radial_zone']:.4f}"),
     "时钟 MAE（越低越好）": (st["clock_circular_mae_sectors"],
                             f"{st['clock_circular_mae_sectors']:.4f}"),
     "尺寸 MAE（越低越好）": (st["size_mae_r"], f"{st['size_mae_r']:.4f}"),
     "caption must-hit（越高越好）": (REP["caption"]["must_hit_rate"],
                                     f"{REP['caption']['must_hit_rate']:.4f}")},
    {"`radial_zone`（越高越好）": "max", "时钟 MAE（越低越好）": "min",
     "尺寸 MAE（越低越好）": "min", "caption must-hit（越高越好）": "max"},
    "5.2.5 profile table",
    ["`radial_zone`（越高越好）", "时钟 MAE（越低越好）", "尺寸 MAE（越低越好）",
     "caption must-hit（越高越好）"])

# ------------------------------------------------- 5.2.5 per-class F1 table
pc = {k: v["f1-score"] for k, v in cls["per_class"].items()}
t3 = insert(
    t2, "类别（F1）", AFTER,
    {c: (pc[c], f"{pc[c]:.3f}") for c in ("Donut", "none", "Edge_Ring", "Scratch")},
    "max", "5.2.5 per-class table", ["Donut", "none", "Edge_Ring", "Scratch"])

after_bold = "\n".join(t3).count("**")
if after_bold != before_bold:
    # Each row keeps exactly one bold either way; only the header may add a pair,
    # and the two tables that moved a bold must net to zero.
    print(f"note: bold markers {before_bold} -> {after_bold}")
if after_bold % 2:
    sys.exit("bold markers are odd; a span would run to the end of the document")

DOC.write_text("\n".join(t3), encoding="utf-8")
print(f"written: {DOC} ({len(t3)} lines)")
