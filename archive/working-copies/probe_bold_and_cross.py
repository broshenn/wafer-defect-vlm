"""Two things nothing reads: the 476 table's cells against the main table's, and its bold.

LIMITATIONS.md carries two nine-column tables. The main one (line 247) is now read cell
by cell. The second (line 476) holds four of the same rows for the same runs -- and its
bold is *stated* to mean "该行最好的一档" (the best value of that row, per the direction
named in the row label):

    | `radial_zone`（越高越好） | ... | **0.4603** |
    | 时钟 MAE（越低越好）      | ... | 1.6860     |   <- bold on the minimum instead

So the emphasis is a claim, and a claim about which cell holds an extreme is exactly the
kind that stays byte-for-byte true while its subject moves: add a run with a lower clock
MAE and the bold is on the wrong cell with every number in the table still correct.

`add_run_columns.py` recomputes the bold when it appends a column, so the intended
pipeline maintains it. Nothing verifies it. This probe checks (a) that the bold sits on
the row's true best value under the direction the label states, (b) exactly one cell per
row is bold, and (c) the four shared rows agree with the main table's values, so the two
tables cannot come to contradict each other with both individual rows "correct".
"""
import json
import re
import pathlib

REP = pathlib.Path("/root/autodl-fs/wafer-vlm/outputs/reports")
DOC = pathlib.Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
text = DOC.read_text(encoding="utf-8")

RUNS = {
    "SFT": "qwen35_9b_adapter",
    "GRPO(G=4) lr1e-5": "qwen35_9b_grpo",
    "GRPO(G=4) lr5e-5": "qwen35_9b_grpo_lr5e5",
    "GSPO(G=4) lr5e-5": "qwen35_9b_gspo_g4_lr5e5",
    "GSPO(G=8) lr1e-5": "gspo_lr1e5",
    "GSPO(G=8) lr5e-5": "qwen35_9b_gspo_v1",
    "GSPO(G=32) lr5e-5": "qwen35_9b_gspo_g32",
    "GSPO(G=4) lr1e-5": "qwen35_9b_gspo_g4_lr1e5",
    "GSPO(G=32) lr1e-5": "qwen35_9b_gspo_g32_lr1e5",
}
P = {}
for n, t in RUNS.items():
    p = REP / f"{t}__report.json"
    if p.is_file():
        d = json.loads(p.read_text(encoding="utf-8"))
        P[n] = dict(
            radial=d["structured"]["field_accuracy"]["radial_zone"],
            clock=d["structured"]["clock_circular_mae_sectors"],
            size=d["structured"]["size_mae_r"],
            must_hit=d["caption"]["must_hit_rate"],
        )

BOLD = re.compile(r"^\*\*(.+?)\*\*$")
LBL = re.compile(r"^(.*?)（(越高越好|越低越好)）$")


def col_key(c):
    c = re.sub(r"\s+", " ", c).strip()
    m = re.fullmatch(r"([A-Za-z]+)\s*\(G=(\d+),\s*(lr[\d.eE+-]+)\)", c)
    return f"{m.group(1)}(G={m.group(2)}) {m.group(3)}" if m else c


def tables():
    """Every table whose header row starts with `| 指标 |` -- as (line, head, rows)."""
    lines = text.split("\n")
    out = []
    for i, l in enumerate(lines):
        if not (l.startswith("| 指标 |") and i + 1 < len(lines)
                and set(lines[i + 1].replace("|", "").replace("-", "").strip()) == set()):
            continue
        j, r = i + 2, {}
        while j < len(lines) and lines[j].startswith("|"):
            c = [x.strip() for x in lines[j].strip().strip("|").split("|")]
            r.setdefault(c[0].strip().strip("*").replace("`", "").strip(), c[1:])
            j += 1
        out.append((i, [c.strip() for c in l.strip().strip("|").split("|")][1:], r))
    return out


def isb(c):
    return bool(BOLD.match(c.strip()))


def val(c):
    return float(BOLD.match(c.strip()).group(1).strip()) if isb(c) else float(c.strip())


# Selected by width first: the 5.2.2 table carries the same row *labels* as the main one
# (it is the SFT/GRPO/delta view), so matching on a label alone picks a three-column
# table and reads it against eight columns -- the mistake patch106 made on the first try.
_nine = [t for t in tables() if len(t[1]) == 8]
main = [t for t in _nine if "retrieval mAP@10" in t[2]][0]
second = [t for t in _nine if t is not main][0]
mcols = [col_key(c) for c in main[1]]
scols = [col_key(c) for c in second[1]]
print(f"main table line {main[0] + 1}: {len(mcols)} cols, {len(main[2])} rows")
print(f"second table line {second[0] + 1}: {len(scols)} cols, {len(second[2])} rows")
print(f"same column order: {mcols == scols}")
print(f"  main   : {mcols}")
print(f"  second : {scols}")
print()

print("--- (c) the four shared rows agree between the tables ---")
# By run key, never by position: the two tables list the two GSPO(G=8) runs in opposite
# orders, so a positional comparison across them compares one run's value against
# another's -- the same mistake in the other direction.
MAIN_LABEL = {
    "radial_zone（越高越好）": "structured radial_zone",
    "时钟 MAE（越低越好）": "clock circular MAE",
    "尺寸 MAE（越低越好）": "size MAE (R)",
    "caption must-hit（越高越好）": "caption must-hit",
}
cross_bad = 0
for lab in second[2]:
    m = MAIN_LABEL.get(lab)
    if m is None or m not in main[2]:
        print(f"  {lab}: no counterpart in the main table ({m!r})")
        cross_bad += 1
        continue
    a = {c: round(val(v), 4) for c, v in zip(mcols, main[2][m])}
    b = {c: round(val(v), 4) for c, v in zip(scols, second[2][lab])}
    if a == b:
        print(f"  {lab:<24} agree ({len(b)} cells, compared by run)")
    else:
        cross_bad += 1
        print(f"  {lab:<24} DISAGREE")
        for c in b:
            if a.get(c) != b[c]:
                print(f"      {c:<22} main={a.get(c)} second={b[c]}")
print()

print("--- (a)/(b) the second table's bold is the row's best value ---")
bold_bad = 0
for lab, cells in second[2].items():
    m = LBL.match(lab)
    if not m:
        print(f"  {lab}: label does not state a direction")
        continue
    name, direction = m.group(1), m.group(2)
    best = min if direction == "越低越好" else max
    marks = [i for i, c in enumerate(cells) if isb(c)]
    vals = [val(c) for c in cells]
    target = best(range(len(vals)), key=lambda i: vals[i])
    if len(marks) != 1:
        bold_bad += 1
        print(f"  {lab:<24} {len(marks)} bold cell(s); the document says 加粗 = 该行最好的一档")
    elif marks[0] != target:
        bold_bad += 1
        print(f"  {lab:<24} MISBOLD: bold on {mcols[marks[0]]} ({vals[marks[0]]}), "
              f"best is {mcols[target]} ({vals[target]})")
    else:
        print(f"  {lab:<24} ok: {mcols[target]} {vals[target]} is the "
              f"{'minimum' if direction == '越低越好' else 'maximum'}")
    if len(scols) == len(mcols):
        exp = [round(P[c][{"radial_zone": "radial", "时钟 MAE": "clock",
                           "尺寸 MAE": "size", "caption must-hit": "must_hit"}[name]], 4)
               for c in scols]
        if [round(v, 4) for v in vals] != exp:
            bold_bad += 1
            print(f"      values disagree with the records: {vals} vs {exp}")
print()
print(f"cross-table disagreements: {cross_bad}; bold findings: {bold_bad}")
