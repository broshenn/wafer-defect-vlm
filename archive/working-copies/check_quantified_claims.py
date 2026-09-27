"""Recompute the document's claims *about the run set*, because adding a run breaks them.

A handful of sentences in LIMITATIONS.md do not state a value. They state a fact about
the whole set of runs:

    "六个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 GRPO lr1e-5"
    "G=32 的 0.4603 在七个 run 里最高"
    "在六个 RL run 里，恰好只有两个有类别归零"
    "是六个 RL run 里最低的"

Every one was true when written. Every one fails in the way this project keeps finding
(section 8 item 9): the numbers inside the sentence stay individually correct, the run
set moves underneath them, and the artefact cannot show that it happened. The tables
survive this -- `tools/add_run_columns.py` appends a column and recomputes the bold --
but the prose does not, and the document says so about itself in 5.2.5 ("在任何新 run
落地后，下面四句里的「七个 run」都必须重核而不是沿用").

This is that re-check, run in the same pass that lands the run. It recomputes each fact
from `outputs/reports/*__report.json` and compares it with what the sentence says --
the count, and where the sentence names a run, which run. It also cross-checks the two
tables' column sets against the reports on disk, which is the same question asked from
the other side: a landing that wrote a report but no column is caught here too.

The claims are of two kinds, and the tool treats them differently on purpose:

  a **count** is arithmetic -- "七个 run" became "八个 run" because a run landed. That
  is mechanical, and `--fix` rewrites the numeral in place.

  a **fact** is an interpretation -- which run is highest, which runs lose a class,
  whether the winner is still unique. `--fix` never touches one. Auto-editing those
  would produce a sentence nobody chose, the same failure as the unsubstituted
  placeholders (item 14), where the artefact read as complete while carrying nothing.

`--fix` also refuses to touch a count while any fact has moved: the numeral in "N 个 RL
run 里唯一 ... 的是 X" is not independent of the fact in the same sentence, and writing
"九个 ... 里唯一 ... 的是 X" when it is no longer unique would be exactly the defect
this tool exists to catch.

Exit status: 0 everything matches; 2 count(s) only -- `--fix` will settle it; 1 a fact
moved and the sentence has to be rewritten by hand.

Run from the repository root:
    venvs/wafer/bin/python tools/check_quantified_claims.py [--fix]
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
REP = ROOT / "outputs/reports"
# Overridable so the --fix path can be exercised against a copy. It writes the
# document, and an auto-editor that has never been run on something disposable is an
# auto-editor whose first action is on the real thing.
DOC = Path(os.environ.get("WAFER_DOC", str(ROOT / "LIMITATIONS.md")))

CLASSES = ("Donut", "none", "Edge_Ring", "Scratch")

# Display name -> report tag. The display name is the column header the two tables
# use; the tag is the file stem. Every run in the comparison is listed here, so a run
# that lands and is not listed here shows up as a table column with no record behind
# it -- which is the point of the cross-check below.
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

COL = re.compile(r"^([A-Za-z]+)\s*\(\s*G=(\d+)\s*[,)]?\s*lr\s*([0-9.eE+-]+)\s*\)?$")
CN = "一二三四五六七八九十"

# A numeral the document might write either way: 六个 or 6 个. Non-capturing on purpose
# -- every pattern below wraps it in "(" + NUM + ")" for the capture it wants, and a
# group inside NUM would shift every index by one.
NUM = r"(?:[一二三四五六七八九十两]+|\d+)"


def cn(n: int) -> str:
    """The numeral the document writes for n here (一..十, then 十X)."""
    if n <= 10:
        return CN[n - 1]
    if n < 20:
        return "十" + CN[n - 11]
    return str(n)


def cn2int(s: str) -> int:
    """Read either numeral back. The document writes counts both ways -- 六个 in 5.2.5
    and "6 个 run" in section 9 -- and a pattern that accepts both must be able to read
    both, or it fails on the ones it accepted."""
    s = s.strip()
    if s.isdigit():
        return int(s)
    if s in ("两", "二"):
        return 2
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + CN.index(s[1]) + 1
    if s.endswith("十"):
        return (CN.index(s[0]) + 1) * 10
    if s not in CN:
        raise SystemExit(f"cannot read {s!r} as a count; the pattern that captured it "
                         f"and the reader that parses it disagree")
    return CN.index(s) + 1


def load(tag: str):
    p = REP / f"{tag}__report.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    cls, st, cap = d["classification"], d["structured"], d["caption"]
    return {
        "macro_f1": cls["macro_f1"],
        "radial_zone": st["field_accuracy"]["radial_zone"],
        "clock_mae": st["clock_circular_mae_sectors"],
        "size_mae": st["size_mae_r"],
        "must_hit": cap["must_hit_rate"],
        "per_class": {k: v["f1-score"] for k, v in cls["per_class"].items()},
    }


records = {name: load(tag) for name, tag in RUNS.items()}
present = {n: v for n, v in records.items() if v is not None}
runs = [n for n in RUNS if n in present]          # all runs, in declaration order
rl = [n for n in runs if n != "SFT"]              # the RL runs
print(f"records on disk: {len(runs)} runs ({len(rl)} RL + SFT); "
      f"absent: {sorted(set(RUNS) - set(runs)) or 'none'}")

text = DOC.read_text(encoding="utf-8")
facts, counts = [], []           # names, by kind
fixes = []                       # (pattern, group index, new count) for --fix


def anchor(name: str, pattern: str):
    """Match, or say so. A missing anchor is a finding, not a pass: a sentence this
    tool was written against has changed shape, and nothing was verified."""
    m = re.search(pattern, text, re.S)
    if not m:
        print(f"  STALE  {name}")
        print("         the sentence this check was written against is not in the "
              "document any more; nothing was verified")
        facts.append(name)
        return None
    return m


def cmp(name: str, asserted, computed, kind: str = "fact", pattern: str = None,
        group: int = None, why: str = "") -> None:
    ok = asserted == computed
    print(f"  {'ok   ' if ok else 'STALE'}  {name}   [{kind}]")
    print(f"         document says: {asserted}")
    print(f"         records say  : {computed}")
    if why and not ok:
        print(f"         {why}")
    if ok:
        return
    if kind == "count":
        counts.append(name)
        if pattern is not None:
            fixes.append((pattern, group, computed))
    else:
        facts.append(name)


def label_of(value: float, by: str, runs_: list[str], reverse: bool = False) -> str:
    """The run that actually holds this extreme -- not the run the sentence names."""
    best = (min if reverse else max)(runs_, key=lambda n: present[n][by])
    return best if abs(present[best][by] - value) < 5e-5 else f"<{value} not any run's {by}>"


def doc_name(label: str) -> str:
    """How 5.2.4 names a run in prose: without the group size it repeats everywhere.

    Not a formatting nicety -- the sentence says "唯一...的是 GRPO lr1e-5" and the
    column is "GRPO(G=4) lr1e-5", so asking whether the document names the same run
    means asking whether it names it the way that sentence does.
    """
    return re.sub(r"\s+", " ", re.sub(r"\(G=\d+\)", "", label)).strip()


# ---------------------------------------------------------------- the run columns
# The two tables that carry every run. Reading their headers is the same question from
# the other side: does the document list exactly the runs that have records?
def header_of(first_cell_starts: str, sentinel_row: str) -> list[str]:
    lines = text.split("\n")
    for i, l in enumerate(lines):
        if not l.startswith(first_cell_starts):
            continue
        rows, j = [], i + 2
        while j < len(lines) and lines[j].startswith("|"):
            rows.append(lines[j].strip().strip("|").split("|")[0].strip())
            j += 1
        if sentinel_row in rows:
            return [c.strip() for c in l.strip().strip("|").split("|")[1:]]
    return []


profile_head = header_of("| 指标 | SFT |", "时钟 MAE（越低越好）")
reward_head = header_of("| 奖励的组内标准差", "`WaferClass`")
p_cols = [c for c in profile_head if COL.match(c) or c == "SFT"]
r_cols = [c for c in reward_head if COL.match(c)]
print(f"profile table: {len(p_cols)} run columns; reward table: {len(r_cols)}")
cmp("the profile table's columns match the records on disk",
    sorted(p_cols), sorted(runs),
    why="a report with no column (or a column with no record) is a landing that half "
        "happened")
cmp("the reward table's columns match the RL runs on disk",
    sorted(r_cols), sorted(rl),
    why="this table carries no SFT column -- it is the runs with a reward signal")

sft_f1 = present["SFT"]["macro_f1"]

# ------------------------------------------------- 5.2.4: which RL runs beat SFT
# "N 个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 X（A 对 B）"
P_ABOVE = (r"(" + NUM + r")\s*个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 "
           r"([^（\n]+)（([\d.]+) 对 ([\d.]+)）")
m = anchor("5.2.4: the only RL run whose macro-F1 is above SFT", P_ABOVE)
if m:
    n, who, val, base = (cn2int(m.group(1)), doc_name(m.group(2).strip()),
                         float(m.group(3)), float(m.group(4)))
    above = sorted((n_ for n_ in rl if present[n_]["macro_f1"] > sft_f1),
                   key=lambda n_: -present[n_]["macro_f1"])
    cmp("5.2.4: the RL-run count", n, len(rl), kind="count",
        pattern=P_ABOVE, group=1)
    cmp("5.2.4: the set of RL runs above SFT, by name",
        [who], [doc_name(x) for x in above] or ["（none: no RL run beats SFT）"])
    cmp("5.2.4: the value quoted for it and for SFT", (val, base),
        (round(present[above[0]]["macro_f1"], 4), round(sft_f1, 4)) if above
        else (None, round(sft_f1, 4)))

# "GRPO lr5e-5 与 lr 5e-5 的三个 GSPO run（G=4 A、G=8 B、G=32 C）都低于 SFT"
g = anchor("5.2.4: the three GSPO lr 5e-5 runs quoted beside it",
           r"lr 5e-5 的三个 GSPO run（G=4 ([\d.]+)、G=8 ([\d.]+)、G=32 ([\d.]+)）")
if g:
    three = sorted((n for n in rl if n.startswith("GSPO") and n.endswith("lr5e-5")),
                   key=lambda n_: int(re.search(r"G=(\d+)", n_).group(1)))
    cmp("5.2.4: the three GSPO lr 5e-5 runs (count and values)",
        tuple(round(float(x), 4) for x in g.groups()),
        tuple(round(present[n]["macro_f1"], 4) for n in three))

# --------------------------------------- 5.2.5 (8.9): the extremes of the profile table
P_RADIAL = (r"G=32 的 ([\d.]+) 在(" + NUM + r")个 run 里最高，\s*是 SFT 的 "
            r"([\d.]+) 倍")
g = anchor("8.9: the radial_zone maximum", P_RADIAL)
if g:
    best = label_of(present["GSPO(G=32) lr5e-5"]["radial_zone"], "radial_zone", runs)
    cmp("8.9: the run holding the radial_zone maximum", best, "GSPO(G=32) lr5e-5")
    cmp("8.9: the radial_zone maximum and its ratio to SFT",
        (float(g.group(1)), float(g.group(3))),
        (round(present["GSPO(G=32) lr5e-5"]["radial_zone"], 4),
         round(present["GSPO(G=32) lr5e-5"]["radial_zone"] / present["SFT"]["radial_zone"], 2)))
    cmp("8.9: the run count in the radial_zone sentence", cn2int(g.group(2)), len(runs),
        kind="count", pattern=P_RADIAL, group=2)

P_SIZE = r"G=32 的 ([\d.]+) 在(" + NUM + r")个 run 里最高（即最差）"
g = anchor("8.9: the size-MAE maximum (the worst)", P_SIZE)
if g:
    best = label_of(present["GSPO(G=32) lr5e-5"]["size_mae"], "size_mae", runs)
    cmp("8.9: the run holding the size-MAE maximum", best, "GSPO(G=32) lr5e-5")
    cmp("8.9: the size-MAE maximum", float(g.group(1)),
        round(present["GSPO(G=32) lr5e-5"]["size_mae"], 4))
    cmp("8.9: the run count in the size-MAE sentence", cn2int(g.group(2)), len(runs),
        kind="count", pattern=P_SIZE, group=2)

# ------------------------ 5.2.5: which RL runs lose a class, and which lose none
P_ZEROED = r"在(" + NUM + r")个 RL run 里，\*\*恰好只有(" + NUM + r")个有类别归零\*\*"
g = anchor("5.2.5: the number of RL runs with a zeroed class", P_ZEROED)
if g:
    zeroed = sorted(n for n in rl
                    if any(present[n]["per_class"].get(c, 1.0) == 0.0 for c in CLASSES))
    cmp("5.2.5: the RL-run count (zeroed-class sentence)", cn2int(g.group(1)), len(rl),
        kind="count", pattern=P_ZEROED, group=1)
    cmp("5.2.5: the number of RL runs with a zeroed class", cn2int(g.group(2)),
        len(zeroed),
        why="which runs lose a class is a fact, not a count -- a new run that zeroes "
            "one changes the sentence, not just its numeral")
    for n in zeroed:
        print(f"         {n} zeroes: "
              f"{[c for c in CLASSES if present[n]['per_class'].get(c, 1.0) == 0.0]}")

g = anchor("5.2.5: how many lr 1e-5 runs keep both classes",
           r"而(" + NUM + r")个 lr 1e-5 的 run 两个类别都在")
if g:
    low = [n for n in rl if n.endswith("lr1e-5")]
    cmp("5.2.5: the number of lr 1e-5 RL runs", cn2int(g.group(1)), len(low),
        why="the sentence also names the values of those runs, so a third one is a "
            "rewrite, not a count bump")
    print(f"         they are: {'、'.join(low)}")

# The scope here is the *lr 5e-5* runs, not all RL runs: the sentence contrasts them
# with the two zeroed columns, which are both lr 5e-5. Reading it as "all RL runs" was
# this tool's first bug -- it reported four intact runs against a sentence that says
# two, and the sentence was right.
g = anchor("5.2.5: the lr 5e-5 runs that lose no class",
           r"但有(" + NUM + r")个 run 一个类别都没丢：([^（\n]+)（run 43）")
if g:
    five = [n for n in rl if n.endswith("lr5e-5")]
    intact = sorted(n for n in five
                    if all(present[n]["per_class"].get(c, 0.0) > 0.0 for c in CLASSES))
    cmp("5.2.5: the number of lr 5e-5 runs that lose no class",
        cn2int(g.group(1)), len(intact))
    cmp("5.2.5: which lr 5e-5 runs they are",
        sorted(x.strip() for x in re.split(r"与|、", g.group(2).strip())),
        sorted(intact))
    zeroed5 = [n for n in five if n not in intact]
    print(f"         of the {len(five)} lr 5e-5 runs, zeroed: "
          f"{'、'.join(zeroed5)}")

# --------------------------------- 5.2.5: the lowest macro-F1 among the RL runs
P_LOW = r"是(" + NUM + r")个 RL run 里最低的"
g = anchor("5.2.5: the lowest RL macro-F1", P_LOW)
if g:
    best = min(rl, key=lambda n: present[n]["macro_f1"])
    cmp("5.2.5: the RL-run count (lowest-F1 sentence)", cn2int(g.group(1)), len(rl),
        kind="count", pattern=P_LOW, group=1)
    cmp("5.2.5: the run with the lowest RL macro-F1",
        (best, round(present[best]["macro_f1"], 4)), ("GSPO(G=4) lr5e-5", 0.491))

# ------------------------------- 5.2.5: the reward table's lowest-of-N per reward row
P_REWARD = r"在 4 个奖励里有 (" + NUM + r") 个是(" + NUM + r")组最低"
g = anchor("5.2.5: how many reward rows G=32 is lowest in", P_REWARD)
if g:
    low = 0
    for k in ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock"):
        vals = {}
        for n in r_cols:
            p = REP / f"{RUNS[n]}_train_result.json"
            if p.is_file():
                vals[n] = json.loads(p.read_text(encoding="utf-8"))[
                    "reward_signal"][k]["mean_std_across_steps"]
        if vals and min(vals, key=vals.get) == "GSPO(G=32) lr5e-5":
            low += 1
    cmp("5.2.5: how many of the 4 reward rows G=32 is lowest in", cn2int(g.group(1)), low)
    cmp("5.2.5: the number of columns 'lowest' is measured over",
        cn2int(g.group(2)), len(r_cols), kind="count",
        pattern=P_REWARD, group=2)

# ------------------------------------------------------------------------ verdict
print()
if not facts and not counts:
    print(f"all {len(runs)} runs accounted for; every quoted count, extreme and set "
          f"matches the records")
    sys.exit(0)

if facts:
    print(f"{len(facts)} fact(s) about the run set have moved. The document is not "
          f"wrong about any number; it is describing a run set that has changed, and "
          f"the sentence has to be rewritten:")
    for s in facts:
        print(f"  - {s}")
    if counts:
        print(f"\n{len(counts)} count(s) are also stale. NOTHING WAS WRITTEN: the "
              f"numeral in a sentence like 「N 个 run 里唯一 … 的是 X」 is not "
              f"independent of the fact beside it, so fixing the count while the fact "
              f"is in question would produce exactly the defect this tool is for.")
    sys.exit(1)

print(f"{len(counts)} count(s) are stale and everything else matches:")
for s in counts:
    print(f"  - {s}")
if "--fix" not in sys.argv[1:]:
    print("\nre-run with --fix to rewrite them in place (counts only; no fact above "
          "has moved).")
    sys.exit(2)

out = text
for pattern, group, value in fixes:
    mm = re.search(pattern, out, re.S)
    if not mm:
        sys.exit(f"the anchor for a count fix stopped matching between passes "
                 f"({pattern!r}); nothing written")
    s, e = mm.span(group)
    out = out[:s] + cn(value) + out[e:]
DOC.write_text(out, encoding="utf-8")
print(f"\nwrote {len(fixes)} count(s) into {DOC.name} ({len(out.splitlines())} lines). "
      f"Re-run without --fix to confirm.")
