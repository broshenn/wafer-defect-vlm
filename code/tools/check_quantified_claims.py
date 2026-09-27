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

import ast
import json
import os
import re
import sys
from pathlib import Path

# An uncaught exception is not a verdict on the document. The exit codes below are
# load-bearing -- 1 means "a fact has moved, rewrite the sentence", 2 means "counts only,
# --fix may write", 0 means green -- and Python's default exit code for an uncaught
# exception is 1, which a caller cannot tell apart from a moved fact. This happened: an
# insertion landed 40 lines too high, referenced a module-level name not yet assigned,
# and the tool exited 1 having printed zero stale items. `tools/land_finish.sh` reads
# only the exit code and printed "a fact about the run set has moved" -- an assertion
# about the document that nothing had computed. Crashes exit 3 so no caller can confuse
# the two, and the sentence above this line is now true.
def _on_crash(kind, value, tb):
    import traceback
    traceback.print_exception(kind, value, tb)
    print("\nCHECKER DID NOT FINISH (exit 3): the failure above is in this tool, not a "
          "finding about LIMITATIONS.md. Nothing here says a fact has moved.")
    sys.exit(3)


sys.excepthook = _on_crash

ROOT = Path("/root/autodl-fs/wafer-vlm")
REP = ROOT / "outputs/reports"
# Overridable so the --fix path can be exercised against a copy. It writes the
# document, and an auto-editor that has never been run on something disposable is an
# auto-editor whose first action is on the real thing.
DOC = Path(os.environ.get("WAFER_DOC", str(ROOT / "LIMITATIONS.md")))

# The class list is derived, not declared -- see the block after the records are read.
# A hardcoded subset only asks about the classes that had been seen to zero, which is
# the one thing a run that lands later is free to change without the sentence beside it
# moving. The four names that used to be here (Donut, none, Edge_Ring, Scratch) are the
# classes the sentences happen to name; they are a subset of the answer, not the answer.


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
        # The main comparison table's remaining rows. Each is kept as None when the
        # report carries no such block, so a cell can tell "not measured" from
        # "measured as zero" -- the table renders those two differently (a dash against
        # a number) and only one of them is a measurement.
        "accuracy": cls.get("accuracy"),
        "f1_ci": cls.get("macro_f1_95ci"),
        "defect_type": st["field_accuracy"].get("defect_type"),
        "robustness": (d.get("robustness") or {}).get("accuracy"),
        "flip": (d.get("robustness") or {}).get("flip_rate_vs_clean"),
        "retrieval": (d.get("retrieval") or {}).get("mAP@10"),
        "per_class": {k: v["f1-score"] for k, v in cls["per_class"].items()},
    }


records = {name: load(tag) for name, tag in RUNS.items()}
present = {n: v for n, v in records.items() if v is not None}
runs = [n for n in RUNS if n in present]          # all runs, in declaration order
rl = [n for n in runs if n != "SFT"]              # the RL runs
print(f"records on disk: {len(runs)} runs ({len(rl)} RL + SFT); "
      f"absent: {sorted(set(RUNS) - set(runs)) or 'none'}")

# Every class any report carries, sorted. Derived from the records so that a class no
# run had zeroed when this tool was written is still looked at by the zeroing sentences
# below -- those sentences say "a class", not "one of four classes".
CLASSES = tuple(sorted({c for r in present.values() for c in r["per_class"]}))

text = DOC.read_text(encoding="utf-8")
facts, counts = [], []           # names, by kind
fixes = []                       # (pattern, group index, new count) for --fix


def anchor_any(name: str, patterns: tuple):
    """Match the first of several shapes a sentence may legitimately take.

    A single fixed pattern makes this tool a second author of the document: the prose
    cannot change shape without the check going STALE, and STALE reads as "nothing was
    verified" and stops the landing. Some of these sentences have more than one honest
    shape -- 5.2.4's uniqueness claim stops being writable as soon as a second run
    clears SFT -- so each shape is spelled out and every one of them is verified the
    same way: against the computed set of runs and the computed values, never against
    the wording.
    """
    for pat in patterns:
        m = re.search(pat, text, re.S)
        if m:
            return m, pat
    print(f"  STALE  {name}")
    print("         the sentence this check was written against is not in the "
          "document any more, in any of the shapes it has; nothing was verified")
    facts.append(name)
    return None, None


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


# A run's report is `{tag}__report.json` and its training record is
# `{tag}_train_result.json` -- except for the two runs that predate the convention and
# named their record after the algorithm. That exception used to be handled by an
# `if p.is_file()` inside the reward-row loop, which dropped the column silently and
# ranked "lowest of N" over a subset while the document said N. It happened to give the
# same answer, which made it unfalsifiable rather than wrong.
TRAIN_RESULT = {
    "GRPO(G=4) lr1e-5": "grpo_train_result",
    "SFT": "sft_train_result",
}


def train_record(name: str) -> Path:
    """Where this run's training record lives. Not a guess: a wrong name here shows up
    as the missing-record finding below, never as a quiet subset."""
    # The map holds a whole file stem, and so does the fallback. The first version of
    # this appended the suffix to the map's value as well, producing
    # `grpo_train_result_train_result.json` -- a wrong lookup in the harmless direction,
    # which is precisely why a missing record is a reported finding and not a skip.
    return REP / f"{TRAIN_RESULT.get(name, RUNS[name] + '_train_result')}.json"


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

# ------------------------------------------- the two tables, read cell by cell
# The tables are enumerated from their own headers, so a row is always read through the
# header that introduced it. That matters here and is not a stylistic choice: the
# document has two eight-column tables and they list the two GSPO(G=8) runs in opposite
# orders, so reading a row through the other table's header puts every value in the
# wrong run's column while each number stays correct. The first version of this item did
# exactly that -- it read the main table's retrieval row through `p_cols`, which is the
# second table's order, and reported a transposition in a document where neither table
# is wrong.
def _row_key(cell: str) -> str:
    """A row label without its emphasis.

    `**零优势步占比（实测）**` and the backticked row labels are the same label as the
    bare text; a reader that did not strip them would report a row as absent from a
    table it is sitting in.
    """
    return cell.strip().strip("*").replace("`", "").strip()


def _col_key(cell: str) -> str:
    """A column header as the display name RUNS declares.

    The two tables write the same run two ways -- `GRPO (G=4, lr1e-5)` in the main table
    and `GRPO(G=4) lr1e-5` in the second -- so the cell is normalised before it is used
    to look a record up. Without this every main-table column would read as an unknown
    run and every row would compare against nothing.
    """
    c = re.sub(r"\s+", " ", cell).strip()
    m = re.fullmatch(r"([A-Za-z]+)\s*\(G=(\d+),\s*(lr[\d.eE+-]+)\)", c)
    return f"{m.group(1)}(G={m.group(2)}) {m.group(3)}" if m else c


def _tables():
    """Every table in the document as (header cells, {row label: cells}).

    A table starts at a `| 指标 |` header whose next line is a separator row. Starting
    from the header is what keeps a row inside its own table.
    """
    lines = text.split("\n")
    out = []
    for i, l in enumerate(lines):
        if not l.startswith("| 指标 |") or i + 1 >= len(lines):
            continue
        if lines[i + 1].strip().strip("|").replace("-", "").replace("|", "").strip():
            continue
        rows, k = {}, i + 2
        while k < len(lines) and lines[k].startswith("|"):
            c = [x.strip() for x in lines[k].strip().strip("|").split("|")]
            rows.setdefault(_row_key(c[0]), c[1:])
            k += 1
        out.append(([c.strip() for c in l.strip().strip("|").split("|")][1:], rows))
    return out


_BOLD = re.compile(r"^\*\*(.+?)\*\*$")
_DIR = re.compile(r"^(.*?)\s*（(越高越好|越低越好)）$")


def _bare(cell: str) -> str:
    v = _BOLD.match(cell.strip())
    return (v.group(1) if v else cell).strip()


def _num(cell: str):
    """A cell as a number, or None when it is the not-measured dash.

    `add_run_columns.py` writes an unmeasured cell as an em dash on purpose; a checker
    that raised on it would be a checker that gets removed.
    """
    v = _bare(cell)
    return None if v in ("", "—", "-", "n/a") else float(v)


def _pct(cell: str):
    v = _bare(cell)
    return None if v in ("", "—", "-", "n/a") else round(float(v.rstrip("%")), 2)


_N = len(runs)

# Selected by width first, then by a row only the main table has: 5.2.2's table repeats
# most of the main table's row labels, so a label alone picks a three-column table.
_wide = [t for t in _tables() if len(t[0]) == _N]
_main = next((t for t in _wide if "retrieval mAP@10" in t[1]), None)
_second = next((t for t in _wide if t is not _main), None)

if _main is None:
    cmp("the main comparison table is findable",
        "an eight-column table with a retrieval mAP@10 row", "none",
        why="everything below reads that table by its own header; if the table cannot be "
            "found then nothing was verified, and an empty check is not a pass")
else:
    _mhead, _mrows = _main
    _mcols = [_col_key(c) for c in _mhead]
    cmp("the main table's columns match the records on disk", sorted(_mcols), sorted(runs),
        why="a report with no column, or a column with no record, is a landing that half "
            "happened -- and this table's column set was compared by nothing before")
    cmp("every row of the main table has one cell per column",
        sorted({len(v) for v in _mrows.values()}), [len(_mhead)],
        why="a row whose length does not match the header is a table that lost a cell, "
            "and reading it positionally would misalign every value after the gap")

    # row label -> the record field it carries. Every numeric row of the table is listed:
    # a row in the document and not here is a metric family that can go stale with this
    # tool green, which is what the last item below looks for.
    TABLE_FIELDS = (
        ("分类准确率", "accuracy"),
        ("macro-F1", "macro_f1"),
        ("structured defect_type", "defect_type"),
        ("structured radial_zone", "radial_zone"),
        ("clock circular MAE", "clock_mae"),
        ("size MAE (R)", "size_mae"),
        ("caption must-hit", "must_hit"),
        ("robustness accuracy", "robustness"),
        ("flip rate", "flip"),
        ("retrieval mAP@10", "retrieval"),
    )
    _CI_ROW = "macro-F1 95% CI"
    _IDLE_ROW = "零优势步占比（实测）"

    def _idle_rate(name: str):
        """A run's idle-step share, from its training record.

        Not from a report: this row is the one row of the table that is a property of
        training rather than of scoring. SFT has no reward signal, so its record carries
        no such field and its cell is the dash -- which is why a missing field reads as
        None rather than as zero.
        """
        p = train_record(name)
        if not p.is_file():
            return None
        v = json.loads(p.read_text(encoding="utf-8")).get("mean_frac_reward_zero_std")
        return v if isinstance(v, (int, float)) else None

    _bad = 0
    for _lab, _field in TABLE_FIELDS:
        if _lab not in _mrows:
            cmp(f"the main table's {_lab} row", "the row", "absent",
                why="a row this item reads is not in the table any more, so nothing "
                    "about that metric was verified")
            _bad += 1
            continue
        _got = [_num(c) for c in _mrows[_lab]]
        _exp = [None if present[c][_field] is None else round(present[c][_field], 4)
                for c in _mcols]
        _bad += _got != _exp
        cmp(f"the main table's {_lab} row, cell by cell", _got, _exp,
            why="a re-scored run rewrites its report and its column here; this row is "
                "compared against the record that produced it, one run at a time")

    if _CI_ROW in _mrows:
        def _ci(cell: str):
            m = re.fullmatch(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", cell.strip())
            return None if not m else [round(float(m.group(1)), 4),
                                       round(float(m.group(2)), 4)]
        _got = [_ci(c) for c in _mrows[_CI_ROW]]
        _exp = [None if present[c]["f1_ci"] is None
                else [round(present[c]["f1_ci"][0], 4), round(present[c]["f1_ci"][1], 4)]
                for c in _mcols]
        _bad += _got != _exp
        cmp("the main table's macro-F1 95% CI row, cell by cell", _got, _exp,
            why="an interval that no longer brackets the point estimate beside it is a "
                "re-scored run; the pair moves together, and a numeral fix would break it")
    else:
        _bad += 1
        cmp("the main table's macro-F1 95% CI row", "the row", "absent", why="as above")

    if _IDLE_ROW in _mrows:
        _got = [_pct(c) for c in _mrows[_IDLE_ROW]]
        _exp = [None if _idle_rate(c) is None else round(100 * _idle_rate(c), 2)
                for c in _mcols]
        _bad += _got != _exp
        cmp("the main table's idle-step row, cell by cell", _got, _exp,
            why="this row is a property of training, so it is read from the training "
                "record -- and SFT's dash is the honest cell there, not a missing value")
    else:
        _bad += 1
        cmp("the main table's idle-step row", "the row", "absent", why="as above")

    _unread = [_k for _k in _mrows
               if _k not in dict(TABLE_FIELDS) and _k not in (_CI_ROW, _IDLE_ROW)]
    cmp("every row of the main table is read by an item here", _unread, [],
        why="a row present in the document and absent from this list is a metric family "
            "whose numbers no item compares against a record -- the blind spot this "
            "whole block exists to close")

    print(f"         main table: {len(_mrows)} rows x {len(_mcols)} columns read cell "
          f"by cell against the reports ({_bad} disagreement(s))")

# ------------------------------------- the second table: agreement and the bold
# Its four rows restate values the main table carries, and its emphasis is a stated
# claim (5.2.5: 加粗 = 该行最好的一档). Neither was checked. The cross-table comparison
# goes by run name: the two tables list the two GSPO(G=8) runs in opposite orders, so a
# positional comparison across them compares one run's value against another's.
_REPEATS = {
    "radial_zone（越高越好）": "structured radial_zone",
    "时钟 MAE（越低越好）": "clock circular MAE",
    "尺寸 MAE（越低越好）": "size MAE (R)",
    "caption must-hit（越高越好）": "caption must-hit",
}
if _main is not None and _second is not None:
    _shead, _srows = _second
    _scols = [_col_key(c) for c in _shead]
    cmp("the second table's columns match the records on disk", sorted(_scols),
        sorted(runs),
        why="as for the main table: two tables over one run set must cover the same runs")
    cmp("every row of the second table has a counterpart here",
        [_k for _k in _srows if _k not in _REPEATS], [],
        why="a row added to this table and not to this list is a claim about its cells "
            "that nothing compares")
    for _lab, _mainlab in _REPEATS.items():
        if _lab not in _srows or _mainlab not in _mrows:
            continue
        cmp(f"the two tables agree on {_lab}",
            {c: round(_num(v), 4) for c, v in zip(_scols, _srows[_lab])},
            {c: round(_num(v), 4) for c, v in zip(_mcols, _mrows[_mainlab])},
            why="two tables carrying different values for one run leave a reader unable "
                "to tell which is authoritative, with each table internally consistent")
        # The bold is the claim "this cell holds the row's best value", in the direction
        # the label states. A tie is accepted: bolding any argmax is honest, and a check
        # that demanded one particular tied cell would fail on a run set that ties.
        _m = _DIR.match(_lab)
        if not _m:
            continue
        _marks = [i for i, c in enumerate(_srows[_lab]) if _BOLD.match(c.strip())]
        _vals = [round(_num(c), 4) for c in _srows[_lab]]
        _low = _m.group(2) == "越低越好"
        _pick = (min if _low else max)(range(len(_vals)), key=lambda i: _vals[i])
        cmp(f"{_lab}: exactly one cell is bold", len(_marks), 1,
            why="the document says 加粗 = 该行最好的一档, so the emphasis is a claim: two "
                "bold cells make it ambiguous and none makes it absent")
        cmp(f"{_lab}: the bold is on the row's {'minimum' if _low else 'maximum'}",
            _vals[_marks[0]] if _marks else None, _vals[_pick],
            why="a bold that stays where it was when a more extreme value lands is a "
                "number correct in a cell whose emphasis asserts the opposite")
        if _marks and _vals[_marks[0]] != _vals[_pick]:
            print(f"         bold is on {_scols[_marks[0]]} ({_vals[_marks[0]]}); the "
                  f"extreme is {_scols[_pick]} ({_vals[_pick]})")
    print(f"         second table: {len(_srows)} rows x {len(_scols)} columns, checked "
          f"against the main table per run and against its own bold claim")
elif _second is None:
    cmp("the second table is findable", "a second eight-column table", "none",
        why="its rows and its bold are read here; not finding it means nothing below "
            "ran, and an empty check is not a pass")

# Compared before the class sets are used: a report that lost a class would make every
# "does this run keep every class" answer below refer to a smaller population, and the
# only visible symptom would be a class missing from a table nobody re-counted.
_class_sets = {n: tuple(sorted(v["per_class"])) for n, v in present.items()}
_variants = sorted(set(_class_sets.values()))
cmp("every report carries the same class set", _variants, _variants[:1],
    why="two reports that disagree on the classes cannot be compared per class; and a "
        "class that is simply absent reads as a class with no score, which the intact "
        "checks below would count as a lost class")

sft_f1 = present["SFT"]["macro_f1"]

# ------------------------------------------------- 5.2.4: which RL runs beat SFT
# Two shapes, because the claim has two shapes: one run above SFT is a uniqueness
# claim, two or more is a list. A pattern that knew only the first would go STALE the
# moment the second became true.
#   shape 1: "N 个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 X（A 对 B）"
#   shape 2: "N 个 RL run 里有 K 个 macro-F1 点估计高于 SFT：X（A 对 B）与 Y（A 对 B）"
P_ABOVE_ONE = (r"(" + NUM + r")\s*个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 "
               r"([^（\n]+)（([\d.]+) 对 ([\d.]+)）")
P_ABOVE_MANY = (r"(" + NUM + r")\s*个 RL run 里有\s*(" + NUM + r")\s*个 macro-F1 点估计"
                r"高于 SFT：([^\n]+)")
m, used = anchor_any("5.2.4: which RL runs' macro-F1 is above SFT",
                     (P_ABOVE_ONE, P_ABOVE_MANY))
if m:
    above = sorted((n_ for n_ in rl if present[n_]["macro_f1"] > sft_f1),
                   key=lambda n_: -present[n_]["macro_f1"])
    want = [doc_name(x) for x in above]
    cmp("5.2.4: the RL-run count", cn2int(m.group(1)), len(rl), kind="count",
        pattern=used, group=1)
    if used is P_ABOVE_ONE:
        cmp("5.2.4: the set of RL runs above SFT, by name",
            [doc_name(m.group(2).strip())], want or ["（none: no RL run beats SFT）"])
        cmp("5.2.4: the value quoted for it and for SFT",
            (float(m.group(3)), float(m.group(4))),
            (round(present[above[0]]["macro_f1"], 4), round(sft_f1, 4)) if above
            else (None, round(sft_f1, 4)))
    else:
        # The list is "name（value 对 SFT）" repeated, separated by 、. The name may
        # contain ASCII parens (it is a column label like GSPO(G=4) lr1e-5) but never
        # the full-width ones the pairs are wrapped in.
        listed = re.findall(r"([^（、）\n]+)（([\d.]+) 对 ([\d.]+)）", m.group(3))
        cmp("5.2.4: the number of RL runs above SFT", cn2int(m.group(2)), len(above),
            why="how many runs clear SFT is a fact and not a count: the sentence is "
                "about which runs, not about the size of the list")
        cmp("5.2.4: the set of RL runs above SFT, by name, in the order written",
            [doc_name(x.strip()) for x, _, _ in listed], want)
        cmp("5.2.4: the value quoted for each of them and for SFT",
            [(float(v), float(b)) for _, v, b in listed],
            [(round(present[x]["macro_f1"], 4), round(sft_f1, 4)) for x in above])

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
    # The sentence asserts a property of these runs -- "两个类别都在" -- so the number it
    # is checked against has to be the number that keeps every class, not the size of the
    # set. While no lr 1e-5 run zeroes a class the two are equal and this passed either
    # way; once one does, comparing against the size would certify a sentence saying the
    # opposite. Which classes count as "every" is CLASSES, derived above from the reports.
    intact_low = [n for n in low
                  if all(present[n]["per_class"].get(c, 0.0) > 0.0 for c in CLASSES)]
    cmp("5.2.5: the number of lr 1e-5 runs that keep every class",
        cn2int(g.group(1)), len(intact_low),
        why="the sentence also names the values of those runs, so a third one is a "
            "rewrite, not a count bump; and it claims they keep both classes, so a run "
            "in the set that zeroes one is a rewrite too")
    if len(intact_low) == len(low):
        print(f"         they are: {'、'.join(low)}")
    else:
        print(f"         they are: {'、'.join(low)} -- of which keeping every class: "
              f"{'、'.join(intact_low) or 'none'}")

g = anchor("5.2.5: the lr 1e-5 runs the parenthetical enumerates",
           r"（`none`：([^；）]+)；`Donut`：([^；）]+)；")
if g:
    _low = [n for n in rl if n.endswith("lr1e-5")]
    for _list, _cls in ((g.group(1), "none"), (g.group(2), "Donut")):
        _items = {}
        for _it in _list.split("、"):
            _nm, _, _v = _it.strip().rpartition(" ")
            _items[_nm] = float(_v)
        cmp(f"5.2.5: the runs whose `{_cls}` F1 the parenthetical enumerates",
            sorted(_items), sorted(_low),
            why="the numeral beside this list counts the lr 1e-5 runs that keep every "
                "class; the list is what the reader actually sees. A run that joins the "
                "population without joining the list leaves a correct numeral over an "
                "enumeration of the old population")
        cmp(f"5.2.5: the `{_cls}` values it prints for them", _items,
            {n: round(present[n]["per_class"][_cls], 3) for n in sorted(_items)})

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
    missing = []
    for k in ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock"):
        vals = {}
        for n in r_cols:
            p = train_record(n)
            if not p.is_file():
                missing.append(n)
                continue
            vals[n] = json.loads(p.read_text(encoding="utf-8"))[
                "reward_signal"][k]["mean_std_across_steps"]
        if vals and min(vals, key=vals.get) == "GSPO(G=32) lr5e-5":
            low += 1
    # After the loop, not inside it: the question is about the reward table's columns,
    # or it is about one reward row, and it is the first. Asking it per row reported the
    # same missing column four times.
    if missing:
        cmp("5.2.5: a training record for every reward-table column", [], missing,
            why="a column with no training record cannot be ranked. Dropping it would "
                "rank 'lowest' over a smaller set than the sentence beside it names, "
                "and the result would look like a document error rather than a lookup "
                "one")
    cmp("5.2.5: how many of the 4 reward rows G=32 is lowest in", cn2int(g.group(1)), low)
    cmp("5.2.5: the number of columns 'lowest' is measured over",
        cn2int(g.group(2)), len(r_cols), kind="count",
        pattern=P_REWARD, group=2)
    # The same sentence quotes the claim it rejects, and the quotation names the
    # population again. That second numeral is where the population moved without the
    # sentence moving with it: run 45 added the seventh reward-table column, the first
    # numeral became 七, and this one stayed 六 -- so one sentence asserted two
    # populations four words apart and this item read only the first. Read as a count
    # for the same reason the first one is: a run landing bumps both numerals.
    P_QUOTED = r"「(" + NUM + r")组最低的标准差」"
    _q = re.search(P_QUOTED, text)
    if _q:
        cmp("5.2.5: the population in the claim that sentence quotes",
            cn2int(_q.group(1)), len(r_cols), kind="count",
            pattern=P_QUOTED, group=1)
    else:
        cmp("5.2.5: the quoted claim's population", "the quoted phrase", "absent",
            why="the sentence quotes the claim it rejects and the quotation carries the "
                "population; if that phrase is rewritten nothing reads the numeral")

# --------------------------------------- 5.2.2: the Scratch maximum and who holds it
# Nothing read this sentence. Its value is a claim about the run set, but its subject is
# the fragile part: it names the run 「G=32」, which is unambiguous only while one G=32
# run exists -- and a second one is training now. Checked as a fact, because the repair
# is to name the run, not to change a number.
P_SCRATCH = r"G=32 \+ lr 5e-5 在 `Scratch` 上拿到全部 run 最高的 ([\d.]+)\s*（SFT ([\d.]+)）"
# --------------- 5.2.2: the same-algorithm same-group-size learning-rate pairs
# The population is every (algorithm, group size) carrying both learning rates. The
# sentence states it twice and enumerates its members; all three are read here. Nothing
# read it before, and run 41 -- GSPO(G=32) lr1e-5 -- adds a fourth pair.
P_PAIRS = r"同算法同组大小的(" + NUM + r")条对照"
gp = anchor("5.2.2: the same-algorithm same-group-size lr pairs", P_PAIRS)
if gp:
    _by = {}
    for _n in rl:
        _m = re.fullmatch(r"([A-Za-z]+)\(G=(\d+)\) (lr[\d.eE+-]+)", _n)
        if _m:
            _by.setdefault((_m.group(1), int(_m.group(2))), {})[_m.group(3)] = _n
    _pairs = {k: v for k, v in _by.items() if len(v) == 2}
    _rates, _raised, _lowered, _missing = {}, [], [], []
    for _k, _v in sorted(_pairs.items()):
        _r = {}
        for _lr, _n in _v.items():
            _p = train_record(_n)
            if not _p.is_file():
                _missing.append(_n)
                continue
            _val = json.loads(_p.read_text(encoding="utf-8")).get(
                "mean_frac_reward_zero_std")
            if isinstance(_val, (int, float)):
                _r[_lr] = round(100 * _val, 2)
        if len(_r) == 2:
            _rates[_k] = (_r["lr1e-5"], _r["lr5e-5"])
            (_raised if _r["lr5e-5"] > _r["lr1e-5"] else _lowered).append(
                f"{_k[0]}(G={_k[1]})")
    if _missing:
        cmp("5.2.2: a training record for every run in those pairs", [], _missing,
            why="a pair whose record is missing cannot be ranked; dropping it would "
                "count the population over a smaller set than the sentence names")
    _label = lambda k: f"{k[0]}(G={k[1]})"
    cmp("5.2.2: the pair count (the numeral before the enumeration)",
        cn2int(gp.group(1)), len(_pairs),
        kind="count", pattern=P_PAIRS, group=1)
    # The second statement of the same population, in the sentence that draws the
    # conclusion from it. Note the 条 after the numeral and before the bold.
    P_PAIRS_SPLIT = r"(" + NUM + r")条\*\*同算法、同组大小\*\*的对照里"
    _gs = re.search(P_PAIRS_SPLIT, text)
    if _gs:
        cmp("5.2.2: the pair count (the second numeral, in the conclusion)",
            cn2int(_gs.group(1)), len(_pairs),
            kind="count", pattern=P_PAIRS_SPLIT, group=1)
    else:
        cmp("5.2.2: the pair count in the conclusion", "the phrase", "absent",
            why="the sentence states the population twice and the conclusion is drawn "
                "over the second statement; if this phrase cannot be found then this "
                "item is reading nothing, which is not the same as the document being "
                "correct -- the phrase is 「N条**同算法、同组大小**的对照里」")
    # The enumeration, keyed by (algorithm, group size) rather than read in order.
    _seen = {}
    for _m in re.finditer(r"([A-Za-z]+)\(G=(\d+)\)\s*([\d.]+)%（\d+/\d+）\s*→\s*"
                          r"([\d.]+)%（\d+/\d+）", text):
        _seen[(_m.group(1), int(_m.group(2)))] = (float(_m.group(3)), float(_m.group(4)))
    cmp("5.2.2: the rates enumerated for each pair",
        {_label(k): v for k, v in _seen.items()},
        {_label(k): v for k, v in _rates.items()},
        why="a pair landing appends to this enumeration; the numeral alone would leave "
            "it listing the old pairs")
    _up_txt = re.search(r"lr 5e-5 把空转步推高的有 (" + NUM + r") 条（([^）]*)）", text)
    _low_txt = re.search(r"压低的 (" + NUM + r") 条（([^）]*)）", text)
    if _up_txt and _low_txt:
        # Sorted on both sides. The item is about *which* pairs, and the `why` says so
        # ("a rewrite of the names"); the document reads them off the four-pair
        # enumeration directly above, which is the order the enumeration exists to give.
        # Sorting one side only demanded that the prose be in lexicographic order --
        # invisible while the split had one pair, and a demand the sentence never made.
        _norm = lambda s: sorted(x.strip() for x in s.split("、") if x.strip())
        cmp("5.2.2: how many pairs lr 5e-5 raises the idle-step share in",
            cn2int(_up_txt.group(1)), len(_raised), kind="count",
            pattern=r"lr 5e-5 把空转步推高的有 (" + NUM + r") 条", group=1)
        cmp("5.2.2: which pairs lr 5e-5 raises it in, named", _norm(_up_txt.group(2)),
            sorted(_raised),
            why="the sentence names the pairs, so a landing that changes the split is a "
                "rewrite of the names and not of the numeral")
        cmp("5.2.2: how many pairs lr 5e-5 lowers the idle-step share in",
            cn2int(_low_txt.group(1)), len(_lowered), kind="count",
            pattern=r"压低的 (" + NUM + r") 条", group=1)
        cmp("5.2.2: which pairs lr 5e-5 lowers it in, named", _norm(_low_txt.group(2)),
            sorted(_lowered),
            why="as above")
        print("         lr pairs by idle-step share: "
              + "、".join(f"{_label(_k)} {_v[0]:.2f}%->{_v[1]:.2f}%"
                          for _k, _v in sorted(_rates.items())))
    else:
        cmp("5.2.2: the pairs lr 5e-5 raises or lowers the idle-step share in",
            "the split phrases", "absent",
            why="the conclusion is drawn from this split; if these phrases are rewritten "
                "nothing reads what it was drawn from")

# ------------- the paired-significance section: what the document quotes from it
# Four sentences quote outputs/reports/paired_significance.json and section 9 calls it the
# instrument that answers what the marginal CI rule cannot. No item read any of the values.
# The record is regenerated by the pipeline, so every one of these is live.
PR = REP / "paired_significance.json"
_BENCH = ROOT / "benchmarks/wafer_bench_v1/classification.jsonl"
if not PR.is_file():
    cmp("the paired-significance record", "present", "absent",
        why="three sentences in this document quote values from it")
else:
    _pr = json.loads(PR.read_text(encoding="utf-8"))

    def _tag(lbl):
        """The record's tag for a run label: GSPO(G=8) lr5e-5 -> GSPO_G8_lr5e5."""
        _m = re.fullmatch(r"([A-Za-z]+)\(G=(\d+)\) (lr[\d.eE+-]+)", lbl)
        return (f"{_m.group(1)}_G{_m.group(2)}_{_m.group(3).replace('-', '')}"
                if _m else None)

    _T2L = {t: n for n in runs if (t := _tag(n))}
    _GRP = lambda lbl: f"{lbl.split('(')[0]}(G={lbl.split('G=')[1].split(')')[0]})"
    _nb = (sum(1 for _l in _BENCH.read_text(encoding="utf-8").split("\n") if _l.strip())
           if _BENCH.is_file() else None)
    NUMF = r"[−\-]?[\d.]+"

    def _f(s):
        return float(s.replace("−", "-"))

    def _dec(s):
        """The precision the document quotes a value at, so the comparison is made
        there. The record carries six decimals and the document four: comparing the
        record's -0.031746 against the quoted -0.0317 at record precision would report
        a stale sentence where the sentence is simply rounded."""
        return len(s.split(".")[1]) if "." in s else 0

    def _quoted(s, val):
        return _f(s) == round(val, _dec(s))

    def _on_grid(v, n):
        """Is v an integer multiple of 1/n, to the record's 6 decimals?"""
        return n is not None and abs(round(v * n) - v * n) <= 1e-3

    # ------------------------------------------- cross_algorithm_paired.lr5e-5
    _caps = _pr.get("cross_algorithm_paired", {}).get("lr5e-5")
    _gm = re.search(r"的两个成员是 \*\*([A-Za-z]+\(G=\d+\)) 与 ([A-Za-z]+\(G=\d+\))\*\*",
                    text)
    if _caps and _gm:
        cmp("5.2.2: the two members of cross_algorithm_paired.lr5e-5",
            sorted([_gm.group(1), _gm.group(2)]),
            sorted([_GRP(_T2L[_caps["grpo_token"]]), _GRP(_T2L[_caps["gspo_sequence"]])]),
            why="the record's two headings name the runs; the sentence's members must be "
                "the same two, or the -0.0278 quoted beside them is some other contrast")
        _gd = re.search(r"那个 ([−\-][\d.]+) 是「G=8 对 G=4」的准确率差", text)
        if _gd:
            cmp("5.2.2: the cross_algorithm_paired accuracy delta quoted",
                _f(_gd.group(1)),
                round(_caps["accuracy_delta_gspo_minus_grpo"], 4),
                kind="count", pattern=r"那个 ([−\-][\d.]+) 是「G=8 对 G=4」的准确率差",
                group=1)
        else:
            cmp("5.2.2: the cross_algorithm_paired delta", "the phrase", "absent",
                why="the sentence quotes it; if this phrase is gone nothing reads it")

    # ------------------------------------------------- paper_setting_paired
    _psp = _pr.get("paper_setting_paired")
    if _psp:
        _a32 = present[_T2L[_psp["to"]]]["accuracy"] if _T2L.get(_psp["to"]) else None
        _a04 = (present[_T2L[_psp["from"]]]["accuracy"]
                if _T2L.get(_psp["from"]) else None)
        cmp("5.2.2: the two runs' aggregate accuracies are actually equal",
            _a32 == _a04, True,
            why="the paragraph's entire argument is that equality of aggregates does not "
                "imply equality per row; if the aggregates are not equal the argument is "
                "about a different situation")
        _ge = re.search(r"完全相同，都是 ([\d.]+)（Δ = ([−\-]?[\d.]+)）", text)
        if _ge:
            cmp("5.2.2: the equal accuracy the paragraph quotes",
                _f(_ge.group(1)), round(_a32, _dec(_ge.group(1))), kind="count",
                pattern=r"完全相同，都是 ([\d.]+)（Δ = ([−\-]?[\d.]+)）", group=1)
            cmp("5.2.2: the delta it quotes beside them", _f(_ge.group(2)),
                round(_psp["accuracy_delta_g32_minus_grpo4"], _dec(_ge.group(2))),
                kind="count",
                pattern=r"完全相同，都是 ([\d.]+)（Δ = ([−\-]?[\d.]+)）", group=2)
        else:
            cmp("5.2.2: the equal accuracy", "the phrase", "absent",
                why="the sentence quotes an accuracy and a zero delta; both are unread if "
                    "this phrase is rewritten")
        # The delta, both CI ends and p, from the sentence that introduces section 9's
        # instrument.
        P_PSP = (r"`paper_setting_paired`：Δ = (" + NUMF + r")、95% CI \[(" + NUMF +
                 r"), \+([\d.]+)\]\s*、?\s*`p` = ([\d.]+)")
        _gp = re.search(P_PSP, text)
        if _gp:
            _ci = _psp["delta_ci95_rows"]
            for _i, (_lab, _val) in enumerate((
                    ("the accuracy delta", _psp["accuracy_delta_g32_minus_grpo4"]),
                    ("the lower CI end", _ci[0]),
                    ("the upper CI end", _ci[1]),
                    ("p", _psp["mcnemar"]["p_exact_two_sided"])), start=1):
                cmp(f"5.2.2: paper_setting_paired {_lab}", _f(_gp.group(_i)),
                    round(_val, _dec(_gp.group(_i))),
                    kind="count", pattern=P_PSP, group=_i)
        else:
            cmp("5.2.2: paper_setting_paired's delta, CI and p", "the phrase", "absent",
                why="this is the sentence that introduces the paired instrument; if it "
                    "cannot be found, three of its numbers are read by nothing")
        # The population, the differing rows, and the complementarity.
        P_PSP_ROWS = (r"但两者在 (" + NUM + r") 行里有 (" + NUM + r") 行判断不同，恰好 (" +
                      NUM + r") 对 (" + NUM + r") 互补")
        _gr = re.search(P_PSP_ROWS, text)
        _mc = _psp["mcnemar"]
        _two = [_mc["grpo_wrong_gspo_right"], _mc["grpo_right_gspo_wrong"]]
        if _gr:
            cmp("5.2.2: the row count the paired test is over", cn2int(_gr.group(1)), _nb,
                kind="count", pattern=P_PSP_ROWS, group=1)
            cmp("5.2.2: the rows the two runs classify differently", cn2int(_gr.group(2)),
                sum(_two), kind="count", pattern=P_PSP_ROWS, group=2)
            cmp("5.2.2: the complementarity of those rows (the pair, in order)",
                [cn2int(_gr.group(3)), cn2int(_gr.group(4))], _two,
                why="the sentence's point is that the differing rows are exactly "
                    "complementary; swapping the two directions keeps the sum and loses "
                    "the claim")
        else:
            cmp("5.2.2: the paired row count and its split", "the phrase", "absent",
                why="the 252-row unit and the 64-row split are unread without it")
        _g64 = re.search(r"会漏掉这 (" + NUM + r") 行的互换", text)
        if _g64:
            cmp("5.2.2: the same 64 rows, the third time the sentence states them",
                cn2int(_g64.group(1)), sum(_two), kind="count",
                pattern=r"会漏掉这 (" + NUM + r") 行的互换", group=1)
        else:
            cmp("5.2.2: the differing-row count repeated after the em dash", "the phrase",
                "absent",
                why="the sentence states the count three times; the three must agree")
        # The macro-F1 pair, and which run the second value belongs to.
        _g32, _oto = _T2L.get(_psp["to"]), _T2L.get(_psp["from"])
        # The numeral is on the next line: the sentence breaks after 低. A pattern with a
        # literal space there matches nothing, and -- with no absent-branch -- would have
        # read zero of these four numbers while reporting green.
        P_MF1 = r"macro-F1 上 G=32 低\s*([\d.]+)（([\d.]+) 对 ([\d.]+)）"
        _gmf = re.search(P_MF1, text)
        if _gmf and _g32 and _oto:
            _mf = {_n: present[_n]["macro_f1"] for _n in (_g32, _oto)}
            # The sentence names the second run's value but not the run. Mapped back
            # through every report: it must be the pair's other member. Two lines above
            # stands the G=8 run, whose macro-F1 is 0.5014, and reading the sentence as
            # "the neighbouring run" is the natural mistake.
            _mapped = [n for n in runs
                       if round(present[n]["macro_f1"], 4)
                       == round(float(_gmf.group(3)), 4)]
            cmp("5.2.2: the run the second macro-F1 in that pair belongs to", _mapped,
                [_oto],
                why="the sentence does not name it; it has to be the other member of the "
                    "same pair, not the G=8 run quoted two lines above")
            cmp("5.2.2: G=32's macro-F1 in that pair", float(_gmf.group(2)),
                round(_mf[_g32], 4), kind="count",
                pattern=P_MF1, group=2)
            cmp("5.2.2: the macro-F1 quoted against it", float(_gmf.group(3)),
                round(_mf[_oto], 4), kind="count",
                pattern=P_MF1, group=3)
            cmp("5.2.2: how much lower G=32's macro-F1 is", float(_gmf.group(1)),
                round(abs(_mf[_g32] - _mf[_oto]), 4), kind="count",
                pattern=P_MF1, group=1)
        else:
            cmp("5.2.2: the macro-F1 pair", "the phrase", "absent",
                why="three numbers and an unnamed subject; all four are unread without it")

    # ------------------------------------------------------ is_level_paired
    _isl = _pr.get("is_level_paired")
    P_ISL = (r"准确率差（sequence − token）= \*\*(" + NUMF + r")\*\*，95% 置信区间 \*\*\[(" +
             NUMF + r"), \+([\d.]+)\]\*\*（包含 0），\s*McNemar 精确检验 `p = ([\d.]+)`（token "
             r"错而 sequence 对 (" + NUM + r") 行，反之 (" + NUM + r") 行）")
    _gi = re.search(P_ISL, text)
    if _isl and _gi:
        _ci = _isl["delta_ci95_rows"]
        _mc = _isl["mcnemar"]
        for _i, (_lab, _val) in enumerate((
                ("the accuracy delta", _isl["accuracy_delta_sequence_minus_token"]),
                ("the lower CI end", _ci[0]),
                ("the upper CI end", _ci[1]),
                ("p", _mc["p_exact_two_sided"]),
                ("the rows token got wrong and sequence right",
                 _mc["token_wrong_sequence_right"]),
                ("the rows token got right and sequence wrong",
                 _mc["token_right_sequence_wrong"])), start=1):
            _v = cn2int(_gi.group(_i)) if _i >= 5 else round(_f(_gi.group(_i)), 6)
            _want = _val if _i >= 5 else round(_val, _dec(_gi.group(_i)))
            cmp(f"5.2.2: is_level_paired {_lab}", _v, _want,
                kind="count", pattern=P_ISL, group=_i)
        cmp("5.2.2: p is quoted to six decimals but the split is not sorted",
            _f(_gi.group(4)) == round(_mc["p_exact_two_sided"], 6), True,
            why="the two directions are named separately, so sorting them would accept "
                "the claim with its sides swapped")
    else:
        cmp("5.2.2: the is_level_paired sentence", "the phrase", "absent",
            why="the only single-variable IS-level contrast in the document; its six "
                "numbers are read by nothing without it")

    # The row grid: every delta and CI end in this record is k/252 for the benchmark's
    # 252 rows. Checked here rather than trusted, because the record is regenerated by
    # the pipeline and the benchmark is still draft_pending_human_review: if it gains a
    # row and the record is not recomputed, these stop being multiples of 1/n.
    _off = []
    for _name, _blk in (("cross_algorithm_paired.lr5e-5",
                         _pr.get("cross_algorithm_paired", {}).get("lr5e-5")),
                        ("paper_setting_paired", _pr.get("paper_setting_paired")),
                        ("is_level_paired", _pr.get("is_level_paired"))):
        if not _blk:
            continue
        _vals = [v for k, v in _blk.items() if "delta" in k and isinstance(v, float)]
        _vals += [v for k, v in _blk.items() if k == "delta_ci95_rows" for v in v]
        for _v in _vals:
            if not _on_grid(_v, _nb):
                _off.append((_name, _v))
    cmp(f"the paired record's values lie on the 1/{_nb} row grid of the benchmark",
        [], _off,
        why="accuracies paired over the benchmark's rows move in steps of 1/n; a value "
            "off the grid means the record was computed over a different row set than "
            "the sentence names")
    for _name in ("cross_algorithm_paired.lr5e-5", "paper_setting_paired",
                  "is_level_paired"):
        _blk = (_pr.get("cross_algorithm_paired", {}).get("lr5e-5")
                if _name.startswith("cross") else _pr.get(_name))
        if _blk:
            print(f"         {_name}: delta={_blk.get('accuracy_delta_gspo_minus_grpo', _blk.get('accuracy_delta_g32_minus_grpo4', _blk.get('accuracy_delta_sequence_minus_token')))} "
                  f"ci={_blk['delta_ci95_rows']} p={_blk['mcnemar']['p_exact_two_sided']}")

# ------------------- the KL/length record: its runs, its 286 steps, and its KL pairs
# Cited three times, quoted twice, read never. The record's entries are keyed by the
# script's own tags (one of which, `grpo`, is not the run stem), so runs are matched to
# entries by what the entry records about itself -- group size, learning rate and IS level
# -- rather than by tag name, which the document does not pin down either.
_KL = REP / "kl_length_confound.json"
if not _KL.is_file():
    cmp("the KL/length confound record", "present", "absent",
        why="two sentences quote its numbers and a third names its field path")
else:
    _kl = json.loads(_KL.read_text(encoding="utf-8"))
    _kruns = _kl.get("runs") or {}
    _steps = {t: (e.get("kl_identity") or {}).get("idle_steps_tested")
              for t, e in _kruns.items()}
    # A cell can hold two runs -- the second seed of a configuration is a re-run, not a
    # second learning rate. The record declares which run each repeat repeats, so this
    # map can be built from the cell owners instead of from whichever entry a dict
    # comprehension happened to write last. (Found when run 42 landed: the lookup took
    # the seed's KL, the pair-versus-run item correctly reported the two halves of the
    # record disagreeing, and the document -- which quotes the run its pair is built
    # from -- looked like the stale one.) The two items below check the declaration
    # itself, so "repeat" cannot become a way to drop a run out of every population.
    def _cellkey(_e):
        return (_e["G"], round(float(_e["lr"]), 12), _e["is_level"])

    _repeats = {t: e.get("repeat_of") for t, e in _kruns.items() if e.get("repeat_of")}
    _by_ident = {_cellkey(e): (t, e) for t, e in _kruns.items() if t not in _repeats}
    cmp("every declared seed repeat names a run the record holds", [],
        [t for t, r in _repeats.items() if r not in _kruns],
        why="the declaration removes the repeat from the cell map; a repeat naming a run "
            "that is not there would leave the cell it shares with nobody")
    cmp("every declared seed repeat shares the cell of the run it repeats", [],
        [t for t, r in _repeats.items() if r in _kruns
         and _cellkey(_kruns[t]) != _cellkey(_kruns[r])],
        why="otherwise the word repeat would excuse a run from the cell map while it "
            "sits in a cell of its own, and its KL would leave every population below "
            "without any sentence changing")

    def _ident(lbl):
        _m = re.fullmatch(r"([A-Za-z]+)\(G=(\d+)\) (lr[\d.eE+-]+)", lbl)
        if not _m:
            return None
        return (int(_m.group(2)), float(_m.group(3)[2:]),
                "sequence" if _m.group(1) == "GSPO" else "token")

    _unparsed = [n for n in rl if _ident(n) is None]
    cmp("every RL run's label parses into (algorithm, group size, learning rate)",
        [], _unparsed,
        why="the runs that do not would be silently outside every population computed "
            "here -- the pair sets, the KL pairs, the record's coverage")
    _covered = {n: _by_ident.get(_ident(n)) for n in rl if _ident(n)}
    _absent = [n for n in rl if _ident(n) and not _covered[n]]
    _finished = [n for n in _absent if train_record(n).is_file()]
    print(f"         the KL record covers {len(rl) - len(_absent)} of {len(rl)} RL runs")
    for _n in _absent:
        print(f"           {_n}: absent from the record"
              + (" -- FINISHED, so the record is behind the run set"
                 if _n in _finished else " (no training record yet)"))
    if _finished:
        cmp("the KL record covers every finished RL run", [], _finished,
            why="the record is one landing behind: its 六个 run and 286 空转步 describe the "
                "record, but the document's KL sentence draws a conclusion over 「同算法同组"
                "大小の对照」 -- and the run set has one more of those than the record does. "
                "The repair is to regenerate the record (tools/kl_length_confound.py has "
                "no caller; nothing regenerates it), then re-read this and the sentences "
                "below")

    # The identity and its two statements of 「六个 run 的全部 286 个空转步」.
    # `\s*` between the numeral and 个, because the document writes counts both ways:
    # 「六个 run」 with no space and 「7 个 run」 with one. A pattern that hardcodes either
    # convention matches one of them and silently reads nothing on the other -- which is
    # what cn2int's own docstring is about, and what this item's absent-branch caught.
    P_IDENT = (r"(" + NUM + r")\s*个 run 的全部 (" + NUM + r")\s*个空转步均满足（逐 run 实测为"
               r"\s*([\d+]+)")
    _gi = re.search(P_IDENT, text)
    if _gi:
        cmp("8: the KL identity's run count", cn2int(_gi.group(1)), len(_kruns),
            kind="count", pattern=P_IDENT, group=1)
        cmp("8: the KL identity's idle-step total", cn2int(_gi.group(2)),
            sum(v for v in _steps.values() if isinstance(v, int)),
            kind="count", pattern=P_IDENT, group=2)
        cmp("8: the per-run idle-step counts listed beside it",
            sorted(int(x) for x in _gi.group(3).split("+")),
            sorted(v for v in _steps.values() if isinstance(v, int)),
            why="the sentence lists one count per run; it does not claim an order, so "
                "this compares them as a bag -- a reordering of the record would "
                "otherwise fire this for a sentence that is still true")
        cmp("8: the total actually equals the per-run counts it lists",
            sum(int(x) for x in _gi.group(3).split("+")),
            sum(v for v in _steps.values() if isinstance(v, int)),
            why="a per-run list and a total in the same parenthesis, added up")
    else:
        cmp("8: the KL identity's run count and idle-step total", "the phrase", "absent",
            why="the 「初稿写 271」 correction and the identity's 286 both hang off this "
                "sentence; if it is rewritten nothing reads either")
    P_IDENT2 = (r"(" + NUM + r")\s*个 run 全部 (" + NUM + r")\s*个空转步上的逐值核对")
    _gj = re.search(P_IDENT2, text)
    if _gj:
        cmp("9: the KL identity's run count, second statement", cn2int(_gj.group(1)),
            len(_kruns), kind="count", pattern=P_IDENT2, group=1)
        cmp("9: the KL identity's idle-step total, second statement",
            cn2int(_gj.group(2)), sum(v for v in _steps.values() if isinstance(v, int)),
            kind="count", pattern=P_IDENT2, group=2)
    else:
        cmp("9: the KL identity's second run count and total", "the phrase", "absent",
            why="the same 286 is stated in two sections; only one was read before")

    # 5.2.4's KL pairs: a population over the record, not the run set.
    # `ALGO(G=n) lo → hi`, repeated. Read by parsing every occurrence inside the
    # parenthesis into a dict keyed by (algorithm, group size) -- the same way patch109
    # reads this population in 5.2.2. Positional groups were the first attempt and they
    # reported the sentence "absent" as soon as it named a third pair: a pattern that
    # stops matching and reports a fact is a broken probe, not a stale document.
    _PAIR_ITEM = r"([A-Za-z]+)\(G=(\d+)\) ([\d.]+) → ([\d.]+)"

    def _parsed(s):
        return {(m.group(1), int(m.group(2))): (float(m.group(3)), float(m.group(4)))
                for m in re.finditer(_PAIR_ITEM, s)}

    _cells = {}
    for _lbl in rl:
        _id = _ident(_lbl)
        if _id and _id in _by_ident:
            _cells.setdefault((_lbl.split("(")[0], _id[0]), {})[_id[1]] = _by_ident[_id][1]
    _kpairs = {k: v for k, v in _cells.items() if len(v) == 2 and
               all(isinstance(v[x].get("mean_kl"), (int, float)) for x in v)}
    # The record's own pair list. `_kpairs` above is assembled from `runs.<tag>` entries
    # (which carry `mean_kl`); the lengths and the idle shares exist only per pair, under
    # `matched_pairs`, whose keys are the script's names for them ("GRPO G=4"). Reading a
    # length out of a run entry is a KeyError -- which is how this item came to exist.
    _mp = {}
    for _nm, _e in (_kl.get("matched_pairs") or {}).items():
        _mm = re.fullmatch(r"([A-Za-z]+) G=(\d+)", _nm)
        if _mm:
            _mp[(_mm.group(1), int(_mm.group(2)))] = _e
    cmp("the record's pair list covers every pair its runs imply",
        [], sorted(set(_kpairs) - set(_mp)),
        why="the document's population sentence counts pairs; the count is assembled from "
            "the runs while the lengths and idle shares it quotes come from the pair "
            "entries. If the two lists disagree, every item below reads a smaller "
            "population than the sentence names and reports green")
    _mp2 = {k: v for k, v in _mp.items() if k in _kpairs}
    # A pair entry holds the two runs side by side in two-element lists, so position is
    # what carries the meaning: index 0 is the lower learning rate. Read, not assumed --
    # if the tool wrote them the other way, every value below would still belong to the
    # pair and the items would still print ok, while the sentence's claims about
    # *direction* (KL falls, the completion shortens) would be about the wrong way round.
    _orient = [k for k, v in _mp2.items()
               if not (isinstance(v.get("lr"), list) and len(v["lr"]) == 2
                       and v["lr"][0] < v["lr"][1])]
    cmp("every pair entry's two runs are listed at the lower rate first", [], _orient,
        why="the sentences compare the two members of each pair; that comparison only "
            "means what it says if the position means what this item checks")
    _mismatch = []
    for _k, _v in _mp2.items():
        _r = _kpairs.get(_k) or {}
        _want = [round(_r[x]["mean_kl"], 6) for x in (1e-05, 5e-05) if x in _r]
        if isinstance(_v.get("kl"), list) and len(_want) == 2:
            if [round(x, 6) for x in _v["kl"]] != _want:
                _mismatch.append(f"{_k[0]}(G={_k[1]})")
    cmp("the pair entries and the run entries agree on the KL, in the same order", [],
        _mismatch,
        why="the only place the two halves of the record are compared directly. The KL "
            "items read `runs` and the length and idle items read `matched_pairs`; if the "
            "two disagree there is no way to say which half a sentence is quoting")
    # What the record says about each pair, in the shapes the sentences quote it in.
    _rec_kl = {k: (round(v[1e-05]["mean_kl"], 4), round(v[5e-05]["mean_kl"], 4))
               for k, v in _kpairs.items()}
    _rec_len = {k: (round(v["completion_length"][0], 1), round(v["completion_length"][1], 1))
                for k, v in _mp2.items()}
    # Section 8 quotes the lengths exactly, so the percentage printed beside them is
    # required to follow from them; 5.2.4 quotes them to 1 dp and prints no percentage.
    _rec_len2 = {k: (round(v["completion_length"][0], 2),
                     round(v["completion_length"][1], 2),
                     round(100 * (v["completion_length"][1]
                                  / v["completion_length"][0] - 1), 1))
                 for k, v in _mp2.items()}
    _rec_idle = {k: (round(100 * v["idle"][0], 2), round(100 * v["idle"][1], 2),
                     round(v["idle"][1] - v["idle"][0], 4))
                 for k, v in _mp2.items()}
    P_KLP = r"同算法同组大小的(" + NUM + r")个对照里 KL 都是下降的\s*（([^）]*)）"
    _gk = re.search(P_KLP, text)
    if _gk:
        _named = _parsed(_gk.group(2))
        cmp("5.2.4: the number of same-algorithm same-group-size KL pairs",
            cn2int(_gk.group(1)), len(_kpairs),
            kind="count", pattern=P_KLP, group=1)
        cmp("5.2.4: which pairs the KL sentence enumerates", sorted(_named),
            sorted(_rec_kl),
            why="the numeral and the list are one claim. A landing that grows the "
                "population must name the new pair, and until it is named this fact is "
                "red -- which is what stops `--fix` from bumping the numeral alone, and "
                "is the intended outcome: which pairs the sentence is evidence about is "
                "a decision, not a count")
        for _k in sorted(_named):
            cmp(f"5.2.4: {_k[0]}(G={_k[1]})'s KL at the lower and higher learning rate",
                list(_named[_k]), list(_rec_kl.get(_k, ())),
                why="the sentence's argument is that raising the learning rate lowers the "
                    "mean per-token KL; these two numbers are its evidence")
            cmp(f"5.2.4: {_k[0]}(G={_k[1]})'s KL does fall when the learning rate rises",
                _rec_kl.get(_k, (0.0, 0.0))[1] < _rec_kl.get(_k, (0.0, 0.0))[0], True,
                why="「都下降」 compares the two numbers; it is stated of every pair the "
                    "sentence names, so a pair enters the claim when it is written down")
    else:
        cmp("5.2.4: the same-algorithm same-group-size KL pairs", "the phrase", "absent",
            why="the sentence states a population and then names its members; without this "
                "phrase nothing reads either")

    # The other measurement of the same pairs: the completions the sentence calls shorter.
    P_KLP_LEN = r"5e-5 的补全却明显更短\s*（([^）]*)）"
    _gkl = re.search(P_KLP_LEN, text)
    if _gkl:
        cmp("5.2.4: the completion lengths enumerated beside the KL pairs",
            _parsed(_gkl.group(1)), _rec_len,
            why="the sentence's point is that the KL and the length are not the same "
                "quantity; that needs both measurements read, not just the KL")
    else:
        cmp("5.2.4: the completion lengths beside the KL pairs", "the phrase", "absent",
            why="the KL sentence's second half; if it is rewritten nothing reads the "
                "lengths it quotes")

    # Section 8 restates the lengths, and adds the percentage change. The percentage is
    # computed from the two rounded lengths -- the arithmetic a reader would do -- rather
    # than read from the record's `length_change_pct`, which would be the record's own
    # summary of the same pair of numbers, i.e. the claim checked against itself.
    P_LEN_PCT = r"但 5e-5 那(" + NUM + r")次的补全明显更短\s*（([^）]*)）"
    _gl = re.search(P_LEN_PCT, text)
    if _gl:
        _litems = re.findall(_PAIR_ITEM + r"(?: token)?，\s*(−?[\d.]+)%", _gl.group(2))
        _ltxt = {(m[0], int(m[1])): (float(m[2]), float(m[3]), -float(m[4].lstrip("−")))
                 for m in _litems}
        _lwant = _rec_len2
        cmp("8: the number of pairs whose completions the sentence calls shorter",
            cn2int(_gl.group(1)), len(_ltxt), kind="count", pattern=P_LEN_PCT, group=1)
        cmp("8: the completion lengths and the percentage changes beside them", _ltxt,
            _lwant,
            why="the percentage is computed here from the two lengths rather than read "
                "from the record's `length_change_pct`, so the item is the arithmetic a "
                "reader would do with the printed pair -- and comparing the lengths "
                "themselves is what makes that arithmetic reproducible: 1 dp is not "
                "enough precision for the GSPO(G=8) pair, whose -4.6% follows from "
                "109.16/114.40 but not from 109.2/114.4")
    else:
        cmp("8: the completion lengths with their percentage changes", "the phrase",
            "absent",
            why="this is where the lengths are quoted with a percentage; the arithmetic "
                "is only checkable while the phrase is found")

    # Section 8 states this population once more, in words patch109's two patterns do not
    # reach (they require 条 and a 、 after 同算法; this sentence says 个 and runs the two
    # words together). Its 两个 was stale the same way 5.2.4's was, and the third pair
    # belongs to its conclusion.
    P_IDLE_PAIRS = (r"(" + NUM + r")个\*\*同算法同组大小\*\*的对照方向[^：]*：\s*"
                    r"([^。]*)。")
    _gip = re.search(P_IDLE_PAIRS, text)
    if _gip:
        # Same enumeration, percentages rather than values: `_PAIR_ITEM` cannot cross
        # the `%` between the first number and the arrow, and a findall that matches
        # nothing returns an empty set, which is not the same as a document that
        # enumerates nothing -- the count item beside this one would still pass.
        _IPCT = (r"([A-Za-z]+)\(G=(\d+)\) ([\d.]+)% → ([\d.]+)%"
                 r"（`idle_delta` ([−+]?[\d.]+)）")
        _iitems = re.findall(_IPCT, _gip.group(2))
        _itxt = {(m[0], int(m[1])): (float(m[2]), float(m[3]),
                                    float(m[4].lstrip("−").replace("+", ""))
                                    * (-1 if m[4].startswith("−") else 1))
                 for m in _iitems}
        cmp("8: the number of same-algorithm same-group-size idle-step pairs",
            cn2int(_gip.group(1)), len(_kpairs), kind="count",
            pattern=P_IDLE_PAIRS, group=1)
        cmp("8: which pairs the idle-step sentence enumerates", sorted(_itxt),
            sorted(_rec_idle),
            why="same population, third statement, first reader")
        cmp("8: the idle-step shares and deltas it quotes for them", _itxt, _rec_idle,
            why="the passage's point is that the direction is not the learning rate's "
                "alone, which is a claim about these deltas")
        cmp("8: the pairs its conclusion says the learning rate raises the share in",
            sorted([k for k, v in _rec_idle.items() if v[2] > 0]),
            sorted([k for k, v in _itxt.items() if v[2] > 0]),
            why="「只在 G=8 上成立」 is a set, and it is the sentence's conclusion; the "
                "third pair lowers the share, so it belongs to this set too")
    else:
        cmp("8: the same-algorithm same-group-size idle-step pairs", "the phrase", "absent",
            why="a fourth statement of the pair population, read by none of the others")


# ---------------------------- 5.2.4: the G=32 column's evidence, quoted from the record
_PV = REP / "paired_significance.json"
if _PV.is_file():
    _pv = json.loads(_PV.read_text(encoding="utf-8"))
    _SFT = _pv.get("comparisons_vs_SFT", {})
    _g32 = [n for n in rl if "G=32" in n and "lr5e-5" in n]
    _g8 = [n for n in rl if "G=8" in n and "lr5e-5" in n]
    P_LOWER = (r"\*\*G=32 \+ lr 5e-5 那一列的强度需要下调。\*\* 它比分低于 SFT，但成对检验 `p` =\s*"
               r"([\d.]+)、\s*置信区间\*\*包含 0\*\*")
    _gl = re.search(P_LOWER, text)
    if _gl and len(_g32) == 1 and len(_g8) == 1:
        _c = _SFT["GSPO_G32_lr5e5"]
        cmp("5.2.4: the p quoted for the G=32 column", float(_gl.group(1)),
            round(_c["mcnemar"]["p_exact_two_sided"], 3), kind="count",
            pattern=P_LOWER, group=1)
        cmp("5.2.4: that the G=32 column scores below SFT", _c["delta_vs_sft"] < 0, True,
            why="「比分低于 SFT」 is a claim about the sign of the paired delta")
        cmp("5.2.4: that its confidence interval contains zero",
            _c["delta_ci95_rows"][0] < 0 < _c["delta_ci95_rows"][1], True,
            why="computed from the interval itself rather than from the record's "
                "`excludes_zero` flag, which would be the same claim checked against "
                "itself")
        cmp("5.2.4: that this column's evidence is weaker than GSPO(G=8) lr5e-5's",
            _c["mcnemar"]["p_exact_two_sided"]
            > _SFT["GSPO_G8_lr5e5"]["mcnemar"]["p_exact_two_sided"], True,
            why="「弱于」 is a comparison of the two p values; the two runs are the only "
                "two in the sentence and both are fixed, so this is a checkable claim")
        print(f"         G=32 lr5e-5 vs SFT: p={_c['mcnemar']['p_exact_two_sided']:.6f} "
              f"ci={_c['delta_ci95_rows']} | G=8 lr5e-5: "
              f"p={_SFT['GSPO_G8_lr5e5']['mcnemar']['p_exact_two_sided']:.6f}")
    elif not _gl:
        cmp("5.2.4: the G=32 column's quoted p and interval", "the phrase", "absent",
            why="the sentence quotes a p and a CI position; both are unread without it")
    # This was a tripwire, and it fired: it asserted that only one G=32 run existed, so
    # `「G=32 那一列」` identified a column, and run 41 made it two. The sentence now names
    # the learning rate, which retires the item -- but not the question it was asking.
    # What is worth checking in its place is the opposite: that the cell the sentence
    # names is still a single cell, since that is what makes the p and the interval
    # beside it describe one run rather than an average of two.
    cmp("5.2.4: the G=32 lr 5e-5 cell the column sentence names is a single cell",
        len([n for n in runs if "G=32" in n and "lr5e-5" in n]), 1,
        why="the sentence now says 「G=32 + lr 5e-5 那一列」; a second run in that cell "
            "would make the column ambiguous again, in the way run 41 made G=32 ambiguous")

# ---------------- 5.2.4: the per-reward rule std, and which run holds its minimum
# A superlative over a population. `3` is only right while the population is the eight
# RL runs *and* the run it is about is the one the sentence names -- and run 41 made
# the sentence's 「G=32」 name neither. The two tripwires that fired covered the column
# sentence and the Scratch maximum; this sentence was not covered by either, and here
# the two readings disagree about the claim itself rather than about its precision.
# Groups: algorithm, G, lr, the reward population, the named cell's count, the
# sibling clause's population, and the sibling's count. The population is captured
# twice on purpose: the second clause restates the first one's population, so a
# sentence that changed it in only one place is caught by the pair.
P_STD = (r"([A-Za-z]+)\(G=(\d+)\) \+ lr (\S+) 同时继承了高学习率的退化与大组下的"
         r"多样性下降\s*——\s*组内奖励标准差\s*在 (" + NUM + r") 个奖励里有 ("
         + NUM + r") 个是八组最低"
         r"（同组大小的 run 41 在这 (" + NUM + r") 个奖励里最低的有 (" + NUM + r") 个")
g = anchor("5.2.4: the per-reward std superlative", P_STD)
if g:
    # label -> result file, declared once, in the tool that already reconciles logs
    # against records; `grpo_train_result.json` is why this is read and not written.
    _stem = {}
    # `_IDLE_TOOL` is assigned below, just before section 9's audit, so it does not
    # exist in this scope yet -- a module-level name is not forward-referenced. Binding
    # it here rather than moving the other one leaves section 9's line untouched.
    _IDLE_TOOL = ROOT / "tools/idle_step_table.py"
    if _IDLE_TOOL.is_file():
        for _node in ast.parse(_IDLE_TOOL.read_text(encoding="utf-8")).body:
            if (isinstance(_node, ast.Assign) and _node.targets
                    and getattr(_node.targets[0], "id", None) == "RUNS"):
                for _e in _node.value.elts:
                    _lab, _s, _lg = ast.literal_eval(_e)
                    _stem[_lab] = _s if _s.endswith(".json") else _s + ".json"
    # The eight RL runs in the order 5.2.4's table lists them, so the population the
    # superlative is over is the population the table shows.
    _STD_ORDER = ["GRPO G=4 lr1e-5", "GRPO G=4 lr5e-5", "GSPO G=4 lr5e-5",
                  "GSPO G=4 lr1e-5", "GSPO G=8 lr1e-5", "GSPO G=8 lr5e-5",
                  "GSPO G=32 lr5e-5", "GSPO G=32 lr1e-5"]
    # The key is the label itself. Prefixing every label with "GSPO" was a no-op for
    # seven of the eight rows and a collision for the two GRPO ones -- whose keys it made
    # equal to their namesakes at G=4 in GSPO, so the second assignment replaced the
    # first and both GRPO runs left the population without changing either verdict.
    _cells = [(l, l) for l in _STD_ORDER]
    _series, _miss = {}, []
    for _lab, _key in _cells:
        _f = REP / _stem.get(_lab, "")
        if _f.is_file():
            _series[_key] = json.loads(_f.read_text(encoding="utf-8"))["reward_signal"]
        else:
            _miss.append(_lab)
    if _miss:
        cmp("5.2.4: every RL run's reward_signal is readable", [], _miss,
            why="the std superlative is computed over these runs; one unreadable "
                "record would silently shrink the population and change which run is "
                "the minimum")
    else:
        cmp("5.2.4: the number of RL groups the std comparison runs over",
            len(_series), len(_STD_ORDER),
            why="the population is assembled from another structure, so its size is a "
                "claim: the key collision that dropped both GRPO runs left every value "
                "in this block correct and both verdicts right -- only the size gave it "
                "away, and only because it was printed")
        _rw = sorted(_series[next(iter(_series))])
        def _mincount(cell):
            return sum(1 for r in _rw
                       if abs(_series[cell][r]["mean_std_across_steps"]
                              - min(_series[c][r]["mean_std_across_steps"]
                                    for c in _series)) < 1e-12)
        _named = f"{g.group(1)} G={g.group(2)} lr{g.group(3)}"
        _sib = f"{g.group(1)} G={g.group(2)} lr1e-5"
        cmp("5.2.4: the reward count the std superlative is measured over",
            cn2int(g.group(4)), len(_rw),
            why="the sentence says \"of N rewards\"; the population is whatever the "
                "records render")
        cmp("5.2.4: the cell the std superlative names", _named, "GSPO G=32 lr5e-5",
            why="run 41 is a second G=32 run, so 「G=32」 alone names neither and the "
                "two readings disagree: the lr 5e-5 cell holds the minimum in 3 of 4 "
                "rewards and the lr 1e-5 cell in none of them")
        cmp("5.2.4: the rewards in which it holds the lowest std",
            cn2int(g.group(5)), _mincount(_named),
            why="a superlative over a population is not a numeral that can be checked "
                "against the records without them")
        cmp("5.2.4: the sibling clause names the same reward population",
            cn2int(g.group(6)), cn2int(g.group(4)),
            why="the two clauses describe one population; if only one of them were "
                "updated the two numerals would differ, which is checkable without "
                "either clause being right about the records")
        cmp("5.2.4: the rewards in which the same group size at lr 1e-5 holds it",
            cn2int(g.group(7)), _mincount(_sib),
            why="the sentence states this count as the reason the collapse follows the "
                "learning rate rather than the group size, so it carries the argument "
                "and not only the illustration. It was read as the population numeral "
                "when this item was first written -- the sentence said the count in "
                "words, and the reader took the number that was beside it")
        for _c in (_named, _sib):
            print(f"         {_c}: lowest of the {len(_series)} RL groups in "
                  f"{_mincount(_c)} of {len(_rw)} rewards"
                  + ("" if _c in _series else "  (no record)"))


g = anchor("5.2.2: the Scratch per-class maximum", P_SCRATCH)
if g:
    _sc = {n: present[n]["per_class"].get("Scratch") for n in runs}
    _sc = {n: v for n, v in _sc.items() if v is not None}
    _best = max(_sc, key=_sc.get)
    cmp("5.2.2: the run holding the Scratch per-class maximum", _best,
        "GSPO(G=32) lr5e-5",
        why="the sentence names a run as the one with the highest Scratch F1; another "
            "run taking that maximum makes it false in a way no numeral can fix")
    cmp("5.2.2: the Scratch per-class maximum", float(g.group(1)), round(_sc[_best], 3))
    cmp("5.2.2: SFT's Scratch F1 quoted beside it", float(g.group(2)),
        round(_sc["SFT"], 3))
    # The same tripwire as 5.2.4's, fired by the same run for the same reason. The
    # sentence now names its run; what remains checkable is that the cell is unique, so
    # that "the maximum" is the maximum of a population the sentence can name.
    cmp("5.2.2: the G=32 lr 5e-5 cell holding the Scratch maximum is a single cell",
        len([n for n in runs if "G=32" in n and "lr5e-5" in n]), 1,
        why="the sentence says 「G=32 + lr 5e-5 在 Scratch 上拿到全部 run 最高的 …」; "
            "with a second run in that cell the phrase would name neither")
    print(f"         Scratch F1 by run: "
          + "、".join(f"{_k} {_v:.3f}" for _k, _v in
                      sorted(_sc.items(), key=lambda kv: -kv[1])))

# --------------- 5.2: every run's gradient accumulation, and the table of wrong records
# The whole 5.2 confound section rests on this claim: accumulation == group size is what
# makes an effective batch 4x, 16x or 64x its group. It was verified by hand over the
# runs that existed when it was written; the population has grown since, so it is read
# from each run's own log here, and the eighth run is inside the quantifier.
P_GACC = (r"\*\*每个 run 的梯度累积都设成了与组大小相等\*\*（4/4、8/8、32/32）")
g = anchor("5.2: every run's gradient accumulation equals its group size", P_GACC)
if g:
    _decl = {}
    if _IDLE_TOOL.is_file():
        for _node in ast.parse(_IDLE_TOOL.read_text(encoding="utf-8")).body:
            if (isinstance(_node, ast.Assign) and _node.targets
                    and getattr(_node.targets[0], "id", None) == "RUNS"):
                for _e in _node.value.elts:
                    _lab, _s, _lg = ast.literal_eval(_e)
                    _decl[_lab] = (_s if _s.endswith(".json") else _s + ".json", list(_lg))
    _gacc, _gsize, _acc_miss, _pending = {}, {}, [], []
    for _lab, (_stem, _logs) in sorted(_decl.items()):
        _got = None
        for _lg in _logs:
            _f = ROOT / "logs" / _lg
            if _f.is_file():
                _m = re.findall(r"(?<![A-Za-z0-9_])gradient_accumulation_steps=?(\d+)",
                                _f.read_text(encoding="utf-8", errors="replace"))
                if _m:
                    _got = int(_m[-1])
                    break
        _m = re.search(r"G=(\d+)", _lab)
        if _got is None or not _m:
            # Declared and not launched yet is not the failure this item is for. The
            # record decides: `outcome == completed` and `steps_logged ==
            # max_steps_requested` is how the rest of the project reads "this run
            # finished". An unfinished run is printed, not counted -- the same line
            # `idle_step_table` prints for an entry whose log has no values yet. A
            # finished run whose log lacks the line still fails, which is the point.
            _rp = REP / _stem
            _done = False
            if _rp.is_file():
                try:
                    _r = json.loads(_rp.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    _r = {}
                _done = (_r.get("outcome") == "completed"
                         and _r.get("steps_logged") == _r.get("max_steps_requested"))
            if not _done:
                _pending.append(_lab)
            else:
                _acc_miss.append(_lab)
            continue
        _gacc[_lab] = _got
        _gsize[_lab] = int(_m.group(1))
    if _pending:
        print("         declared, no finished run yet (not counted here): "
              + "、".join(_pending))
    if _acc_miss:
        cmp("5.2: every run's log states its gradient accumulation", [], _acc_miss,
            why="a run whose log cannot be read is a run outside the quantifier, and the "
                "quantifier is what the section's confound argument rests on")
    else:
        cmp("5.2: the gradient accumulation every run's own log records", _gacc, _gsize,
            why="the sentence says every run's accumulation was set equal to its group "
                "size and writes three example pairs; this is the claim over all of them "
                "that the effective-batch table depends on")
        if _gacc != _gsize:
            print(f"         logs vs group sizes: "
                  + "、".join(f"{k} {_gacc.get(k)}/{_gsize.get(k)}"
                              for k in sorted(_gsize)))

# ------------------------- 11: the table of runs whose record disagrees with its log
# A table of defects is a population, and run 41 was missing from it while its own
# record disagreed with its own log (4 written, 32 logged). Three columns are read back
# against the runs: the labels, the accumulations the logs show, the values the records
# write, and which rows are marked wrong -- the mark and the disagreement have to be the
# same set, or the mark is a claim about the log that nothing computes.
# The table lives inside a numbered item, so every one of its lines is indented by four
# spaces -- and this pattern was written as though the table began at column 0. It has
# therefore never matched: patch118 wrote this pattern and the table it reads in the same
# run, and its own log shows the STALE line below, followed by the caller logging `ok`.
# Nothing has read item 11's table since. `\s*` before a line is the whole fix; what the
# check reads, and what it compares the table against, are unchanged.
P_I11 = (r"\s*\| run \| 实际累积步（该 run 自己的日志）\| 记录写的 \| \|\s*\n"
         r"\s*\| --- \| --- \| --- \| --- \|\s*\n"
         r"((?:\s*\|[^|\n]*G=\d+[^|\n]*\|[^\n]*\n)+)")
g = anchor("11: the table of runs whose record disagrees with its own log", P_I11)
if g:
    _decl = {}
    if _IDLE_TOOL.is_file():
        for _node in ast.parse(_IDLE_TOOL.read_text(encoding="utf-8")).body:
            if (isinstance(_node, ast.Assign) and _node.targets
                    and getattr(_node.targets[0], "id", None) == "RUNS"):
                for _e in _node.value.elts:
                    _lab, _s, _lg = ast.literal_eval(_e)
                    _decl[_lab] = (_s if _s.endswith(".json") else _s + ".json", list(_lg))
    _rows, _rec, _gacc = {}, {}, {}
    for _ln in g.group(1).splitlines():
        _c = [x.strip() for x in _ln.strip().strip("|").split("|")]
        if len(_c) >= 4 and re.search(r"G=\d+", _c[0]):
            _rows[_c[0]] = (_c[1], _c[2].strip("*").strip(), "错" in _c[3])
    for _lab, (_stem, _logs) in sorted(_decl.items()):
        _f = REP / _stem
        if _f.is_file():
            _w = (json.loads(_f.read_text(encoding="utf-8")).get("config") or {}).get(
                "grad_accum")
            if _w is not None:
                _rec[_lab] = int(_w)
        for _lg in _logs:
            _lf = ROOT / "logs" / _lg
            if _lf.is_file():
                _m = re.findall(r"(?<![A-Za-z0-9_])gradient_accumulation_steps=?(\d+)",
                                _lf.read_text(encoding="utf-8", errors="replace"))
                if _m:
                    _gacc[_lab] = int(_m[-1])
                    break
    cmp("11: the runs the table lists, against the runs that have a record",
        sorted(_rows), sorted(_rec),
        why="a run missing from a table of wrong records is a record vouched for by "
            "omission; run 41 was the instance")
    cmp("11: the accumulations it prints from the logs",
        {k: int(v[0]) for k, v in _rows.items() if v[0].isdigit()},
        {k: _gacc[k] for k in _rows if k in _gacc})
    cmp("11: the accumulations it prints for the records",
        {k: int(v[1]) for k, v in _rows.items() if v[1].isdigit()},
        {k: _rec[k] for k in _rows if k in _rec})
    cmp("11: the runs it marks as wrong",
        sorted(k for k, v in _rows.items() if v[2]),
        sorted(k for k in _rows if k in _rec and k in _gacc and _rec[k] != _gacc[k]),
        why="the mark has to be computed from the log; a mark beside two numbers is a "
            "reader's inference printed as a check")

# ------------------- 5.2.5: the idle-step rate, its worst run, and its population
# Nothing read this sentence, so when run 45 added a seventh idle-step column the
# "六组" beside it stayed six and the tool said ok. The population is not the run set in
# general: it is the runs whose training record carries the rate, which is the same set
# the 5.2.2 idle-step row has a column for.
P_IDLE = (r"G=32 的空转步 ([\d.]+)%（(\d+)/(\d+)）是(" + NUM + r")组里最差的")
g = anchor("5.2.5: the idle-step population and its worst run", P_IDLE)
if g:
    idle = {}
    for _n in rl:
        _p = train_record(_n)
        if not _p.is_file():
            continue
        _v = json.loads(_p.read_text(encoding="utf-8")).get("mean_frac_reward_zero_std")
        if isinstance(_v, (int, float)):
            idle[_n] = _v
    if idle:
        _worst = max(idle, key=idle.get)
        cmp("5.2.5: the run with the worst idle-step rate", _worst, "GSPO(G=32) lr5e-5",
            why="the sentence names a run; a different run holding the maximum makes it "
                "false in a way a numeral cannot fix")
        cmp("5.2.5: the worst idle-step rate", float(g.group(1)),
            round(100 * idle[_worst], 2))
        cmp("5.2.5: the idle-step population", cn2int(g.group(4)), len(idle),
            kind="count", pattern=P_IDLE, group=4)
        print(f"         idle-step rates: "
              + "、".join(f"{_k} {100 * _v:.2f}%" for _k, _v in
                          sorted(idle.items(), key=lambda kv: -kv[1])))

# --------------------- 9: the idle-step audit's own coverage, read off the tool
# The sentence in section 9 says how many runs the audit covers and lists their rates in
# that tool's RUNS order. The population is the tool's, not the tables': it includes the
# second seed, which has no column anywhere. So the list is parsed out of the tool's
# source and filtered to the entries whose record is on disk -- never restated here,
# because a second copy of it is a second thing that can drift.
_IDLE_TOOL = ROOT / "tools/idle_step_table.py"
P_IDLE_AUDIT = (r"（当前 (" + NUM + r") 个 run 全部一致，按该工具 `RUNS` 的顺序：\s*"
                r"([\d.]+%(?:\s*、\s*[\d.]+%)*)）")
g = anchor("9: the idle-step audit's coverage", P_IDLE_AUDIT)
if g and _IDLE_TOOL.is_file():
    _entries = []
    for _node in ast.parse(_IDLE_TOOL.read_text(encoding="utf-8")).body:
        if (isinstance(_node, ast.Assign) and _node.targets
                and getattr(_node.targets[0], "id", None) == "RUNS"):
            for _e in _node.value.elts:
                _lab, _stem = ast.literal_eval(_e)[0], ast.literal_eval(_e)[1]
                _entries.append((_lab, _stem if _stem.endswith(".json")
                                 else _stem + ".json"))
    _covered = []
    for _lab, _stem in _entries:
        _p = REP / _stem
        if not _p.is_file():
            continue
        _v = json.loads(_p.read_text(encoding="utf-8")).get("mean_frac_reward_zero_std")
        if isinstance(_v, (int, float)):
            _covered.append((_lab, _v))
    if _covered:
        cmp("9: the idle-step audit's coverage (count)", cn2int(g.group(1)),
            len(_covered), kind="count", pattern=P_IDLE_AUDIT, group=1)
        cmp("9: the rates the audit lists, in the tool's RUNS order",
            [float(x) for x in re.split(r"\s*、\s*", g.group(2).rstrip("%").replace(
                "%", "")) if x],
            [round(100 * _v, 2) for _, _v in _covered],
            why="a run the audit picked up is an entry appended to this list; the "
                "numeral alone would leave the list naming the old population")
        print(f"         the audit covers {len(_covered)} run(s): "
              + "、".join(f"{_l} {100 * _v:.2f}%" for _l, _v in _covered))

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
