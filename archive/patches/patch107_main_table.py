"""The main comparison table is read by nothing, cell by cell.

Every item in this checker reads a *sentence*. `p_cols` -- the document's column order --
is used once, for a sorted set comparison against the records, and no row is read
through it. So the twelve-row table at line 247 is checked only where its numbers happen
to be repeated in prose, and three of its rows are repeated nowhere:

    robustness accuracy   0 mentions anywhere in this file
    flip rate             0
    retrieval mAP@10      0

A re-scored run rewrites its report and its column in both tables. For those three rows
the document would quote a value no report contains and this tool would say ok. That is
the blind spot section 8 item 9 describes -- a number that is right where it is checked
and wrong where it is not, with the artefact unable to show the difference.

Two tables, opposite column orders, and the trap in reading them
---------------------------------------------------------------
The document carries two eight-column tables and they list the two GSPO(G=8) runs in
OPPOSITE orders (line 247 puts lr5e-5 first; line 476 puts lr1e-5 first):

    main   : ... GSPO(G=4) lr1e-5 | GSPO(G=8) lr5e-5 | GSPO(G=8) lr1e-5 | GSPO(G=32) lr5e-5
    second : ... GSPO(G=4) lr1e-5 | GSPO(G=8) lr1e-5 | GSPO(G=8) lr5e-5 | GSPO(G=32) lr5e-5

Both are internally correct -- each row follows its own header. But a row read through
the *other* table's header attributes every value to the wrong run: 0.3607 and 0.3782
are each byte-for-byte right and land in each other's column. That is this project's
defect class in its purest form, and the first version of this item committed it. It read
the retrieval row (line 259, in the main table) through `p_cols`, which comes from the
second table, and reported a transposition in a document where neither table is wrong.

So each row is read as part of the table whose header introduced it -- the tables are
enumerated from their headers, and a row belongs to the block below one. No other
table's header is in scope to reach for, and the cross-table check below compares by run
name rather than by position for the same reason. The retrieval row is not selected by
being the first match for its label either: `retrieval mAP@10` is a row of the
three-column 5.2.2 table as well, so a first-match reader compares a three-cell row
against eight columns. It is selected by its table's width.

What the second table adds
--------------------------
Its four rows repeat values the main table already carries, and its bold is *stated* to
mean something (5.2.5: 加粗 = 该行最好的一档). Both are claims about the relationship
between cells, which is the kind that stays true-looking while its subject moves: land a
run with a lower clock MAE and the bold is on the wrong cell with every number in the
table still correct. `add_run_columns.py` recomputes the bold when it appends a column,
so the pipeline maintains it; nothing verified it. Now three things do -- that the two
tables agree per run, that exactly one cell per row is bold, and that the bolded cell
holds that row's extreme in the direction its own label names.

Everything here is a **fact**, never a count: a re-measured value is not a numeral that
moved, and `--fix` rewriting a cell would produce a number nobody computed.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
BAK = "/tmp/check_quantified_claims.py.bak-patch107"

# ---------------------------------------------------------- 1. the fields the rows need
OLD_LOAD = '''        "must_hit": cap["must_hit_rate"],
        "per_class": {k: v["f1-score"] for k, v in cls["per_class"].items()},
    }'''

NEW_LOAD = '''        "must_hit": cap["must_hit_rate"],
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
    }'''

# ---------------------------------------------------------------- 2. the table reader
ANCHOR = '''cmp("the reward table's columns match the RL runs on disk",
    sorted(r_cols), sorted(rl),
    why="this table carries no SFT column -- it is the runs with a reward signal")
'''

BLOCK = r'''
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
'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, BAK)

for old, new, what in ((OLD_LOAD, NEW_LOAD, "load()'s return"),
                       (ANCHOR, ANCHOR + BLOCK, "the insertion point")):
    if c.count(old) != 1:
        sys.exit(f"{what} appears {c.count(old)} times, expected 1; nothing written")
    c = c.replace(old, new, 1)

# The block calls `train_record` and `present`, both defined above the insertion point.
for _need, _where in (("def _tables():", "the inserted block"),
                      ("present = {n: v for n, v in records.items()", "above the block"),
                      ("def train_record(", "above the block")):
    if c.count(_need) != 1:
        sys.exit(f"{_need!r} appears {c.count(_need)} times; {_where} is not as "
                 f"expected; nothing written")
if c.find("present = {n: v for n, v in records.items()") > c.find("def _tables():"):
    sys.exit("the block is inserted above the records it reads; nothing written")

CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(BAK, CHECKER)
    sys.exit(f"the patched checker does not compile, restored:\n{r.stderr}")

print("the main table is now read cell by cell, and the second table's agreement with it "
      "and its bold are read too.")
print("\n--- re-running the checker ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = r.stdout.strip().splitlines()
for ln in out:
    s = ln.strip()
    if ("table" in s or "bold" in s or "STALE" in s):
        print("  " + s)
if out:
    print("  " + out[-1].strip())
if r.returncode != 0:
    shutil.copy2(BAK, CHECKER)
    sys.exit(f"\nthe checker is not green (exit {r.returncode}); RESTORED -- the tables "
             f"and the reports disagree, and that is the finding, not the patch")
print("\nchecker green: every row of both tables is read from the records, the two tables "
      "agree per run, and the bold is on the value its own label says it marks")
