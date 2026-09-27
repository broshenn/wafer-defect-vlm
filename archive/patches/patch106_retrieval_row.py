"""Retrieval is the one metric family the claims checker does not read at all.

`grep -c 'retrieval\\|mAP' check_quantified_claims.py` returns 0. The document quotes
retrieval numbers in two places, one of which is a prose comparison:

    5.2.2:  | retrieval mAP@10 | **0.3873** | 0.3742 | −0.013（变差） |
    profile:| retrieval mAP@10 | 0.3873 | 0.3742 | 0.4260 | 0.4138 | 0.3748 | 0.3607 | 0.3782 | 0.4337 |

Every other metric family has a row of cells that some item compares against the
records: classification, the structured fields, clock, size, caption, idle steps. This
one has nothing, so a re-scored run -- which rewrites its report and nothing else --
would leave the document quoting a number no report contains, with the tool green.

The check maps the row through the *document's* column order (`p_cols`, read from the
table header), not through `RUNS`' declaration order, because they differ: the document
lists GSPO(G=4) before GSPO(G=8) and the map does not. Reading it in the wrong order
would compare the right numbers against the wrong runs, which is the failure mode of a
check like this one -- so the order comes from the artefact being checked.

There are two rows with that label. 5.2.2's has three cells (SFT, GRPO, a delta) and the
profile table's has one per column, so the row is selected by cell count rather than by
being the first match: taking the first would have compared a three-cell row against an
eight-column table and reported the wrong thing. That selection is the only thing this
patch got wrong on the first attempt, which is why the count is asserted before the
values are.

A cell that is not a number (the em dash a never-measured metric is written as) is
compared as None rather than raising: `add_run_columns.py` writes it that way on purpose,
and a checker that crashed on it would be a checker that gets removed.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"

# ---------------------------------------------------------------- 1. load() gains it
OLD_LOAD = '''        "must_hit": cap["must_hit_rate"],
        "per_class": {k: v["f1-score"] for k, v in cls["per_class"].items()},
    }'''
NEW_LOAD = '''        "must_hit": cap["must_hit_rate"],
        # Kept as None when the report has no retrieval block, so the row check below
        # can tell "not measured" from "measured as zero".
        "retrieval": (d.get("retrieval") or {}).get("mAP@10"),
        "per_class": {k: v["f1-score"] for k, v in cls["per_class"].items()},
    }'''

# ------------------------------------------------------------- 2. the row itself
OLD_AFTER_TABLES = '''# Compared before the class sets are used: a report that lost a class would make every
# "does this run keep every class" answer below refer to a smaller population, and the
# only visible symptom would be a class missing from a table nobody re-counted.'''

NEW_AFTER_TABLES = '''# ------------------------------------------- the retrieval row, the unread family
# Mapped through p_cols -- the document's own column order -- rather than RUNS' order,
# because the two differ (the document puts GSPO(G=4) before GSPO(G=8)). Comparing the
# right numbers against the wrong runs is what a check like this gets wrong.
#
# Two rows carry this label: 5.2.2's, which is three cells wide (SFT, one run, a delta),
# and the profile table's, which has one per column. The row is therefore chosen by its
# cell count. Taking the first match instead read the three-cell row and reported a
# length mismatch against the table it was never from.
_ret_rows = []
for _ln in text.split("\\n"):
    if _ln.startswith("| retrieval mAP@10 |"):
        _ret_rows.append([c.strip() for c in _ln.strip().strip("|").split("|")][1:])


def _cell(v):
    """A table cell as a number, or None when it is the not-measured dash."""
    return None if v.strip() in ("", "—", "-", "n/a") else float(v.strip())


if _ret_rows and p_cols:
    _ret = [r for r in _ret_rows if len(r) == len(p_cols)]
    if not _ret:
        cmp("the profile table's retrieval mAP@10 row has a cell per column",
            [len(r) for r in _ret_rows], [len(p_cols)] * len(_ret_rows),
            why="a row whose length does not match the header is a table that lost a "
                "cell, and reading it positionally would misalign every value after it")
    else:
        cmp("the profile table's retrieval mAP@10 row",
            [_cell(v) for v in _ret[0]],
            [None if present[c]["retrieval"] is None
             else round(present[c]["retrieval"], 4) for c in p_cols],
            why="retrieval is the one metric family with no other item reading it; a "
                "re-scored run rewrites its report and would leave this row quoting a "
                "value no report contains")
        print(f"         retrieval row: {len(_ret[0])} cell(s) over "
              f"{len(_ret_rows)} row(s) with this label")

# Compared before the class sets are used: a report that lost a class would make every
# "does this run keep every class" answer below refer to a smaller population, and the
# only visible symptom would be a class missing from a table nobody re-counted.'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, "/tmp/check_quantified_claims.py.bak-patch106")

for old, new, what in ((OLD_LOAD, NEW_LOAD, "load()'s return"),
                       (OLD_AFTER_TABLES, NEW_AFTER_TABLES, "the insertion point")):
    if c.count(old) != 1:
        sys.exit(f"{what} appears {c.count(old)} times, expected 1; nothing written")
    c = c.replace(old, new, 1)

CHECKER.write_text(c, encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch106", CHECKER)
    sys.exit(f"the patched checker does not compile, restored:\n{r.stderr}")

print("the retrieval mAP@10 row is now compared against the reports, mapped through the "
      "document's own column order.")
print()
print("--- re-running the checker ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
for ln in r.stdout.strip().splitlines()[-12:]:
    print("  " + ln)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch106", CHECKER)
    sys.exit(f"\nthe checker is not green (exit {r.returncode}); RESTORED -- the row and "
             f"the reports disagree and that is the finding, not the patch")
print()
print("checker green: every metric family in the profile table is now read from the "
      "records, retrieval included")
