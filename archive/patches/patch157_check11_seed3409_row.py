"""The check-11 table is one row short, for the same reason it existed at all.

`land_finish.sh`'s second step refused the third seed's landing with:

    STALE  11: the runs the table lists, against the runs that have a record
    document says: [9 runs]
    records say  : [10 runs, the extra being GRPO G=4 lr1e-5 seed3409]
    1 fact(s) about the run set have moved. The document is not wrong about any
    number; it is describing a run set that has changed, and the sentence has to be
    rewritten.

That is the check working as designed, and the refusal is the honest outcome: the table
in section 8 item 11 lists the runs whose record's gradient accumulation is compared with
their own log, and the third seed -- which landed minutes earlier -- has such a record
and was not in the table. A run missing from a table of wrong records is a record
vouched for by omission.

The row is added with the values the checker's own output printed for that run (its log
records 4, its record says 4, so it is not one of the wrong ones), and the cell wording
is copied from the neighbouring GRPO G=4 row rather than invented. `land_finish.sh` is
then run again; the row alone is what the refusal asked for, and if the check finds
anything else it will say so rather than be talked past.
"""
import pathlib
import re
import shutil
import sys

DOC = pathlib.Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
TOKEN = re.compile(r"(?<![.\d])-?\d+\.\d{3,6}")

ANCHOR = "    | GRPO G=4 lr1e-5 seed3408 | 4 | 4 | 对（常量碰巧等于实测值） |\n"
ROW = "    | GRPO G=4 lr1e-5 seed3409 | 4 | 4 | 对（常量碰巧等于实测值） |\n"

s = DOC.read_text(encoding="utf-8")
if ROW in s:
    sys.exit("the third seed's row is already in the table; nothing written")
if s.count(ANCHOR) != 1:
    sys.exit(f"the row above the insertion point matches {s.count(ANCHOR)} times; "
             f"nothing written")
if TOKEN.findall(ROW):
    sys.exit("the new row quotes a decimal; nothing written")

# The row has to satisfy the checker's own row pattern, not merely look like a row:
# `\s*\|[^|\n]*G=\d+[^|\n]*\|[^\n]*\n`, parsed as label | log | record | verdict.
ROW_PAT = re.compile(r"\s*\|[^|\n]*G=\d+[^|\n]*\|[^\n]*\n")
if not ROW_PAT.fullmatch(ROW):
    sys.exit("the new row does not match the pattern the checker reads rows with; "
             "nothing written")
cells = [c.strip() for c in ROW.strip().strip("|").split("|")]
if cells[0] != "GRPO G=4 lr1e-5 seed3409" or cells[1] != "4" or cells[2] != "4":
    sys.exit(f"the row parses as {cells}, not the label and the two values; nothing "
             f"written")
if "错" in cells[3]:
    sys.exit("the row would mark the run as wrong; its log and its record both say 4, "
             "so it is not one of the wrong ones -- nothing written")

shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch157")
before = s.count("**")
s = s.replace(ANCHOR, ANCHOR + ROW, 1)
if (s.count("**") - before) % 2:
    sys.exit("bold markers changed by an odd number; nothing written")
DOC.write_text(s, encoding="utf-8")

# Read the table back the way the checker reads it and print what it now holds.
tbl = re.search(r"\| run \| 实际累积步（该 run 自己的日志）\| 记录写的 \| \|\s*\n"
                r"\s*\| --- \| --- \| --- \| --- \|\s*\n"
                r"((?:\s*\|[^|\n]*G=\d+[^|\n]*\|[^\n]*\n)+)", s)
if not tbl:
    sys.exit("the table no longer parses after the write; restore from "
             "/tmp/LIMITATIONS.md.bak-patch157")
rows = []
for line in tbl.group(1).splitlines():
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    rows.append((c[0], c[1], c[2], "错" in c[3]))
print(f"the table now lists {len(rows)} runs:")
for label, log, rec, wrong in rows:
    print(f"  {label:28s} log {log:>2s}  record {rec:>2s}  {'WRONG' if wrong else 'ok'}")
print(f"\nthird seed: {'present' if any(r[0].endswith('seed3409') for r in rows) else 'MISSING'}"
      f" ({len(s.splitlines())} lines)")
