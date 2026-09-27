"""The item-11 table item must not depend on which of two shapes the table is in.

patch123 added an item that reads item 11's table of wrong records. It anchored on the
header that **patch118 rewrites the table to have** -- a fourth column carrying the
verdict -- and patch118 cannot run until pid 3699 exits, because it edits the training
script that pid is reading. So the item went straight to its absent branch:

    STALE  11: the table of runs whose record disagrees with its own log
           the sentence this check was written against is not in the document any more

which is the right thing for it to say and the wrong thing for it to be: a guard that
only functions after a later patch is a guard whose failure looks like a document
problem. The absent branch earned its keep again -- this is the fourth time today a
sentence-shaped check reported "not there" instead of quietly matching nothing -- but the
fix is to read the table, not to reorder the patches.

So the header is matched on its stable part (`| run | 实际累积步... | 记录写的 |`) with the
fourth column optional, and the row parse accepts three or four cells. The two columns
that exist in both shapes are checked in both: every run with a record must appear, the
「实际累积步」 column must equal the log, and the 「记录写的」 column must equal the record.
The verdict column, when it exists, is compared against the disagreement computed here --
and when it does not exist there is nothing to compare it with, which is said rather than
silently passed, because the current table states which records are wrong only by
juxtaposition (8 beside 4) and a reader of it has to notice.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
CB = "/tmp/check_quantified_claims.py.bak-patch124"

P_OLD = '''P_I11 = (r"\\| run \\| 实际累积步（该 run 自己的日志）\\| 记录写的 \\| \\|\\s*\\n"
         r"\\| --- \\| --- \\| --- \\| --- \\|\\s*\\n((?:\\s*\\|[^\\n]*\\|[^\\n]*\\n)+)")'''
P_NEW = '''# The header's fourth column is optional: patch118 adds it, and this item must read the
# table before and after. Matching the shape patch118 creates would make the item depend
# on the order the patches run in, and its absent branch would report a document problem
# for a check that is simply early.
P_I11 = (r"\\| run \\| 实际累积步(?:（该 run 自己的日志）)? \\| 记录写的 \\|(?: \\|)?\\s*\\n"
         r"\\| --- \\| --- \\| --- \\|(?: --- \\|)?\\s*\\n((?:\\s*\\|\\s*G=\\d+[^\\n]*\\n)+)")'''

ROW_OLD = '''    _doc_rows = {}
    for _ln in g.group(1).splitlines():
        _c = [x.strip() for x in _ln.strip().strip("|").split("|")]
        if len(_c) >= 4 and re.search(r"G=\\d+", _c[0]):
            _doc_rows[_c[0]] = (_c[1], _c[2].strip("*").strip(), "错" in _c[3])'''
ROW_NEW = '''    _doc_rows, _has_verdict = {}, False
    for _ln in g.group(1).splitlines():
        _c = [x.strip() for x in _ln.strip().strip("|").split("|")]
        if len(_c) >= 3 and re.search(r"G=\\d+", _c[0]):
            if len(_c) >= 4:
                _has_verdict = True
                _doc_rows[_c[0]] = (_c[1], _c[2].strip("*").strip(), "错" in _c[3])
            else:
                _doc_rows[_c[0]] = (_c[1], _c[2].strip("*").strip(), None)
    if not _has_verdict:
        print("         (the table has no verdict column yet: which records are wrong "
              "is stated only by juxtaposition, so the marks cannot be read)")'''
MARKS_OLD = '''    cmp("11: the runs it marks as wrong", sorted(k for k, v in _doc_rows.items() if v[2]),
        sorted(_lab for _lab in _doc_rows if _vs_log.get(_lab, 0) != 0),
        why="the mark and the disagreement have to be the same set; a mark that is not "
            "computed from the log is a claim about the log")'''
MARKS_NEW = '''    if _has_verdict:
        cmp("11: the runs it marks as wrong",
            sorted(k for k, v in _doc_rows.items() if v[2]),
            sorted(_lab for _lab in _doc_rows if _vs_log.get(_lab, 0) != 0),
            why="the mark and the disagreement have to be the same set; a mark that is "
                "not computed from the log is a claim about the log")'''
NUM_OLD = '''    cmp("11: the accumulation values the table prints for the records",
        {k: int(v[1]) for k, v in _doc_rows.items() if v[1].lstrip("-").isdigit()},
        {k: _rec[k] for k in _doc_rows if k in _rec})
    cmp("11: the accumulation values it prints from the logs",
        {k: int(v[0]) for k, v in _doc_rows.items() if v[0].lstrip("-").isdigit()},
        {k: _gacc[k] for k in _doc_rows if k in _gacc})'''
NUM_NEW = '''    # The two columns the table has in either shape, checked in both.
    _pr = {k: int(v[1].strip("*").strip()) for k, v in _doc_rows.items()
           if v[1].strip("*").strip().isdigit()}
    _pl = {k: int(v[0]) for k, v in _doc_rows.items() if v[0].isdigit()}
    cmp("11: the runs the table lists, against the runs that have a record",
        sorted(_doc_rows), sorted(set(_rec) | set(_doc_rows)),
        why="a run missing from a table of wrong records is a record vouched for by "
            "omission -- run 41 was the instance this item exists for")
    cmp("11: the accumulation values the table prints for the records", _pr,
        {k: _rec[k] for k in _pr})
    cmp("11: the accumulation values it prints from the logs", _pl,
        {k: _gacc[k] for k in _pl})'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, CB)
edits = [(P_OLD, P_NEW), (ROW_OLD, ROW_NEW), (MARKS_OLD, MARKS_NEW), (NUM_OLD, NUM_NEW)]
for old, new in edits:
    if c.count(old) != 1:
        sys.exit(f"anchor {old[:46]!r} matches {c.count(old)} times; nothing written")
    c = c.replace(old, new, 1)
# The two later items referenced _missing, which the "lists" item above now subsumes.
c = c.replace('''    _missing = sorted(set(_rec) - set(_doc_rows))
    cmp("11: every run with a record appears in the table of wrong records",
        [], _missing,
        why="the table lists which records are wrong; a run missing from it is a run "
            "whose record is vouched for by omission")
''', "", 1)
CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(CB, CHECKER)
    sys.exit(f"does not compile; restored:\n{r.stderr}")

r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = (r.stdout or "").strip().splitlines()
print("--- the item-11 items ---")
for ln in out:
    s = ln.strip()
    if (s.startswith("11:") or "verdict column yet" in s
            or "gradient accumulation" in s):
        print("  " + s[:160])
n_stale = sum(1 for ln in out if ln.strip().startswith("STALE"))
for i, ln in enumerate(out):
    if ln.strip().startswith("STALE"):
        for nx in out[i:i + 3]:
            print("  " + nx.rstrip())
print("--- tail ---")
for ln in out[-3:]:
    print("  " + ln.strip())
if r.stderr.strip():
    print("--- stderr ---")
    print(r.stderr.strip()[-500:])
print(f"\nchecker exit {r.returncode}; {n_stale} stale item(s)")
