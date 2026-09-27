"""Two run-set claims the checker did not read, both of which grew today.

**5.2 says 「每个 run 的梯度累积都设成了与组大小相等（4/4、8/8、32/32）」.** A claim about
every run, with the population written as three example pairs -- and it is the claim the
whole 5.2 confound section rests on, because if one run's accumulation had not equalled
its group size, the effective batch of that run would not be 4x/16x/64x its group size
and every "group size" reading in the section would be a reading of something else. It
was verified by hand when it was written, over the runs that existed then. It is read
here from each run's own log, over every run, so that the eighth run is inside the
quantifier rather than assumed to be.

**Item 11's table lists which records are wrong**, and it listed five rows for eight
runs: run 41 -- whose record writes `grad_accum: 4` while its log says
`gradient_accumulation_steps 32` -- was not in it. A table of defects is a population
like any other, and an omitted row is a defect (or a run) that reads as absent. So the
table's three columns are read back and compared against the logs and the records: the
run labels, the accumulations the logs actually show, the values the records write, and
which rows are marked wrong. The 「记录写的」 column is allowed to disagree with the log
-- that is what the column is for -- but the *set of rows that disagree* must be exactly
the set of rows marked 错, and the marks must not be the only place the disagreement
lives.

The mapping from a table row's label to a result file and a log is read from
`tools/idle_step_table.py`'s `RUNS`, as in the std block: it is the one place those three
are declared together, and the one that already knows a run's record is not always named
after its tag.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
CB = "/tmp/check_quantified_claims.py.bak-patch123"

BLOCK = '''
# --------------- 5.2: every run's gradient accumulation, and the table of wrong records
# The whole 5.2 confound section rests on this claim: accumulation == group size is what
# makes an effective batch 4x, 16x or 64x its group. It was verified by hand over the
# runs that existed when it was written; the population has grown since, so it is read
# from each run's own log here, and the eighth run is inside the quantifier.
P_GACC = (r"\\*\\*每个 run 的梯度累积都设成了与组大小相等\\*\\*（4/4、8/8、32/32）")
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
    _gacc, _gsize, _acc_miss = {}, {}, []
    for _lab, (_stem, _logs) in sorted(_decl.items()):
        _got = None
        for _lg in _logs:
            _f = ROOT / "logs" / _lg
            if _f.is_file():
                _m = re.findall(r"(?<![A-Za-z0-9_])gradient_accumulation_steps=?(\\d+)",
                                _f.read_text(encoding="utf-8", errors="replace"))
                if _m:
                    _got = int(_m[-1])
                    break
        _m = re.search(r"G=(\\d+)", _lab)
        if _got is None or not _m:
            _acc_miss.append(_lab)
            continue
        _gacc[_lab] = _got
        _gsize[_lab] = int(_m.group(1))
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

# Item 11's table of wrong records: a table of defects is a population too, and run 41
# was missing from it while its own record disagreed with its own log.
P_I11 = (r"\\| run \\| 实际累积步（该 run 自己的日志）\\| 记录写的 \\| \\|\\s*\\n"
         r"\\| --- \\| --- \\| --- \\| --- \\|\\s*\\n((?:\\s*\\|[^\\n]*\\|[^\\n]*\\n)+)")
g = anchor("11: the table of runs whose record disagrees with its own log", P_I11)
if g:
    _doc_rows = {}
    for _ln in g.group(1).splitlines():
        _c = [x.strip() for x in _ln.strip().strip("|").split("|")]
        if len(_c) >= 4 and re.search(r"G=\\d+", _c[0]):
            _doc_rows[_c[0]] = (_c[1], _c[2].strip("*").strip(), "错" in _c[3])
    _rec, _marks, _vs_log = {}, [], {}
    for _lab, (_stem, _logs) in sorted(_decl.items()):
        _f = REP / _stem
        if not _f.is_file():
            continue
        _w = (json.loads(_f.read_text(encoding="utf-8")).get("config") or {}).get(
            "grad_accum")
        if _w is None:
            continue
        _rec[_lab] = int(_w)
        if _lab in _doc_rows:
            if _gacc.get(_lab) is not None:
                _vs_log[_lab] = int(_w) - _gacc[_lab]
    _missing = sorted(set(_rec) - set(_doc_rows))
    cmp("11: every run with a record appears in the table of wrong records",
        [], _missing,
        why="the table lists which records are wrong; a run missing from it is a run "
            "whose record is vouched for by omission")
    cmp("11: the runs it marks as wrong", sorted(k for k, v in _doc_rows.items() if v[2]),
        sorted(_lab for _lab in _doc_rows if _vs_log.get(_lab, 0) != 0),
        why="the mark and the disagreement have to be the same set; a mark that is not "
            "computed from the log is a claim about the log")
    cmp("11: the accumulation values the table prints for the records",
        {k: int(v[1]) for k, v in _doc_rows.items() if v[1].lstrip("-").isdigit()},
        {k: _rec[k] for k in _doc_rows if k in _rec})
    cmp("11: the accumulation values it prints from the logs",
        {k: int(v[0]) for k, v in _doc_rows.items() if v[0].lstrip("-").isdigit()},
        {k: _gacc[k] for k in _doc_rows if k in _gacc})
'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, CB)
# Insert before the section comment that introduces the idle-step block. Matched on the
# line's text rather than on the banner's dashes, because the dash count is not part of
# the anchor's meaning and counting it wrong is a mistake with no upside.
MARK = "the idle-step rate, its worst run, and its population"
_lines = c.splitlines(keepends=True)
_hits = [i for i, ln in enumerate(_lines) if MARK in ln]
if len(_hits) != 1:
    sys.exit(f"the insertion line matches {len(_hits)} time(s); nothing written")
_lines.insert(_hits[0], BLOCK.lstrip("\n") + "\n")
CHECKER.write_text("".join(_lines), encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(CB, CHECKER)
    sys.exit(f"does not compile; restored:\n{r.stderr}")

r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = (r.stdout or "").strip().splitlines()
print("--- the new items ---")
for ln in out:
    s = ln.strip()
    if ("gradient accumulation" in s or "grad_accum" in s
            or s.startswith("11:") or "record appears in the table" in s
            or "marks as wrong" in s or "values the table prints" in s
            or "values it prints from the logs" in s or "log states its" in s):
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
    print(r.stderr.strip()[-600:])
print(f"\nchecker exit {r.returncode}; {n_stale} stale item(s)")
