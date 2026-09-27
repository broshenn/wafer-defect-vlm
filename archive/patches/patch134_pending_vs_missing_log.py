"""A declared run that has not run yet is not a run outside the quantifier.

`check_quantified_claims`'s 5.2 item reads every run declared in `idle_step_table.RUNS`
and requires its log to state `gradient_accumulation_steps`, because the section's
confound argument rests on accumulation == group size holding for *every* run it quotes.
Declaring seed 3409 in that table before the run exists (patch133, deliberately: a run
declared late shows up as UNACCOUNTED) made that item fail with

    document says: []
    records say  : ['GRPO G=4 lr1e-5 seed3409']

-- the checker reporting a *document* defect because a run it had been told about has no
log yet. The document is right, the run is not wrong, and the item's own words are what
shows the distinction was missing: "a run whose log cannot be read is a run outside the
quantifier". A run that has not been launched is not a run whose log cannot be read.

`idle_step_table` already draws this line and prints it -- a declared entry with no log
prints "no log with values yet" and is left out of the rows rather than counted as a
zero. The checker is given the same test, with the record as the authority on whether the
run finished (outcome completed, steps_logged == max_steps_requested, the same reading
`kl_length_confound` uses to decide which runs "all runs" means): an unfinished declared
run is printed as not yet measurable and skipped; a finished one whose log lacks the line,
or whose log is missing, still fails -- which is what the item exists to catch.

Why this had to be fixed within the quarter hour: run 42's landing runs this checker.
A red item there stops the landing, and the landing is what restarts the end-of-day
chain -- so a declaration made in good faith would have taken the evening down, with the
only symptom a message about a run that had not started.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
CHECK = ROOT / "tools/check_quantified_claims.py"

OLD = '''    _gacc, _gsize, _acc_miss = {}, {}, []
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
        _gsize[_lab] = int(_m.group(1))'''

NEW = '''    _gacc, _gsize, _acc_miss, _pending = {}, {}, [], []
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
              + "、".join(_pending))'''

s = CHECK.read_text(encoding="utf-8")
shutil.copy2(CHECK, "/tmp/check_quantified_claims.py.bak-patch134")
if s.count(OLD) != 1:
    sys.exit(f"the checker anchor matches {s.count(OLD)} times; nothing written")
CHECK.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECK)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch134", CHECK)
    sys.exit(f"the checker does not compile; restored:\n{r.stderr}")
print("check_quantified_claims.py: a declared run that has not finished is printed, "
      "not counted as a log that cannot be read.")

r = subprocess.run([sys.executable, str(CHECK)], capture_output=True, text=True, cwd=str(ROOT))
out = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
print(f"\n--- check_quantified_claims: exit {r.returncode} ---")
for ln in out:
    if ("gradient accumulation" in ln or "no finished run yet" in ln
            or "moved" in ln or "stale" in ln.lower() or ln.strip().startswith("- ")):
        print("  " + ln.strip()[:170])
