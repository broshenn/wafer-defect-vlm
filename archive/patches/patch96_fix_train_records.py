"""A column was being dropped from the reward-row check without saying so.

`check_quantified_claims.py` verifies 5.2.5's sentence "在 4 个奖励里有 3 个是六组最低"
by ranking each reward row over the reward table's columns and asking how many G=32 is
lowest in. To rank a column it needs that run's training record, and it looked for
`{tag}_train_result.json` -- true for every run except two early ones, whose records are
named after the algorithm instead: `qwen35_9b_grpo` writes `grpo_train_result.json`, and
SFT writes `sft_train_result.json`.

The lookup was `if p.is_file()` inside the loop, so a column with no file was skipped and
the ranking ran over a subset. The sentence says six columns; the check measured five.

It gave the same answer, which is the worst way for this to go: an unfalsifiable pass. Had
GRPO lr1e-5 been the lowest in one of the rows G=32 wins, the record would have said three
and the check would have said two, and the difference would have looked like a document
error rather than a lookup error. And nothing in the output distinguished "six columns,
none lowest for G=32" from "five columns, one of them missing".

Two changes:

  * the exception is written down as a map, so the file is found rather than not looked
    for;
  * a column with no training record is now a *finding*, not a skip. `cmp` is called with
    the missing names against the empty list, so the check fails loudly and names them.
    The same reasoning as every other check here: the failure to look must be as visible
    as a failure to match.

The same lookup, copied into the preview digest, is corrected in the same pass.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
DIGEST = ROOT / "tools/fact_digest.py"

MAP = '''# A run's report is `{tag}__report.json` and its training record is
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
    return REP / f"{TRAIN_RESULT.get(name, RUNS[name])}_train_result.json"


'''

OLD_LOOP = '''                vals[n] = json.loads(p.read_text(encoding="utf-8"))[
                    "reward_signal"][k]["mean_std_across_steps"]'''
NEW_LOOP = '''                vals[n] = json.loads(p.read_text(encoding="utf-8"))[
                    "reward_signal"][k]["mean_std_across_steps"]
        if missing:
            cmp("5.2.5: a training record for every reward-table column", [], missing,
                why="a column with no training record cannot be ranked. Dropping it "
                    "would rank 'lowest' over a smaller set than the sentence beside it "
                    "names, and the result would look like a document error rather than "
                    "a lookup one")'''

OLD_LOOKUP = '''            p = REP / f"{RUNS[n]}_train_result.json"
            if p.is_file():
'''
NEW_LOOKUP = '''            p = train_record(n)
            if not p.is_file():
                missing.append(n)
                continue
            if p.is_file():
'''
OLD_INIT = '''    low = 0
    for k in ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock"):'''
NEW_INIT = '''    low = 0
    missing = []
    for k in ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock"):'''

# --- the digest's copy of the same lookup -----------------------------------------
D_OLD_LOOKUP = '''        p = REP / f"{tag}_train_result.json"
        if p.is_file():'''
D_NEW_LOOKUP = '''        p = REP / f"{TRAIN_RESULT.get(n, tag)}_train_result.json"
        if p.is_file():'''
D_OLD_HEAD = '''rec = {}
for name, tag in RUNS.items():'''
D_NEW_HEAD = '''# Two runs predate the `{tag}_train_result.json` convention and named their record
# after the algorithm. Kept in step with check_quantified_claims.py on purpose: if the
# two lists of what counts as a run's record drift apart, the digest and the gate
# disagree and neither is evidence.
TRAIN_RESULT = {
    "GRPO(G=4) lr1e-5": "grpo_train_result",
    "SFT": "sft_train_result",
}

rec = {}
for name, tag in RUNS.items():'''

s = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, "/tmp/check_quantified_claims.py.bak-patch96")
if "TRAIN_RESULT" in s:
    sys.exit("the checker already has TRAIN_RESULT; nothing written")
if s.count(OLD_INIT) != 1:
    sys.exit(f"the reward-row init appears {s.count(OLD_INIT)} times; nothing written")
if s.count(OLD_LOOKUP) != 1:
    sys.exit(f"the reward-row lookup appears {s.count(OLD_LOOKUP)} times; nothing written")
if s.count(OLD_LOOP) != 1:
    sys.exit(f"the reward-row read appears {s.count(OLD_LOOP)} times; nothing written")

s = s.replace(OLD_LOOP, NEW_LOOP, 1)
s = s.replace(OLD_LOOKUP, NEW_LOOKUP, 1)
s = s.replace(OLD_INIT, NEW_INIT, 1)
# the map goes next to the other name maps, above the columns that use it
anchor = "def label_of("
if s.count(anchor) != 1:
    sys.exit("cannot place TRAIN_RESULT; nothing written")
s = s.replace(anchor, MAP + anchor, 1)
CHECKER.write_text(s, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch96", CHECKER)
    sys.exit(f"the patched checker does not compile, restored:\n{r.stderr}")

# the digest, same correction
d = DIGEST.read_text(encoding="utf-8")
shutil.copy2(DIGEST, "/tmp/fact_digest.py.bak-patch96")
for old, new in ((D_OLD_HEAD, D_NEW_HEAD), (D_OLD_LOOKUP, D_NEW_LOOKUP)):
    if d.count(old) != 1:
        sys.exit(f"the digest anchor {old[:40]!r} appears {d.count(old)} times; "
                 f"nothing written (the checker is already patched)")
    d = d.replace(old, new, 1)
DIGEST.write_text(d, encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(DIGEST)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/fact_digest.py.bak-patch96", DIGEST)
    sys.exit(f"the patched digest does not compile, restored:\n{r.stderr}")

print("both now find the two records by name, and both report a missing one loudly")
