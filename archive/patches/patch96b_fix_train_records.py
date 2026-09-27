"""Two mistakes in the fix I just made, both of the kind it was written to prevent.

  1. `train_record` appended `_train_result.json` to the value the map returns, but the
     map's values are whole file stems -- `grpo_train_result`, not `grpo`. So it looked
     for `grpo_train_result_train_result.json`. The fix reported the column as missing,
     which is the safe direction to be wrong in and is exactly why the missing-record
     finding exists rather than a silent skip: had it been wrong in the other direction
     it would have computed a ranking over a file that is not that run's record.
  2. the missing-record check was inside the per-reward-row loop, so it fired four times
     with an accumulating list. It belongs after the loop: the question is about the
     columns, not about each reward row.

Both are corrected here, and the digest's copy is corrected with them.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
DIGEST = ROOT / "tools/fact_digest.py"

OLD_FN = '''    return REP / f"{TRAIN_RESULT.get(name, RUNS[name])}_train_result.json"'''
NEW_FN = '''    # The map holds a whole file stem, and so does the fallback. The first version of
    # this appended the suffix to the map's value as well, producing
    # `grpo_train_result_train_result.json` -- a wrong lookup in the harmless direction,
    # which is precisely why a missing record is a reported finding and not a skip.
    return REP / f"{TRAIN_RESULT.get(name, RUNS[name] + '_train_result')}.json"'''

OLD_CMP = '''        if missing:
            cmp("5.2.5: a training record for every reward-table column", [], missing,
                why="a column with no training record cannot be ranked. Dropping it "
                    "would rank 'lowest' over a smaller set than the sentence beside it "
                    "names, and the result would look like a document error rather than "
                    "a lookup one")'''
NEW_CMP = '''    if missing:
        cmp("5.2.5: a training record for every reward-table column", [], missing,
            why="a column with no training record cannot be ranked. Dropping it would "
                "rank 'lowest' over a smaller set than the sentence beside it names, "
                "and the result would look like a document error rather than a lookup "
                "one")'''

D_OLD = '''        p = REP / f"{TRAIN_RESULT.get(n, tag)}_train_result.json"'''
D_NEW = '''        p = REP / f"{TRAIN_RESULT.get(n, tag + '_train_result')}.json"'''

s = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, "/tmp/check_quantified_claims.py.bak-patch96b")
for old, new, what in ((OLD_FN, NEW_FN, "the lookup"),
                       (OLD_CMP, NEW_CMP, "the missing-record check")):
    if s.count(old) != 1:
        sys.exit(f"{what} appears {s.count(old)} times, expected 1; nothing written")
    s = s.replace(old, new, 1)
CHECKER.write_text(s, encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch96b", CHECKER)
    sys.exit(f"the checker does not compile, restored:\n{r.stderr}")

d = DIGEST.read_text(encoding="utf-8")
shutil.copy2(DIGEST, "/tmp/fact_digest.py.bak-patch96b")
if d.count(D_OLD) != 1:
    sys.exit(f"the digest lookup appears {d.count(D_OLD)} times; nothing written "
             f"(the checker is already corrected)")
DIGEST.write_text(d.replace(D_OLD, D_NEW, 1), encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(DIGEST)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/fact_digest.py.bak-patch96b", DIGEST)
    sys.exit(f"the digest does not compile, restored:\n{r.stderr}")

print("corrected: the stem is used as a stem, and the missing-record check runs once")
