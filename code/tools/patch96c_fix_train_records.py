"""Correct the reward-row block: the lookup, and where the missing-record check sits.

State being corrected: the map exists, `train_record` appends `_train_result.json` to a
value that already is one, and the missing-record check sits inside the per-reward-row
loop. The last edit also dedented it without moving the lines below, which put
`low += 1` inside `if missing:` -- so the check reported 0 rows where the digest, reading
the same records, reported 3.

That is worth naming, because it is this project's defect class turning up in the tool
instead of the document: the count was still a number, still printed, still plausible,
and it now answered a question nobody asked ("how many rows is G=32 lowest in, given a
missing record"). Nothing about the output said so. The digest disagreed, which is the
only reason it was caught -- and the digest only disagreed because it had been corrected
in the same pass.

The replacement moves the check out of the loop to where the question belongs and fixes
the stem handling. The whole block is replaced at once rather than in pieces, because
patching indentation-sensitive code in pieces is what produced the bug.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"

OLD_FN = '''    return REP / f"{TRAIN_RESULT.get(name, RUNS[name])}_train_result.json"'''
NEW_FN = '''    # The map holds a whole file stem, and so does the fallback. The first version of
    # this appended the suffix to the map's value as well, producing
    # `grpo_train_result_train_result.json` -- a wrong lookup in the harmless direction,
    # which is precisely why a missing record is a reported finding and not a skip.
    return REP / f"{TRAIN_RESULT.get(name, RUNS[name] + '_train_result')}.json"'''

OLD = '''    low = 0
    missing = []
    for k in ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock"):
        vals = {}
        for n in r_cols:
            p = train_record(n)
            if not p.is_file():
                missing.append(n)
                continue
            if p.is_file():
                vals[n] = json.loads(p.read_text(encoding="utf-8"))[
                    "reward_signal"][k]["mean_std_across_steps"]
        if missing:
            cmp("5.2.5: a training record for every reward-table column", [], missing,
                why="a column with no training record cannot be ranked. Dropping it "
                    "would rank 'lowest' over a smaller set than the sentence beside it "
                    "names, and the result would look like a document error rather than "
                    "a lookup one")
        if vals and min(vals, key=vals.get) == "GSPO(G=32) lr5e-5":
            low += 1
'''

NEW = '''    low = 0
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
'''

s = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, "/tmp/check_quantified_claims.py.bak-patch96c")
for old, new, what in ((OLD_FN, NEW_FN, "the stem lookup"), (OLD, NEW, "the reward block")):
    if s.count(old) != 1:
        sys.exit(f"{what} appears {s.count(old)} times, expected 1; nothing written")
    s = s.replace(old, new, 1)
CHECKER.write_text(s, encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch96c", CHECKER)
    sys.exit(f"does not compile, restored:\n{r.stderr}")

# Read it back the way the document's own numbers are read: from a run of the tool.
out = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                     env={**__import__("os").environ})
for line in out.stdout.splitlines():
    if "reward rows G=32" in line or "training record for every" in line \
       or "how many of the 4 reward rows" in line:
        print("  " + line.strip())
i = out.stdout.find("how many of the 4 reward rows")
print(out.stdout[i:i + 400] if i >= 0 else "the reward check did not run")
print("rc =", out.returncode)
