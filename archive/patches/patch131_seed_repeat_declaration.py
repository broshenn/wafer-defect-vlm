"""A cell can hold two runs, and every reader of the record was resolving cells by guess.

Run 42 is the second seed of GRPO G=4 lr 1e-5. Declaring it in the confound record
(patch130) made the record complete -- and immediately turned two of the checker's facts
red, in the direction that looks like the document is wrong:

    5.2.4: GRPO(G=4)'s KL at the lower and higher learning rate   [fact]
      document says: [1.1068, 1.0435]
      records say  : [1.1244, 1.0435]

The document is right. 1.1068 is the first seed's KL, and 5.2.4's whole argument is the
*paired* learning-rate contrast -- whose low-rate member is that first seed, named
explicitly by the record's own `matched_pairs` entry ("from": "grpo"). The checker
resolves a document label like `GRPO(G=4) lr1e-5` into a run by building
`{(G, lr, is_level): (tag, entry)}` from `runs`, and with two runs sharing that key the
dict keeps the last one -- the seed. The fact that came back red was the pair-versus-run
consistency item, which is the record's own cross-check: the pair says 1.1068, the cell
lookup said 1.1244, and the item correctly reported that the two halves of the record
disagree. Nothing was wrong with the record; the lookup had no way to know which of two
runs owns the cell.

The fix is a declaration rather than a tie-break rule, for the reason this project keeps
rediscovering: "the last one wins" is a fact about a dict, and every tie-break invented
here (lowest tag, first in RUNS, fewest idle steps) would be a guess that reads correct
until a third seed lands. So the record says which run each repeat repeats, in a
`SEED_REPEATS` map beside the `RUNS` list that already owns the run set's membership,
and the record carries `repeat_of` per run. The checker drops declared repeats from its
cell map and asserts two properties of the declaration itself: that it names a run the
record holds, and that the run it names really is in the same cell -- otherwise the word
"repeat" would be a way to remove a run from every population below.

Order matters here and is the reason this is one patch: the record has to carry
`repeat_of` before the checker can read it, and the landing's finish pass runs both.
"""
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
TOOL = ROOT / "tools/kl_length_confound.py"
CHECK = ROOT / "tools/check_quantified_claims.py"
REC = ROOT / "outputs/reports/kl_length_confound.json"

# ------------------------------------------------- 1. the record declares its repeats
T_OLD = '''        "qwen35_9b_grpo_lr1e5_seed3408"]'''
T_NEW = '''        "qwen35_9b_grpo_lr1e5_seed3408"]

# A configuration can be re-run with another seed. A repeat is not a second learning
# rate: the cell it belongs to already has an owner, and `matched_pairs` compares the two
# runs that define the cell's learning-rate contrast. Declaring which run a repeat
# repeats is what lets a reader that resolves `runs` by (algorithm, group size, learning
# rate) know which entry owns the cell -- without it, that lookup returns whichever entry
# the dict ended with, and after run 42 landed that was the seed: 5.2.4's GRPO(G=4) KL
# read 1.1244 instead of 1.1068, and the checker reported the *document* stale while the
# document was quoting the run its own pair is built from.
SEED_REPEATS = {"qwen35_9b_grpo_lr1e5_seed3408": "grpo"}'''

E_OLD = '''    out["runs"][tag] = {
        "G": cfg.get("num_generations"),'''
E_NEW = '''    out["runs"][tag] = {
        "G": cfg.get("num_generations"),
        # None for every run that owns its cell. `check_quantified_claims` reads this to
        # keep a repeat out of the cell map it builds from this record; the repeat's own
        # numbers stay in `runs`, and are counted by every sentence that counts runs.
        "repeat_of": SEED_REPEATS.get(tag),'''

s = TOOL.read_text(encoding="utf-8")
shutil.copy2(TOOL, "/tmp/kl_length_confound.py.bak-patch131")
for i, (old, new) in enumerate(((T_OLD, T_NEW), (E_OLD, E_NEW)), 1):
    if s.count(old) != 1:
        sys.exit(f"edit {i}: the anchor matches {s.count(old)} times; nothing written")
    s = s.replace(old, new, 1)
TOOL.write_text(s, encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(TOOL)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/kl_length_confound.py.bak-patch131", TOOL)
    sys.exit(f"kl_length_confound.py does not compile; restored:\n{r.stderr}")
print("kl_length_confound.py: repeats are declared (SEED_REPEATS -> repeat_of).")

# -------------------------------------------- 2. the checker uses the declaration
C_OLD = '''    _by_ident = {(e["G"], round(float(e["lr"]), 12), e["is_level"]): (t, e)
                 for t, e in _kruns.items()}'''
C_NEW = '''    # A cell can hold two runs -- the second seed of a configuration is a re-run, not a
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
            "without any sentence changing")'''

c = CHECK.read_text(encoding="utf-8")
shutil.copy2(CHECK, "/tmp/check_quantified_claims.py.bak-patch131")
if c.count(C_OLD) != 1:
    sys.exit(f"the checker anchor matches {c.count(C_OLD)} times; nothing written "
             f"(the record edit above stands)")
CHECK.write_text(c.replace(C_OLD, C_NEW, 1), encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECK)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch131", CHECK)
    sys.exit(f"the checker does not compile; restored:\n{r.stderr}")
print("check_quantified_claims.py: cell lookup reads the declaration.")

# --------------------------------------------- 3. regenerate, then read it back
shutil.copy2(REC, "/tmp/kl_length_confound.json.bak-patch131")
r = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True, cwd=str(ROOT))
if r.returncode != 0:
    shutil.copy2("/tmp/kl_length_confound.json.bak-patch131", REC)
    sys.exit(f"kl_length_confound.py failed (rc={r.returncode}); record restored:\n"
             f"{r.stdout[-1500:]}\n{r.stderr[-800:]}")
rec = json.loads(REC.read_text(encoding="utf-8"))
rep = {t: e.get("repeat_of") for t, e in rec["runs"].items() if e.get("repeat_of")}
print(f"record: {len(rec['runs'])} runs; declared repeats: {rep}")
print(f"        cell owners: {[t for t in rec['runs'] if t not in rep]}")

# ------------------------------------------------- 4. the checker, on the real text
r = subprocess.run([sys.executable, str(CHECK)], capture_output=True, text=True, cwd=str(ROOT))
out = ((r.stdout or "") + (r.stderr or "")).strip()
lines = out.splitlines()
print(f"\n--- check_quantified_claims: exit {r.returncode} "
      f"({sum(1 for l in lines if l.strip().startswith('ok'))} ok) ---")
for ln in lines[-14:]:
    print("  " + ln[:170])
