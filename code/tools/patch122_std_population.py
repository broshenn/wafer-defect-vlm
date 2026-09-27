"""My new item shrank its own population from eight runs to six, and printed the six.

    _cells = [(l, "GSPO " + l.split(" ", 1)[1]) for l in _STD_ORDER]

The key was built by prefixing every label with "GSPO". For `GRPO G=4 lr1e-5` that
yields `GSPO G=4 lr1e-5` -- which is also the key of the *GSPO* G=4 lr1e-5 row further
down the list. Two runs, one key, and the second assignment silently replaced the first,
so both GRPO runs were absent from the population the minimum was taken over. The item
then printed 「lowest of the 6 RL groups」 beside a sentence about eight.

Every value in the block was still correct -- the minima it computed are the minima of
the six, and the six contain the two cells the sentence names -- so both verdicts came
out right and nothing went red. That is the defect this document catalogues, committed
by the item written to prevent it, one hour after the same class was fixed in the prose:
**a correct numeral over an invisible population.** The sizes agreed because the truth
happened to survive the loss.

Two things are fixed, and the second matters more than the first:

  * the key is the label itself. The table's labels and the sentence's cell names already
    share one format (`GSPO G=32 lr5e-5`), so the prefix was never needed -- it was a
    no-op for seven rows and a collision for two, which is why it read as harmless.
  * the population's size is now an item. A reader notices 「6」 only if it looks, and
    every value beside it was right; an item is what makes the loss loud. This is the
    same reason the pair-coverage item exists for `matched_pairs`: when a set is
    assembled from another structure, the size of what was assembled is a claim.

The general shape, worth one line because it has now happened twice in two hours: the
guard for a defect has to be *run against the state that produces the defect*, and a
generator that is green on a correct document is not evidence about a wrong one. Here the
loss was invisible because it did not change the answer -- which is also the only reason
it was found at all, since the printed size was the one thing that did change.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
CB = "/tmp/check_quantified_claims.py.bak-patch122"

OLD = '''    _cells = [(l, "GSPO " + l.split(" ", 1)[1]) for l in _STD_ORDER]'''
NEW = '''    # The key is the label itself. Prefixing every label with "GSPO" was a no-op for
    # seven of the eight rows and a collision for the two GRPO ones -- whose keys it made
    # equal to their namesakes at G=4 in GSPO, so the second assignment replaced the
    # first and both GRPO runs left the population without changing either verdict.
    _cells = [(l, l) for l in _STD_ORDER]'''

SIZE_OLD = '''        _rw = sorted(_series[next(iter(_series))])'''
SIZE_NEW = '''        cmp("5.2.4: the number of RL groups the std comparison runs over",
            len(_series), len(_STD_ORDER),
            why="the population is assembled from another structure, so its size is a "
                "claim: the key collision that dropped both GRPO runs left every value "
                "in this block correct and both verdicts right -- only the size gave it "
                "away, and only because it was printed")
        _rw = sorted(_series[next(iter(_series))])'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, CB)
for old, new in ((OLD, NEW), (SIZE_OLD, SIZE_NEW)):
    if c.count(old) != 1:
        sys.exit(f"anchor {old[:50]!r} matches {c.count(old)} times; nothing written")
    c = c.replace(old, new, 1)
CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(CB, CHECKER)
    sys.exit(f"does not compile; restored:\n{r.stderr}")
print("the population keeps all eight runs, and its size is now an item.")

r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = (r.stdout or "").strip().splitlines()
print("\n--- the std items ---")
for ln in out:
    s = ln.strip()
    if ("std superlative" in s or "lowest std" in s or "holds it" in s
            or "RL groups in" in s or "reward population" in s
            or "std comparison runs over" in s):
        print("  " + s[:150])
n_stale = sum(1 for ln in out if ln.strip().startswith("STALE"))
for i, ln in enumerate(out):
    if ln.strip().startswith("STALE"):
        for nx in out[i:i + 3]:
            print("  " + nx.rstrip())
print("--- tail ---")
for ln in out[-3:]:
    print("  " + ln.strip())
print(f"\nchecker exit {r.returncode}; {n_stale} stale item(s)")
