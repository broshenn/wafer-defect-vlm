"""A population that run 41 is about to grow, in a sentence nothing reads.

5.2.2 (lines 318-322), one claim in two halves:

    同算法同组大小的三条对照：GRPO(G=4) 36.00% → 32.67%、
    GSPO(G=8) 15.33% → 38.00%、GSPO(G=4) 33.33% → 26.67%。
    三条**同算法、同组大小**的对照里，lr 5e-5 把空转步推高的有 1 条（GSPO(G=8)）、
    压低的 2 条（GRPO(G=4)、GSPO(G=4)）—— 方向随算法与组大小改变，
    **不是学习率单方面的效应**。

The population is the set of (algorithm, group size) pairs where both learning rates have
a run. There are three today, which is why the sentence says three. Run 41 is
`GSPO(G=32) lr1e-5`: it pairs with the existing `GSPO(G=32) lr5e-5` and makes a fourth.
The sentence then names three pairs while four exist, in a sentence no item read -- and its
*conclusion* ("the direction changes with the algorithm and the group size") is drawn from
the split, which the fourth pair can move.

Three things are asserted and all three are read here: the population (a count, stated
twice), the enumeration, and the split (facts -- a pair is appended to them, and a numeral
bump alone would leave the sentence naming the old pairs, while `--fix` refuses to touch a
count once a fact has moved, which is the intended outcome here).

The population is computed from `mean_frac_reward_zero_std`, the same training-record field
the idle-step items use, so this cannot disagree with them about a rate.

The enumeration is compared as a **dict keyed by (algorithm, group size)**, not as a list.
The document's order happens to match the run set's declaration order today, but nothing
makes that true: `RUNS` declares `GSPO(G=4) lr5e-5` before `GSPO(G=4) lr1e-5`, so that pair
only completes at the last entry. A positional comparison would make the item depend on an
ordering coincidence and would fail on a reordering that changed nothing about the claim.

Two notes on how this patch is written, both from its first attempt, because the same
mistake would be easy to make again:

  * The insertion point is a **block boundary**. The first attempt inserted after an item
    inside the reward block, and the block's next statement read `g` -- the name anchor()
    binds -- which the inserted code had reassigned, so the *pre-existing* item died with
    IndexError. This block uses `gp` and goes before a fresh `anchor(` call, where nothing
    before it is still live.

  * The population's second numeral is written 「三条**同算法、同组大小**的对照里」: the
    numeral is followed by 条 and only then by the bold. A pattern that goes straight from
    the numeral to `**` cannot match it, and a pattern that fails silently -- as a fact
    reporting "absent" -- looks like a stale document rather than a broken probe. The
    absent-branch below exists so that failure is visible, and it is what caught this.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
BAK = "/tmp/check_quantified_claims.py.bak-patch109"

# Inserted BEFORE this line: the start of the 5.2.2 Scratch block, which is a clean
# boundary -- the reward block's statements that use `g` are all behind it.
ANCHOR = 'g = anchor("5.2.2: the Scratch per-class maximum", P_SCRATCH)\n'

BLOCK = r'''# --------------- 5.2.2: the same-algorithm same-group-size learning-rate pairs
# The population is every (algorithm, group size) carrying both learning rates. The
# sentence states it twice and enumerates its members; all three are read here. Nothing
# read it before, and run 41 -- GSPO(G=32) lr1e-5 -- adds a fourth pair.
P_PAIRS = r"同算法同组大小的(" + NUM + r")条对照"
gp = anchor("5.2.2: the same-algorithm same-group-size lr pairs", P_PAIRS)
if gp:
    _by = {}
    for _n in rl:
        _m = re.fullmatch(r"([A-Za-z]+)\(G=(\d+)\) (lr[\d.eE+-]+)", _n)
        if _m:
            _by.setdefault((_m.group(1), int(_m.group(2))), {})[_m.group(3)] = _n
    _pairs = {k: v for k, v in _by.items() if len(v) == 2}
    _rates, _raised, _lowered, _missing = {}, [], [], []
    for _k, _v in sorted(_pairs.items()):
        _r = {}
        for _lr, _n in _v.items():
            _p = train_record(_n)
            if not _p.is_file():
                _missing.append(_n)
                continue
            _val = json.loads(_p.read_text(encoding="utf-8")).get(
                "mean_frac_reward_zero_std")
            if isinstance(_val, (int, float)):
                _r[_lr] = round(100 * _val, 2)
        if len(_r) == 2:
            _rates[_k] = (_r["lr1e-5"], _r["lr5e-5"])
            (_raised if _r["lr5e-5"] > _r["lr1e-5"] else _lowered).append(
                f"{_k[0]}(G={_k[1]})")
    if _missing:
        cmp("5.2.2: a training record for every run in those pairs", [], _missing,
            why="a pair whose record is missing cannot be ranked; dropping it would "
                "count the population over a smaller set than the sentence names")
    _label = lambda k: f"{k[0]}(G={k[1]})"
    cmp("5.2.2: the pair count (the numeral before the enumeration)",
        cn2int(gp.group(1)), len(_pairs),
        kind="count", pattern=P_PAIRS, group=1)
    # The second statement of the same population, in the sentence that draws the
    # conclusion from it. Note the 条 after the numeral and before the bold.
    P_PAIRS_SPLIT = r"(" + NUM + r")条\*\*同算法、同组大小\*\*的对照里"
    _gs = re.search(P_PAIRS_SPLIT, text)
    if _gs:
        cmp("5.2.2: the pair count (the second numeral, in the conclusion)",
            cn2int(_gs.group(1)), len(_pairs),
            kind="count", pattern=P_PAIRS_SPLIT, group=1)
    else:
        cmp("5.2.2: the pair count in the conclusion", "the phrase", "absent",
            why="the sentence states the population twice and the conclusion is drawn "
                "over the second statement; if this phrase cannot be found then this "
                "item is reading nothing, which is not the same as the document being "
                "correct -- the phrase is 「N条**同算法、同组大小**的对照里」")
    # The enumeration, keyed by (algorithm, group size) rather than read in order.
    _seen = {}
    for _m in re.finditer(r"([A-Za-z]+)\(G=(\d+)\)\s*([\d.]+)%（\d+/\d+）\s*→\s*"
                          r"([\d.]+)%（\d+/\d+）", text):
        _seen[(_m.group(1), int(_m.group(2)))] = (float(_m.group(3)), float(_m.group(4)))
    cmp("5.2.2: the rates enumerated for each pair",
        {_label(k): v for k, v in _seen.items()},
        {_label(k): v for k, v in _rates.items()},
        why="a pair landing appends to this enumeration; the numeral alone would leave "
            "it listing the old pairs")
    _up_txt = re.search(r"lr 5e-5 把空转步推高的有 (" + NUM + r") 条（([^）]*)）", text)
    _low_txt = re.search(r"压低的 (" + NUM + r") 条（([^）]*)）", text)
    if _up_txt and _low_txt:
        _norm = lambda s: [x.strip() for x in s.split("、") if x.strip()]
        cmp("5.2.2: how many pairs lr 5e-5 raises the idle-step share in",
            cn2int(_up_txt.group(1)), len(_raised), kind="count",
            pattern=r"lr 5e-5 把空转步推高的有 (" + NUM + r") 条", group=1)
        cmp("5.2.2: which pairs lr 5e-5 raises it in, named", _norm(_up_txt.group(2)),
            sorted(_raised),
            why="the sentence names the pairs, so a landing that changes the split is a "
                "rewrite of the names and not of the numeral")
        cmp("5.2.2: how many pairs lr 5e-5 lowers the idle-step share in",
            cn2int(_low_txt.group(1)), len(_lowered), kind="count",
            pattern=r"压低的 (" + NUM + r") 条", group=1)
        cmp("5.2.2: which pairs lr 5e-5 lowers it in, named", _norm(_low_txt.group(2)),
            sorted(_lowered),
            why="as above")
        print("         lr pairs by idle-step share: "
              + "、".join(f"{_label(_k)} {_v[0]:.2f}%->{_v[1]:.2f}%"
                          for _k, _v in sorted(_rates.items())))
    else:
        cmp("5.2.2: the pairs lr 5e-5 raises or lowers the idle-step share in",
            "the split phrases", "absent",
            why="the conclusion is drawn from this split; if these phrases are rewritten "
                "nothing reads what it was drawn from")

'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, BAK)

for what, cond in (("the insertion point", c.count(ANCHOR) == 1),
                   ("the pair items", "P_PAIRS" not in c)):
    if not cond:
        sys.exit(f"{what} is not as expected; nothing written")

c = c.replace(ANCHOR, BLOCK + ANCHOR, 1)
CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(BAK, CHECKER)
    sys.exit(f"the patched checker does not compile, restored:\n{r.stderr}")

print("the same-algorithm same-group-size pairs are now read from the training records: "
      "the population (both numerals), the rates, and the split the conclusion rests on.")
print("\n--- re-running the checker ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = r.stdout.strip().splitlines()
shown = 0
for ln in out:
    s_ = ln.strip()
    if "5.2.2" in s_ and ("pair" in s_ or "Scratch" in s_) or "lr pairs by" in s_:
        print("  " + s_)
        shown += 1
if not shown:
    print("  (none of the new items printed -- the anchor did not match; the block is "
          "inert)")
if r.stderr.strip():
    print("--- stderr ---")
    for ln in r.stderr.strip().splitlines()[-10:]:
        print("  " + ln)
print("  ...")
for ln in out[-6:]:
    print("  " + ln.strip())
if r.returncode != 0:
    shutil.copy2(BAK, CHECKER)
    sys.exit(f"\nthe checker is not green (exit {r.returncode}); RESTORED -- the pairs "
             f"and the records disagree, and that is the finding")
print("\nchecker green: the pair population, its rates and its split are all read; run 41 "
      "will grow the population, and this will say so instead of staying quiet")
