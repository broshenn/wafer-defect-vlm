"""Make every bootstrap interval depend only on the two runs it compares.

WHY. The file drew its resamples from one shared Random consumed in file order.
That makes a contrast's interval a function of everything computed before it, so
adding a run anywhere above silently moves unrelated intervals. It has now done
that twice in this project:

  * patch66 added a contrast and moved group_size_paired's lower bound, caught only
    because LIMITATIONS.md quotes that number;
  * run 43 landing today moved it again, from -0.027778 to -0.031746 -- a shift of
    exactly one row (1/252) in the 2.5% quantile -- and also moved the lower bound
    of the GRPO lr-effect interval, which is a contrast run 43 is not part of.

Two more runs (queue 41, queue 42) are still coming, so without this the same thing
happens again twice more, each time to numbers already written into the documents.

THE FIX. Each call site constructs its own Random(SEED). An interval is then a
function of (a, b, SEED) alone. All six call sites share the same seed, which also
makes the resampling a common-random-numbers design across contrasts -- the draws
are identical, which is a fixed property of the design rather than a dependency
between blocks.

This moves some intervals once, by at most one row. Every number that moves is
reported afterwards against the documents, not left to be discovered.
"""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/paired_significance.py")
s = P.read_text(encoding="utf-8")

if "There is deliberately no shared bootstrap stream" in s:
    print("already applied; nothing to do")
    sys.exit(0)

# ---------------------------------------------------------------- 1. the call sites
OLD_CALL, NEW_CALL = ", rng)", ", random.Random(SEED))"
n = s.count(OLD_CALL)
if n != 6:
    sys.exit(f"expected 6 shared-stream call sites, found {n}; refusing to guess "
             f"which is which")
s = s.replace(OLD_CALL, NEW_CALL)

# ------------------------------------------------------- 2. remove the shared stream
OLD_DEF = 'rng = random.Random(SEED)\nsft = correct["SFT"]'
NEW_DEF = '''# There is deliberately no shared bootstrap stream. Every contrast constructs its
# own Random(SEED) where it is used, so an interval depends only on the two runs it
# compares -- never on which other runs exist, nor on the order of the blocks above.
# With one shared stream, adding a run silently moved the 2.5% quantile of unrelated
# contrasts by a whole row (1/252). That happened twice here, the second time to a
# number LIMITATIONS.md already quotes.
sft = correct["SFT"]'''
if s.count(OLD_DEF) != 1:
    sys.exit(f"shared-stream definition appears {s.count(OLD_DEF)} times, expected 1")
s = s.replace(OLD_DEF, NEW_DEF)

# ------------------------------------------------------------------- 3. the stale note
OLD_NOTE = '''print("  NOTE: any NEW contrast added to this file must use its own Random(SEED),")
print("  not the shared `rng` -- the shared stream is consumed in file order, so")
print("  drawing from it here would shift every CI below and change values that")
print("  are already recorded and quoted in LIMITATIONS.md.")'''
NEW_NOTE = '''print("  NOTE: every contrast here draws from its own Random(SEED), so no interval")
print("  depends on which other runs exist or on the order of the blocks. Keep it")
print("  that way: one shared stream makes each new contrast silently move the")
print("  intervals of the ones below, including values already quoted in")
print("  LIMITATIONS.md.")'''
if s.count(OLD_NOTE) != 1:
    sys.exit(f"shared-stream note appears {s.count(OLD_NOTE)} times, expected 1")
s = s.replace(OLD_NOTE, NEW_NOTE)

ast.parse(s)
P.write_text(s, encoding="utf-8")
print(f"paired_significance.py: {n} call sites now build their own Random(SEED); "
      f"shared stream removed")
