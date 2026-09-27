"""Stop the added contrast from perturbing every bootstrap CI computed after it.

The tool draws all of its bootstrap resamples from one shared `random.Random(SEED)`
stream, so inserting a contrast anywhere but the end shifts the stream for every
contrast below it. That silently changed group_size_paired's CI from
[-0.0278, +0.0833] to [-0.0317, +0.0833] -- a number the document already quotes.

A separate stream for the new block restores the old values exactly and makes the
block's position irrelevant, so the hazard does not return the next time a contrast
is added.
"""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/paired_significance.py")
s = P.read_text(encoding="utf-8")

OLD = """for a, b in (("GRPO_G4_lr5e5", "GSPO_G32_lr5e5"),):
    va, vb = correct[a], correct[b]
    d = acc_here[b] - acc_here[a]
    lo, hi = boot_ci(va, vb, rng)
"""

NEW = """for a, b in (("GRPO_G4_lr5e5", "GSPO_G32_lr5e5"),):
    va, vb = correct[a], correct[b]
    d = acc_here[b] - acc_here[a]
    # A private stream, deliberately: the shared `rng` is consumed in file order,
    # so drawing from it here would shift every contrast below and silently change
    # values already written into the record and quoted in the document. With its
    # own stream this block is order-independent and the existing numbers are
    # reproduced exactly.
    lo, hi = boot_ci(va, vb, random.Random(SEED))
"""

if s.count(OLD) != 1:
    sys.exit(f"anchor appears {s.count(OLD)} times, expected 1")
s = s.replace(OLD, NEW)

# The hazard also applies to the group-size contrast already in the file: it draws
# from the shared stream. Leave it as it is (changing it would move its own CI, the
# value the document quotes), but record the constraint where the next editor looks.
OLD2 = """print()
print("the paper's own setting against the baseline group size, at matched lr.")
"""
NEW2 = """print()
print("the paper's own setting against the baseline group size, at matched lr.")
print("  NOTE: any NEW contrast added to this file must use its own Random(SEED),")
print("  not the shared `rng` -- the shared stream is consumed in file order, so")
print("  drawing from it here would shift every CI below and change values that")
print("  are already recorded and quoted in LIMITATIONS.md.")
"""
if s.count(OLD2) != 1:
    sys.exit(f"print anchor appears {s.count(OLD2)} times, expected 1")
s = s.replace(OLD2, NEW2)

ast.parse(s)
P.write_text(s, encoding="utf-8")
print("paired_significance.py: added contrast now uses a private RNG stream")
