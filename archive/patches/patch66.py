"""Add the paper's own contrast to the paired tool: GSPO(G=32) vs GRPO(G=4) at lr 5e-5.

Section 5.2.5 asserts something about this pair, and nothing in the record measured
it. The two runs' aggregate accuracies happen to be identical, which is exactly the
case where a reader would assume the paired test is unnecessary -- it is not, because
equal aggregates do not imply equal per-row predictions. The test is recorded here
rather than reasoned about in prose.

Purely additive: one new top-level key, no existing key touched.
"""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/paired_significance.py")
s = P.read_text(encoding="utf-8")

ANCHOR = """    print(f"  lr{lr:5s} GSPO(G=8) - GRPO(G=4): dacc={d:+.4f} "
          f"CI=[{lo:+.4f},{hi:+.4f}] n01={n01} n10={n10} p={p:.5f}")
"""

NEW_BLOCK = """

print()
print("the paper's own setting against the baseline group size, at matched lr.")
print("  GSPO(G=32, sequence) vs GRPO(G=4, token) at lr 5e-5. Section 5.2.5 makes")
print("  a claim about this pair; it is also the most confounded contrast in the")
print("  record -- group size, IS level and effective batch all move together.")
for a, b in (("GRPO_G4_lr5e5", "GSPO_G32_lr5e5"),):
    va, vb = correct[a], correct[b]
    d = acc_here[b] - acc_here[a]
    lo, hi = boot_ci(va, vb, rng)
    n01, n10, p = mcnemar_exact(va, vb)
    out["paper_setting_paired"] = {
        "from": a, "to": b,
        "accuracy_delta_g32_minus_grpo4": round(d, 6),
        "delta_ci95_rows": [round(lo, 6), round(hi, 6)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "mcnemar": {"grpo_wrong_gspo_right": n01, "grpo_right_gspo_wrong": n10,
                    "p_exact_two_sided": pval(p)},
        "aggregate_accuracy_equal": bool(acc_here[a] == acc_here[b]),
        "note": ("The two runs' aggregate accuracies are equal. Equality of "
                 "aggregates does not imply equality of per-row predictions, so "
                 "the paired test is not redundant here; this records it."),
        "confound": "group size (4 vs 32), IS level (token vs sequence) and "
                    "effective batch (16 vs 1024) all move together, so this "
                    "contrast identifies none of the three",
    }
    print(f"  GSPO(G=32) - GRPO(G=4) lr5e-5: dacc={d:+.4f} "
          f"CI=[{lo:+.4f},{hi:+.4f}] n01={n01} n10={n10} p={p:.5f}")
"""

if s.count(ANCHOR) != 1:
    sys.exit(f"anchor appears {s.count(ANCHOR)} times, expected 1")
s = s.replace(ANCHOR, ANCHOR + NEW_BLOCK)

ast.parse(s)
assert s.count("paper_setting_paired") == 1
P.write_text(s, encoding="utf-8")
print("paired_significance.py: paper_setting_paired contrast added")
