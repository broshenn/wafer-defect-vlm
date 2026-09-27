"""Teach the paired tool about the IS-level pair and stop it claiming a missing cell
that has arrived.

Run 43 supplies GRPO(G=4) vs GSPO(G=4) at lr 5e-5 — a pair differing only in
--importance_sampling_level. Two consequences:

  * the contrast is computed and recorded, since it is the first one in this project
    whose explanation cannot be group size or effective batch;
  * the tool's standing argument that "no such pair exists" becomes false the moment
    the report lands. Left as-is it would keep asserting it, which is exactly the
    stale-claim class this project keeps finding.

The new block uses its own Random(SEED): the shared stream is consumed in file order,
so drawing from it here would move every CI below.
"""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/paired_significance.py")
s = P.read_text(encoding="utf-8")

# ---------------------------------------------------------------- 1. run registry
OPT_OLD = '''OPTIONAL = [("GSPO_G32_lr1e5", "qwen35_9b_gspo_g32_lr1e5")]'''
OPT_NEW = '''OPTIONAL = [("GSPO_G32_lr1e5", "qwen35_9b_gspo_g32_lr1e5"),
            ("GSPO_G4_lr5e5", "qwen35_9b_gspo_g4_lr5e5")]'''
if s.count(OPT_OLD) != 1:
    sys.exit(f"OPTIONAL anchor appears {s.count(OPT_OLD)} times")
s = s.replace(OPT_OLD, OPT_NEW)

# ------------------------------------------------- 2. the single-variable contrast
ANCHOR = """print()
print("group size at matched algorithm and lr (GSPO, lr 5e-5): G=8 -> G=32.")
"""

CONTRAST = '''
# ------------------------------------------------- the single-variable IS contrast
# Run 43 exists to be the pair this tool has been asking for. Both runs are G=4 with
# accumulation 4 (effective batch 16), lr 5e-5, same seed, steps, data and rewards;
# only --importance_sampling_level differs (token vs sequence). Verified against the
# reference run's own config by scripts/43_gspo_g4_lr5e5.sh step 3, which exits if
# anything else differs.
if "GSPO_G4_lr5e5" in correct:
    _a, _b = "GRPO_G4_lr5e5", "GSPO_G4_lr5e5"
    _va, _vb = correct[_a], correct[_b]
    _d = acc_here[_b] - acc_here[_a]
    _lo, _hi = boot_ci(_va, _vb, random.Random(SEED))
    _n01, _n10, _p = mcnemar_exact(_va, _vb)
    out["is_level_paired"] = {
        "from": _a, "to": _b,
        "accuracy_delta_sequence_minus_token": round(_d, 6),
        "delta_ci95_rows": [round(_lo, 6), round(_hi, 6)],
        "excludes_zero": bool(_lo > 0 or _hi < 0),
        "mcnemar": {"token_wrong_sequence_right": _n01,
                    "token_right_sequence_wrong": _n10,
                    "p_exact_two_sided": pval(_p)},
        "single_variable": True,
        "held_fixed": "group size 4, accumulation 4 (effective batch 16), lr 5e-5, "
                      "seed 3407, 150 steps, dataset, reward functions and weights",
        "only_variable": "--importance_sampling_level (token vs sequence)",
        "reading": ("the two IS levels are not separable at G=4 / lr 5e-5"
                    if not (_lo > 0 or _hi < 0) else
                    "the two IS levels differ at G=4 / lr 5e-5, in favour of "
                    + ("sequence" if _d > 0 else "token")),
        "does_not_settle": "the proposition is about how the lr EFFECT differs by IS "
                           "level. One lr cannot show a difference in tolerance; it "
                           "shows only the levels' relative position at that lr.",
    }
    print()
    print("IS level with everything else held fixed (the first such pair here):")
    print("  GRPO(G=4, token) vs GSPO(G=4, sequence), both lr 5e-5, accum 4.")
    print(f"  dacc(seq - tok)={_d:+.4f} CI=[{_lo:+.4f},{_hi:+.4f}] "
          f"n01={_n01} n10={_n10} p={_p:.5f}")

'''

if s.count(ANCHOR) != 1:
    sys.exit(f"group-size anchor appears {s.count(ANCHOR)} times, expected 1")
s = s.replace(ANCHOR, CONTRAST + ANCHOR)

# ------------------------------------------- 3. the missing-cell text, conditional
OLD_TAIL = '''print("WHAT WOULD TEST THE PAPER'S PROPOSITION, and is not in this record:")
print("  a pair differing ONLY in IS level. Every token-level run here is G=4")
print("  and every sequence-level run is G=8 or G=32, so IS level never moves")
print("  alone. The cheapest missing cell is GSPO at G=4 (i.e. GRPO with")
print("  --importance_sampling_level sequence at the same group size as the")
print("  token-level run), which is a ~40 minute run at G=4. Until it exists,")
print("  the proposition is untested rather than refuted.")
out["paper_claim_status"] = {
    "proposition": "sequence-level normalisation tolerates a higher learning "
                   "rate than token-level",
    "status": "untested in this record",
    "why": "no pair differs only in IS level; token-level is always G=4 and "
           "sequence-level always G=8 or G=32",
    "missing_cell": "GSPO (IS=sequence) at G=4, matching the token-level GRPO "
                    "run's group size; approx. 40 minutes of training",
    "what_is_established": "both IS levels degrade at lr 5e-5 (GRPO p=0.04356, "
                           "GSPO G=8 p=0.000324), so sequence-level does not "
                           "visibly tolerate the higher lr better than "
                           "token-level does at the group sizes measured"}'''

NEW_TAIL = '''# Whether the proposition has a test is a fact about this record, not a standing
# argument. Run 43 supplies the pair differing only in IS level, so the old
# unconditional text would keep asserting a missing cell that has arrived -- the
# stale-claim class this project keeps finding. Both branches are recorded so the
# artefact states which world it is in.
_isl = out.get("is_level_paired")
if _isl:
    print("THE PAPER'S PROPOSITION HAS ITS SINGLE-VARIABLE IS PAIR:")
    print("  GRPO(G=4, token) vs GSPO(G=4, sequence) at lr 5e-5 -- group size,")
    print("  effective batch, seed, steps, data and rewards all held fixed.")
    print("  " + _isl["reading"])
    out["paper_claim_status"] = {
        "proposition": "sequence-level normalisation tolerates a higher learning "
                       "rate than token-level",
        "status": "the single-variable IS-level pair is in this record",
        "single_variable_pair": _isl["from"] + " vs " + _isl["to"],
        "result": _isl["reading"],
        "what_this_pair_establishes": "at G=4 and lr 5e-5 the two IS levels are "
                                      "compared with group size and effective "
                                      "batch fixed, so the confound that limited "
                                      "every earlier contrast is absent here",
        "missing_cell": "tolerance is a statement about the SIZE of the lr effect "
                        "at each IS level, which needs the same 2x2 at G=4. This "
                        "record has token at both lrs but sequence at G=4 only at "
                        "lr 5e-5, so the cell still missing is GSPO (IS=sequence) "
                        "at G=4 with lr 1e-5",
        "what_is_established": "both IS levels degrade at lr 5e-5 (GRPO p=0.04356, "
                               "GSPO G=8 p=0.000324), so sequence-level does not "
                               "visibly tolerate the higher lr better than "
                               "token-level does at the group sizes measured"}
else:
    print("WHAT WOULD TEST THE PAPER'S PROPOSITION, and is not in this record:")
    print("  a pair differing ONLY in IS level. Every token-level run here is G=4")
    print("  and every sequence-level run is G=8 or G=32, so IS level never moves")
    print("  alone. The cheapest missing cell is GSPO at G=4 (i.e. GRPO with")
    print("  --importance_sampling_level sequence at the same group size as the")
    print("  token-level run), which is a ~40 minute run at G=4. Until it exists,")
    print("  the proposition is untested rather than refuted.")
    out["paper_claim_status"] = {
        "proposition": "sequence-level normalisation tolerates a higher learning "
                       "rate than token-level",
        "status": "untested in this record",
        "why": "no pair differs only in IS level; token-level is always G=4 and "
               "sequence-level always G=8 or G=32",
        "missing_cell": "GSPO (IS=sequence) at G=4, matching the token-level GRPO "
                        "run's group size; approx. 40 minutes of training",
        "what_is_established": "both IS levels degrade at lr 5e-5 (GRPO "
                               "p=0.04356, GSPO G=8 p=0.000324), so "
                               "sequence-level does not visibly tolerate the "
                               "higher lr better than token-level does at the "
                               "group sizes measured"}'''

if s.count(OLD_TAIL) != 1:
    sys.exit(f"tail anchor appears {s.count(OLD_TAIL)} times, expected 1")
s = s.replace(OLD_TAIL, NEW_TAIL)

ast.parse(s)
P.write_text(s, encoding="utf-8")
print("paired_significance.py: IS-level contrast + conditional claim status")
