"""Three draws are a range; the rule for it was written before the third number existed.

`tools/seed_variance.py` says the limit of run 42 in its own docstring: two draws give
one difference and no confidence interval on the spread, so what it reports is a lower
bound on seed variation. Queue 43 adds a third draw of the same configuration. That
turns one difference into a range -- still not a variance estimate, but the smallest
set that can show whether a third draw lands *inside* the pair or outside it, which a
difference structurally cannot show.

Two things this patch is careful about:

  * THE RULE IS FIXED BEFORE THE NUMBER EXISTS. The threshold for the range is the
    same pre-registered one -- HALF of the within-GRPO learning-rate effect -- applied
    to the widest pair rather than to the one pair. It is written here at 17:30, while
    the third draw is a process waiting to start training, and the record marks it
    `declared_before_the_extra_draw`. A threshold chosen after seeing the spread would
    be worth nothing, which is what the old comment about "not a knob to turn" means.

  * THE TWO-DRAW READING IS NOT REWRITTEN. `ratio_seed_to_lr_effect` and
    `rule_triggered` keep their meaning and their values: patch94 quotes them into
    5.2.7, and a record that changed what a written sentence refers to would leave the
    document describing a computation that no longer exists -- the defect section 8
    records over and over. The spread is a second statement carried beside them, and
    the record says whether the two agree.

A draw counts only when both halves of it are on disk: the predictions file and the
report the tool checks its parsing against. Half a draw is not a draw. When the third
draw's files are absent -- which is the case for every run of this tool until queue
43's eval finishes -- the block reports it as pending and every number above is
unchanged, so run 42's landing (which calls this tool in `--pre`) is unaffected.

Verified before it is used: 5.2.7 and 5.2.8 in LIMITATIONS.md are written from this
record, and a record whose `seed_spread` block miscounts draws would put a wrong range
in the document with every count in it still correct.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
TOOL = ROOT / "tools/seed_variance.py"

# --------------------------------------------------------------- 1. the docstring
DOC_OLD = '''Exits nonzero, writing nothing, if the seed-2 record is absent, so that a missing
run cannot be mistaken for a small difference.
"""'''

DOC_NEW = '''Exits nonzero, writing nothing, if the seed-2 record is absent, so that a missing
run cannot be mistaken for a small difference.

A THIRD DRAW (queue 43) extends the difference above into a range. The rule for the
range is the same pre-registered one applied to the widest pair, and it was written
here before the third draw's number existed. Three draws still do not give a variance
estimate: a range is the distance between two order statistics of the draws that
exist, so it can only grow as draws are added and remains a lower bound on the spread
the configuration actually has. It is more than a difference (which cannot show
whether a third draw lands inside the pair or outside it) and less than a variance.
When the third run's predictions are absent, the block that reports it says so and
every two-seed number below is unchanged. The two-draw fields keep their meaning --
5.2.7 quotes them -- so the spread is carried beside them, not in place of them.
"""'''

# ------------------------------------------------------------------ 2. the draws
SEEDS_OLD = '''SEED1 = ("qwen35_9b_grpo", "GRPO_G4_lr1e5_seed1")
SEED2 = ("qwen35_9b_grpo_lr1e5_seed3408", "GRPO_G4_lr1e5_seed3408")'''

SEEDS_NEW = '''SEED1 = ("qwen35_9b_grpo", "GRPO_G4_lr1e5_seed1")
SEED2 = ("qwen35_9b_grpo_lr1e5_seed3408", "GRPO_G4_lr1e5_seed3408")
# The third draw of the same configuration (queue 43). Optional by construction: a
# draw whose two files are not both on disk is reported as pending and left out of
# every number, so this tool produces the two-seed record it was written to produce
# until there is a third draw to fold in.
SEED3 = ("qwen35_9b_grpo_lr1e5_seed3409", "GRPO_G4_lr1e5_seed3409")
EXTRA = [SEED3]'''

# ------------------------------------------------------- 3. load the extra draws
LOAD_OLD = '''v1, r1 = load(SEED1[0])
v2, r2 = load(SEED2[0])
vs, rs = load(SFT_TAG)
v5, r5 = load(LR5E5_TAG)

# Reproduction check, as in paired_significance.py: if this script's reading of
# the responses does not reproduce the project's own scorer, its numbers are
# about misparsed text and must not be published.
for lbl, vec, rep in ((SEED1[1], v1, r1), (SEED2[1], v2, r2), ("SFT", vs, rs),
                      ("GRPO_lr5e5", v5, r5)):'''

LOAD_NEW = '''v1, r1 = load(SEED1[0])
v2, r2 = load(SEED2[0])
vs, rs = load(SFT_TAG)
v5, r5 = load(LR5E5_TAG)


# A draw counts only when both halves of it are on disk: the predictions, and the
# report this script reproduces its parsing against. A predictions file without its
# report cannot be checked, and a draw that cannot be checked must not enter a range.
def ready(tag):
    return ((BASE / (tag + ".jsonl")).is_file()
            and (REPORTS / (tag + "__report.json")).is_file())


EXTRA_PENDING = [lbl for tag, lbl in EXTRA if not ready(tag)]
EXTRA_LOADED = []           # (tag, label, correct-vector, report)
for _tag, _lbl in EXTRA:
    if ready(_tag):
        _v, _r = load(_tag)
        EXTRA_LOADED.append((_tag, _lbl, _v, _r))
if EXTRA_PENDING:
    print(f"NOT YET: {', '.join(EXTRA_PENDING)} declared but not evaluated; the "
          f"spread below covers only the draws that exist")

# Reproduction check, as in paired_significance.py: if this script's reading of
# the responses does not reproduce the project's own scorer, its numbers are
# about misparsed text and must not be published.
_checks = [(SEED1[1], v1, r1), (SEED2[1], v2, r2), ("SFT", vs, rs),
           ("GRPO_lr5e5", v5, r5)]
_checks += [(lbl, v, r) for _, lbl, v, r in EXTRA_LOADED]
for lbl, vec, rep in _checks:'''

# -------------------------------------------- 4. the spread and the extra draws
SPREAD_ANCHOR = '''# The two facts the replication sentence is computed from.
_same_side = (acc1 - accs >= 0) == (d_vs_sft_2 >= 0)
_sig = bool(pval(ps1) < 0.05 or pval(ps) < 0.05)'''

SPREAD_BLOCK = '''# ------------------------------------------------------------- the spread itself
# Two draws give one difference; three give a range. The rule for the range is the
# same pre-registered rule applied to the widest pair instead of to the one pair: the
# spread is not separable from seed noise if it reaches HALF of |lr effect|. Written
# before the third draw's number existed -- the third draw is a process that has not
# started training -- because a threshold chosen after seeing the value decides
# nothing. `ratio` above is NOT recomputed from the range: it is what the record was
# built with and what 5.2.7 quotes.
_draws = [(SEED1[1], acc1), (SEED2[1], acc2)]
_draws += [(lbl, sum(v) / len(v)) for _, lbl, v, _r in EXTRA_LOADED]
_best = max(_draws, key=lambda kv: kv[1])
_worst = min(_draws, key=lambda kv: kv[1])
spread = _best[1] - _worst[1]
half_lr = HALF * abs(d_lr)
trig2 = bool(ratio >= HALF)
spread_trig = bool(spread >= half_lr)
print("draws: " + ", ".join(f"{l} {a:.4f}" for l, a in _draws))
print(f"spread over {len(_draws)} draws: {spread:.4f} "
      f"({_worst[0]} {_worst[1]:.4f} .. {_best[0]} {_best[1]:.4f}); half the lr "
      f"effect = {half_lr:.4f} -> "
      f"{'AT OR ABOVE the threshold' if spread_trig else 'below the threshold'}")
if trig2 != spread_trig:
    print("  NOTE: the one-difference rule and the range rule do not agree here; the "
          "record says which one triggered, and the extra draw is what moved it")

# Per extra draw: the same paired apparatus seed 2 got, against seed 1 and against
# SFT, plus the same marginal metrics the two seeds are compared on.
_extra_rows = {}
for _tag, _lbl, _v, _rp in EXTRA_LOADED:
    _a = sum(_v) / len(_v)
    _n01, _n10, _p = mcnemar_exact(v1, _v)
    _lo, _hi = boot_ci(v1, _v)
    _so1, _so10, _ps = mcnemar_exact(vs, _v)
    _extra_rows[_lbl] = {
        "accuracy": round(_a, 6),
        "delta_vs_seed1": round(_a - acc1, 6),
        "ci95_rows": [round(_lo, 6), round(_hi, 6)],
        "excludes_zero": bool(_lo > 0 or _hi < 0),
        "mcnemar_vs_seed1": {"seed1_wrong_draw_right": _n01,
                             "seed1_right_draw_wrong": _n10,
                             "p_exact_two_sided": pval(_p)},
        "vs_sft": {"delta": round(_a - accs, 6), "p_exact_two_sided": pval(_ps)},
        "structured_and_caption": {lbl: round(metric(_rp, *path), 6) for lbl, path in KEYS},
    }
    print(f"  {_lbl}: acc {_a:.4f} (vs seed1 {_a - acc1:+.4f}, "
          f"95% CI [{_lo:+.4f},{_hi:+.4f}], McNemar p={_p:.5f}); "
          f"vs SFT {_a - accs:+.4f} (p={_ps:.5f})")

# The reading, computed from the values rather than written to cover every case (the
# failure section 8 item 14 records: a sentence true however the run came out carries
# nothing). The interesting case is the one where the extra draw changes the verdict,
# because that is the case in which a claim already written in 5.2.2 has to be re-read.
if len(_draws) == 2:
    _range_reading = (
        "two draws give a range equal to their own difference, so this block reports "
        "the same number as the pair above until a third draw exists")
    _range_caveat = (
        "two draws give a difference, not a spread and not a variance estimate: the "
        "range above is the difference itself, and it cannot show whether a third "
        "draw would land inside it or outside")
else:
    _range_reading = (
        f"over {len(_draws)} draws the range is {spread:.4f} "
        f"({_worst[0]} {_worst[1]:.4f} .. {_best[0]} {_best[1]:.4f}), "
        + ("at or above" if spread_trig else "below")
        + f" half the learning-rate effect ({half_lr:.4f})"
        + (", the same verdict the one-difference rule reached"
           if trig2 == spread_trig else
           f", which is NOT the verdict the one-difference rule reached "
           f"(|{d_seed:+.4f}| / |{d_lr:+.4f}| = {ratio:.3f}) -- the extra draw is what "
           f"moved the reading, and 5.2.2 has to be re-read against it"))
    _range_caveat = (
        f"{len(_draws)} draws give a range, not a variance estimate: a range is the "
        f"distance between the extreme order statistics of the draws that exist, so it "
        f"can only grow as draws are added and remains a lower bound on the spread the "
        f"configuration actually has. It is more than a difference and less than a "
        f"variance, and {len(_draws)} draws are still few")

# The two facts the replication sentence is computed from.
_same_side = (acc1 - accs >= 0) == (d_vs_sft_2 >= 0)
_sig = bool(pval(ps1) < 0.05 or pval(ps) < 0.05)'''

# ------------------------------------------------------------ 5. the out record
OUT_OLD = '''    "ratio_seed_to_lr_effect": round(ratio, 6),
    "rule_triggered": bool(ratio >= HALF),'''

OUT_NEW = '''    "ratio_seed_to_lr_effect": round(ratio, 6),
    "rule_triggered": bool(ratio >= HALF),
    # The same pre-registered rule applied to the widest pair over every draw that
    # exists. Carried beside the one-difference fields rather than replacing them:
    # 5.2.7 quotes those, and a record that changed what a written sentence refers to
    # would leave the document describing a computation that no longer exists.
    # `declared_before_the_extra_draw` is the point of the block -- the threshold is
    # only worth something if it predates the number it judges.
    "seed_spread": {
        "n_draws": len(_draws),
        "draws": {lbl: round(a, 6) for lbl, a in _draws},
        "range": round(spread, 6),
        "widest_pair": [_worst[0], _best[0]],
        "threshold_half_lr_effect": round(half_lr, 6),
        "rule_triggered": spread_trig,
        "rule_triggered_by_the_pair": trig2,
        "rules_agree": bool(trig2 == spread_trig),
        "declared_before_the_extra_draw": True,
        "reading": _range_reading,
        "what_a_range_cannot_give": _range_caveat,
    },
    "extra_seeds": {
        "present": [lbl for _, lbl, _v, _r in EXTRA_LOADED],
        "declared_but_not_evaluated": EXTRA_PENDING,
        "per_draw": _extra_rows,
        "note": "a draw enters this block only when both its predictions and the "
                "report they are parsed against are on disk; half a draw is not a "
                "draw and cannot be checked",
    },'''

EDITS = [
    ("the docstring", DOC_OLD, DOC_NEW),
    ("the seed tags", SEEDS_OLD, SEEDS_NEW),
    ("the load / parse-check block", LOAD_OLD, LOAD_NEW),
    ("the spread block", SPREAD_ANCHOR, SPREAD_BLOCK),
    ("the record", OUT_OLD, OUT_NEW),
]

s = TOOL.read_text(encoding="utf-8")
shutil.copy2(TOOL, "/tmp/seed_variance.py.bak-patch135")
for what, old, new in EDITS:
    if s.count(old) != 1:
        sys.exit(f"{what}: the anchor matches {s.count(old)} times, expected 1; "
                 f"nothing written")
    s = s.replace(old, new, 1)
TOOL.write_text(s, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(TOOL)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/seed_variance.py.bak-patch135", TOOL)
    sys.exit(f"the patched seed_variance.py does not compile, restored:\n{r.stderr}")
print("seed_variance.py: the spread over every draw that exists is computed beside the "
      "one-difference reading, and a missing draw is reported as pending")

# ------------------------------------------------------------- the run, right now
r = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True,
                   cwd=str(ROOT))
print(f"\n--- seed_variance.py: exit {r.returncode} ---")
for ln in ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-14:]:
    print("  " + ln[:170])
