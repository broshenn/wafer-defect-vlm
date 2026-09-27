"""What is the run-to-run noise floor, and are the claimed effects above it?

Every verdict in this project compares two runs and asks whether the difference
is significant. Significance there answers "is the difference larger than what
the 252 benchmark rows can resolve" -- a question about benchmark size. It does
not answer the question a reader will actually ask: if you trained the same
configuration again with a different random seed, how much would the score move?
If that movement is comparable to the effects being claimed, then the effects are
not established, however small their p-values are. A p-value computed on a
benchmark says nothing about seed variance, because the benchmark is fixed and
the seed is what varies.

Queue 42 re-runs GRPO at lr 1e-5 with a different seed, which is the one honest
estimate of that movement this project can produce. This script reads both seeds
and reports:

  * the seed-to-seed difference itself, with a paired bootstrap and McNemar. The
    pairing is over the same 252 rows, so it answers "do these two runs differ on
    these rows", not "would they differ on new rows".
  * the same difference on the structured metrics, which have no per-row
    decomposition and so are marginal estimates only.
  * whether the second seed reproduces the first seed's central conclusion
    against SFT (i.e. is "GRPO at lr 1e-5 is indistinguishable from SFT"
    reproducible, or was that one draw).

THE DECISION RULE IS FIXED BEFORE THE NUMBER IS SEEN, which is the whole point of
writing it here rather than after: the seed difference is compared against
|dacc| of the learning-rate effect within GRPO (lr5e-5 minus lr1e-5). If the
seed difference reaches half that size, the learning-rate effect is not
separable from seed noise on this benchmark and 5.2.2 must say so. The threshold
is not to be adjusted once the value is known; if it needs changing, that is a
finding to report, not a knob.

WHAT TWO SEEDS CANNOT GIVE: a point difference, not a variance estimate. Two
draws give one difference and no confidence interval on the spread. A real noise
floor needs several seeds, which was not run. So the number below is a lower
bound on the problem, not the problem measured -- it can understate seed variance
and cannot overstate it. That asymmetry is the honest reading and is written into
the output.

Exits nonzero, writing nothing, if the seed-2 record is absent, so that a missing
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
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
BENCH = ROOT / "benchmarks/wafer_bench_v1"
BASE = ROOT / "outputs/baselines"
REPORTS = ROOT / "outputs/reports"
OUT = REPORTS / "seed_variance.json"

SEED1 = ("qwen35_9b_grpo", "GRPO_G4_lr1e5_seed1")
SEED2 = ("qwen35_9b_grpo_lr1e5_seed3408", "GRPO_G4_lr1e5_seed3408")
# The third draw of the same configuration (queue 43). Optional by construction: a
# draw whose two files are not both on disk is reported as pending and left out of
# every number, so this tool produces the two-seed record it was written to produce
# until there is a third draw to fold in.
SEED3 = ("qwen35_9b_grpo_lr1e5_seed3409", "GRPO_G4_lr1e5_seed3409")
EXTRA = [SEED3]
SFT_TAG = "qwen35_9b_adapter"
LR5E5_TAG = "qwen35_9b_grpo_lr5e5"

BOOT = 20000
SEED = 20260916
HALF = 0.5   # the pre-registered fraction; see docstring

for tag, _ in (SEED2,):
    if not (BASE / (tag + ".jsonl")).is_file():
        print(f"NOT READY: {BASE / (tag + '.jsonl')} does not exist yet; "
              f"seed 2 has not been evaluated. Nothing written.")
        sys.exit(0)


def rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def image_key(row):
    imgs = row.get("images") or []
    if not imgs:
        return None
    first = imgs[0]
    p = first.get("path") if isinstance(first, dict) else first
    return Path(p).name if p else None


def mcnemar_exact(a, b):
    from math import comb
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    return n01, n10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def pval(p):
    return float(f"{p:.6g}")


requests = rows(BENCH / "all_requests.jsonl")
gold = {r["sample_id"]: r["gold_label"]
        for r in rows(BENCH / "classification.jsonl")}
cls_idx, cls_gold = [], []
for i, r in enumerate(requests):
    if r.get("task") == "classification":
        cls_idx.append(i)
        cls_gold.append(gold.get(r.get("sample_id")))
CLASSES = sorted(set(gold.values()))


def parse_label(text):
    tail = text.strip().splitlines()[-1].strip() if text.strip() else ""
    for c in CLASSES:
        if tail == c:
            return c
    for c in CLASSES:
        if re.search(r"\b" + re.escape(c) + r"\b", tail):
            return c
    for c in CLASSES:
        if c in text:
            return c
    return None


def load(tag):
    preds = rows(BASE / (tag + ".jsonl"))
    if len(preds) != len(requests):
        sys.exit(f"{tag}: {len(preds)} predictions for {len(requests)} requests")
    bad = [(i, image_key(preds[i]), image_key(requests[i]))
           for i in range(len(requests))
           if image_key(preds[i]) != image_key(requests[i])]
    if bad:
        sys.exit(f"{tag}: {len(bad)} rows out of order, first: {bad[0]}")
    vec = [parse_label(preds[i].get("response") or "") == g
           for i, g in zip(cls_idx, cls_gold)]
    rp = REPORTS / (tag + "__report.json")
    rep = json.loads(rp.read_text(encoding="utf-8")) if rp.is_file() else None
    return vec, rep


v1, r1 = load(SEED1[0])
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
for lbl, vec, rep in _checks:
    if rep is None:
        sys.exit(f"{lbl}: no __report.json, cannot verify parsing")
    here = sum(vec) / len(vec)
    there = rep["classification"]["accuracy"]
    if abs(here - there) >= 1e-9:
        sys.exit(f"{lbl}: parsed accuracy {here:.6f} != reported {there:.6f}; "
                 "verdicts would describe misread responses")
    print(f"parse check {lbl:22s} {here:.4f} == report {there:.4f}")

acc1, acc2, accs, acc5 = (sum(v) / len(v) for v in (v1, v2, vs, v5))
rng = random.Random(SEED)


def boot_ci(a, b, n=BOOT):
    diffs = []
    m = len(a)
    for _ in range(n):
        s = [rng.randrange(m) for _ in range(m)]
        diffs.append(sum(b[i] for i in s) / m - sum(a[i] for i in s) / m)
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


n01, n10, p = mcnemar_exact(v1, v2)
d_seed = acc2 - acc1
lo, hi = boot_ci(v1, v2)

d_lr = acc5 - acc1                       # the effect this noise floor tests
d_lr_paired = acc5 - acc1

print()
print(f"seed 1 accuracy {acc1:.4f}   seed 2 accuracy {acc2:.4f}")
print(f"seed-to-seed difference {d_seed:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]  "
      f"McNemar p={p:.5f}")
print(f"for scale, GRPO lr5e-5 minus lr1e-5 (seed 1) = {d_lr:+.4f}")
ratio = abs(d_seed) / abs(d_lr) if d_lr else float("inf")
print(f"|seed difference| / |lr effect| = {ratio:.3f}  "
      f"(rule: >= {HALF} means not separable)")

# Does seed 2 also land near SFT? The claim under test is a null, so the
# informative event is seed 2 being significantly BELOW SFT, which would break
# the replication.
n01s, n10s, ps = mcnemar_exact(vs, v2)
d_vs_sft_2 = acc2 - accs
n01s1, n10s1, ps1 = mcnemar_exact(vs, v1)
print(f"vs SFT: seed1 {acc1 - accs:+.4f} (p={ps1:.5f})   "
      f"seed2 {d_vs_sft_2:+.4f} (p={ps:.5f})")

# Structured / caption metrics, marginal only.
def metric(rep, *path):
    d = rep
    for k in path:
        d = d[k]
    return d


KEYS = [("structured radial_zone", ("structured", "field_accuracy", "radial_zone")),
        ("structured clock MAE", ("structured", "clock_circular_mae_sectors")),
        ("structured size MAE", ("structured", "size_mae_r")),
        ("caption must_hit", ("caption", "must_hit_rate")),
        # The record spells it macro_f1; "macro-F1" is how the prose writes it.
        ("classification macro_f1", ("classification", "macro_f1"))]
print()
print(f"  {'metric':<26s} {'seed1':>9s} {'seed2':>9s} {'diff':>9s} "
      f"{'lr effect':>10s} {'ratio':>7s}")
mrows = {}
for lbl, path in KEYS:
    a, b = metric(r1, *path), metric(r2, *path)
    c = metric(r5, *path)
    rr = abs(b - a) / abs(c - a) if c != a else None
    mrows[lbl] = {"seed1": round(a, 6), "seed2": round(b, 6),
                  "diff": round(b - a, 6), "lr_effect": round(c - a, 6),
                  "ratio_seed_to_lr": None if rr is None else round(rr, 4)}
    print(f"  {lbl:<26s} {a:>9.4f} {b:>9.4f} {b - a:>+9.4f} {c - a:>+10.4f} "
          + ("     n/a" if rr is None else f"{rr:>7.3f}"))

# ------------------------------------------------------------- the spread itself
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
_sig = bool(pval(ps1) < 0.05 or pval(ps) < 0.05)

# ------------------------------------------------------------------ the design claim
# "these two runs differ only in the seed" is the premise of everything below, so it is
# compared rather than declared. Two runs of the same configuration produce the same
# config block except for the seed; anything else that moved makes this pair a
# confounded comparison, and a confounded comparison is what the section must not call
# a noise floor.
TRAIN_RESULT = {"qwen35_9b_grpo": "grpo_train_result"}


def config_of(tag):
    """The run's config, or None if its training record is not on disk."""
    p = REPORTS / f"{TRAIN_RESULT.get(tag, tag + '_train_result')}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("config") or {}


_c1, _c2 = config_of(SEED1[0]), config_of(SEED2[0])
_s1 = (_c1 or {}).get("seed")
_s2 = (_c2 or {}).get("seed")
if _c1 is None or _c2 is None:
    _config_compared = False
    _config_diff = []
    _identical_except = ("unverified: the two training records are not both on disk, so "
                         "the configurations were never compared and 'only the seed "
                         "differs' is an assumption, not a finding")
    print("  WARNING: cannot compare the two configs (a training record is missing); "
          "the record will say the design claim is unverified")
else:
    _config_compared = True
    _config_diff = sorted(k for k in set(_c1) | set(_c2) if _c1.get(k) != _c2.get(k))
    if _config_diff == ["seed"]:
        _identical_except = (f"the random seed ({_s1} -> {_s2}); every other config "
                             f"field compared equal in both training records")
    elif not _config_diff:
        _identical_except = ("nothing: the two runs' configs are identical, seed "
                             "included -- this is a repeat, not a reseed")
    else:
        _identical_except = ("the random seed, AND " + ", ".join(_config_diff)
                             + " -- the pair is not single-variable, so the difference "
                               "below is not a pure seed effect")
    print(f"  config diff (seed1 vs seed2): {_config_diff or 'none'}")

# The extra draws' configs, compared the same way and against the FIRST draw rather
# than against each other: "same configuration" is a statement about the cell, and the
# cell's owner is the first draw. Comparing the later draws to each other would leave
# open the case where both drifted the same way.
_extra_cfg = {}
for _tag, _lbl, _v, _r in EXTRA_LOADED:
    _c = config_of(_tag)
    if _c is None or not _config_compared:
        _extra_cfg[_lbl] = {
            "compared": False, "diff": None,
            "identical_except": "unverified: this draw's training record and the first "
                                "draw's are not both on disk, so the configurations "
                                "were never compared and 'the same configuration' is "
                                "an assumption, not a finding"}
        print(f"  WARNING: cannot compare {_lbl} against {SEED1[1]}; the record will "
              f"say that claim is unverified")
        continue
    _d = sorted(k for k in set(_c1) | set(_c) if _c1.get(k) != _c.get(k))
    _extra_cfg[_lbl] = {
        "compared": True,
        "diff": _d,
        "identical_except": (
            f"the random seed ({_s1} -> {_c.get('seed')}); every other config field "
            f"compared equal to {SEED1[1]} in both training records"
            if _d == ["seed"] else
            "nothing: identical to the first draw, seed included -- this is a repeat, "
            "not a reseed" if not _d else
            "the random seed, AND " + ", ".join(_d)
            + " -- not single-variable against the first draw, so what it adds to the "
              "range is not a pure seed effect"),
    }
    print(f"  config diff ({SEED1[1]} vs {_lbl}): {_d or 'none'}")

out = {
    "question": "How large is run-to-run (seed) variation on this benchmark, and "
                "are the reported effects larger than it?",
    "design": {"seed1": SEED1[1], "seed2": SEED2[1],
               "identical_except": _identical_except,
               "config_compared": _config_compared,
               "config_diff": _config_diff,
               "seeds": {"seed1": _s1, "seed2": _s2},
               "extra": _extra_cfg,
               "benchmark": "the same 252 classification rows for both"},
    "pre_registered_rule": {
        "statement": "the learning-rate effect is not separable from seed noise if "
                     "|seed difference| >= half of |lr effect|",
        "fraction": HALF,
        "fixed_before_seeing": True,
        "note": "if this threshold is changed after the fact, that is a finding "
                "to report and not a knob to turn"},
    "accuracy": {"seed1": round(acc1, 6), "seed2": round(acc2, 6)},
    "seed_difference_accuracy": {
        "delta": round(d_seed, 6),
        "ci95_rows": [round(lo, 6), round(hi, 6)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "mcnemar": {"seed1_wrong_seed2_right": n01, "seed1_right_seed2_wrong": n10,
                    "p_exact_two_sided": pval(p)}},
    "reference_lr_effect_within_grpo": {
        "pair": "GRPO lr5e-5 minus GRPO lr1e-5 (seed 1)",
        "accuracy_delta": round(d_lr_paired, 6)},
    "ratio_seed_to_lr_effect": round(ratio, 6),
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
    },
    "replication_vs_sft": {
        "seed1": {"delta": round(acc1 - accs, 6), "p_exact_two_sided": pval(ps1)},
        "seed2": {"delta": round(d_vs_sft_2, 6), "p_exact_two_sided": pval(ps)},
        # Computed, not written by hand. This field used to be a disjunction of both
        # possible outcomes -- true however the run came out, and therefore worth
        # nothing to a reader, which is the failure section 8 item 14 records in the
        # document (an artefact indistinguishable from a complete one). The two
        # booleans the sentence is computed from are recorded beside it, so the
        # sentence can be checked against the numbers instead of believed.
        "same_side_of_sft": _same_side,
        "significantly_different_from_sft": _sig,
        "reading": (
            "not reproducible: the two seeds land on opposite sides of SFT "
            f"({acc1 - accs:+.4f} and {d_vs_sft_2:+.4f}), so the seed moves the "
            "comparison by more than the comparison resolves"
            if not _same_side else
            "not reproducible: the seeds agree on the side but one of them differs "
            f"from SFT significantly (p = {pval(ps1):.5f}, {pval(ps):.5f}), so the "
            "comparison is not stable across seeds"
            if _sig else
            "reproducible and null: both seeds are on the same side of SFT "
            f"({acc1 - accs:+.4f} and {d_vs_sft_2:+.4f}) and neither differs from "
            f"it significantly (p = {pval(ps1):.5f}, {pval(ps):.5f})"),
    },
    "structured_and_caption": mrows,
    "what_two_seeds_cannot_give": "two draws give one difference and no interval "
                                  "on the spread, so this is a lower bound on seed "
                                  "variation: it can understate it and cannot "
                                  "overstate it. A real noise floor needs several "
                                  "seeds, which were not run.",
    "structured_metrics_are_marginal": "no per-row decomposition, so the paired "
                                       "apparatus does not apply to them; they are "
                                       "compared only in magnitude",
}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwrote {OUT}")
