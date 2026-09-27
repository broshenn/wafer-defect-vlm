"""Is the lr 5e-5 drop real? A paired test, where the CI rule cannot answer.

tools/final_report.py decides whether a run is worse than SFT with
`run_ci_hi < sft_ci_lo` -- two marginal 95% intervals, compared for overlap.
That rule is conservative even for independent samples, and these are not
independent: every run answered the same 252 benchmark rows from the same
prompts, so the paired structure is exactly the part of the design that carries
the power. Comparing marginal intervals throws it away. With it thrown away, the
completed 2x2 reads "every RL run overlaps SFT", which is true as stated and
much weaker than what the data can support.

So test the differences directly. Two paired instruments:

  * McNemar's exact test on the discordant pairs -- for the rows SFT and the run
    disagree on, is the split even? This asks whether one is more often right
    than the other on the same rows.
  * A bootstrap over rows of the accuracy difference, resampling rows and not
    runs, which is the unit that varies here.

Neither speaks to macro-F1 directly: macro-F1 is a function of the whole
confusion matrix, not a per-row average, so it has no per-row decomposition to
pair on. Accuracy does, and accuracy and macro-F1 move together across these
runs, so accuracy is the paired quantity and macro-F1 stays a marginal estimate.
That asymmetry is stated in the output rather than papered over.

Alignment is verified, not assumed. Prediction files carry no sample_id; the
join is positional against all_requests.jsonl, so every row's image path is
compared against the request it is supposed to answer. If any row is out of
order the script fails instead of scoring misaligned data.

Parsing is validated by reproduction: the accuracy computed here must equal the
accuracy the project's own scorer wrote in the __report.json. If it does not,
this script's reading of the responses is wrong and its verdicts are void.

A positive control is asserted. The base model scores far below SFT; a paired
test that cannot see a difference that large cannot see anything, and every
null result below would be uninterpretable. If the control does not come out
significant the script exits rather than reporting.

Writes outputs/reports/paired_significance.json.
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
OUT = REPORTS / "paired_significance.json"

RUNS = [("BASE_zero_shot", "qwen35_9b_zero_shot"),
        ("SFT", "qwen35_9b_adapter"),
        ("GRPO_G4_lr1e5", "qwen35_9b_grpo"),
        ("GRPO_G4_lr5e5", "qwen35_9b_grpo_lr5e5"),
        ("GSPO_G8_lr1e5", "gspo_lr1e5"),
        ("GSPO_G8_lr5e5", "qwen35_9b_gspo_v1"),
        ("GSPO_G32_lr5e5", "qwen35_9b_gspo_g32")]

# GSPO at G=32 and lr 1e-5 is produced by queue 41 and may not exist yet. It is
# the only run that gives the lr contrast at a second group size, so its absence
# must not pass unremarked: an unrecorded change in the comparison set is the
# exact defect this project keeps finding (section 8, item 9). So it is optional
# and its presence or absence is written into the output.
OPTIONAL = [("GSPO_G32_lr1e5", "qwen35_9b_gspo_g32_lr1e5"),
            ("GSPO_G4_lr5e5", "qwen35_9b_gspo_g4_lr5e5"),
            ("GSPO_G4_lr1e5", "qwen35_9b_gspo_g4_lr1e5")]

# Runs on disk that are deliberately not cells of this record, each with its reason.
# Written down rather than left out: a run that lands and changes nothing here is
# indistinguishable, from the record's side, from a run that was never evaluated.
EXCLUDE = {
    "qwen35_9b_grpo_lr1e5_seed3408":
        "the second seed of GRPO_G4_lr1e5. Same configuration, so it is not a new "
        "cell in any contrast here; its purpose is the replication check in "
        "tools/seed_variance.py, and it carries a column of its own in the "
        "comparison tables.",
    "qwen35_9b_grpo_lr1e5_seed3409":
        "the third draw of the same cell. Not a new cell either: it changes no "
        "contrast in this record, and no contrast here can answer the question it "
        "exists for. One difference is not a spread, so tools/seed_variance.py reports "
        "this run as the third point of the range rather than as a second learning "
        "rate. Like the second seed it is a row in the comparison tables and not a "
        "column in LIMITATIONS.md's, which the landing runs with --no-column so that "
        "one configuration is not counted twice.",
}

BOOT = 20000
SEED = 20260916


def rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def image_key(row):
    """The path of the first image, whichever shape the row uses."""
    imgs = row.get("images") or []
    if not imgs:
        return None
    first = imgs[0]
    p = first.get("path") if isinstance(first, dict) else first
    return Path(p).name if p else None


def mcnemar_exact(a, b):
    """Two-sided exact McNemar on the discordant pairs."""
    from math import comb
    n01 = sum(1 for x, y in zip(a, b) if not x and y)   # a wrong, b right
    n10 = sum(1 for x, y in zip(a, b) if x and not y)   # a right, b wrong
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * tail)


def pval(p):
    """Store a p-value at 6 significant figures, not 8 decimal places.

    round(2.83e-26, 10) is 0.0, which is not a small number -- it is a wrong
    one, and it renders in prose as "p = 0". Significant figures keep the
    magnitude of a decisive result instead of discarding it.
    """
    return float(f"{p:.6g}")


requests = rows(BENCH / "all_requests.jsonl")
gold = {r["sample_id"]: r["gold_label"] for r in rows(BENCH / "classification.jsonl")}

cls_idx, cls_gold = [], []
for i, r in enumerate(requests):
    if r.get("task") == "classification":
        cls_idx.append(i)
        cls_gold.append(gold.get(r.get("sample_id")))
missing = [i for i, g in zip(cls_idx, cls_gold) if g is None]
if missing:
    sys.exit(f"classification rows without gold: {missing[:5]}")

CLASSES = sorted(set(gold.values()))
print(f"classification requests: {len(cls_idx)}   classes: {len(CLASSES)}")


def parse_label(text):
    """The response ends with the label, but the model may wrap it in a think
    block, so take a class name from the tail rather than assuming a position."""
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


correct, acc_here, acc_reported = {}, {}, {}
alignment = {}
AVAILABLE = list(RUNS) + [(n, t) for n, t in OPTIONAL
                          if (BASE / (t + ".jsonl")).is_file()]
ABSENT = [n for n, t in OPTIONAL if not (BASE / (t + ".jsonl")).is_file()]

# Every report on disk must be one of RUNS, one of OPTIONAL, or a named exclusion.
# Without this, adding a run to the project changes nothing here and nothing says so.
_known = {t for _, t in RUNS} | {t for _, t in OPTIONAL} | set(EXCLUDE)
_unaccounted = sorted(p.name[: -len("__report.json")]
                      for p in REPORTS.glob("*__report.json")
                      if p.name[: -len("__report.json")] not in _known)
if _unaccounted:
    sys.exit("report(s) on disk that are neither a run of this record nor a named "
             f"exclusion: {_unaccounted}. Add each to RUNS/OPTIONAL, or to EXCLUDE "
             "with the reason it is not a contrast here. A run that lands and "
             "changes nothing in this record is the defect this guard exists for.")

# The grid is enumerated from the run names, which already carry algorithm, group
# size and learning rate. A hand-written list of cells was here before and would
# have kept the G=4 cell of the lr axis out of a record that still looked complete.
CELL = re.compile(r"^(GRPO|GSPO)_G(\d+)_lr(1e5|5e5)$")


def lr_cells(names):
    """(label, lr1e-5 run, lr5e-5 run) for every algorithm x group size that has
    both learning rates present, ordered by algorithm then group size. The label is
    the one this record has always used, so existing keys are unchanged."""
    have = {}
    for n in names:
        m = CELL.match(n)
        if m:
            have.setdefault((m.group(1), int(m.group(2))), set()).add(m.group(3))
    return [(f"{a} G={g}", f"{a}_G{g}_lr1e5", f"{a}_G{g}_lr5e5")
            for (a, g), lrs in sorted(have.items()) if {"1e5", "5e5"} <= lrs]
if ABSENT:
    print(f"\nNOTE: optional runs not yet evaluated and therefore excluded: "
          f"{', '.join(ABSENT)}")
    print("  Their absence is recorded in the output; downstream readers must "
          "not read the results below as covering them.")

for name, tag in AVAILABLE:
    preds = rows(BASE / (tag + ".jsonl"))
    if len(preds) != len(requests):
        sys.exit(f"{tag}: {len(preds)} predictions for {len(requests)} requests")

    bad = [(i, image_key(preds[i]), image_key(requests[i]))
           for i in range(len(requests))
           if image_key(preds[i]) != image_key(requests[i])]
    if bad:
        sys.exit(f"{tag}: {len(bad)} rows out of order, first: {bad[0]}")
    alignment[name] = f"{len(preds)}/{len(requests)} rows aligned by image path"

    correct[name] = [parse_label(preds[i].get("response") or "") == g
                     for i, g in zip(cls_idx, cls_gold)]
    acc_here[name] = sum(correct[name]) / len(correct[name])

    rp = REPORTS / (tag + "__report.json")
    acc_reported[name] = (json.loads(rp.read_text(encoding="utf-8"))
                          ["classification"]["accuracy"]) if rp.is_file() else None

print()
print("self-check: accuracy computed here vs the project's own scorer")
ok = True
for name, _ in AVAILABLE:
    a, b = acc_here[name], acc_reported[name]
    agree = b is not None and abs(a - b) < 1e-9
    ok &= agree
    print(f"  {name:16s} here={a:.4f}  report="
          + ("n/a" if b is None else format(b, ".4f"))
          + ("  match" if agree else "  MISMATCH"))
if not ok:
    sys.exit("parsing does not reproduce the scorer; verdicts below would be "
             "about misread responses, so nothing is written")

# There is deliberately no shared bootstrap stream. Every contrast constructs its
# own Random(SEED) where it is used, so an interval depends only on the two runs it
# compares -- never on which other runs exist, nor on the order of the blocks above.
# With one shared stream, adding a run silently moved the 2.5% quantile of unrelated
# contrasts by a whole row (1/252). That happened twice here, the second time to a
# number LIMITATIONS.md already quotes.
sft = correct["SFT"]


def boot_ci(a, b, rng, n=BOOT):
    diffs = []
    for _ in range(n):
        s = [rng.randrange(len(a)) for _ in range(len(a))]
        diffs.append(sum(b[i] for i in s) / len(s) - sum(a[i] for i in s) / len(s))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


# Positive control: the base model scores far below SFT. If a paired test on
# these 252 rows cannot see a difference that large, it cannot see anything, and
# every null below is uninterpretable. Asserted, not merely reported.
_n01, _n10, _p = mcnemar_exact(correct["BASE_zero_shot"], sft)
if _p > 0.001:
    sys.exit(f"CONTROL FAILED: base vs SFT gives p={_p:.4f}; the instrument "
             "cannot detect a difference this large, so nothing else it "
             "reports is meaningful")
print("\ncontrol: BASE vs SFT dacc="
      + format(acc_here["SFT"] - acc_here["BASE_zero_shot"], "+.4f")
      + " p=" + format(_p, ".3g") + " -- instrument detects it, as it must")

out = {"description": __doc__.strip().split("\n\n")[0],
       "alignment": alignment, "bootstrap_resamples": BOOT, "seed": SEED,
       "unit": "accuracy is paired per benchmark row; macro-F1 has no per-row "
               "decomposition so it stays a marginal estimate",
       "positive_control": {
           "pair": "BASE_zero_shot vs SFT",
           "accuracy_delta": round(acc_here["SFT"] - acc_here["BASE_zero_shot"], 6),
           "p_exact_two_sided": pval(_p),
           "requirement": "p <= 0.001 or the script exits",
           "passed": True},
       "accuracy": {n: round(acc_here[n], 6) for n, _ in AVAILABLE},
       "runs_included": [n for n, _ in AVAILABLE],
       "runs_optional_absent": ABSENT,
       "comparisons_vs_SFT": {}}

print()
print("paired comparison against SFT (same 252 rows)")
print(f"  {'run':16s} {'dacc':>9s} {'95% CI (rows)':>24s} "
      f"{'n01':>4s} {'n10':>4s} {'p':>10s}")
for name, _ in AVAILABLE:
    if name in ("SFT", "BASE_zero_shot"):
        continue
    vec = correct[name]
    d = acc_here[name] - acc_here["SFT"]
    lo, hi = boot_ci(sft, vec, random.Random(SEED))
    n01, n10, p = mcnemar_exact(sft, vec)
    out["comparisons_vs_SFT"][name] = {
        "accuracy": round(acc_here[name], 6),
        "delta_vs_sft": round(d, 6),
        "delta_ci95_rows": [round(lo, 6), round(hi, 6)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "mcnemar": {"sft_wrong_run_right": n01, "sft_right_run_wrong": n10,
                    "p_exact_two_sided": pval(p)}}
    print(f"  {name:16s} {d:>+9.4f} "
          + format(f"[{lo:+.4f},{hi:+.4f}]", ">24s")
          + f" {n01:>4d} {n10:>4d} {p:>10.5f}")

print()
print("paired lr effect within each algorithm (lr5e-5 minus lr1e-5):")
for algo, a, b in lr_cells(correct):
    va, vb = correct[a], correct[b]
    d = acc_here[b] - acc_here[a]
    lo, hi = boot_ci(va, vb, random.Random(SEED))
    n01, n10, p = mcnemar_exact(va, vb)
    out.setdefault("lr_effect_paired", {})[algo] = {
        "from": a, "to": b, "accuracy_delta": round(d, 6),
        "delta_ci95_rows": [round(lo, 6), round(hi, 6)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "mcnemar": {"lr1e5_wrong_lr5e5_right": n01,
                    "lr1e5_right_lr5e5_wrong": n10,
                    "p_exact_two_sided": pval(p)}}
    print(f"  {algo:10s} dacc={d:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  "
          f"n01={n01} n10={n10}  p={p:.5f}")

print()
print("cross-algorithm at matched lr (the paper's claim: sequence-level IS "
      "tolerates a higher lr).")
print("  Confounded: every token-level run is G=4 and every sequence-level run "
      "is G=8,")
print("  so this contrast cannot separate IS level from group size.")
for lr, a, b in (("1e-5", "GRPO_G4_lr1e5", "GSPO_G8_lr1e5"),
                 ("5e-5", "GRPO_G4_lr5e5", "GSPO_G8_lr5e5")):
    va, vb = correct[a], correct[b]
    d = acc_here[b] - acc_here[a]
    lo, hi = boot_ci(va, vb, random.Random(SEED))
    n01, n10, p = mcnemar_exact(va, vb)
    out.setdefault("cross_algorithm_paired", {})["lr" + lr] = {
        "grpo_token": a, "gspo_sequence": b,
        "accuracy_delta_gspo_minus_grpo": round(d, 6),
        "delta_ci95_rows": [round(lo, 6), round(hi, 6)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "mcnemar": {"grpo_wrong_gspo_right": n01, "grpo_right_gspo_wrong": n10,
                    "p_exact_two_sided": pval(p)},
        "confound": "GRPO is G=4 and GSPO is G=8, so IS level and group size "
                    "move together and this contrast cannot separate them"}
    print(f"  lr{lr:5s} GSPO(G=8) - GRPO(G=4): dacc={d:+.4f} "
          f"CI=[{lo:+.4f},{hi:+.4f}] n01={n01} n10={n10} p={p:.5f}")


print()
print("the paper's own setting against the baseline group size, at matched lr.")
print("  NOTE: every contrast here draws from its own Random(SEED), so no interval")
print("  depends on which other runs exist or on the order of the blocks. Keep it")
print("  that way: one shared stream makes each new contrast silently move the")
print("  intervals of the ones below, including values already quoted in")
print("  LIMITATIONS.md.")
print("  GSPO(G=32, sequence) vs GRPO(G=4, token) at lr 5e-5. Section 5.2.5 makes")
print("  a claim about this pair; it is also the most confounded contrast in the")
print("  record -- group size, IS level and effective batch all move together.")
for a, b in (("GRPO_G4_lr5e5", "GSPO_G32_lr5e5"),):
    va, vb = correct[a], correct[b]
    d = acc_here[b] - acc_here[a]
    # A private stream, deliberately: the shared `rng` is consumed in file order,
    # so drawing from it here would shift every contrast below and silently change
    # values already written into the record and quoted in the document. With its
    # own stream this block is order-independent and the existing numbers are
    # reproduced exactly.
    lo, hi = boot_ci(va, vb, random.Random(SEED))
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

print()
print("group size at matched algorithm and lr (GSPO, lr 5e-5): G=8 -> G=32.")
print("  This is the comparison section 5.2.5 reads off two marginal macro-F1")
print("  values as a 2.8x magnitude gap. Magnitude is not significance, so the")
print("  same paired instrument is applied to accuracy here.")
for a, b, label in (("GSPO_G8_lr5e5", "GSPO_G32_lr5e5", "GSPO lr5e-5: G=8 -> G=32"),):
    va, vb = correct[a], correct[b]
    d = acc_here[b] - acc_here[a]
    lo, hi = boot_ci(va, vb, random.Random(SEED))
    n01, n10, p = mcnemar_exact(va, vb)
    out["group_size_paired"] = {
        "label": label, "from": a, "to": b,
        "accuracy_delta_g32_minus_g8": round(d, 6),
        "delta_ci95_rows": [round(lo, 6), round(hi, 6)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "mcnemar": {"g8_wrong_g32_right": n01, "g8_right_g32_wrong": n10,
                    "p_exact_two_sided": pval(p)},
        "note": ("G=32 is present at both learning rates in this record, so the "
                 "lr contrast is available at two group sizes (see "
                 "lr_effect_both_group_sizes). macro-F1 is not covered."
                 if "GSPO_G32_lr1e5" in correct else
                 "one lr only; G=32 at lr 1e-5 is run by queue 41 and is not in "
                 "this record. macro-F1 is not covered.")}
    print(f"  {label}: dacc={d:+.4f} CI=[{lo:+.4f},{hi:+.4f}] "
          f"n01={n01} n10={n10} p={p:.5f}")

# ---------------------------------------------------------- the identification test
# The paper's proposition is that sequence-level normalisation tolerates a higher
# learning rate than token-level does. Testing it needs a pair that differs in IS
# level and nothing else. There is no such pair here: every token-level run is
# G=4 and every sequence-level run is G=8 or G=32. So the proposition is not
# tested by any contrast in this record, and saying which contrast WOULD test it
# is more useful than reporting a confounded one as if it answered the question.
#
# What the new G=32 lr 1e-5 run does buy is a second measurement of the lr effect
# at a different group size. If the lr effect is present at both G=8 and G=32,
# then group size is not what produces it, which removes the leading alternative
# explanation for the lr 5e-5 degradation without ever measuring IS level.
if "GSPO_G32_lr1e5" in correct:
    print()
    print("lr effect at each group size (GSPO, sequence-level): the same change "
          "of learning rate measured at every group size in this record.")
    lr_blocks = {}
    # `lr_cells` yields (label, lr1e-5 run, lr5e-5 run) -- three values, which is what
    # the two other callers in this file unpack. This loop unpacked four, expecting the
    # group size as a fourth element. It ran for the first time when run 41 gave GSPO a
    # second group size, and raised ValueError, so the record did not write. The group
    # size is in the label, which is what the label is for.
    for _label, a, b in lr_cells(correct):
        _algo, _gs = _label.split(" G=")
        g = int(_gs)
        if _algo != "GSPO":
            continue
        va, vb = correct[a], correct[b]
        d = acc_here[b] - acc_here[a]
        lo, hi = boot_ci(va, vb, random.Random(SEED))
        n01, n10, p = mcnemar_exact(va, vb)
        lr_blocks[f"G={g}"] = {
            "from": a, "to": b, "accuracy_delta": round(d, 6),
            "delta_ci95_rows": [round(lo, 6), round(hi, 6)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "mcnemar": {"lr1e5_wrong_lr5e5_right": n01,
                        "lr1e5_right_lr5e5_wrong": n10,
                        "p_exact_two_sided": pval(p)}}
        print(f"  G={g:<3d} dacc={d:+.4f} CI=[{lo:+.4f},{hi:+.4f}] "
              f"n01={n01} n10={n10} p={p:.5f}")
    _sig = [g for g, blk in lr_blocks.items() if blk["excludes_zero"]]
    out["lr_effect_both_group_sizes"] = {
        "blocks": lr_blocks,
        "significant_at": _sig,
        "reading": ("the lr effect is present at "
                    + " and ".join(sorted(_sig)) + ", so it is not produced by "
                    "group size" if len(_sig) == len(lr_blocks) else
                    "the lr effect is not detected at every group size measured, "
                    "so group size is not excluded as the cause"),
        "does_not_settle": "this removes group size as the cause of the lr "
                           "effect. It does NOT test the paper's proposition, "
                           "which needs a pair differing only in IS level."}

    print()
    print("group size at matched algorithm, IS level and lr (GSPO, lr 1e-5): "
          "G=8 -> G=32.")
    va, vb = correct["GSPO_G8_lr1e5"], correct["GSPO_G32_lr1e5"]
    d = acc_here["GSPO_G32_lr1e5"] - acc_here["GSPO_G8_lr1e5"]
    lo, hi = boot_ci(va, vb, random.Random(SEED))
    n01, n10, p = mcnemar_exact(va, vb)
    out["group_size_paired_lr1e5"] = {
        "label": "GSPO lr 1e-5: G=8 -> G=32", "from": "GSPO_G8_lr1e5",
        "to": "GSPO_G32_lr1e5", "accuracy_delta_g32_minus_g8": round(d, 6),
        "delta_ci95_rows": [round(lo, 6), round(hi, 6)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "mcnemar": {"g8_wrong_g32_right": n01, "g8_right_g32_wrong": n10,
                    "p_exact_two_sided": pval(p)}}
    print(f"  dacc={d:+.4f} CI=[{lo:+.4f},{hi:+.4f}] n01={n01} n10={n10} "
          f"p={p:.5f}")

    g32_blocks = [out.get("group_size_paired", {}), out["group_size_paired_lr1e5"]]
    _acts = [b for b in g32_blocks if b and b.get("excludes_zero")]
    out["group_size_verdict"] = {
        "contrasts": [{"label": b.get("label"), "delta": b.get(
            "accuracy_delta_g32_minus_g8"), "p": b.get("mcnemar", {}).get(
                "p_exact_two_sided")} for b in g32_blocks if b],
        "significant_count": len(_acts),
        "reading": ("neither lr shows a group-size effect, so at lr 5e-5 and "
                    "lr 1e-5 alike G=32 is not separable from G=8"
                    if not _acts else
                    "a group-size effect appears at "
                    + ", ".join(b["label"] for b in _acts)),
        "caveat": "two group sizes (8, 32) and two lrs; a non-monotone effect "
                  "between them would not be seen."}

    print()

# The missing-cell argument does not depend on whether queue 41 has landed: it is
# about which pair would be needed to test the proposition at all, and no run in
# this record supplies it. Recorded unconditionally so it is not lost if the
# optional run is absent.
# Whether the proposition has a test is a fact about this record, not a standing
# argument. Run 43 supplies the pair differing only in IS level, so the old
# unconditional text would keep asserting a missing cell that has arrived -- the
# stale-claim class this project keeps finding. Both branches are recorded so the
# artefact states which world it is in.
# Everything the prose below asserts is read from the blocks above rather than
# typed here. The p-values used to be literals in this file, and the two sentences
# that name the missing cell and the established result were written when the G=4
# 2x2 was incomplete -- which is a state this record has now left.
_lr_all = sorted(out.get("lr_effect_paired", {}))
_lr_sig = [k for k in _lr_all if out["lr_effect_paired"][k]["excludes_zero"]]
_five = sorted(n for n in out["comparisons_vs_SFT"] if n.endswith("lr5e5"))
_five_sig = [n for n in _five if out["comparisons_vs_SFT"][n]["excludes_zero"]]
_one = sorted(n for n in out["comparisons_vs_SFT"] if n.endswith("lr1e5"))
_one_sig = [n for n in _one if out["comparisons_vs_SFT"][n]["excludes_zero"]]
_established_txt = (
    "the lr effect is significant in "
    + (f"every cell where both learning rates exist ({', '.join(_lr_all)})"
       if _lr_all and len(_lr_sig) == len(_lr_all) else
       f"{len(_lr_sig)} of the {len(_lr_all)} cells where both learning rates "
       f"exist ({', '.join(_lr_sig) or 'none'})")
    + ", so sequence-level does not visibly tolerate the higher learning rate "
      "better than token-level does at the group sizes measured")

# The 2x2 at G=4 is the only group size at which both IS levels have both learning
# rates. Its four cells are the whole reason the G=4 runs exist, so whether they are
# all present is computed, not assumed.
_G4_2X2 = ("GRPO_G4_lr1e5", "GRPO_G4_lr5e5", "GSPO_G4_lr1e5", "GSPO_G4_lr5e5")
_g4_have = [n for n in _G4_2X2 if n in correct]
_g4_done = len(_g4_have) == 4
_g4_missing_txt = (
    "none: the 2x2 at G=4 is complete -- both IS levels at both learning rates -- "
    "so the lr effect is measured separately for each IS level at one group size, "
    "which is the form in which 'tolerates a higher learning rate' becomes a "
    "comparison rather than an impression. What is still NOT measured is whether "
    "the two effects differ in SIZE: this record holds each effect and its own "
    "test, not a test of the difference between them."
    if _g4_done else
    "tolerance is a statement about the SIZE of the lr effect at each IS level, "
    "which needs the same 2x2 at G=4. Cells present at G=4: "
    + (", ".join(_g4_have) or "none") + "; the 2x2 is not complete, so the lr "
    "effect is not measured for both IS levels at one group size")
if _g4_done:
    print()
    print("the 2x2 at G=4 is complete: lr effect measured separately for each IS "
          "level at one group size.")

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
        "missing_cell": _g4_missing_txt,
        "what_is_established": _established_txt}
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
        "what_is_established": _established_txt}

# macro-F1, marginal only. Recorded here so that numbers cited from this run
# trace to a record rather than to arithmetic done while writing prose -- the
# deltas are the part a reader cannot reconstruct from the reports alone.
print()
print("macro-F1 (marginal, NOT paired -- recorded so the deltas are traceable):")
mf1 = {}
for name, tag in AVAILABLE:
    rp = REPORTS / (tag + "__report.json")
    mf1[name] = (json.loads(rp.read_text(encoding="utf-8"))
                 ["classification"]["macro_f1"]) if rp.is_file() else None
out["macro_f1_marginal"] = {
    "note": "marginal estimates; macro-F1 is not a per-row average so the "
            "paired instruments above do not apply to it",
    "values": {n: (round(v, 6) if v is not None else None) for n, v in mf1.items()}}
print(f"  {'SFT':16s} {mf1['SFT']:.4f}")
for algo, a, b in lr_cells(correct):
    out["macro_f1_marginal"].setdefault("lr_effect", {})[algo] = {
        "lr1e-5": round(mf1[a], 6), "lr5e-5": round(mf1[b], 6),
        "delta": round(mf1[b] - mf1[a], 6)}
    out["macro_f1_marginal"].setdefault("vs_sft", {})[algo] = {
        "lr1e-5_delta": round(mf1[a] - mf1["SFT"], 6),
        "lr5e-5_delta": round(mf1[b] - mf1["SFT"], 6)}
    print(f"  {algo:10s} {mf1[a]:.4f} -> {mf1[b]:.4f}  delta={mf1[b]-mf1[a]:+.4f}"
          f"   vs SFT: {mf1[a]-mf1['SFT']:+.4f} / {mf1[b]-mf1['SFT']:+.4f}")

out["reading"] = (
    "The marginal-CI rule in final_report.py reports that every RL run's "
    "interval overlaps SFT's. Paired on the same 252 rows: "
    + (f"all {len(_five)} runs at lr 5e-5 are significantly below SFT"
       if _five and len(_five_sig) == len(_five) else
       f"{len(_five_sig)} of the {len(_five)} runs at lr 5e-5 are significantly "
       f"below SFT ({', '.join(_five_sig) or 'none'})")
    + "; the lr effect inside a fixed algorithm and group size is significant in "
    + (f"all {len(_lr_all)} cells where both learning rates exist"
       if _lr_all and len(_lr_sig) == len(_lr_all) else
       f"{len(_lr_sig)} of the {len(_lr_all)} cells where both learning rates "
       f"exist ({', '.join(_lr_sig) or 'none'})")
    + "; and "
    + (f"no run at lr 1e-5 is separable from SFT ({len(_one)} tested)"
       if not _one_sig else
       f"{len(_one_sig)} of the {len(_one)} runs at lr 1e-5 differ significantly "
       f"from SFT ({', '.join(_one_sig)})")
    + ". The overlap rule is conservative for independent samples and these are "
      "paired, so it understates differences it should see; the paired test does "
      "not overturn the direction, it supplies the significance the other rule "
      "could not. macro-F1 is not covered by this test.")

# ---------------------------------------------------------------------------
# Group-size series at one algorithm, one IS level and one learning rate.
#
# GSPO, sequence-level, lr 5e-5: G=4 -> 8 -> 32. Every other contrast in this file
# changes at least two things; along this series group size is the only thing that
# moves, and three points can show whether an effect is flat or non-monotone in a
# way two endpoints cannot.
#
# WHAT IT DOES NOT DO: effective batch moves with group size here (these runs were
# launched at their group size without compensating accumulation), so this is a
# series along the paper's configuration axis, not a single-variable experiment.
# The record's one single-variable pair is is_level_paired.
#
# Its own RNG, and the G=8 -> G=32 rung is copied from the block that already
# computed it: sharing the stream would move every CI above, and recomputing would
# let one contrast acquire two different intervals.
# ---------------------------------------------------------------------------
if "GSPO_G4_lr5e5" in correct:
    print()
    print("group-size series, GSPO / sequence-level / lr 5e-5: G=4 -> 8 -> 32 "
          "(group size is the only thing that moves)")
    _rng_series = random.Random(SEED)
    _rungs = {}
    for _lab, _a, _b in [("G=4 -> G=8", "GSPO_G4_lr5e5", "GSPO_G8_lr5e5")]:
        if _a not in correct or _b not in correct:
            continue
        _va, _vb = correct[_a], correct[_b]
        _d = acc_here[_b] - acc_here[_a]
        _lo, _hi = boot_ci(_va, _vb, _rng_series)
        _n01, _n10, _p = mcnemar_exact(_va, _vb)
        _rungs[_lab] = {
            "from": _a, "to": _b, "accuracy_delta": round(_d, 6),
            "delta_ci95_rows": [round(_lo, 6), round(_hi, 6)],
            "excludes_zero": bool(_lo > 0 or _hi < 0),
            "mcnemar": {"lower_wrong_higher_right": _n01,
                        "lower_right_higher_wrong": _n10,
                        "p_exact_two_sided": pval(_p)}}
        print(f"  {_lab:12s} dacc={_d:+.4f} CI=[{_lo:+.4f},{_hi:+.4f}] "
              f"n01={_n01} n10={_n10} p={_p:.5f}")
    if "group_size_paired" in out:
        _g = out["group_size_paired"]
        _rungs["G=8 -> G=32"] = {
            "reused_from": "group_size_paired",
            "from": _g.get("from"), "to": _g.get("to"),
            "accuracy_delta": _g.get("accuracy_delta_g32_minus_g8"),
            "delta_ci95_rows": _g.get("delta_ci95_rows"),
            "excludes_zero": _g.get("excludes_zero"),
            "mcnemar": _g.get("mcnemar")}
    _sig = [k for k, v in _rungs.items() if v.get("excludes_zero")]
    _dl = {k: v.get("accuracy_delta") for k, v in _rungs.items()}
    out["group_size_series_lr5e5"] = {
        "series": "GSPO, sequence-level, lr 5e-5, group size 4 -> 8 -> 32",
        "held_fixed": ["algorithm", "importance_sampling_level", "learning_rate",
                       "data", "rewards", "reward_weights"],
        "moves_with_group_size": ["effective_batch_size"],
        "rungs": _rungs,
        "deltas": _dl,
        "significant_rungs": _sig,
        "reading": ("no rung separates the two group sizes it spans, so along this "
                    "series accuracy is flat in group size from 4 to 32"
                    if not _sig else
                    "the series is not flat: "
                    + ", ".join(f"{k} {_dl[k]:+.4f}" for k in _sig)),
        "does_not_settle": "this is about group size, not IS level. The paper's "
                           "proposition needs a pair differing only in IS level, "
                           "which is is_level_paired; and effective batch moves "
                           "with group size along this series, so it is a series in "
                           "configuration space, not a controlled single-variable "
                           "experiment."}
    # The size of the confound, not just its existence: these runs set gradient
    # accumulation equal to the group size, so effective batch is group size
    # squared and the series moves it by a factor of 64 across its span. Read from
    # the correction record so the figure is the one the rest of the project uses.
    _ebf = REPORTS / "effective_batch_correction.json"
    if _ebf.is_file():
        _by_g = (json.loads(_ebf.read_text(encoding="utf-8"))
                 .get("design_confound", {}).get("effective_batch_by_G", {}))
        if _by_g:
            out["group_size_series_lr5e5"]["effective_batch_by_G"] = _by_g
            out["group_size_series_lr5e5"]["confound_size"] = (
                "gradient accumulation was set equal to the group size in every run "
                "of this series, so effective batch is group size squared: G=4 -> "
                f"{_by_g.get('4')}, G=8 -> {_by_g.get('8')}, G=32 -> "
                f"{_by_g.get('32')}. The series therefore moves group size 8x and "
                "effective batch 64x at the same time, and no contrast in it is "
                "single-variable.")
    if not _sig:
        print("  no rung separates the two group sizes; the series is flat")
    else:
        print("  significant rung(s): " + ", ".join(_sig))

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote " + str(OUT.relative_to(ROOT)))
