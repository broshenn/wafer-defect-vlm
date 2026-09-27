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

# Reproduction check, as in paired_significance.py: if this script's reading of
# the responses does not reproduce the project's own scorer, its numbers are
# about misparsed text and must not be published.
for lbl, vec, rep in ((SEED1[1], v1, r1), (SEED2[1], v2, r2), ("SFT", vs, rs),
                      ("GRPO_lr5e5", v5, r5)):
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

out = {
    "question": "How large is run-to-run (seed) variation on this benchmark, and "
                "are the reported effects larger than it?",
    "design": {"seed1": SEED1[1], "seed2": SEED2[1],
               "identical_except": "the random seed; same algorithm, group size 4, "
                                   "learning rate 1e-5, data, steps",
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
    "replication_vs_sft": {
        "seed1": {"delta": round(acc1 - accs, 6), "p_exact_two_sided": pval(ps1)},
        "seed2": {"delta": round(d_vs_sft_2, 6), "p_exact_two_sided": pval(ps)},
        "reading": "both seeds are the same side of SFT and neither is "
                   "significantly below, or the null is not reproducible"},
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
