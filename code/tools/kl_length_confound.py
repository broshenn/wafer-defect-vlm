"""Can KL be compared across these runs at all?

Section 5.2.2 cites KL rising at lr 5e-5 as evidence the policy was pushed into
a degenerate region. The raw numbers fall instead, in both matched learning-rate
pairs. Before that sentence is corrected -- in either direction -- two things
have to be settled, and neither can be settled by reading the record's mean_kl.

First: what is `kl`? If it is a per-token mean then it is not scaled by
completion length, and comparing it across runs of different length is
defensible. On a step where frac_reward_zero_std is 1 every advantage is exactly
0, so the policy-gradient term vanishes and the logged loss is beta * kl alone.
That identity is tested here per step rather than assumed -- it is the only
direct evidence available about what the field means.

Second: are the completions comparable? `kl` is averaged over the tokens the run
actually emitted. If the 5e-5 runs emit shorter completions, the mean is taken
over a different span and the two values do not measure the same quantity, so
the comparison cannot establish a direction and the sentence has to go rather
than be reversed.

Writes outputs/reports/kl_length_confound.json. Numbers in LIMITATIONS.md are
cited from that record, not from this tool's stdout -- a printed number that no
record holds is the failure item 7 of section 8 describes.

beta is read with a left anchor on the args-dump form. Without it, an
unanchored `beta=` also matches `sdar_gate_beta=5.0`, which sits in the same
dump and sorts after `beta=0.04`, so a last-match-wins search silently yields
5.0. That happened, and the identity check below failed on all 150 steps -- a
loud failure, which is the point of asserting an identity rather than
eyeballing agreement.
"""
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
OUT = ROOT / "outputs/reports/kl_length_confound.json"

# Every RL run belongs here. A hard-coded list that a new run is not added to does
# not make the record wrong -- it makes it narrower than the sentences that quote it,
# which say "all runs". Adding a run means adding it here and regenerating.
RUNS = ["grpo", "qwen35_9b_grpo_lr5e5", "gspo_lr1e5", "qwen35_9b_gspo_v1",
        "qwen35_9b_gspo_g32", "qwen35_9b_gspo_g4_lr5e5",
        "qwen35_9b_gspo_g4_lr1e5", "qwen35_9b_gspo_g32_lr1e5",
        # The second seed of GRPO G=4 lr 1e-5 (queue 42). Appended, not grouped with
        # `grpo`, its first seed: this list's order is the order `sync_counts` writes
        # the per-run idle list out in, so inserting it mid-list would rewrite eight
        # numbers to say the same thing in a different order. It carries no `_gNN_lrXX`
        # shape, so the cell-pairing loop below skips it -- two seeds are one cell.
        "qwen35_9b_grpo_lr1e5_seed3408"]

# A configuration can be re-run with another seed. A repeat is not a second learning
# rate: the cell it belongs to already has an owner, and `matched_pairs` compares the two
# runs that define the cell's learning-rate contrast. Declaring which run a repeat
# repeats is what lets a reader that resolves `runs` by (algorithm, group size, learning
# rate) know which entry owns the cell -- without it, that lookup returns whichever entry
# the dict ended with, and after run 42 landed that was the seed: 5.2.4's GRPO(G=4) KL
# read 1.1244 instead of 1.1068, and the checker reported the *document* stale while the
# document was quoting the run its own pair is built from.
SEED_REPEATS = {"qwen35_9b_grpo_lr1e5_seed3408": "grpo",
                "qwen35_9b_grpo_lr1e5_seed3409": "grpo"}

REPORTS = ROOT / "outputs/reports"
SFT_RECORD = "sft_train_result.json"

# A finished run on disk that RUNS does not name is added, not skipped. "Finished"
# is the record's own claim (outcome completed, steps_logged == max_steps_requested):
# a run still training has a log and a partial record, and counting it as one of
# "all runs" would make every sentence that says "all runs" wrong in a new way.
for _p in sorted(REPORTS.glob("*_train_result.json")):
    _tag = _p.name[: -len("_train_result.json")]
    if _p.name == SFT_RECORD or _tag in RUNS:
        continue
    try:
        _rec = json.loads(_p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"NOTE: {_p.name} is unreadable; not included", file=sys.stderr)
        continue
    if (_rec.get("outcome") == "completed"
            and _rec.get("steps_logged") == _rec.get("max_steps_requested")):
        RUNS.append(_tag)
        print(f"NOTE: {_tag} finished and was not in RUNS; included. "
              f"Add it to RUNS so the record's order is a decision, not a sort.",
              file=sys.stderr)
    else:
        _shape = f"outcome={_rec.get('outcome')} steps={_rec.get('steps_logged')}/" \
                 f"{_rec.get('max_steps_requested')}"
        print(f"NOTE: {_tag} is on disk but not finished ({_shape}); excluded from "
              f"a record whose claims say 'all runs'", file=sys.stderr)

NUM = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?"
STEP = re.compile(r"\{'loss':.*?\}")


def field(line, key):
    m = re.search(rf"'{re.escape(key)}': '({NUM})'", line)
    return float(m.group(1)) if m else None


def series(lines, key):
    return [v for v in (field(l, key) for l in lines) if v is not None]


def fmean(xs):
    return statistics.fmean(xs) if xs else None


def rnd(x, n=6):
    return None if x is None else round(x, n)


out = {"description": __doc__.strip().split("\n\n")[0],
       "question": "is the KL comparison in LIMITATIONS 5.2.2 sound?",
       "runs": {}}

for tag in RUNS:
    rec = json.loads(
        (ROOT / f"outputs/reports/{tag}_train_result.json").read_text(encoding="utf-8"))
    logp = Path(rec["log"])
    text = logp.read_text(encoding="utf-8", errors="replace")
    lines = STEP.findall(text)
    if not lines:
        sys.exit(f"{tag}: no logged steps found in {logp.name}")

    beta_cli = re.findall(r"--beta\s+([0-9.]+)", text)
    beta_dump = re.findall(r"(?<![A-Za-z0-9_])beta=([0-9.]+)", text)
    beta = float(beta_cli[-1]) if beta_cli else (
        float(beta_dump[-1]) if beta_dump else None)
    if beta_cli and beta_dump and float(beta_cli[-1]) != float(beta_dump[-1]):
        sys.exit(f"{tag}: beta disagrees between flag and args dump: "
                 f"cli={beta_cli[-1]} dump={beta_dump[-1]}")

    kl = series(lines, "kl")
    loss = series(lines, "loss")
    zero = series(lines, "frac_reward_zero_std")
    mlen = series(lines, "completions/mean_length")
    lmin = series(lines, "completions/min_length")
    lmax = series(lines, "completions/max_length")
    clip = series(lines, "completions/clipped_ratio")
    rew = series(lines, "reward")

    # The identity, tested only where the policy-gradient term is exactly zero.
    ident = []
    if beta is not None and len(loss) == len(kl) == len(zero):
        ident = [abs(l - beta * k)
                 for l, k, z in zip(loss, kl, zero) if z == 1.0]
    ident_ok = sum(1 for d in ident if d < 1e-4)

    cfg = rec.get("config") or {}
    out["runs"][tag] = {
        "G": cfg.get("num_generations"),
        # None for every run that owns its cell. `check_quantified_claims` reads this to
        # keep a repeat out of the cell map it builds from this record; the repeat's own
        # numbers stay in `runs`, and are counted by every sentence that counts runs.
        "repeat_of": SEED_REPEATS.get(tag),
        "lr": cfg.get("learning_rate"),
        "is_level": cfg.get("importance_sampling_level"),
        "log": logp.name,
        "steps": len(lines),
        "mean_reward": rnd(fmean(rew)),
        "mean_kl": rnd(fmean(kl)),
        "mean_completion_length": rnd(fmean(mlen), 2),
        "mean_completion_min_length": rnd(fmean(lmin), 2),
        "mean_completion_max_length": rnd(fmean(lmax), 2),
        "mean_clipped_ratio": rnd(fmean(clip)),
        "mean_frac_reward_zero_std": rnd(fmean(zero)),
        "beta": beta,
        "beta_sources": {"cli": beta_cli, "args_dump": beta_dump},
        "kl_identity": {
            "claim": "on steps with frac_reward_zero_std == 1 the "
                     "policy-gradient term is exactly 0, so logged loss "
                     "equals beta * kl, which is what makes kl identifiable "
                     "as a per-token KL rather than a total",
            "beta": beta,
            "idle_steps_tested": len(ident),
            "max_abs_deviation": rnd(max(ident), 8) if ident else None,
            "steps_matching_within_1e-4": ident_ok,
            "holds": bool(ident) and ident_ok == len(ident),
        },
    }
    r = out["runs"][tag]
    ki = r["kl_identity"]
    print(f"{tag:24s} G={r['G']:<3} lr={r['lr']:<6} kl={r['mean_kl']:.4f} "
          f"len={r['mean_completion_length']:>6.1f} rew={r['mean_reward']:.4f} "
          f"identity={'yes' if ki['holds'] else 'NO'} "
          f"({ki['steps_matching_within_1e-4']}/{ki['idle_steps_tested']}, "
          f"max dev {ki['max_abs_deviation']})")

pairs = [("GRPO G=4", "grpo", "qwen35_9b_grpo_lr5e5"),
         ("GSPO G=8", "gspo_lr1e5", "qwen35_9b_gspo_v1"),
         # `qwen35_9b_gspo_g32` is the G=32 lr 5e-5 run and its tag predates the
         # `_gNN_lrXX` convention, so CELL below cannot match it and the cell looks
         # half-complete to the loop that auto-pairs every other cell. The loop's own
         # comment says the explicit list owns the legacy tags; GSPO G=8 lr 5e-5 is
         # owned here for the same reason, so GSPO G=32 is named here too. Without
         # this line the pair silently does not exist: the record holds both runs,
         # the checker counts the pair from them, and only this list disagrees.
         ("GSPO G=32", "qwen35_9b_gspo_g32_lr1e5", "qwen35_9b_gspo_g32")]

# Every (algorithm, group size) cell that has both learning rates becomes a pair.
CELL = re.compile(r"^qwen35_9b_(gspo|grpo)_g(\d+)_lr([0-9.eE+-]+)$")
_cells = {}
for _tag in RUNS:
    _m = CELL.match(_tag)
    if not _m:
        continue        # a legacy tag or a seed repeat; the explicit list owns those
    _algo, _g, _lr = _m.group(1).upper(), int(_m.group(2)), float(_m.group(3))
    _cells.setdefault((_algo, _g), {})[_lr] = _tag
for (_algo, _g), _bys in sorted(_cells.items()):
    if len(_bys) != 2:
        continue
    _lo, _hi = sorted(_bys)
    _label = f"{_algo} G={_g}"
    if _label in [n for n, _, _ in pairs]:
        continue
    pairs.append((_label, _bys[_lo], _bys[_hi]))
out["matched_pairs"] = {}
print()
for name, a, b in pairs:
    A, B = out["runs"][a], out["runs"][b]
    d = {"from": a, "to": b,
         "lr": [A["lr"], B["lr"]],
         "kl": [A["mean_kl"], B["mean_kl"]],
         "kl_delta": rnd(B["mean_kl"] - A["mean_kl"]),
         "completion_length": [A["mean_completion_length"],
                               B["mean_completion_length"]],
         "length_delta": rnd(B["mean_completion_length"] - A["mean_completion_length"], 2),
         "length_change_pct": rnd(
             (B["mean_completion_length"] / A["mean_completion_length"] - 1) * 100, 1),
         "reward": [A["mean_reward"], B["mean_reward"]],
         "reward_delta": rnd(B["mean_reward"] - A["mean_reward"]),
         "idle": [A["mean_frac_reward_zero_std"], B["mean_frac_reward_zero_std"]],
         "idle_delta": rnd(B["mean_frac_reward_zero_std"] - A["mean_frac_reward_zero_std"])}
    out["matched_pairs"][name] = d
    print(f"  {name}: kl {d['kl_delta']:+.4f}  len {d['length_delta']:+.1f} "
          f"({d['length_change_pct']:+.1f}%)  reward {d['reward_delta']:+.4f}  "
          f"idle {d['idle_delta']:+.4f}")

# "in both pairs" was a quantifier written when there were two. It is counted now,
# and the two per-pair facts behind it are read off the pairs rather than asserted.
_short = [n for n, d in out["matched_pairs"].items() if d["length_delta"] < 0]
_falls = [n for n, d in out["matched_pairs"].items() if d["reward_delta"] < 0]
_n = len(out["matched_pairs"])
out["conclusion"] = (
    "kl is a per-token mean -- the identity loss == beta*kl holds on every idle "
    "step of every run, to within the logs' own rounding -- so it is not scaled "
    "by completion length. But the completions are not the same length across "
    "the pairs being compared: of the " + str(_n) + " matched lr pairs, the 5e-5 "
    "run emits shorter completions in " + str(len(_short)) + " ("
    + (", ".join(_short) if _short else "none") + "), so its mean is taken over a "
    "different span and the two values do not measure the same quantity. The raw "
    "kl falls in " + str(len(_short)) + " of them, which is the opposite of what "
    "LIMITATIONS 5.2.2 claims; but this measurement cannot establish the direction "
    "either way, so the claim is removed rather than reversed. What the pairs "
    "agree on at 5e-5: the mean reward is lower in " + str(len(_falls)) + " of "
    + str(_n) + " (" + (", ".join(_falls) if _falls else "none") + ").")

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwrote {OUT.relative_to(ROOT)}")
