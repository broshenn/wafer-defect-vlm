"""Add the three-point group-size series at one lr and one algorithm.

Run 43 (GSPO G=4 lr 5e-5) completes a series that no other set of runs in this
record can form: GSPO, sequence-level, lr 5e-5, at group sizes 4, 8 and 32. Along
it the algorithm, the IS level, the learning rate, the data, the rewards and the
step count are all fixed, so group size is the only thing that moves.

Two points can only say whether two group sizes differ. Three points at one lr can
also say whether the relation is monotone or flat or turns -- and the existing
records contain both a two-point group-size contrast at lr 5e-5 and one at lr 1e-5,
each of which is silent about the shape between the endpoints.

The block this adds:
  * uses its own random.Random(SEED). All bootstrap resamples above come from one
    stream consumed in file order, so drawing here would shift every CI in the
    file. That has already happened once in this project and was caught only
    because a number quoted in the documents moved.
  * copies the G=8 -> G=32 rung out of the block that already computed it instead
    of recomputing it, so one contrast cannot acquire two different intervals.
  * states what it does not establish: effective batch moves with group size along
    the series, so it is a series in the paper's configuration axis, not a
    controlled single-variable experiment.

Inert until run 43's predictions exist: everything is inside one guard on run 43
being present.
"""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/paired_significance.py")
s = P.read_text(encoding="utf-8")

ANCHOR = 'OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")'

BLOCK = '''# ---------------------------------------------------------------------------
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

'''

if s.count(ANCHOR) != 1:
    sys.exit(f"anchor appears {s.count(ANCHOR)} times, expected 1")
if "group_size_series_lr5e5" in s:
    print("already applied; nothing to do")
    sys.exit(0)

s = s.replace(ANCHOR, BLOCK + ANCHOR)
ast.parse(s)
P.write_text(s, encoding="utf-8")
print("paired_significance.py: three-point group-size series added (own RNG, inert "
      "until run 43 lands)")
