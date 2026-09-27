"""Every fact about the run set that the document's prose depends on, in one place.

`check_quantified_claims.py` answers "did a sentence stop being true" -- it prints the
sentences that moved and stops. That is the right behaviour for a gate, and the wrong
shape for writing the replacement text: to rewrite a passage that names four runs you
need the values of all four, not only the two that mismatched, and reading them out of
four separate mismatch blocks is how a rewrite ends up quoting a stale neighbour.

So this prints the whole picture and writes nothing: the macro-F1 ordering, which runs
lose which class, who holds each extreme in the profile table, which reward row each
column is lowest in. Both are read from the same records the checker reads, so a
reported fact here and a finding there cannot disagree.

Not a tool for editing. The one kind of claim these sentences make that no record can
settle -- is the winner still *unique* -- is left to be read off the numbers below.

    venvs/wafer/bin/python tools/fact_digest.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
REP = ROOT / "outputs/reports"

CLASSES = ("Donut", "none", "Edge_Ring", "Scratch")
REWARDS = ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock")

# Same map as check_quantified_claims.py, and it has to stay the same map: two lists of
# what counts as an RL run drifting apart is the defect both tools exist to catch.
RUNS = {
    "SFT": "qwen35_9b_adapter",
    "GRPO(G=4) lr1e-5": "qwen35_9b_grpo",
    "GRPO(G=4) lr5e-5": "qwen35_9b_grpo_lr5e5",
    "GSPO(G=4) lr5e-5": "qwen35_9b_gspo_g4_lr5e5",
    "GSPO(G=8) lr1e-5": "gspo_lr1e5",
    "GSPO(G=8) lr5e-5": "qwen35_9b_gspo_v1",
    "GSPO(G=32) lr5e-5": "qwen35_9b_gspo_g32",
    "GSPO(G=4) lr1e-5": "qwen35_9b_gspo_g4_lr1e5",
    "GSPO(G=32) lr1e-5": "qwen35_9b_gspo_g32_lr1e5",
}

# Two runs predate the `{tag}_train_result.json` convention and named their record
# after the algorithm. Kept in step with check_quantified_claims.py on purpose: if the
# two lists of what counts as a run's record drift apart, the digest and the gate
# disagree and neither is evidence.
TRAIN_RESULT = {
    "GRPO(G=4) lr1e-5": "grpo_train_result",
    "SFT": "sft_train_result",
}

rec = {}
for name, tag in RUNS.items():
    p = REP / f"{tag}__report.json"
    if not p.is_file():
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    rec[name] = {
        "macro_f1": d["classification"]["macro_f1"],
        "per_class": {k: v["f1-score"] for k, v in d["classification"]["per_class"].items()},
        "radial_zone": d["structured"]["field_accuracy"]["radial_zone"],
        "clock_mae": d["structured"]["clock_circular_mae_sectors"],
        "size_mae": d["structured"]["size_mae_r"],
        "must_hit": d["caption"]["must_hit_rate"],
    }

order = [n for n in RUNS if n in rec]
rl = [n for n in order if n != "SFT"]
print(f"runs on disk: {len(order)} ({len(rl)} RL + SFT); "
      f"absent: {sorted(set(RUNS) - set(order)) or 'none'}\n")

sft = rec["SFT"]

print("== macro-F1, high to low ==")
for n in sorted(order, key=lambda x: -rec[x]["macro_f1"]):
    mark = ""
    if n != "SFT":
        mark = "  <-- above SFT" if rec[n]["macro_f1"] > sft["macro_f1"] else ""
    print(f"  {rec[n]['macro_f1']:.4f}  {n}{mark}")
print(f"  the RL runs above SFT: "
      f"{[n for n in rl if rec[n]['macro_f1'] > sft['macro_f1']] or ['(none)']}")
print(f"  lowest RL macro-F1: "
      f"{min(rl, key=lambda n: rec[n]['macro_f1'])} "
      f"{min(rec[n]['macro_f1'] for n in rl):.4f}")

print("\n== per-class F1: classes at zero, and the lr 1e-5 runs' intact pair ==")
for n in order:
    z = [c for c in CLASSES if rec[n]["per_class"].get(c, 1.0) == 0.0]
    tag = f"  ZEROES {z}" if z else ""
    print(f"  {n:22s}{tag}")
low = [n for n in rl if n.endswith("lr1e-5")]
print(f"  lr 1e-5 RL runs ({len(low)}): {'、'.join(low)}")
for n in low:
    pc = rec[n]["per_class"]
    print(f"    {n:22s} none {pc.get('none', float('nan')):.3f}  "
          f"Donut {pc.get('Donut', float('nan')):.3f}"
          + ("   <-- keeps both" if all(pc.get(c, 0.0) > 0.0 for c in CLASSES) else ""))
zeroed = [n for n in rl if any(rec[n]["per_class"].get(c, 1.0) == 0.0 for c in CLASSES)]
print(f"  RL runs with a zeroed class: {len(zeroed)} ({'、'.join(zeroed) or 'none'})")

print("\n== the four profile-table extremes (holder, value, ratio to SFT) ==")
for key, best, label in (("radial_zone", max, "radial_zone (highest)"),
                         ("clock_mae", min, "clock MAE (lowest)"),
                         ("size_mae", max, "size MAE (highest = worst)"),
                         ("size_mae", min, "size MAE (lowest = best)"),
                         ("must_hit", min, "caption must-hit (lowest)")):
    vals = {n: rec[n][key] for n in order}
    w = best(vals, key=vals.get)
    print(f"  {label:28s} {w:22s} {vals[w]:.4f}   "
          f"ratio to SFT {vals[w] / sft[key]:.2f}   "
          f"runner-up {sorted(vals.values())[1 if best is min else -2]:.4f}")

print("\n== reward rows: which column is lowest in each ==")
for k in REWARDS:
    vals = {}
    for n, tag in RUNS.items():
        if n == "SFT":
            continue
        p = REP / f"{TRAIN_RESULT.get(n, tag + '_train_result')}.json"
        if p.is_file():
            vals[n] = json.loads(p.read_text(encoding="utf-8"))["reward_signal"][k][
                "mean_std_across_steps"]
    if not vals:
        continue
    w = min(vals, key=vals.get)
    print(f"  {k:14s} lowest: {w:22s} {vals[w]:.4f}  "
          f"(over {len(vals)} columns)")
