"""Recompute the zero-advantage-step table from the training logs.

Section 5.2.2 of LIMITATIONS.md quotes a rate per run -- 32.67%, 38.00%, 42.00%,
36.00%, 15.33% -- with the fraction behind it (57/150). Transcribing those by hand
is the risk this project keeps finding: a wrong digit that nothing can detect.

Each run wrote two logs (an inner trainer log and an outer wrapper log) and a
`mean_frac_reward_zero_std` field in its own result. That gives three independent
paths to one number. This prints all of them and requires them to agree, so a rate
is quoted only when the logs agree with each other and with the record -- and a
disagreement is reported, never averaged away.

Nothing here is inferred from filenames. The log names are not uniform across runs
(GRPO G=4 lr5e-5 has only an outer log, GRPO G=4 lr1e-5 only an inner one), and the
train_result name is not uniform either (that run's is `grpo_train_result.json`
while its report is `qwen35_9b_grpo__report.json`). A filename-derived guess would
attach a log to the wrong run and look exactly like a correct answer, so every path
is declared below. A log on disk with a real number of steps that is not declared
here is printed as unaccounted rather than skipped.
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
LOGS = ROOT / "logs"
REPORTS = ROOT / "outputs/reports"

VALUE = re.compile(r"'frac_reward_zero_std': '([-0-9.eE]+)'")
MIN_STEPS = 10          # below this it is a smoke test, not a run

# label, train_result filename, [log files]
RUNS = [
    ("GRPO G=4 lr1e-5", "grpo_train_result.json",
     ["24_grpo_qwen35_9b_grpo_v1.log"]),
    ("GRPO G=4 lr5e-5", "qwen35_9b_grpo_lr5e5_train_result.json",
     ["29_qwen35_9b_grpo_lr5e5_train.log"]),
    ("GRPO G=4 lr1e-5 seed3408", "qwen35_9b_grpo_lr1e5_seed3408_train_result.json",
     ["24_grpo_qwen35_9b_grpo_lr1e5_seed3408.log",
      "29_qwen35_9b_grpo_lr1e5_seed3408_train.log"]),
    # Declared before it exists: an entry whose log has no values yet is printed as
    # "no log with values yet" and left out of the row list, so the document's rate list
    # grows only when there is a rate to grow it with. Declared afterwards instead, this
    # run's log would be reported UNACCOUNTED -- this table's way of saying a run exists
    # that nothing measured.
    ("GRPO G=4 lr1e-5 seed3409", "qwen35_9b_grpo_lr1e5_seed3409_train_result.json",
     ["24_grpo_qwen35_9b_grpo_lr1e5_seed3409.log"]),
    ("GSPO G=8 lr5e-5", "qwen35_9b_gspo_v1_train_result.json",
     ["24_grpo_qwen35_9b_gspo_v1.log", "29_grpo_train.log"]),
    ("GSPO G=8 lr1e-5", "gspo_lr1e5_train_result.json",
     ["24_grpo_gspo_lr1e5.log", "29_gspo_lr1e5_train.log"]),
    ("GSPO G=32 lr5e-5", "qwen35_9b_gspo_g32_train_result.json",
     ["24_grpo_qwen35_9b_gspo_g32.log", "29_qwen35_9b_gspo_g32_train.log"]),
    ("GSPO G=32 lr1e-5", "qwen35_9b_gspo_g32_lr1e5_train_result.json",
     ["24_grpo_qwen35_9b_gspo_g32_lr1e5.log",
      "29_qwen35_9b_gspo_g32_lr1e5_train.log"]),
    ("GSPO G=4 lr5e-5", "qwen35_9b_gspo_g4_lr5e5_train_result.json",
     ["24_grpo_qwen35_9b_gspo_g4_lr5e5.log",
      "29_qwen35_9b_gspo_g4_lr5e5_train.log"]),
    ("GSPO G=4 lr1e-5", "qwen35_9b_gspo_g4_lr1e5_train_result.json",
     ["24_grpo_qwen35_9b_gspo_g4_lr1e5.log",
      "29_qwen35_9b_gspo_g4_lr1e5_train.log"]),
]


def rate_of(p):
    """(steps with zero advantage, steps logged) for one log, or None."""
    if not p.is_file():
        return None
    vals = [float(v) for v in VALUE.findall(
        p.read_text(encoding="utf-8", errors="replace"))]
    if len(vals) < MIN_STEPS:
        return None
    return sum(1 for v in vals if v >= 1.0), len(vals)


def main() -> int:
    declared = {n for _, _, logs in RUNS for n in logs}
    unaccounted = sorted(
        p.name for p in LOGS.glob("*.log")
        if p.name not in declared and (rate_of(p) or (0, 0))[1] >= MIN_STEPS)
    if unaccounted:
        print("UNACCOUNTED logs on disk with a real number of steps:")
        for u in unaccounted:
            print(f"  {u}")
        print("  A run missing from this table would look exactly like a run that "
              "was never measured.\n")

    print(f"{'run':<26} {'logs (zero-adv/total)':<32} {'rate':>7} "
          f"{'record':>7}  verdict")
    print("-" * 86)
    rows, bad = [], []
    for label, tr_name, logs in RUNS:
        got = [(n, rate_of(LOGS / n)) for n in logs]
        got = [(n, r) for n, r in got if r]
        if not got:
            print(f"{label:<26} {'no log with values yet':<32}")
            continue
        pairs = {r for _, r in got}
        shown = ", ".join(f"{n1}/{n}" for _, (n1, n) in got)
        if len(pairs) > 1:
            bad.append(f"{label}: the logs disagree with each other: {shown}")
            print(f"{label:<26} {shown:<32} {'':>7} {'':>7}  LOGS DISAGREE")
            continue
        n1, n = got[0][1]
        rate = n1 / n
        rp = REPORTS / tr_name
        rec = (json.loads(rp.read_text(encoding="utf-8")).get(
            "mean_frac_reward_zero_std") if rp.is_file() else None)
        if rec is None:
            print(f"{label:<26} {shown:<32} {rate:>6.2%} {'--':>7}  "
                  f"still training; rate is over the steps logged so far")
            continue
        if abs(rate - rec) < 1e-9:
            print(f"{label:<26} {shown:<32} {rate:>6.2%} {rec:>7.2%}  match")
            rows.append((label, n1, n, rate))
        else:
            bad.append(f"{label}: log says {rate:.4%}, record says {rec:.4%}")
            print(f"{label:<26} {shown:<32} {rate:>6.2%} {rec:>7.2%}  MISMATCH")

    print()
    print(f"{len(rows)} run(s) verified: every log agrees with every other log of "
          f"that run and with its recorded value." if not bad else
          f"{len(bad)} PROBLEM(S):")
    for b in bad:
        print("  " + b)
    if bad:
        print("  Do not quote these numbers until resolved.")
    print("\nsection 5.2.2 cells, in this table's order:")
    print("| " + " | ".join(f"{r[3]:.2%}" for r in rows) + " |")
    print("| " + " | ".join(f"({r[1]}/{r[2]})" for r in rows) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
