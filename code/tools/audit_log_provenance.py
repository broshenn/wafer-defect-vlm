"""Does each run record's cited log actually belong to that run?

Every train record names a `log`, and until now nothing checked that the log is
the run's own. Two ways that goes wrong, both live in this project:

  * projects/wafer-defect-vlm/scripts/29_grpo_train.sh defaults RUN_TAG to the
    bare string "grpo", so a run launched without an explicit tag writes to
    29_grpo_train.log -- the same path an earlier run used. The GSPO-v1 run did
    exactly this, overwriting the original GRPO run's 29_ log. No record cites
    that path today, so nothing on disk is wrong; the hazard is latent, not
    realised.
  * The log a record cites can be empty. In that case a reader notices. In the
    first the file is full, the numbers are real, and they belong to another
    run -- the failure mode section 8 of LIMITATIONS.md exists to list.

Both flag forms are searched, because ms-swift prints its command line
(`--learning_rate 5e-5`) and its parsed-args dump (`learning_rate=5e-05`). The
dump form is anchored on the left, because names ending in these keys exist:
`sdar_gate_beta=5.0` and `data_seed=42` both appear in the dump, and an
unanchored `seed=` search matches `data_seed=42` as well. That search currently
returns the right value only because the dump is printed in alphabetical order,
so `seed=3407` happens to sort after `data_seed=42` and last-match-wins picks it.
Depending on an ordering accident is not a check.

Fields the log does not state at all are reported as UNVERIFIED rather than
passed. Searching only the command-line form once made this tool report
"agrees" for a field it had simply failed to find -- absent evidence read as
agreement, the same bug the tool was written to catch.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
REPORTS = ROOT / "outputs" / "reports"

# `(?<![A-Za-z0-9_])` stops `data_seed=` matching the `seed=` alternative.
PATTERNS = {
    "num_generations":
        r"(?:--num_generations\s+|(?<![A-Za-z0-9_])num_generations=)(\d+)",
    "learning_rate":
        r"(?:--learning_rate\s+|(?<![A-Za-z0-9_])learning_rate=)([0-9][0-9.eE+-]*)",
    "importance_sampling_level":
        r"(?:--importance_sampling_level\s+|(?<![A-Za-z0-9_])importance_sampling_level=)(\w+)",
    "seed":
        r"(?:--seed\s+|(?<![A-Za-z0-9_])seed=)(\d+)",
}
COMPARED = ("num_generations", "learning_rate", "importance_sampling_level", "seed")


def tail_args(text):
    return {k: (re.findall(p, text) or [None])[-1] for k, p in PATTERNS.items()}


def norm(v):
    if v is None:
        return None
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


problems, unverified, recoverable = [], [], []
checked = 0
seed_evidence = []

for p in sorted(REPORTS.glob("*_train_result.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("steps_logged") is None:
        continue                        # SFT record: no RL args to match
    checked += 1
    cfg = d.get("config") or {}
    cited = d.get("log")
    if not cited:
        problems.append((p.name, "no log field"))
        continue
    lp = Path(cited)
    if not lp.is_file():
        problems.append((p.name, f"log does not exist: {cited}"))
        continue
    if lp.stat().st_size == 0:
        problems.append((p.name, f"log is EMPTY: {lp.name}"))
        continue

    found = tail_args(lp.read_text(encoding="utf-8", errors="replace"))
    mism, missing = [], []
    for k in COMPARED:
        want, got = norm(cfg.get(k)), norm(found.get(k))
        if got is None:
            missing.append(k)           # cannot verify: say so, do not pass
        elif want is None:
            recoverable.append((p.name, k, got))
        elif want != got:
            mism.append(f"{k}: record={want} log={got}")
    if found.get("seed"):
        seed_evidence.append((p.name, cfg.get("seed"), found["seed"],
                              cfg.get("seed_source") is not None))

    if mism:
        problems.append((p.name, f"{lp.name}: " + "; ".join(mism)))
    elif missing:
        unverified.append((p.name, f"{lp.name}: " + ", ".join(missing)))
    else:
        print(f"  ok   {p.name:44s} -> {lp.name}")

print()
if recoverable:
    print("record field empty, log states it (recoverable, not a mismatch):")
    for name, k, v in recoverable:
        print(f"  ~    {name:44s} {k}={v}")
    print()
if unverified:
    print("UNVERIFIED -- log does not state these, so agreement is untested:")
    for name, why in unverified:
        print(f"  ?    {name:44s} {why}")
    print()
if seed_evidence:
    print("seed: record vs log, and whether the record flagged it reconstructed")
    for name, rec_seed, log_seed, flagged in seed_evidence:
        mark = "ok " if norm(rec_seed) == norm(log_seed) else "!! "
        note = " (record marks it reconstructed)" if flagged else ""
        print(f"  {mark} {name:44s} record={rec_seed} log={log_seed}{note}")
    print()
if problems:
    print("MISMATCHES:")
    for name, why in problems:
        print(f"  !!   {name:44s} {why}")
    sys.exit(2)

print(f"{checked} RL records checked; none cites another run's log")
