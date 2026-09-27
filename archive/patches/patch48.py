"""Backfill the training seed into records written before it was recorded.

The second GRPO seed run is only worth doing if it can be shown to differ from
the first. The seed is now forwarded to ms-swift and written into the run record
for that reason -- but the five runs already on disk were launched before that
change, so none of their records carries a seed, and the check would report
"cannot prove the runs differ" instead of proving it.

The value is not a guess. Until it was made overridable, the launcher passed a
literal `--seed 3407` (projects/wafer-defect-vlm/scripts/24_train_grpo.sh line
115, still visible at commit b1092dc, the last commit in which the flag was a
literal). No launcher set a seed any other way, so every run that went through
that script used 3407. That is why this writes a value at all rather than
leaving the fields empty: the alternative is a check that cannot fail, which is
worth less than no check.

The backfilled records say so. A reader must be able to tell a number the run
logged from a number reconstructed afterwards, or the difference between the two
is exactly what this record is supposed to preserve.
"""
import json
import sys
from pathlib import Path

REPORTS = Path("/root/autodl-fs/wafer-vlm/outputs/reports")
SEED = 3407
TARGETS = ["grpo", "gspo_lr1e5", "qwen35_9b_gspo_g32", "qwen35_9b_gspo_v1"]
NOTE = ("Reconstructed, not logged: the launcher passed a literal --seed 3407 "
        "(projects/wafer-defect-vlm/scripts/24_train_grpo.sh, line 115 at commit "
        "b1092dc, the last commit in which the flag was still a literal) and no "
        "launcher set a seed any other way, so every run through that script "
        "used 3407. Written here so a later replication run can be shown to "
        "differ from its original rather than merely asserted to.")

changed = 0
for name in TARGETS:
    path = REPORTS / f"{name}_train_result.json"
    if not path.is_file():
        sys.exit(f"missing record: {path}")
    d = json.loads(path.read_text(encoding="utf-8"))
    config = d.setdefault("config", {})
    if config.get("seed") == SEED:
        print(f"  {name}: already carries seed {SEED}")
        continue
    if config.get("seed") not in (None, ""):
        sys.exit(f"REFUSING: {name} already has a different seed "
                 f"{config['seed']!r}; this backfill would overwrite it")
    config["seed"] = SEED
    config["seed_source"] = NOTE
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {name}: seed {SEED} backfilled, marked as reconstructed")
    changed += 1

print(f"\n{changed} records updated")
