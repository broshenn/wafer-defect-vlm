"""Point each RL record at its own training log, and measure the idle-step share.

The outer log (logs/29_<tag>_train.log) is written by the launcher; the first
GSPO run reused the tag `grpo`, so logs/29_grpo_train.log now holds GSPO output
while outputs/reports/grpo_train_result.json still names it as its log. Anything
that reads that field to measure the GRPO run would silently measure GSPO
instead -- the same "plausible but wrong" failure this project keeps producing.

The per-run inner log (logs/24_grpo_<RUN>.log) is not shared, so it is the
correct artefact. Repoint both records at theirs and recompute the idle share
directly, so the numbers in the report come from a file rather than from prose.
"""
import json
import re
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
REPORTS = ROOT / "outputs/reports"
LOGS = ROOT / "logs"

# record -> the log that actually belongs to it
PAIRS = [
    ("grpo_train_result.json", LOGS / "24_grpo_qwen35_9b_grpo_v1.log"),
    ("gspo_lr5e5_train_result.json", LOGS / "24_grpo_qwen35_9b_gspo_v1.log"),
]
# present only once the second GSPO run finishes
PAIRS.append(("gspo_lr1e5_train_result.json", LOGS / "24_grpo_gspo_lr1e5.log"))


def idle_fraction(log: Path):
    if not log.is_file():
        return None, 0
    vals = [float(v) for v in re.findall(
        r"'frac_reward_zero_std': '([-0-9.eE]+)'",
        log.read_text(encoding="utf-8", errors="replace"))]
    return ((sum(vals) / len(vals)) if vals else None), len(vals)


print(f"{'record':38} {'log':38} {'rows':>5} {'zero-adv':>9}")
for name, log in PAIRS:
    path = REPORTS / name
    if not path.is_file():
        print(f"{name:38} MISSING")
        continue
    frac, rows = idle_fraction(log)
    print(f"{name:38} {log.name:38} {rows:>5} "
          f"{(f'{frac:.1%}' if frac is not None else 'n/a'):>9}")

    data = json.loads(path.read_text(encoding="utf-8"))
    old = data.get("log")
    if old != str(log):
        data["log"] = str(log)
        data["log_note"] = (
            "Repointed from logs/29_<tag>_train.log, which the launcher shares "
            "between runs: the first GSPO run reused the tag `grpo` and "
            "overwrote it, so that file no longer contains this run's output. "
            "This per-run inner log is the artefact the numbers come from.")
    if frac is not None:
        data["mean_frac_reward_zero_std"] = frac
        data["frac_reward_zero_std_note"] = (
            "Share of logged optimizer steps where every generation in the group "
            "scored identically, so the advantage was zero and no gradient was "
            "produced. Recomputed from the per-run log, not copied from prose.")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nrecords repointed and idle share stored in `mean_frac_reward_zero_std`")
