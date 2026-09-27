"""Remove the `gspo_lr5e5` alias: one run must not have two names.

When the first GSPO run finished, I copied its outputs to `gspo_lr5e5__report.json`
/ `gspo_lr5e5.jsonl` / `gspo_lr5e5_train_result.json` for readability, and then
referenced the alias in some places and the original run name in others. Those
files are byte-identical copies of `qwen35_9b_gspo_v1__*`, so every reference is
ambiguous about which one is canonical -- the same shape of problem as the shared
log that made a GRPO lookup read GSPO's numbers, and the shared RESULT path that
overwrote the GRPO record.

Run 1 keeps its original stem `qwen35_9b_gspo_v1`, which is the name of its
checkpoint directory. Run 2's stem is `gspo_lr1e5`, which is likewise the name of
its checkpoint directory -- and that directory is in use by a running job, so it
stays. Different stems, but each matches its own checkpoint directory, which is
the property that actually disambiguates.

The comparison *column* labels stay GSPO_lr5e5 / GSPO_lr1e5: those describe the
learning rate to the reader and are decoupled from the file names here.
"""
import re
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
PROJ = ROOT / "projects/wafer-defect-vlm/scripts"

# ------------------------------------------------------------- 1. the aliases
aliases = [
    ROOT / "outputs/reports/gspo_lr5e5__report.json",
    ROOT / "outputs/reports/gspo_lr5e5_train_result.json",
    ROOT / "outputs/baselines/gspo_lr5e5.jsonl",
]
canonical = [
    ROOT / "outputs/reports/qwen35_9b_gspo_v1__report.json",
    ROOT / "outputs/reports/qwen35_9b_gspo_v1_train_result.json",
    ROOT / "outputs/baselines/qwen35_9b_gspo_v1.jsonl",
]
for alias, orig in zip(aliases, canonical):
    if not orig.is_file():
        sys.exit(f"REFUSING: canonical file missing, would delete the only copy: {orig}")
    if alias.is_file():
        same = alias.read_bytes() == orig.read_bytes()
        if not same:
            sys.exit(f"REFUSING: {alias.name} differs from {orig.name}; not a duplicate")
        alias.unlink()
        print(f"  removed duplicate alias {alias.name} (identical to {orig.name})")
    else:
        print(f"  alias already absent: {alias.name}")

# --------------------------------------------------- 2. every reference to it
# (path, [(old, new), ...], expected occurrences each)
edits = {
    PROJ / "31_after_grpo.sh": [
        ('"$REPORTS/gspo_lr5e5__report.json"', '"$REPORTS/qwen35_9b_gspo_v1__report.json"'),
    ],
    ROOT / "tools/final_report.py": [
        ('("gspo_lr5e5_train_result.json", "GSPO（lr 5e-5，论文设定）")',
         '("qwen35_9b_gspo_v1_train_result.json", "GSPO（lr 5e-5，论文设定）")'),
    ],
    PROJ / "35_after_gspo.sh": [
        ('run_one "gspo_lr5e5" "$ROOT/outputs/checkpoints/qwen35_9b_gspo_v1" \\\n'
         '        "$ROOT/models/Qwen3.5-9B-gspo-lr5e5-merged"',
         'run_one "qwen35_9b_gspo_v1" "$ROOT/outputs/checkpoints/qwen35_9b_gspo_v1" \\\n'
         '        "$ROOT/models/Qwen3.5-9B-gspo-lr5e5-merged"'),
        ('  "GSPO_lr5e5:$REPORTS/gspo_lr5e5__report.json" \\',
         '  "GSPO_lr5e5:$REPORTS/qwen35_9b_gspo_v1__report.json" \\'),
    ],
}
for path, pairs in edits.items():
    src = path.read_text(encoding="utf-8")
    for old, new in pairs:
        n = src.count(old)
        if n != 1:
            sys.exit(f"FAILED in {path.name}: {n} occurrences of {old[:80]!r}")
        src = src.replace(old, new)
    path.write_text(src, encoding="utf-8")
    print(f"  repointed {len(pairs)} reference(s) in {path.name}")

# ------------------------------------------------------------- 3. no leftovers
leftovers = []
for path in list(PROJ.glob("*.sh")) + [ROOT / "tools/final_report.py",
                                       ROOT / "tools/make_report.py"]:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "gspo_lr5e5" in line and "merged" not in line:
            leftovers.append(f"{path.name}:{i}: {line.strip()[:90]}")
if leftovers:
    sys.exit("FAILED: references to the removed alias remain:\n" + "\n".join(leftovers))
print("  verified: no script or tool still reads the removed alias")
