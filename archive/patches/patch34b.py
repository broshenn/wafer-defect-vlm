"""Finish removing the `gspo_lr5e5` alias, keeping the corrections.

patch34 refused to delete the alias because it is not a byte-identical copy: it
is a strict superset. fix_records.py corrected the `log` field and added the
measured idle-step share to the alias only, so the alias holds the truth and the
canonical name holds the stale version. Deleting the alias as planned would have
thrown away the corrections; copying it over the canonical name and then
deleting it keeps them and leaves one file per run.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
PROJ = ROOT / "projects/wafer-defect-vlm/scripts"

alias = ROOT / "outputs/reports/gspo_lr5e5_train_result.json"
canon = ROOT / "outputs/reports/qwen35_9b_gspo_v1_train_result.json"

if alias.is_file():
    if not canon.is_file():
        sys.exit(f"REFUSING: canonical {canon.name} is missing entirely")
    shutil.copy2(alias, canon)
    print(f"  copied corrected {alias.name} -> {canon.name}")
    alias.unlink()
    print(f"  removed alias {alias.name}")
else:
    print(f"  alias already gone: {alias.name}")

# `gspo_lr5e5__report.json` was already removed by patch34 before it aborted.
for name in ("outputs/baselines/gspo_lr5e5.jsonl",):
    p = ROOT / name
    if p.is_file():
        orig = ROOT / name.replace("gspo_lr5e5", "qwen35_9b_gspo_v1")
        if p.read_bytes() != orig.read_bytes():
            sys.exit(f"REFUSING: {name} is not a duplicate of {orig.name}")
        p.unlink()
        print(f"  removed duplicate alias {p.name}")

# --------------------------------------------------- every reference to it
edits = {
    PROJ / "31_after_grpo.sh": [
        ('"$REPORTS/gspo_lr5e5__report.json"', '"$REPORTS/qwen35_9b_gspo_v1__report.json"'),
    ],
    ROOT / "tools/final_report.py": [
        ('("gspo_lr5e5_train_result.json", "GSPO（lr 5e-5，论文设定）")',
         '("qwen35_9b_gspo_v1_train_result.json", "GSPO（lr 5e-5，论文设定）")'),
    ],
    PROJ / "35_after_gspo.sh": [
        ('run_one "gspo_lr5e5" "$ROOT/outputs/checkpoints/qwen35_9b_gspo_v1"',
         'run_one "qwen35_9b_gspo_v1" "$ROOT/outputs/checkpoints/qwen35_9b_gspo_v1"'),
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

# ------------------------------------------------------------- no leftovers
leftovers = []
for path in list(PROJ.glob("*.sh")) + [ROOT / "tools/final_report.py"]:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "gspo_lr5e5" in line and "merged" not in line:
            leftovers.append(f"{path.name}:{i}: {line.strip()[:90]}")
if leftovers:
    sys.exit("FAILED: references to the removed alias remain:\n" + "\n".join(leftovers))
print("  verified: no script or tool still reads the removed alias")
