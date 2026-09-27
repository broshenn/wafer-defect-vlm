"""Cross-tabulate what each run's training log actually ran against what its
record claims.

The record's config block hardcodes "grad_accum": 4 -- it is a constant, not a
measurement, so it cannot disagree with itself and cannot reveal this. The
training logs contain the real swift argv. Where the two differ, the record is
wrong and every effective-batch figure derived from it is wrong.
"""
import json
import re
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
LOGS = R / "logs"
REPORTS = R / "outputs/reports"

ARGS = ("gradient_accumulation_steps", "num_generations", "learning_rate",
        "importance_sampling_level", "per_device_train_batch_size")

# log filename -> (label, record file)
RUNS = [("29_grpo_train.log", "GSPO G=8 lr5e-5 (qwen35_9b_gspo_v1)",
         "qwen35_9b_gspo_v1_train_result.json"),
        ("29_gspo_lr1e5_train.log", "GSPO G=8 lr1e-5", "gspo_lr1e5_train_result.json"),
        ("29_qwen35_9b_grpo_lr5e5_train.log", "GRPO G=4 lr5e-5",
         "qwen35_9b_grpo_lr5e5_train_result.json"),
        ("29_qwen35_9b_gspo_g32_train.log", "GSPO G=32 lr5e-5",
         "qwen35_9b_gspo_g32_train_result.json"),
        ("29_qwen35_9b_gspo_g32_lr1e5_train.log", "GSPO G=32 lr1e-5 (RUNNING)",
         "qwen35_9b_gspo_g32_lr1e5_train_result.json")]


def argv_from_log(p):
    """The real swift argv, taken from the first line that has all the flags."""
    if not p.is_file():
        return None
    txt = p.read_text(encoding="utf-8", errors="replace")
    vals = {}
    for a in ARGS:
        m = re.search(r"--" + a + r"[= ]([^\s,]+)", txt)
        if m:
            vals[a] = m.group(1)
    return vals or None


print(f"{'run':<34} {'arg':<30} {'log (real)':>12} {'record':>10}  verdict")
print("-" * 104)
mismatches = []
for logname, label, recname in RUNS:
    real = argv_from_log(LOGS / logname)
    rec = None
    rp = REPORTS / recname
    if rp.is_file():
        rec = json.loads(rp.read_text(encoding="utf-8")).get("config", {})
    if real is None:
        print(f"{label:<34} {'(no log / no argv)':<30} {'-':>12} "
              f"{str(rec.get('grad_accum') if rec else None):>10}")
        continue
    g = int(real.get("num_generations", 0))
    a = int(real.get("gradient_accumulation_steps", 0))
    pdb = int(real.get("per_device_train_batch_size", 1))
    print(f"{label:<34} {'num_generations':<30} {g:>12} "
          f"{str(rec.get('num_generations') if rec else '-'):>10}")
    print(f"{'':<34} {'gradient_accumulation_steps':<30} {a:>12} "
          f"{str(rec.get('grad_accum') if rec else '-'):>10}"
          + ("   <-- RECORD WRONG" if rec and rec.get("grad_accum") != a else ""))
    print(f"{'':<34} {'effective batch (G x pdb x accum)':<30} {g * pdb * a:>12} "
          f"{(rec.get('num_generations', 0) * rec.get('per_device_batch', 1) * rec.get('grad_accum', 0)) if rec else 0:>10}")
    if rec and rec.get("grad_accum") != a:
        mismatches.append((label, a, rec.get("grad_accum")))
    print(f"{'':<34} {'learning_rate':<30} {real.get('learning_rate'):>12}")
    print(f"{'':<34} {'importance_sampling_level':<30} "
          f"{real.get('importance_sampling_level'):>12}")
    print()

print("=== mismatches ===")
for label, real, claimed in mismatches:
    print(f"  {label}: ran {real}, record claims {claimed}")
if not mismatches:
    print("  none")

print()
print("=== consequence ===")
print("  effective batch = num_generations x per_device_batch x grad_accum.")
print("  If accumulation was scaled with the group size, then effective batch")
print("  grew as the SQUARE of it, and every group-size contrast in this project")
print("  moved group size and effective batch together.")
