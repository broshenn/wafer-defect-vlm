"""Every run's effective batch: what the record claims vs what actually ran.

The G=32 job's live argv shows --gradient_accumulation_steps 32 while every other
run used the default 4. If the records say 4 for all runs, then the record cannot
show the confound, and every group-size contrast involving G=32 is really a
contrast of group size AND effective batch size together.
"""
import json
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm/outputs/reports")

print(f"{'record file':<52} {'G':>4} {'accum':>6} {'eff batch':>10}")
print("-" * 76)
rows = []
for f in sorted(R.glob("*train_result.json")):
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"{f.name:<52} unreadable: {e}")
        continue
    c = d.get("config", {})
    g = c.get("num_generations")
    a = c.get("grad_accum")
    pdb = c.get("per_device_batch", 1)
    eff = (g or 0) * (pdb or 1) * (a or 0)
    rows.append((f.name, g, a, eff))
    print(f"{f.name:<52} {str(g):>4} {str(a):>6} {eff:>10}")

print()
print("=== the launcher's intent, for comparison ===")
print("  queue 41 passes GRAD_ACCUM=32 with NUM_GENERATIONS=32")
print("  queue 42 (grpo seed2) and the G=4/G=8 runs pass neither,")
print("  so GRAD_ACCUM falls through to 24_train_grpo.sh's default of 4.")

print()
print("=== does 29_grpo_train.sh forward GRAD_ACCUM explicitly? ===")
sh = Path("/root/autodl-fs/wafer-vlm/projects/wafer-defect-vlm/scripts/29_grpo_train.sh")
txt = sh.read_text(encoding="utf-8")
print(f"  mentions GRAD_ACCUM: {'GRAD_ACCUM' in txt}")
i = txt.find("24_train_grpo.sh")
if i > 0:
    print("  invocation context:")
    for ln in txt[max(0, i - 300):i + 40].split("\n")[-8:]:
        print(f"    {ln}")

print()
print("=== the live G=32 argv, for the record ===")
import subprocess
out = subprocess.run(["pgrep", "-f", "swift/cli/rlhf.py"], capture_output=True,
                     text=True).stdout.split()
for pid in out[:1]:
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        argv = [a.decode() for a in argv if a]
        for k in ("--num_generations", "--gradient_accumulation_steps",
                  "--per_device_train_batch_size", "--importance_sampling_level",
                  "--learning_rate"):
            if k in argv:
                print(f"  {k} {argv[argv.index(k) + 1]}")
    except Exception as e:
        print(f"  pid {pid}: {e}")
