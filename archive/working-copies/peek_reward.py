"""Compare the group-reward signal across the three RL runs so far.

The point of a larger group is lower-variance advantages, so if G=32 is doing
anything it should show up here first, before any benchmark number exists.
"""
import re
from pathlib import Path

LOGS = Path("/root/autodl-fs/wafer-vlm/logs")
RUNS = [
    ("GRPO      G=4  lr1e-5", "24_grpo_qwen35_9b_grpo_v1.log"),
    ("GSPO      G=8  lr5e-5", "24_grpo_qwen35_9b_gspo_v1.log"),
    ("GSPO      G=8  lr1e-5", "24_grpo_gspo_lr1e5.log"),
    ("GSPO      G=32 lr5e-5", "24_grpo_qwen35_9b_gspo_g32.log"),
]
FUNCS = ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock")


def col(text, key):
    return [float(v) for v in re.findall(
        r"'%s': '([-0-9.eE]+)'" % re.escape(key), text)]


print(f"{'run':22} {'steps':>5} {'idle':>7} " +
      " ".join(f"{f[5:]:>10}" for f in FUNCS) + f" {'KL':>7}")
for label, fname in RUNS:
    p = LOGS / fname
    if not p.is_file():
        print(f"{label:22} MISSING {fname}")
        continue
    text = p.read_text(encoding="utf-8", errors="replace")
    idle = col(text, "frac_reward_zero_std")
    means = [col(text, f"rewards/{f}/std") for f in FUNCS]
    kl = col(text, "kl")
    steps = len(idle)
    if not steps:
        print(f"{label:22} no step rows yet")
        continue
    cells = []
    for series in means:
        cells.append(f"{sum(series) / len(series):>10.4f}" if series else f"{'-':>10}")
    idle_frac = sum(idle) / len(idle)
    kl_mean = f"{sum(kl) / len(kl):>7.3f}" if kl else f"{'-':>7}"
    print(f"{label:22} {steps:>5} {idle_frac:>6.1%} " + " ".join(cells) + f" {kl_mean}")

print()
print("markdown rows for LIMITATIONS 5.2.3:")
for label, fname in RUNS:
    p = LOGS / fname
    if not p.is_file():
        continue
    text = p.read_text(encoding="utf-8", errors="replace")
    vals = [f"{sum(col(text, f'rewards/{f}/std')) / max(1, len(col(text, f'rewards/{f}/std'))):.4f}"
            for f in FUNCS]
    idle = col(text, "frac_reward_zero_std")
    idle_s = f"{sum(idle) / len(idle) * 100:.2f}%" if idle else "-"
    print(f"| {label.strip()} | {idle_s} | " + " | ".join(vals) + " |")

print("\n'idle' = steps where every generation in the group scored alike, so the")
print("advantage was zero and no gradient was produced. Columns are the mean")
print("within-group standard deviation of each reward, averaged over steps.")
