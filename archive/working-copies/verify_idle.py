"""Verify the idle-step figure per RL run, and how the wrong one got quoted.

The report briefly claimed GSPO cut idle steps from 36% to 13%. The 13% came
from a single logged row, not a mean, so it is checked here against the full
distribution before any prose is written from it.
"""
import json
import re
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
RUNS = [
    ("GRPO", "logs/24_grpo_qwen35_9b_grpo_v1.log", "grpo_train_result.json"),
    ("GSPO_lr5e5", "logs/24_grpo_qwen35_9b_gspo_v1.log", "gspo_lr5e5_train_result.json"),
]

for name, log, rec in RUNS:
    text = (ROOT / log).read_text(encoding="utf-8", errors="replace")
    vals = [float(x) for x in re.findall(r"'frac_reward_zero_std': '([-0-9.eE]+)'", text)]
    mean = sum(vals) / len(vals)
    print(f"{name:11} n={len(vals):3d}  mean={mean:.4f}  min={min(vals):.2f}  "
          f"max={max(vals):.2f}  first={vals[0]:.2f}  last={vals[-1]:.2f}")
    print(f"{'':11} steps with 0.0: {sum(1 for v in vals if v == 0)}/{len(vals)}"
          f"   steps with 1.0: {sum(1 for v in vals if v == 1.0)}/{len(vals)}")
    d = json.loads((ROOT / "outputs/reports" / rec).read_text(encoding="utf-8"))
    fs = d["first_step"]["frac_reward_zero_std"]
    ls = d["last_step"]["frac_reward_zero_std"]
    print(f"{'':11} record stores only first_step={fs} last_step={ls}"
          f"  <-- '13%' was one of these, not the mean")
    for fn in ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock"):
        print(f"{'':11} {fn:12} mean group std = "
              f"{d['reward_signal'][fn]['mean_std_across_steps']:.4f}")
    print()
