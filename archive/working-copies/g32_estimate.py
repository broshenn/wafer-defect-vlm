"""Estimate the cost of the paper's group size from the two runs already done.

The report, LIMITATIONS and the commit all claim the paper's G=32 needs 40+ hours
and is therefore infeasible here. That claim was never measured -- it came from
assuming a generation batch of 32 prompts times 32 generations per prompt. The
ms-swift error seen when the first GSPO run was launched ("generation_batch_size
(4) must be evenly divisible by num_generations (16). Valid values: [2, 4]")
implies generation_batch_size is counted in completions, not prompts, so the
real cost should be far lower. Two measured points are enough to fit the line
before spending GPU time on a smoke test.
"""
import json
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm/outputs/reports")

points = []
for fname, label in [("grpo_train_result.json", "GRPO"),
                     ("qwen35_9b_gspo_v1_train_result.json", "GSPO lr5e5")]:
    d = json.loads((R / fname).read_text(encoding="utf-8"))
    c = d["config"]
    g = c["num_generations"]
    steps = d["steps_logged"]
    per_step = d["elapsed_seconds"] / steps
    points.append((g, per_step))
    print(f"{label:12} G={g:2d}  gen_batch_size={c['grad_accum']:2d}  "
          f"steps={steps}  elapsed={d['elapsed_seconds']}s  "
          f"per_step={per_step:.2f}s  peak={d['gpu_peak_mib']} MiB")

(g1, t1), (g2, t2) = points
slope = (t2 - t1) / (g2 - g1)          # seconds per extra generation per step
intercept = t1 - slope * g1            # fixed per-step cost
print(f"\nfitted: per_step = {intercept:.2f}s + {slope:.3f}s x G")

for g in (4, 8, 16, 32):
    est = intercept + slope * g
    print(f"  G={g:2d} -> ~{est:5.1f}s/step, 150 steps ~ {est * 150 / 3600:.2f} h")

print("\nCaveats: the fit is over two points, and it assumes the fixed cost stays")
print("fixed. A larger generation batch also grows the KV cache, so the peak")
print("memory at G=32 is not predicted here -- only a real run measures that.")
