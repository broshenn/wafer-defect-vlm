"""Round the GSPO run's KL and reward in the report.

They printed as 1.0217186666666667 because they go through a different code path
than the GRPO line, which already rounds to 4 places. Same run, same report, two
formats -- and full float noise reads as spurious precision.
"""
import sys
from pathlib import Path

path = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
src = path.read_text(encoding="utf-8")

old = ('''            f"耗时 {rec.get('elapsed_seconds')}s，峰值显存 {rec.get('gpu_peak_mib')} MiB，"
            f"平均 KL {rec.get('mean_kl')}，平均奖励 {rec.get('mean_reward')}")''')
new = ('''            f"耗时 {rec.get('elapsed_seconds')}s，峰值显存 {rec.get('gpu_peak_mib')} MiB，"
            + (f"平均 KL {rec['mean_kl']:.4f}，平均奖励 {rec['mean_reward']:.4f}"
               if rec.get("mean_kl") is not None and rec.get("mean_reward") is not None
               else "平均 KL / 平均奖励 not available"))''')

n = src.count(old)
if n != 1:
    sys.exit(f"FAILED: {n} occurrences")
path.write_text(src.replace(old, new), encoding="utf-8")

import ast
ast.parse(path.read_text(encoding="utf-8"))
print("final_report.py: GSPO KL/reward now rounded, and parses")
