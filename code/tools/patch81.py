"""Correct the one documented bound the RNG isolation moved, and record the guarantee.

patch80 gave every bootstrap interval its own Random(SEED). Four numbers moved by one
row (1/252) as a result. Three of them moved back onto the values the document
already quotes; the fourth is the GRPO(G=4) lr5e-5 vs SFT interval's upper bound,
-0.0079 -> -0.0040. That cell is hand-written in LIMITATIONS.md 5.1, so it has to be
edited; FINAL_REPORT.md regenerates from the record and is not touched here.

The p-value is unchanged (0.04356) because McNemar is exact and uses no resampling,
and the verdict is unchanged (the interval still excludes zero).
"""
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
s = P.read_text(encoding="utf-8")

EDITS = []

# ------------------------------------------------------------------ 1. the bound
OLD1 = "| **GRPO(G=4) lr5e-5** vs SFT | **-0.0556** | **[-0.1071, -0.0079]** | **0.04356** |"
NEW1 = "| **GRPO(G=4) lr5e-5** vs SFT | **-0.0556** | **[-0.1071, -0.0040]** | **0.04356** |"
EDITS.append((OLD1, NEW1, "5.1 table: the moved bootstrap bound"))

# --------------------------------------------------- 2. the guarantee, in section 9
OLD2 = """- 成对显著性 `tools/paired_significance.py`：全部数值见
  `outputs/reports/paired_significance.json`。"""
NEW2 = """- 成对显著性 `tools/paired_significance.py`：全部数值见
  `outputs/reports/paired_significance.json`。**每个对照的重采样各自构造一个
  `Random(SEED)`**，所以某个区间的取值只由它比较的那两个 run 决定，与「盘上还有
  哪些 run」「上文还有哪些对照」都无关。此前所有对照共用一个按文件顺序消费的随机流，
  于是**新增一个 run 会静默移动无关对照的区间分位**（一次一行，1/252）：本项目发生过
  两次，第二次把 5.1 表里 GRPO(G=4) lr5e-5 对 SFT 的上界从 -0.0079 推到 -0.0040，
  已按记录更正（`p` 不变，因为它来自精确检验、不用重采样）。"""
EDITS.append((OLD2, NEW2, "section 9: intervals are now run-set independent"))

before = s.count("**")
for old, new, why in EDITS:
    n = s.count(old)
    if n != 1:
        sys.exit(f"anchor for '{why}' appears {n} times, expected 1; nothing written")
    s = s.replace(old, new)
after = s.count("**")
if (after - before) % 2:
    sys.exit(f"bold markers went {before} -> {after}, an odd change; nothing written")

P.write_text(s, encoding="utf-8")
print(f"LIMITATIONS.md: {len(EDITS)} edit(s) applied ({len(s.splitlines())} lines)")
for _, _, why in EDITS:
    print(f"  - {why}")
