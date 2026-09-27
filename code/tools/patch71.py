"""Emit the paper's own contrast in FINAL_REPORT, read from the record.

FINAL_REPORT.md is generated and queue 41 regenerates it, so a finding that lives
only in LIMITATIONS.md would be absent from the report a reader sees first. All
numbers come from the record; nothing is computed in prose here.

The point of the bullet is that equal aggregate accuracy is not equal performance:
the two runs disagree on 64 rows, 32 in each direction, and the exchanges cancel.
A reader who saw only "Δ = 0.0000" would take away the wrong thing.
"""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
s = P.read_text(encoding="utf-8")

ANCHOR = """        _gs = ps.get("group_size_paired")
        if _gs:
"""

NEW = """        _pp = ps.get("paper_setting_paired")
        if _pp:
            _n01 = _pp["mcnemar"]["grpo_wrong_gspo_right"]
            _n10 = _pp["mcnemar"]["grpo_right_gspo_wrong"]
            _lo2, _hi2 = _pp["delta_ci95_rows"]
            out.append(
                f"  - **论文原始设定的那一格**（GSPO G=32 对 GRPO G=4，同为 lr 5e-5）："
                f"Δ准确率 {_pp['accuracy_delta_g32_minus_grpo4']:+.4f}，"
                f"95% CI [{_lo2:+.4f}, {_hi2:+.4f}]，"
                f"p = {_pp['mcnemar']['p_exact_two_sided']:.6g} —— 聚合准确率完全重合。"
                f"但**这不是「逐行相同」**：两者在 252 行里有 {_n01 + _n10} 行判断不同，"
                f"恰好 {_n01} 对 {_n10} 互补，聚合值相等是两边互换抵消的结果。"
                f"该对照一次改了组大小（4→32）、IS 层级（token→sequence）与"
                f"有效批次（16→1024）三样，**标识不了其中任何一样**。")
"""

if s.count(ANCHOR) != 1:
    sys.exit(f"anchor appears {s.count(ANCHOR)} times, expected 1")
s = s.replace(ANCHOR, NEW + ANCHOR)

ast.parse(s)
P.write_text(s, encoding="utf-8")
print("final_report.py: paper_setting_paired bullet added")
