"""Emit the single-variable IS-level contrast in FINAL_REPORT, read from the record.

Every earlier contrast in this project changes more than the thing under discussion.
This one changes exactly one, so it is the first result whose explanation cannot be
group size or effective batch -- and it says explicitly what it still does not
establish, so the bullet cannot be read as settling the paper's proposition.
"""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
s = P.read_text(encoding="utf-8")

ANCHOR = """        _gs = ps.get("group_size_paired")
        if _gs:
"""

NEW = """        _isl = ps.get("is_level_paired")
        if _isl:
            _lo3, _hi3 = _isl["delta_ci95_rows"]
            _m3 = _isl["mcnemar"]
            out.append(
                f"  - **IS 层级的单变量对照**（GRPO(G=4) 对 GSPO(G=4)：lr 5e-5、累积 "
                f"4/有效批次 16、同种子、同步数、同数据、同奖励函数与权重，"
                f"**只差 `--importance_sampling_level`**）："
                f"Δ准确率（sequence − token）"
                f"{_isl['accuracy_delta_sequence_minus_token']:+.4f}，"
                f"95% CI [{_lo3:+.4f}, {_hi3:+.4f}]，"
                f"p = {_m3['p_exact_two_sided']:.6g} —— {_isl['reading']}。"
                f"这是本项目**第一个只改一个变量的对照**，此前每一个都同时改了 IS "
                f"层级、组大小或有效批次之一以上。但它只覆盖 lr 5e-5 一个学习率，"
                f"而「更耐高学习率」是关于**学习率效应的大小**的比较，"
                f"因此论文的中心主张仍然**未被完整检验**，只是它的一个必要部分"
                f"第一次有了无混淆的读数。")
"""

if s.count(ANCHOR) != 1:
    sys.exit(f"anchor appears {s.count(ANCHOR)} times, expected 1")
s = s.replace(ANCHOR, NEW + ANCHOR)

ast.parse(s)
P.write_text(s, encoding="utf-8")
print("final_report.py: is_level_paired bullet added")
