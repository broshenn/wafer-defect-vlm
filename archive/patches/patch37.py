"""Remove the false "G=32 is infeasible" claim from the report generator.

The same claim appears in three places: LIMITATIONS (already corrected), the
commit that added it, and here. This is the generator, so leaving it would
re-emit the error every time the report is rebuilt.

Two edits: the per-run group-size line, and the closing RL paragraph, which also
still said GSPO showed no advantage at the gradient level. That was true for the
lr 5e-5 run and false once the learning rate is matched: idle steps fall from
36.00% to 15.30%, the signal-level gain is real, and what fails to appear is the
benchmark gain.
"""
import ast
import sys
from pathlib import Path

path = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
src = path.read_text(encoding="utf-8")

# ------------------------------------------------- 1. the group-size footnote
old_g = ('''            out.append(f"  - 组大小 {rec['config']['num_generations']}"
                       f"（论文设定为 32；本环境无 vLLM，G=32 需 40 小时以上，不可行）")''')
new_g = ('''            g = rec["config"]["num_generations"]
            # The note is only for runs that deviate from the paper. The claim
            # that G=32 was infeasible was measured false -- 60s/step, 15.2 GiB
            # -- so the deviation is stated as a choice about budget, not a
            # hardware limit.
            note = "" if g == 32 else (
                "（论文设定为 32。G=32 实测约 60s/步、15.2 GiB 显存，150 步约 2.5 小时，"
                "可以运行 —— 本报告初版曾断言其不可行，实测推翻，见 LIMITATIONS 5.2.1）")
            out.append(f"  - 组大小 {g}{note}")''')

# --------------------------------------------- 2. the closing RL paragraph
old_c = '''        "- 需要说清的是，这**不等于**「GSPO 不如 GRPO」：本文 GSPO 的组大小只有"
        "论文的 1/4，学习率也按论文设定而非按本模型规模调过，两者都可能单独"
        "造成退化。诚实的表述是：**在本文预算与超参下 GSPO 既未带来基准收益，"
        "也未在梯度层面显示优势**（零优势步占比见上，两者基本持平）——"
        "初版曾据一个被误读的单步数字声称 GSPO 改善了学习信号，已撤回。"
        "判定算法本身优劣需要 G=32 或学习率扫描，均超出本次预算。")'''
new_c = '''        "- 需要说清的是，这**不等于**「GSPO 不如 SFT」或「GSPO 不如 GRPO」："
        "三个 RL run 在基准上的差异都落在噪声内，唯一明显的差异来自学习率"
        "（论文的 5e-5 在这套 QLoRA 配置下把模型推进了退化区，见 LIMITATIONS 5.2.2），"
        "而不是来自算法。")
    out.append(
        "- **信号层面的收益是真实的**：把学习率对齐后，GSPO 的零优势步从 GRPO 的 "
        "36.00% 降到 15.30%（见上）。但该收益**没有转化为基准分数**。"
        "这是本次 RL 最值得记下的一条：优势信号的多寡与最终指标的高低，在这里是解耦的。")'''

for old, new in ((old_g, new_g), (old_c, new_c)):
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAILED: {n} occurrences of:\n{old[:150]}")
    src = src.replace(old, new)

ast.parse(src)
path.write_text(src, encoding="utf-8")
print("final_report.py: G=32 claim corrected, RL closing paragraph updated")
