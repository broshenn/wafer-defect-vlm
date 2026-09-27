"""Qualify the learning-rate claim in the report's own verdict section.

LIMITATIONS now discloses that GRPO was never run at lr 5e-5. The generated
report has to say the same thing, or the summary is stronger than the evidence
in the document it summarises.
"""
import ast
import sys
from pathlib import Path

path = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
src = path.read_text(encoding="utf-8")

old = '''        "- 需要说清的是，这**不等于**「GSPO 不如 SFT」或「GSPO 不如 GRPO」："
        "四个 RL run 在基准上的差异都落在噪声内，唯一明显的差异来自学习率"
        "（论文的 5e-5 在这套 QLoRA 配置下把模型推进了退化区，见 LIMITATIONS 5.2.2），"
        "而不是来自算法。")'''
new = '''        "- 需要说清的是，这**不等于**「GSPO 不如 SFT」或「GSPO 不如 GRPO」："
        "四个 RL run 在基准上的差异都落在噪声内，最明显的一项差异来自学习率"
        "（论文的 5e-5 在这套 QLoRA 配置下把模型推进了退化区，见 LIMITATIONS 5.2.2）。"
        "**该判断只覆盖 GSPO 内部两个学习率的对照**：GRPO 仅跑了 lr 1e-5，"
        "缺 lr 5e-5 这一格，因此无法排除「崩的是 GSPO 本身」这一解释；"
        "补齐后才是完整的算法 × 学习率 2×2 对照（见 provenance 的未决事项）。")'''

n = src.count(old)
if n != 1:
    sys.exit(f"FAILED: {n} occurrences (need exactly 1)")
src = src.replace(old, new)
ast.parse(src)
path.write_text(src, encoding="utf-8")
print("final_report.py: learning-rate claim qualified in the verdict section")
