"""Add the G=32 run to the report's RL section.

The generator lists GSPO runs explicitly, so a run that finishes after this file
was written would be missing from the report even though its column is in the
comparison table. Adding it here means the G=32 result appears in both.
"""
import ast
import sys
from pathlib import Path

path = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
src = path.read_text(encoding="utf-8")

old = '''    gspo_runs = [
        ("qwen35_9b_gspo_v1_train_result.json", "GSPO（lr 5e-5，论文设定）"),
        ("gspo_lr1e5_train_result.json", "GSPO（lr 1e-5，与 GRPO 对齐以分离算法与学习率）"),
    ]'''
new = '''    gspo_runs = [
        ("qwen35_9b_gspo_v1_train_result.json", "GSPO（G=8，lr 5e-5）"),
        ("gspo_lr1e5_train_result.json", "GSPO（G=8，lr 1e-5，与 GRPO 对齐以分离算法与学习率）"),
        ("qwen35_9b_gspo_g32_train_result.json", "GSPO（G=32，lr 5e-5，论文设定）"),
    ]'''

n = src.count(old)
if n != 1:
    sys.exit(f"FAILED: {n} occurrences")
src = src.replace(old, new)
ast.parse(src)
path.write_text(src, encoding="utf-8")
print("final_report.py: G=32 run added to the RL section")
