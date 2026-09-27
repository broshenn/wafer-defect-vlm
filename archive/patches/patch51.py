"""Add this round's new checks to section 9, and re-state the audit's result.

The provenance line in section 9 was true when written and is now out of date:
the document has grown two new defect items and several new numbers, and a
paired test has been added that the number-provenance audit cannot see at all
(it checks that a number exists in a record, not that a claim about it is true).
"""
from pathlib import Path
import sys

p = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
s = p.read_text(encoding="utf-8")

OLD = """- 数字溯源核对 `tools/audit_report_numbers.py`：三份文档共 378 个指标数字，其中 4 个
  无法回溯到任何记录，且此 4 个全部是已记录数字之间的差值（算术量），无实测数字缺失。
  复跑：`venvs/wafer/bin/python tools/audit_report_numbers.py`。
- 未能自动化的只有人工双审（见第 1 节）。"""

NEW = """- 数字溯源核对 `tools/audit_report_numbers.py`：三份文档共 378 个指标数字，其中 4 个
  无法回溯到任何记录，且此 4 个全部是已记录数字之间的差值（算术量），无实测数字缺失。
  复跑：`venvs/wafer/bin/python tools/audit_report_numbers.py`。
  **但这一核对只回答「这个数字是否存在于某份记录」，不回答「关于它的话是否成立」**
  ——第 8 节第 9 条记的就是后者失守的一次，而溯源核对结构上不可能发现它。
- 记录与日志的对应关系核对 `tools/audit_log_provenance.py`：5 份 RL 记录，每一份
  引用的日志都确实是它自己的（比对 `--num_generations` / `--learning_rate` /
  `--importance_sampling_level` / `--seed` 四项，命令行与 args dump 两种写法都查，
  args dump 形式加左边界以免 `data_seed=` 混入）。
  同时核出：4 份记录的 `seed` 是事后补写的，其值 3407 与该 run 自己的日志一致。
- KL 与补全长度的对照 `tools/kl_length_confound.py`：结论与全部数值见
  `outputs/reports/kl_length_confound.json`，含 `loss == beta * kl` 恒等式在
  五个 run 全部 271 个空转步上的逐值核对。
- 成对显著性 `tools/paired_significance.py`：全部数值见
  `outputs/reports/paired_significance.json`。
  它以两条自我否定作为前提才允许给出结论：
  （a）解析必须**复现**项目自身 scorer 写下的准确率，六个 run 逐一相等，否则直接退出；
  （b）正对照必须显著 —— BASE 对 SFT 准确率差 +0.4087、`p` = 2.83e-26，
      一个看不出这么大的差的检验，其零结果没有解释力，故该条不通过即退出。
  配对检验与 `final_report.py` 的置信区间重叠规则给出**不同结论**，
  原因与取舍写在 5.2.2；macro-F1 没有逐样本分解，未纳入配对检验。
- 机器人可自动化的仍只有上述核对；人工双审（见第 1 节）无法自动化。"""

if s.count(OLD) != 1:
    sys.exit(f"FAILED: section 9 block appears {s.count(OLD)} times")
s = s.replace(OLD, NEW)
p.write_text(s, encoding="utf-8")
print("section 9: updated with this round's checks")
