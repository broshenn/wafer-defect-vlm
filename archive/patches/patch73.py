"""Point the quoted audit figures at the current measurements.

Re-run after this to check the quoted integers are not themselves counted as metric
numbers -- i.e. that the sentence has a fixed point. If they are, the wording has to
stop quoting a total at all, because otherwise every edit moves its own target.
"""
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
s = P.read_text(encoding="utf-8")

OLD = """- 数字溯源核对 `tools/audit_report_numbers.py`：三份文档共 477 个指标数字，
  **无法回溯到任何记录的：0 个**；其中 5 个是已记录数字之间的差值（算术量）。
  （**这两个数是写作时的快照，随文档每次修改而变** —— 本节这一轮改动中它就从
  472/4 变成了 477/5。引用时必须重跑该工具，不要沿用这里的值。该工具自己声明
  它是「搜索」而非「证明」：它为每个差值给出一对可凑出它的记录数字，可能是巧合，
  只可读作「不是无出处的测量」。）"""

NEW = """- 数字溯源核对 `tools/audit_report_numbers.py`：三份文档共 479 个指标数字，
  **无法回溯到任何记录的：0 个**；其中 5 个是已记录数字之间的差值（算术量）。
  （**这两个数是写作时的快照，随文档与报告每次修改而变** —— 本节这一轮改动中它
  先从 472/4 变成 477/5，补上论文原始设定那一格后又变成 479/5。引用时必须重跑该
  工具，不要沿用这里的值。该工具自己声明它是「搜索」而非「证明」：它为每个差值
  给出一对可凑出它的记录数字，可能是巧合，只可读作「不是无出处的测量」。）"""

if s.count(OLD) != 1:
    sys.exit(f"anchor appears {s.count(OLD)} times, expected 1")
s = s.replace(OLD, NEW)
if s.count("**") % 2 != 0:
    sys.exit("bold markers unbalanced")
P.write_text(s, encoding="utf-8")
print("LIMITATIONS.md: audit figures -> 479/5")
