"""Refresh the audit figures the document quotes, and mark them as a snapshot.

The counts move whenever the documents change, so a bare number here is a value
that will be wrong by the next edit and cannot show it. The caveat is part of the
fix, not decoration.
"""
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
s = P.read_text(encoding="utf-8")

OLD = """- 数字溯源核对 `tools/audit_report_numbers.py`：三份文档共 472 个指标数字，
  **无法回溯到任何记录的：0 个**；其中 4 个是已记录数字之间的差值（算术量）。"""

NEW = """- 数字溯源核对 `tools/audit_report_numbers.py`：三份文档共 477 个指标数字，
  **无法回溯到任何记录的：0 个**；其中 5 个是已记录数字之间的差值（算术量）。
  （**这两个数是写作时的快照，随文档每次修改而变** —— 本节这一轮改动中它就从
  472/4 变成了 477/5。引用时必须重跑该工具，不要沿用这里的值。该工具自己声明
  它是「搜索」而非「证明」：它为每个差值给出一对可凑出它的记录数字，可能是巧合，
  只可读作「不是无出处的测量」。）"""

if s.count(OLD) != 1:
    sys.exit(f"anchor appears {s.count(OLD)} times, expected 1")
s = s.replace(OLD, NEW)

if s.count("**") % 2 != 0:
    sys.exit(f"bold markers unbalanced: {s.count('**')}")

P.write_text(s, encoding="utf-8")
print(f"LIMITATIONS.md: audit figures refreshed to 477/5 with a snapshot caveat "
      f"({s.count(chr(10)) + 1} lines)")
