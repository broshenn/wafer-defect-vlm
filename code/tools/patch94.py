"""Stop the audit's history line from looking like the current count.

Section 9 quotes the number audit's two counts and then warns, in the same
parenthesis, not to reuse them -- and lists the values they passed through
(472/4 -> 477/5 -> 479/5). An earlier version of that sentence ended on 479/5, so a
reader had a stale number directly under a warning that the number is a snapshot.

The two *current* values are now read back from the audit by tools/sync_counts.py,
so they cannot lag. The history stays hand-written -- and says so, and no longer
stops at a value that looks current.
"""
import os
import sys
from pathlib import Path

DOC = Path(os.environ.get("WAFER_DOC", "/root/autodl-fs/wafer-vlm/LIMITATIONS.md"))
s = DOC.read_text(encoding="utf-8")

OLD = """（**这两个数是写作时的快照，随文档与报告每次修改而变** —— 本节这一轮改动中它
  先从 472/4 变成 477/5，补上论文原始设定那一格后又变成 479/5。引用时必须重跑该
  工具，不要沿用这里的值。"""
NEW = """（**这两个数是写作时的快照，随文档与报告每次修改而变** —— 本节这一轮改动中它
  先从 472/4 变成 477/5，补上论文原始设定那一格后又变成 479/5；今天又随 run 43
  落地与总报告重新生成而增大到上面写的当前值。**上面这两个当前值不是手写的**：
  `tools/sync_counts.py` 每次从该工具的摘要里读回来，所以它们不会停在某一次编辑
  上；而这几行历史是手写的，只用于说明「它会变」这件事。引用时必须重跑该
  工具，不要沿用这里的值。"""

if s.count(OLD) != 1:
    sys.exit(f"anchor appears {s.count(OLD)} times; nothing written")
s = s.replace(OLD, NEW, 1)
if s.count("**") % 2:
    sys.exit("bold markers are odd; nothing written")
DOC.write_text(s, encoding="utf-8")
print(f"section 9: the audit-count parenthesis now separates history from current "
      f"({len(s.splitlines())} lines)")
