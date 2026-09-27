"""Item 16's first bullet, with the evidence the first run after the fix produced.

The claim "the counts stayed at whatever the last successful write left" was written from
reading the tool. The finish pass that ran minutes later made it measurable: sync_counts
rewrote nine lines it had not been touching -- the table's column count, the total metric
numbers, the number of RL records, and the idle-step rate list (which gained the second
seed's 38.00% at its own position). Every one of them had been *low*, which is the lag's
direction: the document kept the counts from the last run that could write.

That is worth one sentence in the item, because it converts "this tool was not running"
into "here is the sentence that was wrong for an unknown length of time" -- and because
the defect's symptom is the absence of one. The numbers themselves are deliberately not
restated here: section 9 already carries them and is rewritten from the audit's own
summary, whereas a number in this item would be a claim nothing recomputes.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
DOC = ROOT / "LIMITATIONS.md"

OLD = """      的核对工具，等于没有这个工具**，而它的失败方向是「文档里的数字停在上一次」，
      没有任何东西会因此变红。
"""

NEW = """      的核对工具，等于没有这个工具**，而它的失败方向是「文档里的数字停在上一次」，
      没有任何东西会因此变红。
      它的代价在修好后的第一次运行里就看得见：那次运行改掉了「可复现性核对」一节里
      三处滞后的值（表的列数、指标数字总数、RL 记录份数各一处）—— 在此之前的每一次核对，
      读到的都是同一份更早的计数。**这条缺陷的症状是「什么都没发生」**：它不会自己现形，
      只能被找出来。
"""

s = DOC.read_text(encoding="utf-8")
if "它的代价在修好后的第一次运行里就看得见" in s:
    sys.exit("the evidence sentence is already there; nothing written")
if s.count(OLD) != 1:
    sys.exit(f"the anchor matches {s.count(OLD)} times; nothing written")

shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch142")
DOC.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
print("item 16's first bullet now carries the evidence from the finish pass")
