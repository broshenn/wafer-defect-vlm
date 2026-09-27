"""Item 18 gains the fourth instance -- the one that appeared while the item was being written.

The item as first written named three checks. Between writing it and checking the end-of-day
chain, `tools/post_consolidation.sh`'s `WATCH` list turned out to be one entry short in the
same way: it names the runs whose retrieval the pass repairs, it was written before the
third seed existed, and an omission from it is silent -- the pass prints its completion
marker either way. That is item 16's fourth bullet (a hand-written list), and it is the
second list fixed today (44's `REQUIRED` was the first, patch143).

Leaving it out would have been the reasonable-looking choice: the item was finished, the
measurement was somebody else's file, and the coincidence is not the item's subject. But
the item's whole claim is that recording a shape does not prevent the next instance of it --
and the demonstration of that claim arrived during the writing and would have been dropped
for tidiness. So the heading says four, and the fourth is written down with how it was
found: by reading a file for an unrelated reason.
"""
import pathlib
import re
import shutil
import sys

DOC = pathlib.Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
TOKEN = re.compile(r"(?<![.\d])-?\d+\.\d{3,6}")

OLD_HEAD = ("18. **三个检查，没有一个能拦住它要拦住的东西 —— 同一天出现三次。**")
NEW_HEAD = ("18. **三个检查，没有一个能拦住它要拦住的东西 —— 同一天出现三次；"
            "写这一条的过程中又出现第四处，是一份同样短了一行的清单。**")

CLOSING = ("    三处合起来是今天最该记的一句：**一个拦不住被检查对象的检查，就是一行日志。**")
FOURTH = """    第四处不是检查，和前三处同一天出现，出现在写这一条的过程中。`tools/post_consolidation.sh`
    里的 `WATCH` 列表写明「评估最后发生的那些 run」以及每个 run 的评估被告知的合并目录 ——
    它是三个条目，第三次抽样不在其中：那份名单写在它被启动之前。它漏掉的后果很窄，却完全
    安静：如果第三次抽样的评估没有把检索字段写进报告，这个环节既不修也不报，**它照样印出
    自己的完成标记**。唯一会响的是 `land_run.sh` 在那里等那个字段，上限六个小时然后停下 ——
    从外面看，这和一个还在训练的 run 一模一样。补进去时，条目里的两个字符串是从该 run 自己
    的队列脚本里抄的，不是重新写的，改完照例过了一遍 `bash -n`。

    这是今天第二份因为「写下来时是对的」而短了一行的清单（第一份是 44 号脚本的 `REQUIRED`，
    见第 16 条第四条），也是唯一一处**在被记录的那条缺陷里、边写边撞见的新实例**：
    发现它不是在找它，而是在为别的原因读那份脚本。所以它留在这里，而不是被当成题外话删掉。

"""
CLOSING_NEW = ("    四处合起来是今天最该记的一句：**一个拦不住被检查对象的检查，就是一行日志。**")

s = DOC.read_text(encoding="utf-8")
for what, old in (("the heading", OLD_HEAD), ("the closing sentence", CLOSING)):
    if s.count(old) != 1:
        sys.exit(f"{what} matches {s.count(old)} times; nothing written")
# Keyed on the sentence this patch inserts, not on the file name: the document already
# mentions post_consolidation.sh elsewhere, so a file-name guard refuses on a document
# that has not been touched -- which is what it did on the first run.
if "第四处不是检查" in s:
    sys.exit("the fourth instance is already in the document; nothing written")

bad = TOKEN.findall(FOURTH + NEW_HEAD)
if bad:
    sys.exit(f"the added text quotes {len(bad)} decimal(s) {bad}; nothing written")
if (FOURTH + NEW_HEAD).count("**") % 2:
    sys.exit("the added text has an odd number of bold markers; nothing written")

shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch154")
before = s.count("**")
s = s.replace(OLD_HEAD, NEW_HEAD, 1).replace(CLOSING, FOURTH + CLOSING_NEW, 1)
if (s.count("**") - before) % 2:
    sys.exit("bold markers changed by an odd number; nothing written")
DOC.write_text(s, encoding="utf-8")
print(f"item 18 now names four instances ({len(s.splitlines())} lines, bold {before} -> "
      f"{s.count('**')})")
print("  the fourth: post_consolidation.sh's WATCH list, found by reading the chain")
