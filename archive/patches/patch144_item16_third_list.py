"""Item 16 said four defects. The 44 wait list was the third of the same shape, so it says six.

A title that enumerates is a claim, and this one was about to become false the moment the
third instance of "a list written when it was true" was fixed in the last script of the
chain. The title and the opening sentence are rewritten to match what the item now
contains, and bullet four gains the third instance with the evidence that the guard was
run in both directions rather than only written.

The count in the title is a word, not a numeral, on purpose: this document's number audit
counts metric numbers and traces them to records, and a count of defects in prose is not
a measurement anything recomputes. Section 9's two counts are the ones that are read back
from the audit summary, and the finish pass rewrites them.
"""
import pathlib
import shutil
import sys

DOC = pathlib.Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")

OLD_HEAD = """16. **今天的四处缺陷：两处「从未运行过的分支」，一处「失败被记成 WARNING」，
    一处「靠字典顺序猜出来的答案」。**
    四处都不是数字错，四处都产出过看起来合理的结果。记在这里是因为它们**只有在把那个
    分支真的跑一遍之后才现形** —— 前两类一直静静躺在盘上，第三类被调用方的一句
    `WARNING` 掩盖，第四类只在第二个种子落地的那一刻出现。
"""

NEW_HEAD = """16. **今天六处缺陷：一处失败被记成 WARNING，两处答案由字典顺序或兜底给出，
    三处「写下来时是对的、之后没有人再看它一眼」的清单和条件。**
    六处都不是数字错，六处都产出过看起来合理的结果。记在这里是因为它们**只有在把那个
    分支真的跑一遍之后才现形** —— 它们一直静静躺在盘上，或者只在第二个种子落地的那一刻
    出现。
"""

OLD_TAIL = """      这一条是把 `busy()` 对着**真实进程表**跑出来才看见的。
"""

NEW_TAIL = """      这一条是把 `busy()` 对着**真实进程表**跑出来才看见的。
      同一形状今天出现第三次，而且就在链条的最后一步：`44_final_consolidation.sh`
      等待「所有报告都在」用的也是一张手写的清单，第三个种子的报告不在里面。它的重建
      是从盘上枚举的，所以少的那一列不会掉；真正少掉的是**等待** —— 44 不会等这个 run，
      而输出里没有任何东西能把「等于它没跑完」和「等于已经等过它」分开。已补上该项，
      并加了一道核对：等待清单必须覆盖 `run_set.py` 认账的每一个 run，否则**不打印完成
      标记**（该脚本的取舍是明确的：宁可重建一份说明了缺失的表，也不要什么都没有；
      但两者在下一个观察者眼里必须不同）。这道核对两个方向都跑过 ——
      对着真实清单通过，拿掉一项就打印出缺的是哪一个。
"""

s = DOC.read_text(encoding="utf-8")
if "同一形状今天出现第三次" in s:
    sys.exit("the third list is already in the item; nothing written")
for what, old in (("the title and opening", OLD_HEAD), ("bullet four's tail", OLD_TAIL)):
    if s.count(old) != 1:
        sys.exit(f"{what} matches {s.count(old)} times; nothing written")

shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch144")
before = s.count("**")
s = s.replace(OLD_HEAD, NEW_HEAD, 1).replace(OLD_TAIL, NEW_TAIL, 1)
after = s.count("**")
if (after - before) % 2:
    sys.exit(f"bold markers went {before} -> {after}, an odd change; nothing written")
DOC.write_text(s, encoding="utf-8")
print("item 16 now names six defects and carries the third stale list")
