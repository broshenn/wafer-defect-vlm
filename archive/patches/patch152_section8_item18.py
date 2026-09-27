"""Section 8 item 18: three checks that could not stop the thing they were pointed at.

Written the same evening, because all three happened within about two hours of each other
and they share one property that items 15 and 16 do not name. Item 15 records a mechanism
that exists but never ran while its own self-check passes; item 16's first bullet records a
failure printed as a WARNING by its caller. Tonight's three are instances of both -- and
the reason recording the shapes did not prevent the next instance is the finding: in every
case the check's *failure path printed something*, and nothing about that changed an exit
code. A check that cannot fail what it checks is a log line.

The three: patch75 rewrote the training wrapper with an apostrophe inside its `${:?}` word
and printed the resulting `bash -n` failure while its caller logged ok (the third seed's
run died, and the queue behind it waits three hours for a result that cannot come);
check 11 of check_quantified_claims.py required the table at column 0 while the table is
indented inside a numbered item, so it had never matched anything since the day patch118
wrote it in the same run that generated the table; and audit_report_numbers.py read the
document's quotation of that same wrapper line as an unfilled template field and stopped
run 42's landing.

Every number here is a count, a line number, a clock time or a duration -- no decimal is
quoted, and the patch refuses to write text containing one. That is not tidiness:
LIMITATIONS.md is one of the three documents the audit traces every decimal in back to a
record, so a decimal written here would have to be traceable, and the honest place for
these measurements is this item's prose, not a record. The check is in this script, not in
a comment.
"""
import pathlib
import re
import shutil
import sys

DOC = pathlib.Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
ANCHOR9 = "## 9. 可复现性核对"
MARK = "拦不住被检查对象的检查"

# The audit's own token rule. A decimal of three to six places in this document must be
# traceable to a record; a patch that inserts one it cannot trace is inserting a defect.
TOKEN = re.compile(r"(?<![.\d])-?\d+\.\d{3,6}")

ITEM = """18. **三个检查，没有一个能拦住它要拦住的东西 —— 同一天出现三次。**
    这三处彼此无关，唯一共同点是：**每一个检查在写下的那一天，都没有拿它要读的输入
    真的跑过一次。** 一个读的是自己刚写出来的文件，一个从写下那天起就没匹配上任何东西，
    还有一个读的是文档里引用的一行 shell。代价是一次训练、一段空转，和一次被打断的落地。

    第一处，`tools/patch75_grpo_record.py` 重写了 `29_grpo_train.sh`，把
    `${RUN_TAG:?...}` 那条说明换了个更清楚的写法 —— 改动本身就是第 10 条的修法的一部分。
    bash 用引号规则解析 `:?` 后面那个词，而新写法里有一个撇号：它在第 18 行打开了一个
    字符串，整个文件再没有关上，那一行之后的一切都不再是 shell。patch75 确实跑了
    `bash -n`，也确实把失败印了出来（第 136 行 unexpected EOF）；它调用者那一行看到
    退出码是 0，于是在日志里记下 `ok`。**不到一分钟之后**，队列 43 —— 它一直在等的
    正是这两个补丁 —— 启动了那个文件，第三次抽样的训练在第一行就死了。

    更值得记的是第二层：同目录的 `tools/patch118_grpo_train_fixes.py` **会** 在
    `bash -n` 失败时还原文件，而它上一步就主动让位了，理由是 patch75 已经写进了它要找的
    那个字段。也就是说，唯一能拦住这件事的那个检查，**按构造就不可能被执行到**。

    代价不是猜测：训练进程当场退出，而队列脚本的下一步是等 `train_result.json`，
    上限三个小时。如果没有人在旁边看，这张卡会从那一刻起空转到超时，第三次抽样一次也
    不会跑 —— 而「等到超时」和「正在训练」在日志里长得一样。实际发现得早，空了约五分钟。

    第二处，`tools/check_quantified_claims.py` 的第 11 项检查 —— 那张「记录写的与自己
    日志不符」的表 —— 由 `patch118` 在**同一次运行里**写下，同一次运行也重新生成了它要读
    的那张表。它的模式要求表头出现在行首；而那张表在文档里挂在编号条目下面，每一行都
    缩进四个空格。于是它一次也没有匹配过：锚点失配、打印 STALE、退出 1 —— 而 patch118
    自己的日志里就印着那一行，紧接着调用者记下 `ok`。从那一天起，第 11 条那张表没有任何
    东西读过；**而第 11 条讲的正是一个没有任何东西去核对的「记录」。** 今天把四个空格
    补进模式之后它第一次真的跑起来，四条比对全部通过：表本身是对的，从来没有跑过的是
    读它的那个检查。

    第三处方向相反：`tools/audit_report_numbers.py` 把文档引用的**证据**当成了文档的缺陷。
    `LIMITATIONS.md` 逐字引用了包装脚本那一行（`RUN_TAG="${RUN_TAG:?…}"`），因为那行正是
    一次训练丢失的原因，值得原样留下；而「未填模板字段」这条规则把它读成了一个没人填的
    模板字段，拒绝继续，把 42 号 run 的落地停在那里。**这条规则从没跑过一次「文档引用一行
    shell」的输入。** 修法是在两处收窄：`${...}` 是展开而不是模板字段，且格式说明符从不以
    `?` 开头。收窄用两个方向量过：真正没填的字段 —— 包括写在代码片段里的 —— 仍然会让这个
    环节停下来，而引用那行 shell 不再会。这是三处里唯一一处「检查按设计工作了、而设计本身
    是错的」，它和另外两处是同一个病：没有人拿它要读的输入试过它。

    三处合起来是今天最该记的一句：**一个拦不住被检查对象的检查，就是一行日志。**
    这两种形状本文档都分别记过（第 15 条的「机制存在但从未运行，而它自己的自检通过」、
    第 16 条第一条的「失败被调用者印成 WARNING」）。记下来并没有挡住下一次 —— 因为
    共同点不是形状，而是**每一个检查的失败路径看起来都正常**：失败信息印出来了、
    STALE 印出来了、告警印出来了，而没有任何一步的退出码因此改变。
    所以这三处都不是靠读代码发现的，是靠**让它们跑一次**发现的：patch75 的检查直到
    被拿它自己的输出试过才拦人，第 11 项直到把表交给它才第一次比对，未填字段那条直到
    有人拿一份真文档问它才现形。

"""

if MARK in DOC.read_text(encoding="utf-8"):
    sys.exit("item 18 is already in the document; nothing written")

bad = TOKEN.findall(ITEM)
if bad:
    sys.exit(f"the text quotes {len(bad)} decimal(s) {bad}; every decimal in this document "
             f"has to be traceable to a record, and these are measurements of tonight -- "
             f"nothing written")
if ITEM.count("**") % 2:
    sys.exit(f"the text has {ITEM.count('**')} bold markers, an odd number; nothing written")

s = DOC.read_text(encoding="utf-8")
if s.count(ANCHOR9) != 1:
    sys.exit(f"the section 9 heading appears {s.count(ANCHOR9)} times; nothing written")

shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch152")
before_bold = s.count("**")
s = s.replace(ANCHOR9, ITEM + ANCHOR9, 1)
after_bold = s.count("**")
if (after_bold - before_bold) % 2:
    sys.exit(f"bold markers went {before_bold} -> {after_bold}, an odd change; nothing "
             f"written")
DOC.write_text(s, encoding="utf-8")
print(f"section 8 item 18 written ({len(s.splitlines())} lines, bold {before_bold} -> "
      f"{after_bold})")
print("  no decimal is quoted, so the number audit gains nothing new to trace")
