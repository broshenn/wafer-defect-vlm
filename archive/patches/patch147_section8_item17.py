"""Section 8 item 17: the human-review merge read a typo as an assertion about the data.

Item 16 records today's defects on the training and landing path. This one is on the
human path and deserves its own item for that reason: `tools/merge_review.py` is the step
that turns two people's sheets into the numbers the freeze decision rests on, and its
decisive branches had never executed with a single label present -- the agreement.json on
disk came from two empty sheets, so `pairs` was empty, cohen_kappa returned before its
division, and neither `confirmed` nor `gold_mismatch` was ever incremented.

The defect: an out-of-vocabulary label was sorted into `gold_mismatch` ("the automatic
label is wrong, which is a benchmark defect rather than a reviewer problem") or into
`reviewer_disagreement` ("genuinely ambiguous sample"), and the tool exited 0. Measured on
synthetic sheets in /tmp, then fixed by checking the labels against the core file's own
`failure_type` set. The specific rates are not quoted: they are measurements of a synthetic
fixture, and this document's numbers are audited for traceability to records, so a number
nothing recomputes does not belong in it. What is quoted is the shape -- which counts moved
and by how much in units of samples, and that the exit code said success.

The last paragraph is the missing `import sys` in the patch that added this very error
message: `py_compile` cannot see a name that does not exist, and only running the new
branch reaches the line. It happened while this item was being written.
"""
import pathlib
import shutil
import sys

DOC = pathlib.Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
ANCHOR9 = "## 9. 可复现性核对"
MARK = "把笔误当成对数据的断言"

CROSSREF_OLD = """    分支真的跑一遍之后才现形** —— 它们一直静静躺在盘上，或者只在第二个种子落地的那一刻
    出现。
"""
CROSSREF_NEW = """    分支真的跑一遍之后才现形** —— 它们一直静静躺在盘上，或者只在第二个种子落地的那一刻
    出现。（人工审核那一步的工具今天另有一处，见下一条。）
"""

ITEM = """17. **一个把笔误读成对数据的断言的工具，而且它退出 0。**
    `tools/merge_review.py` 是人工双审之后的合并步骤：读两份评审表，把每个样本分成三类
    —— 两位评审彼此一致且与自动标注一致（确认）、彼此一致但与自动标注不同（**说明自动
    标注是错的，是基准自身的缺陷**）、彼此不一致（样本本身有歧义）。它检查过两份表里的
    `defect_class` 是不是一个合法类别吗？没有：`normalise` 只去掉首尾空白。于是把
    `Edge_Loc` 写成 `Edgeloc` 的样本进的是第二类 —— **报告会说基准的自动标注在那一处
    是错的**，而那只是一处拼写；把 `Center` 写成 `center` 的进的是第三类，报告会说那是
    一个「本身有歧义的样本」。两类都是对数据的断言，来源却是键盘。

    这件事一直没有被看见，是因为这条路径**从来没有在有任何一个标签的情况下跑过一次**：
    盘上的 `agreement.json` 是两份全空的表跑出来的，`pairs` 是空列表，kappa 在除法之前
    就返回了 None，确认与不一致两条计数一次也没有增加过。而基准尚未冻结，所以这一步正是
    **把两个人工的标注变成数字**的那一步。

    已用两份**合成**表把这条路径跑了一遍（在 /tmp 里，绝不进入基准目录，也绝不冒充人工
    审核）：八处不合词汇表的写法把确认与不一致之外的计数整体抬高 —— gold_mismatch 多出
    四处、reviewer_disagreement 多出四处，报告的一致率随之被拉低，而工具**退出 0**。
    一份这样被污染的报告与一份干净的报告，在读者眼里完全一样。修法是：合法词汇表就是
    被合并的那份 core 文件里的 `failure_type` 集合（正是 README 让两位评审从里面挑的那
    九个），词表之外的写法单独成一类、**不进入任何统计**，并且工具退出 3；报告仍然打印，
    因为读者正需要它去修那两份表。

    修完用同一份合成表重跑：一致率回到与该脚本**独立重算**的值逐位一致，八处笔误全部落在
    新的一类里，`--update-core` 只改确认过的那些行、每行只加两个字段，真实的基准目录逐
    字节未动。

    还有一处值得记，因为它正是本条要说的事：**这个修补自己漏了 `import sys`**，而
    `py_compile` 看不出一个不存在的名字 —— 只有把新的错误分支真的跑一遍，才会在需要它的
    那一行上崩掉。它发生在同一天，就在写这条记录的过程中。

"""

s = DOC.read_text(encoding="utf-8")
if MARK in s:
    sys.exit("item 17 is already in the document; nothing written")
if s.count(ANCHOR9) != 1:
    sys.exit(f"the section 9 heading appears {s.count(ANCHOR9)} times; nothing written")
if s.count(CROSSREF_OLD) != 1:
    sys.exit(f"item 16's opening matches {s.count(CROSSREF_OLD)} times; nothing written")

shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch147")
s = s.replace(CROSSREF_OLD, CROSSREF_NEW, 1).replace(ANCHOR9, ITEM + ANCHOR9, 1)
DOC.write_text(s, encoding="utf-8")
print(f"section 8 item 17 written; document now {len(s.splitlines())} lines")
