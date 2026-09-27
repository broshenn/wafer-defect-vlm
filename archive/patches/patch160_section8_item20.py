"""Section 8 item 20: two end-of-chain markers -- an inverted guard, and a sentence
that says it recorded something.

Both were found by running tonight's finishing chain, and both are about the markers the
chain uses to prove itself: 44_final_consolidation.sh withholds its completion marker
when FAILED is set, and post_consolidation.sh waits for that marker. The guard that sets
FAILED had `!` in front of its check with the two branches swapped, so it failed exactly
when it passed -- and since the guard was written at 17:34 and first executed at 18:57,
nothing had ever seen it. The other is a WARNING that says it is recording that the pass
ran without a confirmed predecessor; the word 'predecessor' occurs once in that file, in
that sentence, and nothing is recorded.

No decimal is quoted (the audit traces every decimal in this document to a record), and
the text is checked for that before it is written.
"""
import pathlib
import re
import shutil
import sys

DOC = pathlib.Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
ANCHOR9 = "## 9. 可复现性核对"
MARK = "两个终点标记"
TOKEN = re.compile(r"(?<![.\d])-?\d+\.\d{3,6}")

ITEM = """20. **两个终点标记：一个把两个分支写反的守卫，和一句声明自己记下了什么的话。**
    这两处都在今天收尾的那条链上，也都被那条链自己在同一天晚上撞出来。它们比第 18 条
    记的那些更贴近要害一点：第 18 条记的是**检查**拦不住东西，这两处是**标记**在说谎 ——
    而收尾链从头到尾靠的就是标记。

    第一处，`44_final_consolidation.sh` 的等待名单守卫。它是今天下午 17:34 才写下的，
    针对的正是下午真的发生过的那件事：第三次抽样漏在等待名单外面。守卫的形状是

        if ! "$PY" - ... <<'GUARDPY'
        ...
        GUARDPY
        then
          echo "wait list check: covers the enumerated run set"
        else
          echo "FATAL: the wait list does not cover the enumerated run set (see above)"
          FAILED=1
        fi

    那个 `!` 和两条消息的分支是对调的：**检查通过的时候走 else、印 FATAL、并把 FAILED
    设上，而 FAILED 决定完成标记印不印。** 也就是说，从写下的那一刻到今晚，这个脚本
    **不可能**打印它的完成标记；而 `post_consolidation.sh` 的第一步等的正是那个标记。
    没人看见，因为这个守卫**是第一次执行** —— 17:34 写下，44 号下一次运行就是今晚 18:57。
    今晚它确实印出了两句互相矛盾的话：「wait list covers every run run_set.py accounts
    for (12)」和「FATAL: the wait list does not cover the enumerated run set」。两句都是
    真的，是关于不同东西的真话。

    值得注意的是发现方式：不是读代码，是**把两个方向各跑一次**。改前拿真实的守卫文本
    试，检查通过走向 FATAL；改后同样两个方向，通过走向 covers、失败走向 FATAL。
    同一条守卫在写下的那天如果被拿它自己的两种结果试过一次，它活不到今晚 ——
    这正是第 18 条那句「一个拦不住被检查对象的检查，就是一行日志」的又一处，只是这一次
    **检查本身是反的**，它不是在放行，它是在拦住正确的那个结果。

    第二处方向相反：`tools/post_consolidation.sh` 等不到标记时印的那句 WARNING 说它
    「继续，并记下这一趟是在没有确认前驱的情况下跑的」—— **而它什么也没记**。
    `predecessor` 这个词在整个文件里只出现一次，就在那句话里：没有标志、没有产物，结尾
    那行完成标记对前驱一个字也不说。也就是说，一个读者最需要知道的状态（这些文档是在
    没有确认前驱的情况下重建的）被宣布为「已记录」，然后没有被记录。

    修法是今天能给出的最弱、也是诚实的一种：一个标志，加上一个**带限定语的完成标记** ——
    有前驱时印 `post-consolidation pass complete`，没有时印
    `post-consolidation pass complete (WITHOUT a confirmed predecessor)` 并在下面写明
    为什么。这不是新增一个检查，是撤掉一句假话：那两行字现在说明的状态，和实际发生的
    状态是同一个。

    这两处在今晚之所以值得单独记一条，是因为它们都长在**收尾链自己的标记**上。一条链
    的每一环都用「上一环印了那句话」来确认自己不是在没有前驱的情况下开工，而这一环的
    守卫写反了、那一环的 WARNING 说了自己没做的事 —— 于是从 17:34 到 19:06，没有任何
    东西真正检查过这条链是否完整；它靠一句 WARNING 和一行 FATAL 的措辞来自证清白。
    收尾链重新跑过一遍之后，标记是干净的、等待名单守卫两个方向都验过，但这一条要记的
    不是「修好了」，而是**这条链的完整性一直是由它自己不印的那句话来保证的**。

"""

if MARK in DOC.read_text(encoding="utf-8"):
    sys.exit("item 20 is already in the document; nothing written")

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

shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch160")
before_bold = s.count("**")
s = s.replace(ANCHOR9, ITEM + ANCHOR9, 1)
after_bold = s.count("**")
if (after_bold - before_bold) % 2:
    sys.exit(f"bold markers went {before_bold} -> {after_bold}, an odd change; nothing "
             f"written")
DOC.write_text(s, encoding="utf-8")
print(f"section 8 item 20 written ({len(s.splitlines())} lines, bold {before_bold} -> "
      f"{after_bold})")
print("  no decimal is quoted, so the number audit gains nothing new to trace")
