"""Section 8 item 19: a real disk exhaustion, and the three shapes it hid inside.

Tonight the third seed's merge died with `SafetensorError ... No space left on device
(os error 28)` and left a 4.6 G merged directory that has `.safetensors` in it and no
config -- which is exactly the test `watch_retrieval.sh` uses to decide whether a merge
is needed, so the half-written directory does not merely sit there, it makes every later
attempt skip the merge. The ranking that took it reported `Can't load image processor`,
its caller printed that as `WARNING: GRPO retrieval failed`, the report kept
`retrieval: null` (which the table renders as "not run"), and `land_run.sh` began its
six-hour wait for a field that could not appear.

The fourth finding is independent of the disk and older than the night: the merge stage
of `watch_retrieval.sh` -- the only path that repairs a null retrieval, and the one
`post_consolidation.sh` calls -- passed the checkpoint *root* to `swift export`, which
answered `is not an adapter`. Training writes adapters one level down, so that stage
cannot have worked for any run of this project; the one run whose retrieval was measured
was measured by 30_eval_grpo.sh's own step, which resolves the adapter with a `find`.

Item 19 records these because its subject is not the disk either: it is that the
pipeline's "we already have a product" tests ask whether the file is *there*, and never
whether it is *usable* -- and that a fallback which has never once run looks, in the
file, exactly like a fallback.

No decimal is quoted: LIMITATIONS.md is one of the three documents the number audit
traces every decimal in back to a record, so a decimal here would have to be traceable,
and these measurements belong in the item's prose. The check is in this script.
"""
import pathlib
import re
import shutil
import sys

DOC = pathlib.Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
ANCHOR9 = "## 9. 可复现性核对"
MARK = "一次真实的盘满"

# The audit's own token rule: a decimal of three to six places in this document must be
# traceable to a record. A patch that inserts one it cannot trace inserts a defect.
TOKEN = re.compile(r"(?<![.\d])-?\d+\.\d{3,6}")

ITEM = """19. **一次真实的盘满，和它伪装成的三样东西 —— 外加一条从未跑过一次的修复路径。**
    第三次抽样的评估走到合并那一步时盘满了。报错是 `SafetensorError: ... No space left
    on device (os error 28)`，18:43 写进队列日志 —— 存盘挂在 200 G 的 `/autodl-fs/data`
    上，当时空闲 52 M。这一条记的不是「盘满了」，而是**盘满之后，这套流程把它显示成了
    另外三样东西，而三样都没能把人叫醒**，以及它顺带暴露出的第四件、与盘无关的事。

    第一样，一个**看起来完整的产物**。合并死在写权重的中途，留下一个 4.6 G 的
    `...-seed3409-merged` 目录：`.safetensors` 已经写出来了，缺的是
    `preprocessor_config.json` 和其余配置。而判「要不要重新合并」的那句测试问的恰好是
    **「有没有 `.safetensors`」**。于是这个半成品不但会被当成成品留在那里，还会让下一次
    尝试**跳过合并、直接拿去排序**，并且一直这样做下去。这正是第 14 条那一类：一个与
    完整产物无法区分的产物。与之前记过的几次的区别在于，那几次是「产物在、内容是旧的」，
    这一次是**产物在、却根本不能用**，而唯一能暴露它的线索（少一个配置文件）比要读它的
    那个测试更不起眼。

    第二样，**一次被印成 WARNING 的死因**。排序接手这个目录后报
    `OSError: Can't load image processor`，调用它的那一行把它印成
    `WARNING: GRPO retrieval failed`，然后继续往下走。这是第 16 条第一条的形状；在这里
    代价更具体：那行 WARNING 之后没有任何东西停下。

    第三样，**一个读起来像「没测过」的空单元格**。报告里的 `retrieval` 保持 `null`，
    表格把这个 `null` 渲染成「未运行」—— 也就是「这一项没有做」。真实情况是这一项
    **做了，而且死在半路**。第 14 条记的是「产物与完整产物无法区分」，这里是它的镜像：
    **失败与未尝试无法区分**，而两者的后续处理完全一样 —— 什么都不做。最后 `land_run.sh`
    进入它等这个字段的六个小时上限；**等待超时和正在训练，从外面看一模一样**，这是今天
    第三次撞上这句话（前两次见第 18 条）。

    第四件与盘无关，而且比上面三样都早：**那条唯一能修好检索字段的路径，从未成功运行过
    一次**。`tools/watch_retrieval.sh` 是 `post_consolidation.sh` 用来补 `retrieval` 的
    兜底，做三件事：合并、排序、重新打分。它的合并一步把**检查点根目录**交给了
    `swift export` 的适配器参数，而训练把适配器写在根目录下面一层：
    `<root>/v0-<日期>/checkpoint-150/adapter_config.json`。得到的回答是
    `AssertionError: ... is not an adapter`。也就是说，**这个脚本的合并阶段对本项目的
    任何一个 run 都不成立** —— 早先那些 run 的目录结构完全一样。今天唯一一个检索被真正
    测出来的 run，是被 `30_eval_grpo.sh` 自己的第 3b 步测的，那一步用 `find` 解析适配器，
    所以从来没有碰上这件事。一个**存在、被依赖、却一次也没有跑通**的兜底：第 15 条的
    形状；区别在于它没有自检可以「通过」，它只会在被需要的那一晚印一行失败，然后重试。

    修法和腾空间的做法：适配器的解析改成与 `30_eval_grpo.sh` 同一段 `find`（取目录
    本身、按版本序取最新检查点），并在没有适配器时直接给出「未测量」这个有字的结果，
    而不是拿空参数每分钟调一次 `swift export` 试半小时；腾空间是删掉那个 4.6 G 的半成品
    和三个已经留了排序结果的合并目录，一共约 29 G，**检查点（每个约 19 G，是训练的原始
    记录）和基座模型没有动**。这一条最后要留下的是那句判断：这套流程里「已经有一份产物
    了」的判断不止一处，而它们问的都是**文件在不在**，不是**它能不能用**。

"""

if MARK in DOC.read_text(encoding="utf-8"):
    sys.exit("item 19 is already in the document; nothing written")

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

shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch156")
before_bold = s.count("**")
s = s.replace(ANCHOR9, ITEM + ANCHOR9, 1)
after_bold = s.count("**")
if (after_bold - before_bold) % 2:
    sys.exit(f"bold markers went {before_bold} -> {after_bold}, an odd change; nothing "
             f"written")
DOC.write_text(s, encoding="utf-8")
print(f"section 8 item 19 written ({len(s.splitlines())} lines, bold {before_bold} -> "
      f"{after_bold})")
print("  no decimal is quoted, so the number audit gains nothing new to trace")

# The two anchors the landing's other patch (95) depends on, re-checked now that the
# document has moved: an insertion above section 9 must not disturb them.
P95 = pathlib.Path("/root/autodl-fs/wafer-vlm/tools/patch95_seed3.py")
if P95.exists():
    src = P95.read_text(encoding="utf-8")
    anchors = re.findall(r'^OLD\w*\s*=\s*("""|\'\'\')(.*?)\1', src, re.S | re.M)
    missing = [a[1][:40] for a in anchors if a[1] and a[1] not in s]
    print(f"patch95_seed3.py: {len(anchors)} anchors re-checked against the moved "
          f"document, {len(missing)} missing")
    for m in missing:
        print("  MISSING: " + m.replace("\n", " / "))
    if missing:
        print("  (item 19 sits above section 9 and does not share a line with them)")
