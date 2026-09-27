"""Section 8 item 16: today's defects, which is where they belong rather than in a log.

Four of today's five defects share one shape: a mechanism that exists and whose decisive
branch never runs. sync_counts exited on a dead anchor and its caller printed that as a
WARNING; patch94's seed parenthetical could never appear because the function appends
its own suffix to a stem that was already the file name; wait_for_idle's queue list was
two hand-written command lines and queue 43 was launched after 44 had begun waiting, so
44 would have rewritten LIMITATIONS.md during that queue's own landing -- and the
enumeration that replaced the list first generated a path no process has, because
`/root/autodl-fs` is a symlink. The fifth is different in kind: the checker's cell
lookup took whichever run the dict happened to write last, so a second seed silently
became the owner of a cell whose pair is built from the first -- and the fix was a
declaration, not a tie-break.

None of the four produced a wrong number. All four produced a plausible result. Three of
them were found only by running the branch instead of reading it: rendering patch94
offline and seeing "seeds: (not in the records; omitted)" while both records were on
disk, and calling `busy()` against the live process table instead of describing what it
would match. That is the point worth recording -- this document's recurring failure is
an artefact that cannot be told from a complete one, and the only general defence found
so far is to execute the branch and read its output, not to re-read the source.

Written as a patch so it can be applied when nothing else is writing the document: the
landing and 44_final_consolidation both run sync_counts and the checker over this file,
and two writers to one document is the race the last sub-point is about. After it is
applied, tools/land_finish.sh re-derives the counts it touches -- including the two
numbers in section 9 that sync_counts reads back from the audit summary, which move
whenever this document gains numbers.
"""
import pathlib
import shutil
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
DOC = ROOT / "LIMITATIONS.md"
ANCHOR9 = "## 9. 可复现性核对"
MARK = "今天四处缺陷"

ITEM = """16. **今天的四处缺陷：两处「从未运行过的分支」，一处「失败被记成 WARNING」，
    一处「靠字典顺序猜出来的答案」。**
    四处都不是数字错，四处都产出过看起来合理的结果。记在这里是因为它们**只有在把那个
    分支真的跑一遍之后才现形** —— 前两类一直静静躺在盘上，第三类被调用方的一句
    `WARNING` 掩盖，第四类只在第二个种子落地的那一刻出现。

    - `tools/sync_counts.py` 的**两处锚点早已失配**：它每次都在第一个锚点处 `sys.exit`，
      什么都没写，而调用它的那一行把非零退出打印成 `WARNING` 继续往下走。于是那一轮
      它一个数字也没重算，文档里的计数**保持上一次写入的值** ——
      「计数滞后」与「计数写入失败」在日志里长得一样，只有其中一个可以安全地跨过去。
      还有一层：该工具有 `--check` 模式，锚点活着时它**永远退出 0**，所以「检查通过」
      并不等于「写得出」。已按实际文本放宽两处锚点，并把调用方改成**拒绝就停**
      （`tools/land_finish.sh` 里那段 STOP 就是这次写的）。**一个失败被记成 WARNING
      的核对工具，等于没有这个工具**，而它的失败方向是「文档里的数字停在上一次」，
      没有任何东西会因此变红。
    - `tools/check_quantified_claims.py` 把文档里的格子名（`GRPO(G=4) lr1e-5`）解析成
      run 时用的是 `{(G, lr, is_level): (tag, entry)}` 这样一个字典推导式。**同一个格子
      在第二个种子落地后有两个 run，推导式留下的是最后一个**，于是 5.2.4 的 GRPO(G=4)
      KL 读成了第二个种子的值；记录里那条「配对与格子是否自洽」的核对随即报红 ——
      **看上去是文档错了，实际上文档引用的正是它自己那条配对所用的那个 run**。
      修法是一条**声明**而不是一条仲裁规则：记录里新增 `SEED_REPEATS`（哪个 run 重复
      哪个）与逐 run 的 `repeat_of`，格子表只由非重复项构成，另有两条核对专门验证这条
      声明本身（重复项必须指向记录里存在的 run，且必须与它同格）。
      「最后写入的胜出」是字典的性质，不是任何人的决定；这里想到过的每一种仲裁
      （取 tag 最小、取 `RUNS` 里最靠前、取空转步最少）都是猜，而且都会一直读到
      第三个种子落地才露馅。
    - `tools/idle_step_table.py` 的 `RUNS` 里**预先声明**了还没启动的第三个种子
      （这是故意的：落地后再声明，那个 run 会以 UNACCOUNTED 出现）。这立刻让
      「每个 run 的日志都要声明梯度累积」一条变红，报告的是**文档**的缺陷 ——
      而文档没错，run 只是还没开始跑。这条核对自己的措辞里就有答案：
      「日志读不到的 run 是量词之外的 run」，而**尚未启动的 run 不是那件事**。
      已改成读训练记录的 `outcome` 与 `steps_logged` 来区分两种状态：未跑完的
      **打印出来但不计入**（与 `idle_step_table` 对「还没有值的日志」的处理一致），
      跑完却读不到日志的仍然失败 —— 那正是这条要抓的。时机上它尤其要紧：
      run 42 的落地会跑同一个核对，一次善意的提前声明本来会让整晚的链条停在一句
      「某个还没启动的 run」上。
    - `tools/patch94_seed.py` 里生成「（3407 → 3408）」这个小括号的函数**走不到成功分支**：
      `_seed(stem)` 自己会拼上 `_train_result.json`，而两处调用传的是文件名本身，
      于是它去找 `grpo_train_result_train_result.json`，两次都返回 None，句子就按
      **设计好的兜底**把括号省掉。兜底是对的（宁可不写，也不写一个读者无法核对的
      字面量），问题是**一条路径错误能走到兜底**。发现方式是**离线渲染**：把该补丁指向
      文档的副本、记录指向 /tmp 里的同构夹具跑一遍，打印出
      「seeds: (not in the records; omitted)」，而 `ls` 显示两份记录都在盘上。
      修完再渲染一次，句子才写出「（3407 → 3408）」。
      同一形状今天出现第二次：`tools/wait_for_idle.py` 里等待队列驱动的清单是**手写的
      两行命令行**，而第三个种子的队列是在 44 已经开始等待之后才启动的 —— 那个等待者
      会在该队列自己的落地过程中开始重写同一份 `LIMITATIONS.md`，正是
      `land42_driver.sh` 存在的原因：「两个写者对同一份文档是没有好结果的竞态」。
      已改成**从磁盘枚举** `*_queue_*.sh`。枚举完还要再验一次：第一版用了
      `Path(__file__).resolve()`，而 `/root/autodl-fs` 是指向 `/autodl-fs/data` 的软链接，
      于是它枚举出的是**没有任何进程在用的那个拼写**，一个都不匹配 ——
      与它替换掉的那张手写清单是同一种失败，只是外面多了一层代码。
      这一条是把 `busy()` 对着**真实进程表**跑出来才看见的。

"""

s = DOC.read_text(encoding="utf-8")
if MARK in s:
    sys.exit("item 16 is already in the document; nothing written")
if s.count(ANCHOR9) != 1:
    sys.exit(f"the section 9 heading appears {s.count(ANCHOR9)} times; nothing written")
if s.count("15. **一个用来取代手写清单的机制") != 1:
    sys.exit("item 15 is not where this patch expects it; nothing written")

shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch141")
before = s.count("**")
s = s.replace(ANCHOR9, ITEM + ANCHOR9, 1)
after = s.count("**")
if (after - before) % 2:
    sys.exit(f"bold markers went {before} -> {after}, an odd change; nothing written")
DOC.write_text(s, encoding="utf-8")
print(f"section 8 item 16 written ({len(ITEM.splitlines())} lines, "
      f"document now {len(s.splitlines())})")
