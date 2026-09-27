"""Record the failure caused by editing a script that was still executing.

Section 8 is the project's log of defects that silently produced plausible
results. This one differs from the others in two ways worth writing down: it
produced no result at all rather than a wrong one, and it was caused by the way
this project is worked on -- patches pushed to the server while jobs are running
-- rather than by the code under test.
"""
import sys
from pathlib import Path

p = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
s = p.read_text(encoding="utf-8")

ITEM = '''8. **在脚本仍被执行时修改该脚本，使启动器在训练完成之后、写记录之前崩溃。**
   本轮为让训练种子可覆盖，给 `scripts/29_grpo_train.sh` 插入了一行（patch44，12:23），
   而该脚本有一个 run 是 12:18 启动、当时**仍在执行中**。bash 是按**字节偏移增量读取**
   脚本的：插入的那一行让其后所有内容整体后移，bash 恢复到旧偏移时正好落在内嵌 Python
   代码块的中间，于是拿 Python 当 shell 执行 —— 报
   `line 96: syntax error near unexpected token '('`，脚本在**训练已经正常跑完之后**退出，
   记录从未写出。

   危害不在训练（训练是真的，checkpoint 与完整日志都在），而在于它的表现方式：
   等待该记录的 driver 会一直等到 3 小时超时，而它等的那个文件**永远不会出现**，
   全程没有任何一处报错。**一次编辑让整个下午的队列静默停摆。**

   已用启动器**自身的那段 writer 代码**（从当前脚本第 75–165 行原样抽出）配合该 run
   当时的真实环境变量补出记录，使其出于与其它记录**同一套代码**而非第二套实现；
   并在记录里加了 `record_note`，说明该记录是事后补写的、以及每个字段各自的来源
   （步级字段来自该 run 自己的日志，`elapsed_seconds` 与 `gpu_peak_mib` 来自日志里的
   `train_runtime` 与启动器自己采样的显存峰值）。

   **教训：还有 run 在跑时，不要改动它正在读的脚本。** 若必须改，先等 run 结束，
   或把改动放进 run 读不到的新文件。本项目的其余条款防的是「产出貌似合理的错误数字」，
   这一条防的是「什么都不产出且没人发现」。

'''

anchor = "这几处的共同点："
if s.count(anchor) != 1:
    sys.exit(f"FAILED: anchor appears {s.count(anchor)} times")
s = s.replace(anchor, ITEM + anchor)
p.write_text(s, encoding="utf-8")
print("section 8: item 8 added")
