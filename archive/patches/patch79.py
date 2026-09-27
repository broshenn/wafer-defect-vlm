"""Record today's new verification infrastructure in section 9.

Three additions, each of which closes a way a number could be wrong without anyone
noticing:

  * idle_step_table.py -- the 5.2.2 rates were transcribed by hand; this recomputes
    each from both of the run's logs and compares against the value the run recorded
    for itself, so one number now has three independent sources that must agree.
  * 44_final_consolidation.sh + run_set.py -- the end-of-day rebuild, whose run set is
    enumerated from disk instead of from a hand-written list that goes stale.
  * wait_for_idle.py -- how the rebuild decides the card is free, without the
    filename-vs-launcher self-match that would make it wait forever.
"""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
s = P.read_text(encoding="utf-8")

ANCHOR = "- KL 与补全长度的对照 `tools/kl_length_confound.py`：结论与全部数值见"

NEW = """- 零优势步占比的可复算核对 `tools/idle_step_table.py`：5.2.2 那张表里的每个比率，
  都由该 run 自己的**两份**日志各算一遍，再与该 run 记录的
  `mean_frac_reward_zero_std` 比第三遍，三者必须一致（当前 5 个 run 全部一致：
  36.00%、32.67%、38.00%、15.33%、42.00%）。日志名在各 run 之间并不统一 ——
  GRPO G=4 lr5e-5 的训练日志叫 `29_qwen35_9b_grpo_lr5e5_train.log`，
  不是 `24_grpo_*.log`；GRPO G=4 lr1e-5 的记录叫 `grpo_train_result.json`，
  不是 `<tag>_train_result.json` —— 所以该工具把每条路径**声明**在表里、不从文件名
  推断（按文件名推断会把某份日志接到别的 run 上，而结果看起来与正确答案一模一样），
  并在发现盘上有未声明的、步数达到真实规模的日志时报错而不是跳过。
- 终局重建 `projects/wafer-defect-vlm/scripts/44_final_consolidation.sh`：等卡空闲后用它
  重建比较表。run 集合由 `tools/run_set.py` 从磁盘上的 `*__report.json` **枚举**，
  不读任何手写清单 —— 手写清单在写下时正确、之后静默失效，而「少一列」与「那一列
  从未被测过」在产物里长得完全一样。盘上任何既不在其表内、也不在其排除表内的报告
  都会让它失败，枚举结果作为 provenance 写进 `comparison.json`。
- 等待空闲的判据 `tools/wait_for_idle.py`：不以文件名匹配进程 —— 脚本以 base64 推送执行，
  启动它的那个 shell 的命令行里含有脚本**路径**，按文件名匹配永远清不掉 —— 而是按
  参数特征（`swift/cli/rlhf.py`、`--gradient_accumulation_steps`）与队列驱动的
  **完整命令行精确相等**来判定，两者都是启动器不可能具备的。
"""

if s.count(ANCHOR) != 1:
    sys.exit(f"anchor appears {s.count(ANCHOR)} times, expected 1")
if "idle_step_table" in s:
    print("already applied; nothing to do")
    sys.exit(0)

before = s.count("**")
s = s.replace(ANCHOR, NEW + ANCHOR)
after = s.count("**")
# The added text must contribute whole pairs. An odd delta would leave a bold span
# open, which renders as bold running to the end of the document -- and this
# document has had exactly that bug before (patch67), where an anchor captured only
# a closing marker. Parity, not equality: adding balanced pairs is the normal case.
if (after - before) % 2:
    sys.exit(f"bold markers went {before} -> {after}, an odd change; that would "
             f"leave a span open. Nothing written.")

P.write_text(s, encoding="utf-8")
print(f"LIMITATIONS.md: 3 bullets added ({len(s.splitlines())} lines), "
      f"bold markers {before} -> {after} (+{after - before}, balanced)")
