"""Log the document-level version of the "one fact, two copies" failure.

LIMITATIONS section 8 records defects that silently produced plausible results.
Items 5 and 6 are both "the same fact stored under two names, and only one copy
corrected". This adds item 7 for the same failure at the document layer, found by
the new audit tool, plus a line in the reproducibility section so a reader can
re-run the check.
"""
import ast
import sys
from pathlib import Path

p = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
s = p.read_text(encoding="utf-8")

ITEM = '''7. **文档里有四个数字没有任何记录支撑 —— 它们只存在于散文里。**
   本项目的规矩是「报告里的每个指标都能回溯到一次实测」，但这条规矩此前**没有工具
   在执行**。本轮新增 `tools/audit_report_numbers.py`，把三份文档里每个三位以上小数
   的数字拿去比对 `outputs/` 下的全部记录，首跑就发现四处没有出处。三种漏法各不相同：

   - **同一事实的两份硬编码副本。** 「`Edge_Ring` 的中位质心半径只有 0.063」这句话
     同时写在 `tools/final_report.py` 的正文和 `tools/radial_zone_audit.py` 的模块
     docstring 里，**两处都是字面量，没有一处是算出来的**。数据若变，两句话都会继续
     读得很通顺。现值经重算确认为 0.0631，所以此前没有出错 —— 但那是运气。
   - **两个分母各说各话。** 「全基准 83.7% 如此」用的是 251 行（剔除 1 行未标注），
     而记录里的 `majority_floor` 用的是 252 行、得到 0.8333。两个数各有依据，但
     **它们互相矛盾，而没有任何地方提示这一点**。现已把两个分母都写进记录并具名
     （`of_all` / `of_scored`），正文改为直接打印分数「251 行里有 210 行（83.7%）」。
   - **工具算了、打印了、没存。** `tools/grpo_reward_trace.py` 输出的四个 reward
     分量均值/标准差（`0.9998 / 0.8650 / 0.6283`）只存在于 stdout，而第 6 节正是
     引用这三个数论证「时钟奖励承载训练信号」。已加 `--output`，重跑核对全部吻合。

   另有一处不属于上面三类：**SFT 是唯一没有 `*_train_result.json` 的训练 run**，
   它的汇总（`train_loss 0.3891`、峰值显存 `32.92 GiB`）只存在于 `logs/08_train.log`。
   已新增 `tools/summarise_training_log.py`，从日志**转写**该汇总块，并在记录里注明
   来源文件与行号 —— 转写与编造的区别在于，前者可以被回查。

   四个值经重算**全部属实**，所以这次没有需要更正的结论；需要修的是「引用」这件事本身：
   `radial_zone_audit.py` 现在把每类中位半径和两个分母都算出来写进记录，
   `final_report.py` 改为**读回**而不是复述，且记录缺失时句子降级为「见 JSON」，
   而不是回退到记忆里的旧值。

   **教训：没有工具执行的规范等于没有规范。** 这条规矩此前靠人记着，于是四处失守而
   无人察觉。核对后不可溯源的数字从 31 个降到 4 个；剩下的 4 个全部是两个已记录数字
   之差（Base→SFT 的增幅、5e-5 → 1e-5 的降幅），属于算术而非实测，本就不该有记录。
   与第 5、6 条同源 —— **同一事实存两份，只改一份** —— 只是这次发生在文档层。

'''

anchor = "这几处的共同点："
if s.count(anchor) != 1:
    sys.exit(f"FAILED: closing-paragraph anchor appears {s.count(anchor)} times")
s = s.replace(anchor, ITEM + anchor)
print("  section 8: item 7 added")

old9 = "- 未能自动化的只有人工双审（见第 1 节）。"
new9 = ("- 数字溯源核对 `tools/audit_report_numbers.py`：三份文档共 378 个指标数字，其中 4 个\n"
        "  无法回溯到任何记录，且此 4 个全部是已记录数字之间的差值（算术量），无实测数字缺失。\n"
        "  复跑：`venvs/wafer/bin/python tools/audit_report_numbers.py`。\n"
        "- 未能自动化的只有人工双审（见第 1 节）。")
if s.count(old9) != 1:
    sys.exit(f"FAILED: section 9 anchor appears {s.count(old9)} times")
s = s.replace(old9, new9)
print("  section 9: audit line added")

p.write_text(s, encoding="utf-8")
print("LIMITATIONS updated")
