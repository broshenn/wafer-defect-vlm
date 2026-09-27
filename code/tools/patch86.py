"""Record today's two additions to the defect log (section 8).

Item 9 already covers "the numbers are right but the sentence is wrong". A run
landing today produced four instances of that at once, which is worth recording as one
entry rather than four -- it shows the class fires in bulk, and it names which half of
it is now automated and which half is not.

New item 12 is a different defect: the eval script writes its report *before* the
retrieval stages run, so a report on disk can be a complete-looking report with a null
field. In a table with a `retrieval mAP@10` row, that null renders as a blank cell,
which is indistinguishable from a measured zero.
"""
import os
import sys
from pathlib import Path

DOC = Path(os.environ.get("WAFER_DOC", "/root/autodl-fs/wafer-vlm/LIMITATIONS.md"))
s = DOC.read_text(encoding="utf-8")

# ------------------------------------------------ item 9: four instances at once
E1_OLD = "全部数值见 `outputs/reports/kl_length_confound.json`。"
E1_NEW = """全部数值见 `outputs/reports/kl_length_confound.json`。

   今天又添了一组同类实例，而且是**一次 run 落地同时触发四处**（GSPO(G=4) lr5e-5）：
   (1) 5.2.5 里「时钟 MAE 最低的是 SFT」与「尺寸 MAE 最低的是 SFT」两句同时失效 ——
   新 run 两行都更好，而这两句原本只写在加粗里；
   (2) 5.2.4 的「五个 RL run」这个量词当场过期（应为六个）；
   (3) 5.2.5 说整类塌陷「只在 lr 5e-5 的两格 GSPO 上出现」，而新 run 恰好是
   GSPO + lr 5e-5，却一个类别都没丢 —— 句子被推翻；
   (4) `tools/kl_length_confound.py` 的 run 清单是**写死**的，新 run 没有被加进去，
   于是那份记录仍然只覆盖五个 run，而引用它的句子说的是「全部 run 的空转步」——
   **数字全对，范围已经不对**。
   四条都不是算错，四条都只有在有人把新 run 与旧句子对齐时才会暴露。

   **能自动化的那一半现在自动化了**：表格的每一格由 `tools/add_run_columns.py`
   从记录写入（空值拒绝写入，见第 12 条），对照区间由 `tools/paired_significance.py`
   各自独立采样（所以新增 run 不会再移动无关区间），量词计数改成自描述。
   剩下的那一半 ——**哪句话里含量词、哪个量词会被新数据改变** —— 仍然只能靠人核，
   所以 5.2.5 那张表下面留了一句「此表列数随时间增长，新 run 落地后必须重核」。
   该记录在加入新 run 后已重跑，全部数值见 `outputs/reports/kl_length_confound.json`。"""
if s.count(E1_OLD) != 1:
    sys.exit(f"item 9 anchor appears {s.count(E1_OLD)} times")
s = s.replace(E1_OLD, E1_NEW)

# ------------------------------------- item 11: run 43's row in the grad_accum table
E2_OLD = """    | GSPO G=32 lr5e-5 | 32 | 4 |
"""
E2_NEW = """    | GSPO G=32 lr5e-5 | 32 | 4 |
    | GSPO G=4 lr5e-5（新）| 4 | 4（对，原因同下面一行：常量碰巧等于实测值）|
"""
if s.count(E2_OLD) != 1:
    sys.exit(f"item 11 table anchor appears {s.count(E2_OLD)} times")
s = s.replace(E2_OLD, E2_NEW)

# -------------------------------------------------------------- new item 12
E3_OLD = "## 9. 可复现性核对"
E3_NEW = """12. **产物在 run 结束之前就已经存在 ——「报告在盘上」不等于「这次 run 结束了」。**
    `scripts/30_eval_grpo.sh` 的顺序是：先评测、写出
    `outputs/reports/<tag>__report.json`（第 2b 步），然后才合并 adapter、算检索排序，
    再用检索指标**重写同一份报告**（第 3b 步）。所以在合并与检索跑完之前，盘上那份
    报告**已经是一份看起来完整的报告**，只是 `retrieval` 是 `null`。
    5.2.2 那张表有一行 `retrieval mAP@10`：此刻读它，`null` 会变成一个空格子，
    而空格子与「测出来是 0」在表里长得一模一样。
    今天 run 43 恰好落在这个窗口里 —— 它的报告先以 `retrieval: null` 出现，
    约六分钟后第 3b 步把它重写成 0.4138。

    处理方式不是「记得等一会儿」，而是两道机制：
    (a) `tools/watch43_retrieval.sh`：等该 run 的评测进程退出后读报告，若 `retrieval`
    仍为空就**重建**（合并 → 排序 → 重评，逐级判断中间产物是否已存在，因此可续跑），
    若干次尝试后仍为空则在日志里写 `UNMEASURED`，而不是留白；
    (b) `tools/add_run_columns.py` / `tools/add_table_column.py` 遇到空值时**拒绝写入**
    （除非显式 `--allow-unmeasured`，那时写的是「未测」二字，而不是空白）。
    这次的窗口是自行闭合的（GPU 上还有余量，第 3b 步跑完了），所以没有任何产物受损 ——
    但它闭合靠的是余量，不是设计上的保证；并发跑三个任务时余量正好是最少的时候。

## 9. 可复现性核对"""
if s.count(E3_OLD) != 1:
    sys.exit(f"section 9 heading appears {s.count(E3_OLD)} times")
s = s.replace(E3_OLD, E3_NEW)

if s.count("**") % 2:
    sys.exit("bold markers are odd; nothing written")

DOC.write_text(s, encoding="utf-8")
print(f"section 8: item 9 extended, item 11 table gained a row, item 12 added "
      f"({len(s.splitlines())} lines)")
