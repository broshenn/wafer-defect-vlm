"""Record the second overwrite bug in LIMITATIONS section 8.

Section 8 collects defects that produced plausible-looking results silently.
This is the third instance and the second one to destroy a record: the launcher
passed the per-run result path to its embedded Python, which unpacked it and then
wrote to a hardcoded literal instead. The GRPO record survived twice only because
it had already been committed.
"""
import sys
from pathlib import Path

path = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
src = path.read_text(encoding="utf-8")

anchor = ("这几处的共同点：**失败被伪装成正常结果** —— 返回类型正常、数字看起来合理、\n"
          "不报任何错。记录在此以免复现。")

item6 = '''6. **`scripts/29_grpo_train.sh` 把结果写到写死的路径，第二次覆盖了 GRPO 记录。**
   启动脚本把每次 run 的结果路径作为 `argv[1]` 传给内嵌的 Python，Python 也确实
   把它解包成了 `result` —— **然后完全没有使用它**，写文件时用的是字面量
   `outputs/reports/grpo_train_result.json`。patch29b 只改了喂给 `argv[1]` 的那个
   shell 变量，从名字上看已经参数化，所以第二个 GSPO run 是按「记录已隔离」启动的。
   实际没有隔离：lr 1e-5 的结果覆盖了 GRPO 记录，和第一个 GSPO run 犯的是同一个错。
   **两次都只是因为 GRPO 记录此前已被 git 提交才得以恢复 —— 是运气，不是设计。**
   已改为写 `Path(result)`。
   **教训：`argv[1]` 被解包不等于被使用；「看起来参数化了」与「确实隔离」是两件事。**
   与本条第 5 点合起来说明：**同一份数据被两个名字指代**（共用日志、写死路径）
   是本项目最容易产生「貌似正常」结果的结构，改完一处必须回头查其余同类处。

'''

if anchor not in src:
    sys.exit("FAILED: anchor paragraph not found; section 8 layout changed")
if src.count(anchor) != 1:
    sys.exit("FAILED: anchor is not unique")

src = src.replace(anchor, item6 + anchor)
path.write_text(src, encoding="utf-8")
print("LIMITATIONS section 8: overwrite bug recorded as item 6")
print(f"file now {len(src.splitlines())} lines")
