"""Record the count-sync tool and the defect its first version produced.

Section 8 of LIMITATIONS.md is the defect log. Four sentences in this document carry
"how many runs" counts, and three more runs land today, so a tool now rewrites them
from the records (tools/sync_counts.py). Its first version kept every number right
and still damaged the sentence it edited: the count came out as "当前 六 个 run"
(the line's own style is an Arabic digit, and the numeral was spaced away from 个)
and the per-value percent signs were dropped, leaving one % at the end of the list.
Both are the same shape as item 9 -- the artefact looks fine, the sentence about it
does not -- but the mechanism is new, so it gets its own entry: a tool that rewrites
prose by pattern is itself an artefact that can be wrong while its output is
plausible.

The mitigation is the same one used for the tables: rehearse against the shipped
state and require a no-op. `--check` must report "would change: nothing" before the
write path is allowed to run; that is how the two defects above were caught.
"""
import os
import sys
from pathlib import Path

DOC = Path(os.environ.get("WAFER_DOC", "/root/autodl-fs/wafer-vlm/LIMITATIONS.md"))
s = DOC.read_text(encoding="utf-8")

ANCHOR = "## 9. 可复现性核对"
NEW = """13. **按模式改写的工具自己也是产物 —— 数字全对，句子被改坏。**
    本节四条句子里带「几个 run」的计数，今天还有三个 run 落地，所以计数改成由记录
    生成（`tools/sync_counts.py`：逐 run 空转步与总数读 `kl_length_confound.json`，
    空转率列表读 `idle_step_table.py` 自己那行「in this table's order」——顺序是那个
    工具的决定，不是同步脚本的猜测，RL 记录数按 `*_train_result.json` 数）。
    该工具**第一版**把四条里的一条改坏了，而四个数字一个都没错：
    (a) 句子的计数写成「当前 六 个 run」（该行自己的写法是阿拉伯数字，且「六」与
    「个」之间多了一个空格）；
    (b) 每个数值的百分号被吃掉，只在列表末尾留了一个 `%`。
    两条都不是算错，是**改写本身没有对「改写前后应当相同」这件事做检查** ——
    与第 9 条同一类，但机制是新的：一个按模式重写散文的工具，它自己就是一件可以
    出错而输出看起来很正常的产物。
    处理方式同样是先演习再动手：`tools/sync_counts.py --check` 必须在**当前盘上
    的文档**上报出「would change: nothing」，之后才允许写入；上面两条就是这样被
    抓住并修回的（现在重跑 `--check` 为 no-op，四条句子与记录逐字一致）。
    顺带记下这件事的残余风险：这条句子里的计数仍是**人写的措辞**（「当前 6 个
    run 全部一致」），工具只保证数字与列表正确，不保证措辞仍然成立。

"""
if s.count(ANCHOR) != 1:
    sys.exit(f"section 9 anchor appears {s.count(ANCHOR)} times; nothing written")
s = s.replace(ANCHOR, NEW + ANCHOR, 1)

if s.count("**") % 2:
    sys.exit("bold markers are odd; nothing written")

DOC.write_text(s, encoding="utf-8")
print(f"section 8: item 13 added ({len(s.splitlines())} lines)")
