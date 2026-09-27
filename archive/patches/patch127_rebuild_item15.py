"""Section 8 item 15: the mechanism that replaced the hand-written run lists never ran,
and its own guard passed the whole time.

Found by reading a log line rather than by a check: `WARNING: make_report.py returned
non-zero`, sitting in `logs/after_queue_rebuild.log` since 16:34 and answering a question
nobody had asked. The script's name, its docstring, and the sentence in section 9 that
describes it were all correct -- the tool exists, is called, enumerates the run set from
disk, and prints how many runs it found. It just never got a usable answer back from
`make_report.py`, because the arguments it built were malformed, and it logged that fact
at WARNING severity and carried on.

The malformation is worth recording precisely. `run_set.py --emit runs` prints
`--run Name=path`, one argument *pair* per line. Both readers in these scripts
(`while IFS= read -r line; do RUN_ARGS+=("$line"); done` and `mapfile -t`) split on
NEWLINES, so each line became one argv word containing a space. argparse reads such a
token as the option `--run Name` plus a stray `=path`, and answers "the following
arguments are required: --run". So the array was non-empty, the count of runs was
correct, the guard that existed to catch a bad run set passed, and the rebuild never
happened -- the comparison on disk stayed whatever the last *hand-written* list had
produced. Today that was queue 41's list at 16:33: eight runs, two fewer than the reports
on disk, so FINAL_REPORT.md's "全部超过基线" claims were computed over eight while its own
per-run prose covers ten. That is section 8 item 9's failure, in the one artefact no
checker reads.

Two fixes, and the second is the one that generalises:
  * read the pairs as pairs, and assert the array's *shape* (its first element must be
    the flag itself) rather than only its non-emptiness -- the old check asked a question
    whose answer was never in doubt, which is the same mistake as the population-numeral
    and key-collision items in item 10;
  * no rebuild step prints the completion marker any more unless every step returned
    zero. An unconditional marker reports a state nothing checked, and
    post_consolidation.sh keys on that phrase to decide whether to start work at all.

The general sentence, which is why this is worth a numbered item rather than a commit:
**"the mechanism exists" and "the mechanism ran" are different facts, and only the second
one is ever in a log.**
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
DOC = ROOT / "LIMITATIONS.md"
CHECKER = ROOT / "tools/check_quantified_claims.py"
AUDIT = ROOT / "tools/audit_report_numbers.py"

ANCHOR = """    散文里不存在合法的这种形态，因此不需要人读一遍再判断。

## 9. 可复现性核对"""

ITEM = """    散文里不存在合法的这种形态，因此不需要人读一遍再判断。

15. **一个用来取代手写清单的机制从未运行过一次，而它自己的自检通过了。**
    §9 写着「run 集合由 `tools/run_set.py` 从磁盘上的 `*__report.json` **枚举**，
    不读任何手写清单」。三个重建脚本（`44_final_consolidation.sh`、
    `tools/post_consolidation.sh`、`tools/after_queue_rebuild.sh`）都是照这句话写的：
    枚举、转成 `--run Name=path`、交给 `make_report.py`。但读取那一端写成了
    `while IFS= read -r line; do RUN_ARGS+=("$line"); done` 与 `mapfile -t` ——
    **两者都按行切，而 `--emit runs` 的一行是两个 argv 词**（`--run` 与 `Name=path`）。
    于是 argparse 收到的是一个含空格的词 `"--run Name=path"`，它读成选项
    `--run Name` 加一个多余的 `=path`，回答
    `error: the following arguments are required: --run`。
    **这三个脚本的重建因此从未成功过**：`comparison.json` 一直是「最后一个成功写出它
    的人」的版本，而那个人是某个队列脚本里**手写**的 run 清单 —— 正是这些机制存在
    的意义所在。今天 16:33 最后一次写它的是队列 41 的清单：8 个 run，比盘上少两个
    （GSPO G=4 的两个 run 都在该清单写成之后才落地）。于是 `FINAL_REPORT.md` 里
    「全部超过基线」这类**关于「全部」的断言，是在 8 个 run 上算出来的**，
    而同一文件的逐 run 小节有 10 个 —— 这是第 9 条那一类失败，落在**没有任何核对工具
    读取的文件**里。
    自检为什么没发现：它只问「数组是不是空的」，而整行本身也是一个非空元素；
    **它检查的是一个从未有疑问的性质**（与第 10 条两次失误同形）。
    已修：按「对」读（`while read -r flag spec`），并把**数组形状**当作要断言的东西
    （首元素必须是 `--run`），计数改成对数；同时，任何重建步骤非零退出时不再打印
    完成标记 —— 无条件打印的完成标记报告的是一个从未被检查过的状态，而
    `tools/post_consolidation.sh` 正是靠这句话决定要不要开工。
    发现方式值得记下来：不是核对工具报的，是读日志时看到一行
    `WARNING: make_report.py returned non-zero`。它已经在那里躺了半小时，
    严重性被写成 WARNING，而脚本名、它的 docstring 和 §9 的句子全是对的。
    **「机制存在」与「机制运行过」是两件事，只有后者能被日志证明。**

## 9. 可复现性核对"""

doc = DOC.read_text(encoding="utf-8")
shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch127")
if doc.count(ANCHOR) != 1:
    sys.exit(f"the item-15 anchor matches {doc.count(ANCHOR)} times; nothing written")
DOC.write_text(doc.replace(ANCHOR, ITEM, 1), encoding="utf-8")
print(f"LIMITATIONS.md: item 15 added "
      f"({len(ITEM.splitlines()) - len(ANCHOR.splitlines())} lines, "
      f"{len(DOC.read_text(encoding='utf-8').splitlines())} lines total).")

# ------------------------------------------------------------------- 2. the guards
for tool, args in ((CHECKER, []), (AUDIT, [])):
    r = subprocess.run([sys.executable, str(tool)] + args, capture_output=True, text=True,
                       cwd=str(ROOT))
    out = (r.stdout or "").strip().splitlines()
    print(f"\n--- {tool.name}: exit {r.returncode} ---")
    for ln in out[-6 if tool is AUDIT else -3:]:
        print("  " + ln.strip()[:150])
