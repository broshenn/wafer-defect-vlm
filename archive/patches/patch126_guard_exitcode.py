"""Two things today's guards did wrong, and the one mechanism that let the second hide.

**Recorded in section 8 item 10, not section 9.** Item 10's subject is exactly this: the
tools written to catch the first nine defects, each of which failed at least once. Today
the new checks failed twice more, and both times nothing went red.

  * The new std block keyed its cells `"GSPO " + label`. A no-op for the six GSPO rows,
    a collision for the two GRPO ones -- `GRPO G=4 lr1e-5` became `GSPO G=4 lr1e-5`,
    the same key as the GSPO row below it, so the second assignment replaced the first
    and both GRPO runs left the population the minimum was taken over. Every value the
    block computed was still correct and both of its verdicts still came out right,
    because the two dropped rows are not the cells the sentence names. The only thing
    that changed was the size it printed: 6, beside a sentence about eight.

  * In the same block I read 「在 4 个奖励里有 3 个是最低」's 「4 个」 as the count rather
    than as the population, so the item reported the document wrong when the document
    was right.

Both are this document's own defect class -- a correct numeral over an invisible
population -- committed by the code written to prevent it. They are recorded rather than
quietly fixed because the reason they were harmless is the reason they are dangerous:
they did not change an answer.

**The mechanism.** The checker's exit codes were already overloaded: 1 = a fact moved
(must rewrite), 2 = counts only (--fix may write), 0 = green. Python exits 1 on an
uncaught exception too, so a crash with zero stale items is indistinguishable from a
moved fact to any caller reading only the exit code -- and `tools/land_finish.sh` reads
only the exit code, then prints "STOP: a fact about the run set has moved", which is an
assertion about the document that it never computed. That is the same offence as
everything in section 8, one level up: a mechanism reporting a state it did not observe.

So an uncaught exception now exits 3, and land_finish.sh reports it as a defect in the
checker rather than as a verdict on the document. The explicit `sys.exit("...")` self-abort
paths inside the checker still exit 1 -- converting those is a wider change than today
warrants -- but each prints its own reason, so they are distinguishable by their text if
not by their code. Said plainly here rather than left for a reader to discover.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
FINISH = ROOT / "tools/land_finish.sh"
DOC = ROOT / "LIMITATIONS.md"

# ---------------------------------------------------- 1. a crash is not a verdict
HOOK = '''
# An uncaught exception is not a verdict on the document. The exit codes below are
# load-bearing -- 1 means "a fact has moved, rewrite the sentence", 2 means "counts only,
# --fix may write", 0 means green -- and Python's default exit code for an uncaught
# exception is 1, which a caller cannot tell apart from a moved fact. This happened: an
# insertion landed 40 lines too high, referenced a module-level name not yet assigned,
# and the tool exited 1 having printed zero stale items. `tools/land_finish.sh` reads
# only the exit code and printed "a fact about the run set has moved" -- an assertion
# about the document that nothing had computed. Crashes exit 3 so no caller can confuse
# the two, and the sentence above this line is now true.
def _on_crash(kind, value, tb):
    import traceback
    traceback.print_exception(kind, value, tb)
    print("\\nCHECKER DID NOT FINISH (exit 3): the failure above is in this tool, not a "
          "finding about LIMITATIONS.md. Nothing here says a fact has moved.")
    sys.exit(3)


sys.excepthook = _on_crash
'''

ANCHOR = 'from pathlib import Path\n'
c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, "/tmp/check_quantified_claims.py.bak-patch126")
if c.count(ANCHOR) != 1:
    sys.exit(f"the import anchor matches {c.count(ANCHOR)} times; nothing written")
c = c.replace(ANCHOR, ANCHOR + HOOK, 1)
CHECKER.write_text(c, encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch126", CHECKER)
    sys.exit(f"the patched checker does not compile; restored:\n{r.stderr}")
print("checker: an uncaught exception now exits 3, not 1.")

# ------------------------------------------- 2. land_finish stops confusing the two
FIX_OLD = '''if ! "$PY" "$ROOT/tools/check_quantified_claims.py" --fix; then
  printf '\\nSTOP: a fact about the run set has moved, so no count was written.\\n'
  printf 'The sentence above has to be rewritten with the value printed beside it.\\n'
  exit 1
fi'''
FIX_NEW = '''"$PY" "$ROOT/tools/check_quantified_claims.py" --fix
chk=$?
if [ "$chk" -eq 3 ]; then
  printf '\\nSTOP: the checker did not finish (exit 3). That is a defect in the checker,\\n'
  printf 'not a verdict on the document -- nothing here says a fact has moved.\\n'
  exit 3
fi
if [ "$chk" -ne 0 ]; then
  printf '\\nSTOP: a fact about the run set has moved, so no count was written.\\n'
  printf 'The sentence above has to be rewritten with the value printed beside it.\\n'
  exit 1
fi'''
CONF_OLD = '''if ! "$PY" "$ROOT/tools/check_quantified_claims.py"; then
  printf '\\nSTOP: the confirm pass still fails, so the fix did not settle it.\\n'
  exit 1
fi'''
CONF_NEW = '''"$PY" "$ROOT/tools/check_quantified_claims.py"
chk=$?
if [ "$chk" -eq 3 ]; then
  printf '\\nSTOP: the checker did not finish (exit 3) -- see the confirm pass above.\\n'
  printf 'That is a defect in the checker, not a verdict on the document.\\n'
  exit 3
fi
if [ "$chk" -ne 0 ]; then
  printf '\\nSTOP: the confirm pass still fails, so the fix did not settle it.\\n'
  exit 1
fi'''
f = FINISH.read_text(encoding="utf-8")
shutil.copy2(FINISH, "/tmp/land_finish.sh.bak-patch126")
for old, new in ((FIX_OLD, FIX_NEW), (CONF_OLD, CONF_NEW)):
    if f.count(old) != 1:
        sys.exit(f"a land_finish anchor matches {f.count(old)} times ({old[:40]!r}); "
                 f"nothing written")
    f = f.replace(old, new, 1)
FINISH.write_text(f, encoding="utf-8")
r = subprocess.run(["bash", "-n", str(FINISH)], capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/land_finish.sh.bak-patch126", FINISH)
    sys.exit(f"land_finish.sh fails bash -n; restored:\n{r.stderr}")
print("land_finish.sh: exit 3 is now reported as a checker defect, not a moved fact.")

# --------------------------------------- 3. section 8 item 10, where the subject lives
doc = DOC.read_text(encoding="utf-8")
shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch126")
ANCH = '      已加 `tools/audit_log_provenance.py` 常态化核对「记录引用的日志是不是它自己的」。\n'
if doc.count(ANCH) != 1:
    sys.exit(f"the item-10 anchor matches {doc.count(ANCH)} times; the doc was not "
             f"touched (the checker and land_finish edits above are kept)")

ADD = '''
    - **本节写完这两条之后，当天新写的核对项自己又失败了两次，两次都没变红。**
      `tools/check_quantified_claims.py` 里新加的「组内奖励标准差」块，键写成
      `"GSPO " + 标签`：对 GSPO 的六行是空操作，对 GRPO 的两行却把
      `GRPO G=4 lr1e-5` 变成 `GSPO G=4 lr1e-5` —— 与 GSPO 那一行**同键**，
      后一次赋值覆盖前一次，两个 GRPO run 因此从「取最小值的那个总体」里消失。
      该块算出的每个值、给出的两个判定**全部仍然正确**（丢掉的两行不在它命名的
      两个单元里），所以没有任何东西变红；**变了的只有一个数字** ——
      它打印的总体大小 6，而句子说的是八个。已改为用标签本身作键，
      并把**总体大小本身**加成一个核对项（与 `matched_pairs` 的配对数同理：
      总体是拼出来的，它的规模就是一个断言）。
      同一个块里另一处，我把「在 4 个奖励里有 3 个是最低」的「4 个」读成了计数
      （它其实是总体），于是它报「文档写 4、记录是 0」—— **这次是文档对、核对项错**。
      已把该句改成带数字的形式（「最低的有 0 个」），两个数都读回来。
      这两次都是本文档反复记录的那一类（**正确的数字、看不见的总体**），
      而且是被专门用来防这一类的代码犯下的：它们没有产生错误结论，
      靠的是运气 —— 被丢掉的行恰好不是句子点名的那两个。
    - 一处**机制性**的隐患：**核对工具的退出码被两种含义共享。**
      1 = 某条事实移动了（必须改文档），2 = 只有计数不符（`--fix` 可处理），0 = 通过；
      而 Python 的未捕获异常也退出 1。当天一次插入位置高了 40 行、引用了一个尚未赋值
      的模块级名字，工具**带着「0 条 STALE」崩在中间**，退出码 1 ——
      与「文档里有一条事实移动了」无法区分，而 `tools/land_finish.sh` 正是只看退出码，
      于是它打印「STOP: a fact about the run set has moved」：**一句关于文档的断言，
      而它从未计算过。** 已改为崩溃退出 3，并由 `land_finish.sh` 分开报告
      （工具内显式 `sys.exit("…")` 的自中止路径仍退出 1，但它们各自打印原因）。
'''
doc = doc.replace(ANCH, ANCH + ADD, 1)
DOC.write_text(doc, encoding="utf-8")
print(f"LIMITATIONS.md: item 10 gains two bullets ({len(ADD.splitlines())} lines).")

# ------------------------------------------------------------- 4. both still green
for cmd, name in (([sys.executable, str(CHECKER)], "checker (exit 0 expected)"),
                  (["bash", "-n", str(FINISH)], "land_finish.sh syntax")):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    out = (r.stdout or "").strip().splitlines()
    print(f"\n--- {name}: exit {r.returncode} ---")
    for ln in out[-2:]:
        print("  " + ln.strip()[:150])
