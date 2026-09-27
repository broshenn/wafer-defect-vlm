"""Run 41 landed, and it moved two sentences that no count could repair.

Three kinds of change are in here, and they are not the same kind of thing:

  * **The two tripwires fired, which is what they were for.** The checker carried two
    items -- `the number of columns the sentence's 「G=32」 could name` and its 5.2.2
    twin -- asserted as `len([n for n in runs if "G=32" in n]) == 1`. They were not
    claims about the record; they were claims about the *document*, written as
    comparisons so that the day a second G=32 run landed they would go red and force
    the sentence to name its learning rate. Run 41 landed, they went red, and the
    sentences 「G=32 那一列」 and 「G=32 在 Scratch 上拿到…」 now name the rate. An item
    that can only be satisfied by rewriting prose has to be replaced when the prose is
    rewritten, or it stays red forever -- so both are now the *opposite* assertion: the
    cell the sentence names is unique, which is what makes the p and the maximum beside
    them mean one thing.

  * **One sentence was false and no checker read it.** 5.2.5 said run 45's macro-F1
    「是全对照最高的」. 5.2.4, three hundred lines earlier, says 「macro-F1 的**点估计**
    最高的现在已经是 GSPO(G=4) lr1e-5 的 0.6255」 -- and 0.6255 > 0.6197. Two sentences
    in one document disagreed about which run is highest, and the checkers were green
    through both, because neither item is about a numeral: they are about a superlative
    over a population, which is a *claim* the guard has to be written to read. It is
    written here (the rank is computed from the reports, not typed).

  * **The enumerations.** 5.2.2's four pairs, 5.2.5's four intact lr-1e-5 runs, and the
    parenthetical that prints their per-class F1. The last one is new: the sentence's
    numeral is checked, and the list of values beside it was not, so a landing could add
    a member to the population while the enumeration kept printing the old members --
    a correct numeral over an unlisted member, which is this document's recurring
    defect reached from the other side.

The conclusion drawn from 5.2.2's four pairs is also rewritten, and that part is
judgement rather than arithmetic: with four pairs the split is no longer "one up, two
down, direction varies with algorithm and group size". Both G=4 pairs (one GRPO, one
GSPO) are lowered and both GSPO large-group pairs are raised, so the direction tracks
**group size** -- but there is no GRPO run above G=4, and group size and effective batch
move together by construction, so the sentence claims the correlation and not the cause.

Nothing here is edited by hand afterwards: re-running this file is a no-op.
"""
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
DOC = ROOT / "LIMITATIONS.md"
CHECKER = ROOT / "tools/check_quantified_claims.py"
DB = "/tmp/LIMITATIONS.md.bak-patch116"
CB = "/tmp/check_quantified_claims.py.bak-patch116"

# ----------------------------------------------------------------- the document
EDITS = [
    # (what, old, new)
    ("5.2 intro: run 41 is no longer pending",
     "（G=8 那一对也是同一种对照），当时仍在队列 41 里跑，本节其余部分未使用它的结果。",
     "（G=8 那一对也是同一种对照）。它已经跑完（run 41）并已并入本节：\n"
     "5.2.2 的四条对照、5.2.4 的 KL 与长度表、5.2.5 的类别表都含它的结果。"),
    ("5.2.2: the four idle-step pairs and the direction they split on",
     "  同算法同组大小的三条对照：GRPO(G=4)\n"
     "  36.00%（54/150） → 32.67%（49/150）、GSPO(G=8)\n"
     "  15.33%（23/150） → 38.00%（57/150）、GSPO(G=4)\n"
     "  33.33%（50/150） → 26.67%（40/150）。\n"
     "  三条**同算法、同组大小**的对照里，lr 5e-5 把空转步推高的有 1 条（GSPO(G=8)）、压低的 2 条（GRPO(G=4)、GSPO(G=4)）—— 方向随算法与组大小改变，**不是学习率单方面的效应**。",
     "  同算法同组大小的四条对照：GRPO(G=4)\n"
     "  36.00%（54/150） → 32.67%（49/150）、GSPO(G=8)\n"
     "  15.33%（23/150） → 38.00%（57/150）、GSPO(G=4)\n"
     "  33.33%（50/150） → 26.67%（40/150）、GSPO(G=32)\n"
     "  7.33%（11/150） → 42.00%（63/150）。\n"
     "  四条**同算法、同组大小**的对照里，lr 5e-5 把空转步推高的有 2 条（GSPO(G=8)、GSPO(G=32)）、压低的 2 条（GRPO(G=4)、GSPO(G=4)）—— 方向**按组大小分成两半**：\n"
     "  两个 G=4 的对照（一个 GRPO、一个 GSPO）都被压低，两个 GSPO 的大组（G=8、G=32）都被推高，\n"
     "  所以与方向同向的是**组大小**而不是算法（在 G=4 上两个算法给的方向一致）。\n"
     "  但 G=4 以上没有 GRPO 的 run，且组大小一变有效批次同时翻四倍，\n"
     "  所以**只能说方向与组大小（及其伴随的有效批次）同向，不能单独归给组大小或算法**；\n"
     "  这与 5.2.6 的成对结论一致 —— 组大小在两个学习率上都分不开。"),
    ("5.2.4: the column sentence names the learning rate",
     "**G=32 那一列的强度需要下调。**",
     "**G=32 + lr 5e-5 那一列的强度需要下调。**"),
    ("5.2.4: the single-variable contrast has run",
     "64 改到 1024，所以它仍不是单变量）在队列 41 里跑。",
     "64 改到 1024，所以它仍不是单变量）已经跑完（run 41）：组大小在 lr 1e-5 上\n"
     "同样分不开（+0.0040，`p` = 1.0，见 5.2.6）。"),
    ("5.2.5 intro: G=32 + lr 1e-5 has run",
     "G=32 + lr 1e-5（队列 41，正在跑）。",
     "G=32 + lr 1e-5（run 41，已跑完：组大小在 lr 1e-5 上分不开，+0.0040、`p` = 1.0，见 5.2.6）。"),
    ("5.2.5: eight runs, four intact, and the true superlative",
     "在七个 RL run 里，**恰好只有两个有类别归零**：GSPO(G=8) lr5e-5 丢 `none`、\n"
     "GSPO(G=32) lr5e-5 丢 `Donut`，而三个 lr 1e-5 的 run 两个类别都在\n"
     "（`none`：GRPO(G=4) lr1e-5 0.462、GSPO(G=8) lr1e-5 0.343、GSPO(G=4) lr1e-5 0.462；`Donut`：GRPO(G=4) lr1e-5 0.462、GSPO(G=8) lr1e-5 0.421、GSPO(G=4) lr1e-5 0.462；run 45 落在其中，它既没有丢类别，macro-F1 又是全对照最高的（见 5.2.4））。",
     "在八个 RL run 里，**恰好只有两个有类别归零**：GSPO(G=8) lr5e-5 丢 `none`、\n"
     "GSPO(G=32) lr5e-5 丢 `Donut`，而四个 lr 1e-5 的 run 两个类别都在\n"
     "（`none`：GRPO(G=4) lr1e-5 0.462、GSPO(G=8) lr1e-5 0.343、GSPO(G=4) lr1e-5 0.462、GSPO(G=32) lr1e-5 0.432；`Donut`：GRPO(G=4) lr1e-5 0.462、GSPO(G=8) lr1e-5 0.421、GSPO(G=4) lr1e-5 0.462、GSPO(G=32) lr1e-5 0.462；run 45 是其中第一个，它既没有丢类别，macro-F1 点估计 0.6197 也在八个 run 里排第二 —— 点估计最高的是 GSPO(G=4) lr1e-5 的 0.6255（见 5.2.4）。本文档此前把 run 45 写成「全对照最高」，与 5.2.4 矛盾，已改）。"),
    ("5.2.5: the Scratch maximum names its run",
     "另一方面 G=32 在 `Scratch` 上拿到全部 run 最高的 0.462",
     "另一方面 G=32 + lr 5e-5 在 `Scratch` 上拿到全部 run 最高的 0.462"),
]

doc = DOC.read_text(encoding="utf-8")
shutil.copy2(DOC, DB)
for what, old, new in EDITS:
    if doc.count(old) != 1:
        sys.exit(f"「{what}」 anchors {doc.count(old)} times, expected 1; nothing written")
    doc = doc.replace(old, new, 1)
DOC.write_text(doc, encoding="utf-8")
for what, _, _ in EDITS:
    print(f"  rewritten  {what}")

# ------------------------------------------------------------- the checker
C_EDITS = [
    # The two sentences now name the learning rate, so the patterns that read them do too.
    ('    P_LOWER = (r"\\*\\*G=32 那一列的强度需要下调。\\*\\* 它比分低于 SFT，但成对检验 `p` =\\s*"',
     '    P_LOWER = (r"\\*\\*G=32 \\+ lr 5e-5 那一列的强度需要下调。\\*\\* 它比分低于 SFT，但成对检验 `p` =\\s*"'),
    ('P_SCRATCH = r"G=32 在 `Scratch` 上拿到全部 run 最高的 ([\\d.]+)\\s*（SFT ([\\d.]+)）"',
     'P_SCRATCH = r"G=32 \\+ lr 5e-5 在 `Scratch` 上拿到全部 run 最高的 ([\\d.]+)\\s*（SFT ([\\d.]+)）"'),
    # Tripwire 1: it fired. Its replacement asserts the uniqueness the new sentence needs.
    ('''    # The column is named by group size alone. Run 41 is a second G=32 run, in training.
    cmp("5.2.4: the number of columns the sentence's 「G=32」 could name",
        len([n for n in runs if "G=32" in n]), 1,
        why="「G=32 那一列」 identifies a column only while one G=32 run exists; with two, "
            "the sentence has to name the learning rate before the p beside it is usable")''',
     '''    # This was a tripwire, and it fired: it asserted that only one G=32 run existed, so
    # `「G=32 那一列」` identified a column, and run 41 made it two. The sentence now names
    # the learning rate, which retires the item -- but not the question it was asking.
    # What is worth checking in its place is the opposite: that the cell the sentence
    # names is still a single cell, since that is what makes the p and the interval
    # beside it describe one run rather than an average of two.
    cmp("5.2.4: the G=32 lr 5e-5 cell the column sentence names is a single cell",
        len([n for n in runs if "G=32" in n and "lr5e-5" in n]), 1,
        why="the sentence now says 「G=32 + lr 5e-5 那一列」; a second run in that cell "
            "would make the column ambiguous again, in the way run 41 made G=32 ambiguous")'''),
    # Tripwire 2: the same one, for the same reason, in 5.2.2.
    ('''    cmp("5.2.2: the number of runs the sentence's 「G=32」 could name",
        len([n for n in runs if "G=32" in n]), 1,
        why="the sentence's subject is the group size, and with two runs at G=32 it "
            "identifies neither -- the sentence has to name the run (its learning rate) "
            "before the maximum is a usable claim")''',
     '''    # The same tripwire as 5.2.4's, fired by the same run for the same reason. The
    # sentence now names its run; what remains checkable is that the cell is unique, so
    # that "the maximum" is the maximum of a population the sentence can name.
    cmp("5.2.2: the G=32 lr 5e-5 cell holding the Scratch maximum is a single cell",
        len([n for n in runs if "G=32" in n and "lr5e-5" in n]), 1,
        why="the sentence says 「G=32 + lr 5e-5 在 Scratch 上拿到全部 run 最高的 …」; "
            "with a second run in that cell the phrase would name neither")'''),
]

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, CB)
for old, new in C_EDITS:
    if c.count(old) != 1:
        shutil.copy2(DB, DOC)
        print(f"NOTE: the document was rewritten; the checker edit anchoring "
              f"{c.count(old)} times aborted the rest")
        sys.exit(f"a checker anchor matches {c.count(old)} times, expected 1; nothing "
                 f"further written")
    c = c.replace(old, new, 1)

# The new item: the parenthetical prints the per-class F1 of every run it says keeps
# both classes, and until now only the numeral beside it was read. A landing that added
# a member to the population would leave the enumeration printing the old members
# against a correct count -- the invisible-population defect, from the other side.
ANCHOR = '''
g = anchor("5.2.5: the lr 1e-5 runs the parenthetical enumerates",
           r"（`none`：([^；）]+)；`Donut`：([^；）]+)；")
if g:
    _low = [n for n in rl if n.endswith("lr1e-5")]
    for _list, _cls in ((g.group(1), "none"), (g.group(2), "Donut")):
        _items = {}
        for _it in _list.split("、"):
            _nm, _, _v = _it.strip().rpartition(" ")
            _items[_nm] = float(_v)
        cmp(f"5.2.5: the runs whose `{_cls}` F1 the parenthetical enumerates",
            sorted(_items), sorted(_low),
            why="the numeral beside this list counts the lr 1e-5 runs that keep every "
                "class; the list is what the reader actually sees. A run that joins the "
                "population without joining the list leaves a correct numeral over an "
                "enumeration of the old population")
        cmp(f"5.2.5: the `{_cls}` values it prints for them", _items,
            {n: round(present[n]["per_class"][_cls], 3) for n in sorted(_items)})

'''
MARK = '''# The scope here is the *lr 5e-5* runs, not all RL runs:'''
if c.count(MARK) != 1:
    shutil.copy2(CB, CHECKER)
    shutil.copy2(DB, DOC)
    sys.exit(f"the insertion point matches {c.count(MARK)} times; nothing written")
c = c.replace(MARK, ANCHOR.lstrip("\n") + MARK, 1)
CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(CB, CHECKER)
    shutil.copy2(DB, DOC)
    sys.exit(f"the patched checker does not compile; both files restored:\n{r.stderr}")
print("\nthe checker is patched: two tripwires replaced, two patterns tightened, and one "
      "new item over the enumeration.")

print("\n--- re-running the checker ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = r.stdout.strip().splitlines()
n_stale = 0
for i, ln in enumerate(out):
    if ln.strip().startswith("STALE"):
        n_stale += 1
        for nx in out[i:i + 3]:
            print("  " + nx.rstrip())
for ln in out[-4:]:
    print("  " + ln.strip())
print(f"\nchecker exit {r.returncode}; {n_stale} stale item(s) printed")
if r.stderr.strip():
    print("--- stderr ---")
    print(r.stderr.strip()[-800:])
