"""The item I wrote an hour ago read a population numeral as a count, and said so at once.

The sentence I had just written said

    同组大小的 run 41 在 4 个里**一个都不是最低**

and the item I wrote beside it captured `4` as "the rewards in which it holds the
minimum". It reported `document says: 4 / records say : 0` -- and the *document was
right*: the 4 is the population (the four rewards), not the count of wins, which is the
zero the words spell out. So the tool's first reading of its own new sentence was wrong,
and the failure is the document's own recurring defect arrived at from the other side:
a numeral whose scope the reader mis-assigned. The sentence had spelled the count in
words precisely because it was zero, and that is what made the numeral beside it easy to
misread.

Two changes, and the first is to the sentence rather than to the reader:

  * the sentence now states the count as a numeral -- 「在这 4 个奖励里最低的有 0 个」 --
    so the population and the count are two numbers in one clause instead of one number
    and a phrase. A count of zero written as words is a count a pattern cannot read, and
    the temptation is to write a pattern that reads the number that *is* there, which is
    the wrong number. Fixing the sentence removes the temptation.
  * the item now captures both and asserts both against the records, including that the
    two clauses' populations agree with each other -- the second clause restates the
    first one's population, so a sentence that quietly changed it in only one place is
    caught by the pair rather than by either alone.

This is the third numeral in this family today to be wrong in exactly this way -- the
`--fix` that bumped a count while a fact had moved, the run-45 superlative, and now my
own probe -- which is why the fix is a second numeral rather than a clarification in
prose.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
DOC = ROOT / "LIMITATIONS.md"
CHECKER = ROOT / "tools/check_quantified_claims.py"
DB = "/tmp/LIMITATIONS.md.bak-patch121"
CB = "/tmp/check_quantified_claims.py.bak-patch121"

D_OLD = """在 4 个奖励里有 3 个是八组最低（同组大小的 run 41 在 4 个里**一个都不是最低**，所以这条
下降跟着的是学习率、不是组大小）"""
D_NEW = """在 4 个奖励里有 3 个是八组最低（同组大小的 run 41 在这 4 个奖励里最低的有 0 个，
所以这条下降跟着的是学习率、不是组大小）"""

P_OLD = '''P_STD = (r"([A-Za-z]+)\\(G=(\\d+)\\) \\+ lr (\\S+) 同时继承了高学习率的退化与大组下的"
         r"多样性下降\\s*——\\s*组内奖励标准差\\s*在 (" + NUM + r") 个奖励里有 ("
         + NUM + r") 个是八组最低"
         r"（同组大小的 run 41 在 (" + NUM + r") 个里\\*\\*一个都不是最低\\*\\*")'''
P_NEW = '''# Groups: algorithm, G, lr, the reward population, the named cell's count, the
# sibling clause's population, and the sibling's count. The population is captured
# twice on purpose: the second clause restates the first one's population, so a
# sentence that changed it in only one place is caught by the pair.
P_STD = (r"([A-Za-z]+)\\(G=(\\d+)\\) \\+ lr (\\S+) 同时继承了高学习率的退化与大组下的"
         r"多样性下降\\s*——\\s*组内奖励标准差\\s*在 (" + NUM + r") 个奖励里有 ("
         + NUM + r") 个是八组最低"
         r"（同组大小的 run 41 在这 (" + NUM + r") 个奖励里最低的有 (" + NUM + r") 个")'''

IT_OLD = '''        cmp("5.2.4: the rewards in which the same group size at lr 1e-5 holds it",
            cn2int(g.group(6)), _mincount(_sib),
            why="the sentence states this count as the reason the collapse follows the "
                "learning rate rather than the group size, so it carries the argument "
                "and not only the illustration")'''
IT_NEW = '''        cmp("5.2.4: the sibling clause names the same reward population",
            cn2int(g.group(6)), cn2int(g.group(4)),
            why="the two clauses describe one population; if only one of them were "
                "updated the two numerals would differ, which is checkable without "
                "either clause being right about the records")
        cmp("5.2.4: the rewards in which the same group size at lr 1e-5 holds it",
            cn2int(g.group(7)), _mincount(_sib),
            why="the sentence states this count as the reason the collapse follows the "
                "learning rate rather than the group size, so it carries the argument "
                "and not only the illustration. It was read as the population numeral "
                "when this item was first written -- the sentence said the count in "
                "words, and the reader took the number that was beside it")'''

doc = DOC.read_text(encoding="utf-8")
shutil.copy2(DOC, DB)
if doc.count(D_OLD) != 1:
    sys.exit(f"the sibling clause anchors {doc.count(D_OLD)} times; nothing written")
DOC.write_text(doc.replace(D_OLD, D_NEW, 1), encoding="utf-8")

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, CB)
for old, new in ((P_OLD, P_NEW), (IT_OLD, IT_NEW)):
    if c.count(old) != 1:
        shutil.copy2(DB, DOC)
        sys.exit(f"a checker anchor matches {c.count(old)} times; nothing written")
    c = c.replace(old, new, 1)
CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(CB, CHECKER)
    shutil.copy2(DB, DOC)
    sys.exit(f"does not compile; both restored:\n{r.stderr}")
print("the sibling clause now carries its count as a numeral, and both clauses' "
      "populations are compared with each other.")

r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = (r.stdout or "").strip().splitlines()
print("\n--- the std items ---")
for ln in out:
    s = ln.strip()
    if ("std superlative" in s or "lowest std" in s or "holds it" in s
            or "RL groups in" in s or "reward population" in s):
        print("  " + s[:150])
n_stale = sum(1 for ln in out if ln.strip().startswith("STALE"))
for i, ln in enumerate(out):
    if ln.strip().startswith("STALE"):
        for nx in out[i:i + 3]:
            print("  " + nx.rstrip())
print("--- tail ---")
for ln in out[-3:]:
    print("  " + ln.strip())
print(f"\nchecker exit {r.returncode}; {n_stale} stale item(s)")
