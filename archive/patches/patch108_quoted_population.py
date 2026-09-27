"""The same sentence states two populations four words apart, and the item reads one.

Line 539, one sentence:

    G=32 同时继承了高学习率的退化与大组下的多样性下降 —— 组内奖励标准差
    在 4 个奖励里有 3 个是**七组最低** —— 但**「六组最低的标准差」与「最低的分数」
    并不是同一件事，这里不再用前者给后者作保**。

`P_REWARD` anchors on 「在 4 个奖励里有 3 个是(N)组最低」 and compares N with the reward
table's column count. That is the *first* population in the sentence, and it is correct:
run 45 added the seventh column, the table grew, and this numeral became 七.

The second population -- inside the quotation, naming the claim the sentence goes on to
reject -- did not move with it and still says 六. So the sentence asserts seven groups and
six groups of the same table in the same breath, and the certifying item reads only the
half that was updated. This is section 8 item 9 in its sharpest form: not a stale numeral
in an unread sentence, but a stale numeral *inside a sentence the tool reads*, four words
from the numeral it reads. The tool was green on it.

(The quotation is not a historical citation. It names the claim being rejected -- "the
standard deviation is the lowest of the N groups" -- and that N is the population the
claim ranges over, which is the same table either way. A sentence quoting a claim is
still asserting what the claim's words mean now.)

Second finding, from the same sweep: 5.2.2 says

    G=32 在 `Scratch` 上拿到全部 run 最高的 0.462（SFT 0.286）

Nothing read it. The value and the run holding it are claims about the run set, and a
landing can take the maximum away -- but the sentence's *subject* is the more fragile
part. It names the run as 「G=32」, which is unambiguous only while one G=32 run exists.
Queue 41 is `GSPO(G=32) lr1e-5`: a second G=32 run, training now. When it lands,
「G=32 在 Scratch 上拿到全部 run 最高的 0.462」 has two possible subjects and no way for
a reader to tell which was meant -- and if the new run holds the maximum, the sentence is
true of the wrong one. So the item checks the holder's count as a **fact**, not as a
numeral: the fix is to name the run, not to bump a number.

Both new items are facts. The populations here live in a quotation and in a run's name;
neither is a numeral in a slot `--fix` could rewrite.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
DOC = ROOT / "LIMITATIONS.md"
CHECKER = ROOT / "tools/check_quantified_claims.py"
BAK_DOC = "/tmp/LIMITATIONS.md.bak-patch108"
BAK_CHK = "/tmp/check_quantified_claims.py.bak-patch108"

CN = "一二三四五六七八九十"
NUM = r"(?:[一二三四五六七八九十两]+|\d+)"
STALE = "「六组最低的标准差」"
FIXED = "「七组最低的标准差」"
P_QUOTED = r"「(" + NUM + r")组最低的标准差」"
P_LIVE = r"在 4 个奖励里有 (" + NUM + r") 个是(" + NUM + r")组最低"


def cn2int(s):
    if s.isdigit():
        return int(s)
    if s in ("两", "二"):
        return 2
    if s == "十":
        return 10
    if s.endswith("十"):
        return (CN.index(s[0]) + 1) * 10
    return CN.index(s) + 1


# ---------------------------------------------------------------- 1. the document
s = DOC.read_text(encoding="utf-8")

hm = re.search(r"^\| 奖励的组内标准差[^\n]*$", s, re.M)
if not hm:
    sys.exit("the reward table is not in the document; nothing written")
# The whole header line, outer pipes stripped: the first cell is the label (which reads
# 「奖励的组内标准差（150 步均值）」, so a capture starting after the label's words would
# count the unit as a column), the rest are the runs.
n_cols = len([c for c in hm.group(0).strip().strip("|").split("|")]) - 1

m = re.search(P_LIVE, s)
q = re.search(P_QUOTED, s)
if not m or not q:
    sys.exit(f"the sentence is not in the form this patch reads "
             f"(live half {bool(m)}, quoted half {bool(q)}); nothing written")

live, quoted = cn2int(m.group(2)), cn2int(q.group(1))
print(f"the sentence's live half says {live}组, the claim it quotes says {quoted}组, "
      f"and the reward table has {n_cols} columns")

# If the live half is not already the table's size, the table itself moved and this is a
# rewrite rather than a numeral: fixing only the quoted half would leave the sentence
# agreeing with itself about the wrong population.
if live != n_cols:
    sys.exit(f"the live half says {live} groups and the table has {n_cols} columns -- "
             f"that half is a rewrite, not a numeral, and this patch fixes only the "
             f"quoted half; nothing written")
if quoted == n_cols:
    print("the quoted population already matches the table; nothing written")
    sys.exit(0)
if s.count(STALE) != 1:
    sys.exit(f"{STALE!r} appears {s.count(STALE)} times, expected 1; nothing written")
if s.count(FIXED) != 0:
    sys.exit(f"{FIXED!r} is already present; the document is in an unexpected state; "
             f"nothing written")

shutil.copy2(DOC, BAK_DOC)
s = s.replace(STALE, FIXED, 1)
DOC.write_text(s, encoding="utf-8")
print(f"5.2.2: {STALE} -> {FIXED}")

# ----------------------------------------------------------------- 2. the checker
OLD_TAIL = '''    cmp("5.2.5: the number of columns 'lowest' is measured over",
        cn2int(g.group(2)), len(r_cols), kind="count",
        pattern=P_REWARD, group=2)
'''

NEW_TAIL = '''    cmp("5.2.5: the number of columns 'lowest' is measured over",
        cn2int(g.group(2)), len(r_cols), kind="count",
        pattern=P_REWARD, group=2)
    # The same sentence quotes the claim it rejects, and the quotation names the
    # population again. That second numeral is where the population moved without the
    # sentence moving with it: run 45 added the seventh reward-table column, the first
    # numeral became 七, and this one stayed 六 -- so one sentence asserted two
    # populations four words apart and this item read only the first. Read as a count
    # for the same reason the first one is: a run landing bumps both numerals.
    P_QUOTED = r"「(" + NUM + r")组最低的标准差」"
    _q = re.search(P_QUOTED, text)
    if _q:
        cmp("5.2.5: the population in the claim that sentence quotes",
            cn2int(_q.group(1)), len(r_cols), kind="count",
            pattern=P_QUOTED, group=1)
    else:
        cmp("5.2.5: the quoted claim's population", "the quoted phrase", "absent",
            why="the sentence quotes the claim it rejects and the quotation carries the "
                "population; if that phrase is rewritten nothing reads the numeral")

# --------------------------------------- 5.2.2: the Scratch maximum and who holds it
# Nothing read this sentence. Its value is a claim about the run set, but its subject is
# the fragile part: it names the run 「G=32」, which is unambiguous only while one G=32
# run exists -- and a second one is training now. Checked as a fact, because the repair
# is to name the run, not to change a number.
P_SCRATCH = r"G=32 在 `Scratch` 上拿到全部 run 最高的 ([\\d.]+)\\s*（SFT ([\\d.]+)）"
g = anchor("5.2.2: the Scratch per-class maximum", P_SCRATCH)
if g:
    _sc = {n: present[n]["per_class"].get("Scratch") for n in runs}
    _sc = {n: v for n, v in _sc.items() if v is not None}
    _best = max(_sc, key=_sc.get)
    cmp("5.2.2: the run holding the Scratch per-class maximum", _best,
        "GSPO(G=32) lr5e-5",
        why="the sentence names a run as the one with the highest Scratch F1; another "
            "run taking that maximum makes it false in a way no numeral can fix")
    cmp("5.2.2: the Scratch per-class maximum", float(g.group(1)), round(_sc[_best], 3))
    cmp("5.2.2: SFT's Scratch F1 quoted beside it", float(g.group(2)),
        round(_sc["SFT"], 3))
    cmp("5.2.2: the number of runs the sentence's 「G=32」 could name",
        len([n for n in runs if "G=32" in n]), 1,
        why="the sentence's subject is the group size, and with two runs at G=32 it "
            "identifies neither -- the sentence has to name the run (its learning rate) "
            "before the maximum is a usable claim")
    print(f"         Scratch F1 by run: "
          + "、".join(f"{_k} {_v:.3f}" for _k, _v in
                      sorted(_sc.items(), key=lambda kv: -kv[1])))
'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, BAK_CHK)

if c.count(OLD_TAIL) != 1:
    shutil.copy2(BAK_DOC, DOC)
    sys.exit(f"the reward block's tail appears {c.count(OLD_TAIL)} times, expected 1; "
             f"BOTH files restored")
if "P_QUOTED" in c or "P_SCRATCH" in c:
    shutil.copy2(BAK_DOC, DOC)
    sys.exit("the checker already carries one of these items; BOTH files restored")

c = c.replace(OLD_TAIL, NEW_TAIL, 1)
CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(BAK_CHK, CHECKER)
    shutil.copy2(BAK_DOC, DOC)
    sys.exit(f"the patched checker does not compile; BOTH files restored:\n{r.stderr}")

# The fix and its guards have to agree: the checker must be green on the document this
# patch just wrote, including on the numeral it just changed. It was green on that
# numeral before the patch, which is the finding.
print("\n--- re-running the checker on the corrected document ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = r.stdout.strip().splitlines()
for ln in out[-26:]:
    print("  " + ln.strip())
if r.stderr.strip():
    print("--- stderr ---")
    for ln in r.stderr.strip().splitlines()[-12:]:
        print("  " + ln)
if r.returncode != 0:
    shutil.copy2(BAK_CHK, CHECKER)
    shutil.copy2(BAK_DOC, DOC)
    sys.exit(f"\nthe checker is not green after the fix (exit {r.returncode}); BOTH "
             f"files restored")
print("\nchecker green: both populations in that sentence are now read, and the Scratch "
      "sentence's value, holder and subject are read too")
