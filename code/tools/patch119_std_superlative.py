"""A third 「G=32」, in a superlative no tripwire covered -- and run 41 made it ambiguous.

The two tripwires that fired covered the 5.2.4 column sentence and the 5.2.2 Scratch
maximum. This sentence was not covered:

    G=32 同时继承了高学习率的退化与大组下的多样性下降 —— 组内奖励标准差
    在 4 个奖励里有 3 个是八组最低 —— 但**「八组最低的标准差」与「最低的分数」
    并不是同一件事，这里不再用前者给后者作保**。

It was written while 「G=32」 identified a run. Run 41 is a second G=32 run, so it now
identifies neither -- and unlike the other two, the two readings **disagree about the
claim itself**: computed from the records, the lr 5e-5 cell holds the lowest per-reward
std in 3 of the 4 rewards, and the lr 1e-5 cell (same group size, run 41) holds it in
**none** of them. So the sentence is not merely imprecise; a reader who takes 「G=32」 to
mean the newer run gets the opposite answer. The other two sentences' ambiguity could be
settled by quoting the value beside them; this one has no value to settle it.

The fix names the cell, and the sibling cell's count is written in as well, because it is
the more interesting half of the same fact: the diversity collapse follows the learning
rate, not the group size, which is what the paragraph is arguing. It is a count the
sentence now states, so it is a count the checker now reads.

The item computes both counts from the runs' own `reward_signal` blocks, over the eight
RL runs in the order the document's table lists them. The label -> result-file mapping is
read from `tools/idle_step_table.py`'s `RUNS`, which is where those two are declared
together and which already handles the one run whose record is not named after its tag
(`grpo_train_result.json`). A hand-written mapping here would be a second list to drift.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
DOC = ROOT / "LIMITATIONS.md"
CHECKER = ROOT / "tools/check_quantified_claims.py"
DB = "/tmp/LIMITATIONS.md.bak-patch119"
CB = "/tmp/check_quantified_claims.py.bak-patch119"

D_OLD = """G=32 同时继承了高学习率的退化与大组下的多样性下降 —— 组内奖励标准差
在 4 个奖励里有 3 个是八组最低 —— 但**「八组最低的标准差」与「最低的分数」
并不是同一件事，这里不再用前者给后者作保**。"""
D_NEW = """GSPO(G=32) + lr 5e-5 同时继承了高学习率的退化与大组下的多样性下降 —— 组内奖励标准差
在 4 个奖励里有 3 个是八组最低（同组大小的 run 41 在 4 个里**一个都不是最低**，所以这条
下降跟着的是学习率、不是组大小）—— 但**「八组最低的标准差」与「最低的分数」
并不是同一件事，这里不再用前者给后者作保**。"""

doc = DOC.read_text(encoding="utf-8")
shutil.copy2(DOC, DB)
if doc.count(D_OLD) != 1:
    sys.exit(f"the std sentence anchors {doc.count(D_OLD)} times, expected 1; nothing "
             f"written")
DOC.write_text(doc.replace(D_OLD, D_NEW, 1), encoding="utf-8")
print("5.2.4's std superlative now names its cell and states the sibling's count.")

BLOCK = '''
# ---------------- 5.2.4: the per-reward rule std, and which run holds its minimum
# A superlative over a population. `3` is only right while the population is the eight
# RL runs *and* the run it is about is the one the sentence names -- and run 41 made
# the sentence's 「G=32」 name neither. The two tripwires that fired covered the column
# sentence and the Scratch maximum; this sentence was not covered by either, and here
# the two readings disagree about the claim itself rather than about its precision.
P_STD = (r"([A-Za-z]+)\\(G=(\\d+)\\) \\+ lr (\\S+) 同时继承了高学习率的退化与大组下的"
         r"多样性下降\\s*——\\s*组内奖励标准差\\s*在 (" + NUM + r") 个奖励里有 ("
         + NUM + r") 个是八组最低"
         r"（同组大小的 run 41 在 (" + NUM + r") 个里\\*\\*一个都不是最低\\*\\*")
g = anchor("5.2.4: the per-reward std superlative", P_STD)
if g:
    # label -> result file, declared once, in the tool that already reconciles logs
    # against records; `grpo_train_result.json` is why this is read and not written.
    _stem = {}
    if _IDLE_TOOL.is_file():
        for _node in ast.parse(_IDLE_TOOL.read_text(encoding="utf-8")).body:
            if (isinstance(_node, ast.Assign) and _node.targets
                    and getattr(_node.targets[0], "id", None) == "RUNS"):
                for _e in _node.value.elts:
                    _lab, _s, _lg = ast.literal_eval(_e)
                    _stem[_lab] = _s if _s.endswith(".json") else _s + ".json"
    # The eight RL runs in the order 5.2.4's table lists them, so the population the
    # superlative is over is the population the table shows.
    _STD_ORDER = ["GRPO G=4 lr1e-5", "GRPO G=4 lr5e-5", "GSPO G=4 lr5e-5",
                  "GSPO G=4 lr1e-5", "GSPO G=8 lr1e-5", "GSPO G=8 lr5e-5",
                  "GSPO G=32 lr5e-5", "GSPO G=32 lr1e-5"]
    _cells = [(l, "GSPO " + l.split(" ", 1)[1]) for l in _STD_ORDER]
    _series, _miss = {}, []
    for _lab, _key in _cells:
        _f = REP / _stem.get(_lab, "")
        if _f.is_file():
            _series[_key] = json.loads(_f.read_text(encoding="utf-8"))["reward_signal"]
        else:
            _miss.append(_lab)
    if _miss:
        cmp("5.2.4: every RL run's reward_signal is readable", [], _miss,
            why="the std superlative is computed over these runs; one unreadable "
                "record would silently shrink the population and change which run is "
                "the minimum")
    else:
        _rw = sorted(_series[next(iter(_series))])
        def _mincount(cell):
            return sum(1 for r in _rw
                       if abs(_series[cell][r]["mean_std_across_steps"]
                              - min(_series[c][r]["mean_std_across_steps"]
                                    for c in _series)) < 1e-12)
        _named = f"{g.group(1)} G={g.group(2)} lr{g.group(3)}"
        _sib = f"{g.group(1)} G={g.group(2)} lr1e-5"
        cmp("5.2.4: the reward count the std superlative is measured over",
            cn2int(g.group(4)), len(_rw),
            why="the sentence says \\"of N rewards\\"; the population is whatever the "
                "records render")
        cmp("5.2.4: the cell the std superlative names", _named, "GSPO G=32 lr5e-5",
            why="run 41 is a second G=32 run, so 「G=32」 alone names neither and the "
                "two readings disagree: the lr 5e-5 cell holds the minimum in 3 of 4 "
                "rewards and the lr 1e-5 cell in none of them")
        cmp("5.2.4: the rewards in which it holds the lowest std",
            cn2int(g.group(5)), _mincount(_named),
            why="a superlative over a population is not a numeral that can be checked "
                "against the records without them")
        cmp("5.2.4: the rewards in which the same group size at lr 1e-5 holds it",
            cn2int(g.group(6)), _mincount(_sib),
            why="the sentence states this count as the reason the collapse follows the "
                "learning rate rather than the group size, so it carries the argument "
                "and not only the illustration")
        for _c in (_named, _sib):
            print(f"         {_c}: lowest of the {len(_series)} RL groups in "
                  f"{_mincount(_c)} of {len(_rw)} rewards"
                  + ("" if _c in _series else "  (no record)"))
'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, CB)
MARK = '''            "would make the column ambiguous again, in the way run 41 made G=32 ambiguous")'''
if c.count(MARK) != 1:
    shutil.copy2(DB, DOC)
    sys.exit(f"the insertion anchor matches {c.count(MARK)} times; nothing written")
c = c.replace(MARK, MARK + "\n" + BLOCK, 1)
CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(CB, CHECKER)
    shutil.copy2(DB, DOC)
    sys.exit(f"the patched checker does not compile; both files restored:\n{r.stderr}")
print("the std superlative is now read: its population, its cell, and both counts.")

print("\n--- re-running the checker ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = r.stdout.strip().splitlines()
n_stale = 0
for i, ln in enumerate(out):
    s = ln.strip()
    if "std superlative" in s or "lowest std" in s or "lr 1e-5 holds it" in s \
            or "RL groups in" in s:
        print("  " + s)
    if s.startswith("STALE"):
        n_stale += 1
        for nx in out[i:i + 3]:
            print("  " + nx.rstrip())
for ln in out[-4:]:
    print("  " + ln.strip())
print(f"\nchecker exit {r.returncode}; {n_stale} stale item(s)")
