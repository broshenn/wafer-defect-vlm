"""Generate the KL/length sentences from the record, so the next landing is a re-run.

These six sentences have now been hand-edited twice for the same reason: the record
gained a run, and each sentence states a population over it -- the pair count, the
enumeration of the pairs, the per-run idle-step list, the total, and the conclusion drawn
from the direction of the deltas. Every one of them was correct when written and became
stale at the next landing, which is precisely the defect this document catalogues: a
number byte-for-byte right in a sentence about a population that moved.

The checkers catch the staleness (that is why they exist) but catching it is not the same
as not producing it. So these sentences are now a function of the record, in the same
shape the checker reads, written by this tool:

  A  5.2.4  「同算法同组大小的N个对照里 KL 都是下降的（…）」
  B  5.2.4  「5e-5 的补全却明显更短（…）」
  C  8     「但 5e-5 那N次的补全明显更短（…，−x%；…）」
  D  8     「N个 run 的全部 M个空转步均满足（逐 run 实测为 a+b+…；」
  E  8     「N个 run 全部 M个空转步上的逐值核对」
  F  8     「N个**同算法同组大小**的对照方向并不一致：（…）。即「…」只在 … 上成立，…」

Three things are decisions rather than arithmetic, and each is stated where it is made:

  * **The order of the pairs** is the record's order, which is the tool's own `pairs`
    list followed by the auto-paired cells. Nothing in the document claims an order, and
    the checkers read the enumerations as sets (dicts and sorted lists), so the order is
    readability rather than meaning -- but it is taken from the record so that a
    reordering of the record shows up here instead of being silently normalised.

  * **The comparison with SFT** is not here: it is `patch100_above_list.py`, which owns
    the one list in the document whose *shape* changes with the run set.

  * **C's two decimals** are load-bearing. The percentage is computed from the printed
    pair, so the pair has to be printed at a precision that reproduces it; at one
    decimal the GSPO(G=8) pair prints −4.5% where the record computes −4.6%, because
    109.16/114.40 is not 109.2/114.4. The generator computes the percentage with the
    same rounding the checker does -- round(100*(round(hi,2)/round(lo,2)-1), 1) -- so
    the printed percentage follows from the printed numbers by construction.

If a sentence cannot be read, or the record does not say what the sentence claims (a KL
that rises where the text says all fall), the tool stops without writing: a generated
sentence about the wrong population is not better than a hand-written one, it is just
faster to produce.

Idempotent: run it after every landing. It writes only if something changed.
"""
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
DOC = ROOT / "LIMITATIONS.md"
REC = ROOT / "outputs/reports/kl_length_confound.json"
BAK = "/tmp/LIMITATIONS.md.bak-patch101"

NUM = r"(?:[一二三四五六七八九十两]+|\d+)"
MINUS = "−"          # the document writes minus as U+2212, not ASCII hyphen


def cn(n):
    """一..十, then 十X, then the digits -- the same form the checker's --fix writes."""
    d = "零一二三四五六七八九"
    if n < 10:
        return d[n]
    if n == 10:
        return "十"
    if n < 20:
        return "十" + d[n % 10]
    return str(n)


def sgn(v, nd):
    """+0.2267 / −0.0333 -- a sign always, and the document's minus."""
    s = f"{v:+.{nd}f}"
    return s.replace("-", MINUS)


def label(key):
    a, g = key.split(" G=")
    return f"{a}(G={g})"


rec = json.loads(REC.read_text(encoding="utf-8"))
runs, pairs = rec["runs"], rec["matched_pairs"]
order = list(pairs)
steps = [runs[t]["kl_identity"]["idle_steps_tested"] for t in runs]
total = sum(steps)
n_runs, n_pairs = len(runs), len(pairs)

# The claims the sentences make, computed rather than assumed.
kl_lo = {k: round(v["kl"][0], 4) for k, v in pairs.items()}
kl_hi = {k: round(v["kl"][1], 4) for k, v in pairs.items()}
rises = [k for k in order if kl_hi[k] > kl_lo[k]]
if rises:
    sys.exit(f"the record has {len(rises)} pair(s) whose KL rises with the learning rate "
             f"({rises}); sentence A says they all fall. Rewrite it by hand: the claim is "
             f"no longer what the record says, and no generator should paper over that.")

A_items = "、\n  ".join(f"{label(k)} {kl_lo[k]:.4f} → {kl_hi[k]:.4f}" for k in order)
B_items = "、\n  ".join(
    f"{label(k)} {pairs[k]['completion_length'][0]:.1f} → "
    f"{pairs[k]['completion_length'][1]:.1f}" + (" token" if i == 0 else "")
    for i, k in enumerate(order))
def c_item(k, i):
    """Section 8's form: the pair at the precision that reproduces its own percentage."""
    lo = round(pairs[k]["completion_length"][0], 2)
    hi = round(pairs[k]["completion_length"][1], 2)
    pct = round(100 * (hi / lo - 1), 1)
    return (f"{label(k)} {lo:.2f} → {hi:.2f}" + (" token" if i == 0 else "")
            + f"，{sgn(pct, 1)}%")


C_items = "；".join(c_item(k, i) for i, k in enumerate(order))
D_list = "+".join(str(s) for s in steps)
F_items = "、\n   ".join(
    f"{label(k)} {100 * pairs[k]['idle'][0]:.2f}% → {100 * pairs[k]['idle'][1]:.2f}%"
    f"（`idle_delta` {sgn(pairs[k]['idle_delta'], 4)}）" for k in order)
up = [label(k) for k in order if pairs[k]["idle_delta"] > 0]
dn = [label(k) for k in order if pairs[k]["idle_delta"] < 0]
zero = [label(k) for k in order if pairs[k]["idle_delta"] == 0]
if zero:
    sys.exit(f"{len(zero)} pair(s) have an idle-step delta of exactly zero ({zero}). The "
             f"sentence's conclusion is a two-way split and this tool will not guess a "
             f"third case; rewrite it by hand.")
if up and dn:
    F_conc = ("即「学习率把空转步推高」只在 " + "、".join(up) + " 上成立，"
              + "、".join(dn) + " 都把它压低")
elif up:
    F_conc = ("即「学习率把空转步推高」在全部 " + cn(n_pairs) + " 个对照上都成立")
else:
    F_conc = ("即「学习率把空转步推高」在这 " + cn(n_pairs) + " 个对照上一个也不成立")

SUBS = [
    # A -- the KL pairs
    (re.compile(r"(同算法同组大小的)(" + NUM + r")(个对照里 KL 都是下降的\s*（)([^）]*)(）)"),
     lambda m: m.group(1) + cn(n_pairs) + m.group(3) + A_items + m.group(5)),
    # B -- the same pairs' completion lengths, 1 dp, no percentage
    (re.compile(r"(5e-5 的补全却明显更短\s*（)([^）]*)(）)"),
     lambda m: m.group(1) + B_items + m.group(3)),
    # C -- section 8's restatement, 2 dp, with the percentage each pair implies
    (re.compile(r"(但 5e-5 那)(" + NUM + r")(次的补全明显更短\s*（)([^）]*)(）)"),
     lambda m: m.group(1) + cn(n_pairs) + m.group(3) + C_items + m.group(5)),
    # D -- the identity's run count, step total and per-run counts.
    # Both counts need `\s*` before 个: the document writes 「七个 run」 with no space and
    # 「336 个空转步」 with one, in the same sentence. This is the trap cn2int's docstring
    # describes, and the first version of this pattern fell into it -- it matched
    # 「七个 run」 and then failed on the space after 336, reporting the sentence absent.
    (re.compile(r"(" + NUM + r")\s*(个 run 的全部 )(" + NUM + r")\s*"
                r"(个空转步均满足（逐 run 实测为\s*)([\d+]+)"),
     lambda m: cn(n_runs) + m.group(2) + str(total) + m.group(4) + D_list),
    # E -- the same statement, in the audit section
    (re.compile(r"(" + NUM + r")\s*(个 run 全部 )(" + NUM + r")\s*(个空转步上的逐值核对)"),
     lambda m: cn(n_runs) + m.group(2) + str(total) + m.group(4)),
    # F -- the idle-step pairs and the conclusion drawn from their direction
    (re.compile(r"(" + NUM + r")(个\*\*同算法同组大小\*\*的对照方向[^：]*：\s*)([^。]*)(。\s*\n\s*)"
                r"(即「学习率把空转步推高」[^。]*)(。)"),
     lambda m: cn(n_pairs) + m.group(2) + F_items + m.group(4) + F_conc + m.group(6)),
]

doc = DOC.read_text(encoding="utf-8")
shutil.copy2(DOC, BAK)

# One-time cleanup: C's explanatory note about the two decimals was written inside the
# parenthetical this tool now regenerates, so it would be replaced along with the numbers.
# The generator's docstring carries the explanation instead, where it belongs: nothing
# hand-edits these sentences any more, so there is no reader to warn off.
new, report = doc, []
_note = re.compile(r"；此处取两位小数是必需的[^）]*")
_n = _note.findall(new)
if len(_n) > 1:
    sys.exit(f"the precision note appears {len(_n)} times, expected at most 1; nothing "
             f"written")
if _n:
    new = _note.sub("", new, count=1)
    report.append(("the precision note (now the generator's docstring)", "removed"))

for rx, fn in SUBS:
    hits = rx.findall(new)
    if len(hits) != 1:
        sys.exit(f"a KL sentence anchors {len(hits)} times, expected 1; nothing written.\n"
                 f"pattern: {rx.pattern[:70]}...")
    before = new
    new = rx.sub(fn, new, count=1)
    report.append((rx.pattern.split("(")[0][:34] or rx.pattern[:34],
                   "changed" if new != before else "already"))

if new == doc:
    print("nothing to do: every KL sentence is already what the record implies.")
    for what, state in report:
        print(f"  {state:8s} {what}")
    sys.exit(0)

DOC.write_text(new, encoding="utf-8")
print(f"the KL sentences are regenerated from the record: {n_runs} runs, {total} idle "
      f"steps, {n_pairs} pairs.")
for what, state in report:
    print(f"  {state:8s} {what}")
print(f"\n  the pairs, in the record's order: {'、'.join(label(k) for k in order)}")
print(f"  lr 5e-5 raises the idle-step share in: {'、'.join(up) or '(none)'}")
print("  (the record lives in %s; re-run this after every landing)" % REC.name)
