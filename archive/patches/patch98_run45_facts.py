"""Run 45 (GSPO, G=4, lr 1e-5) moves three sentences in 5.2.4 and 5.2.5.

The run's macro-F1 is 0.6255, above SFT's 0.6114 and above GRPO lr1e-5's 0.6197. Three
claims about the run set stop being true, and none of them is a wrong number:

  1. 5.2.4 says "六个 RL run 里**唯一** macro-F1 点估计高于 SFT 的是 GRPO lr1e-5".
     There are two now, and the one that leads is not the one the sentence names. The
     sentence moves to its list shape, which is the shape `check_quantified_claims.py`
     was extended to accept *before* this run landed -- a tool that knew only the
     uniqueness shape would report STALE, and STALE reads as "nothing verified", which
     is not the same as "wrong".
  2. 5.2.4's verdict says RL produced no measurable benchmark gain. That is a claim about
     the paired test against SFT, not about macro-F1, and the two come apart: a point
     estimate can rise while the paired test still cannot exclude zero. So the verdict is
     rewritten from the record's own comparison for run 45, and which way it branches is
     read off that record rather than assumed from the macro-F1.
  3. 5.2.5 says "两个 lr 1e-5 的 run 两个类别都在" and lists their two class values.
     There are three lr 1e-5 runs now. The count is mechanical; the case where one of
     them has lost a class is not, and is written as its own branch rather than assumed
     away.

Nothing is typed in. The macro-F1 values come from the reports, the class values from
the reports, the paired result from `paired_significance.json`. The patch refuses while
run 45's entry is absent from that record -- which is what keeps it from writing a
sentence about a comparison that has not been computed -- and refuses again if run 45
did not in fact clear SFT, because then the premise of the rewrite is false and the
sentence needs only its numeral, which `--fix` will do.

It also tightens one regex in the checker: the shape-2 name capture allowed "）与 " inside
a name, so a list separated by "与" would have captured the separator as part of the
second run's name and reported a mismatch that was really punctuation. The document uses
"、", which the old pattern handled -- but a check whose correctness depends on which
separator the prose happened to use is a check that will be wrong quietly later.
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
DOC = ROOT / "LIMITATIONS.md"
REP = ROOT / "outputs/reports"
CHECKER = ROOT / "tools/check_quantified_claims.py"

# ---------------------------------------------------------------- the tool first
# Before the guard below: this is a fix to the check rather than part of the landing, so
# it should still happen if run 45's comparison is not in the record yet.
_s = CHECKER.read_text(encoding="utf-8")
_R_OLD = r'''re.findall(r"([^（、\n]+)（([\d.]+) 对 ([\d.]+)）", m.group(3))'''
_R_NEW = r'''re.findall(r"([^（、）\n]+)（([\d.]+) 对 ([\d.]+)）", m.group(3))'''
if _R_NEW in _s:
    print("checker: the shape-2 name capture is already tightened")
elif _s.count(_R_OLD) == 1:
    shutil.copy2(CHECKER, "/tmp/check_quantified_claims.py.bak-patch98")
    CHECKER.write_text(_s.replace(_R_OLD, _R_NEW, 1), encoding="utf-8")
    print("checker: the shape-2 name capture no longer swallows a separator")
else:
    sys.exit(f"the shape-2 name capture appears {_s.count(_R_OLD)} times; nothing written")

# ------------------------------------------------------------------- the numbers
# The seven RL runs, label -> report tag. Same labels the tables use, so that
# `doc_name()` in the checker strips the group size the same way it does for the column.
RL = {
    "GRPO(G=4) lr1e-5": "qwen35_9b_grpo",
    "GRPO(G=4) lr5e-5": "qwen35_9b_grpo_lr5e5",
    "GSPO(G=8) lr5e-5": "qwen35_9b_gspo_v1",
    "GSPO(G=8) lr1e-5": "gspo_lr1e5",
    "GSPO(G=32) lr5e-5": "qwen35_9b_gspo_g32",
    "GSPO(G=4) lr5e-5": "qwen35_9b_gspo_g4_lr5e5",
    "GSPO(G=4) lr1e-5": "qwen35_9b_gspo_g4_lr1e5",
}
NEW = "GSPO(G=4) lr1e-5"


def report(tag):
    p = REP / f"{tag}__report.json"
    if not p.is_file():
        sys.exit(f"no report for {tag!r} ({p.name}); nothing written")
    return json.loads(p.read_text(encoding="utf-8"))["classification"]


F = {n: report(t)["macro_f1"] for n, t in RL.items()}
PC = {n: report(t)["per_class"] for n, t in RL.items()}
FSFT = report("qwen35_9b_adapter")["macro_f1"]
CLASSES = sorted(PC[NEW].keys())

if NEW not in F:
    sys.exit("internal error: the new run is not in the RL map")
ABOVE = [n for n in sorted(RL, key=lambda n: -F[n]) if F[n] > FSFT]
if NEW not in ABOVE:
    sys.exit(f"{NEW} is {F[NEW]:.4f}, not above SFT's {FSFT:.4f}. This patch rewrites the "
             f"sentence into its list shape on the premise that run 45 clears SFT; that "
             f"premise is false, so the sentence needs only its numeral, which "
             f"`check_quantified_claims.py --fix` does. Nothing written.")

PS = json.loads((REP / "paired_significance.json").read_text(encoding="utf-8"))
CVS = PS.get("comparisons_vs_SFT") or {}
KEY45, KEYGRPO = "GSPO_G4_lr1e5", "GRPO_G4_lr1e5"
for k in (KEY45, KEYGRPO):
    if k not in CVS:
        sys.exit(f"paired_significance.json has no comparisons_vs_SFT[{k!r}]; run 45 has "
                 f"not been folded into the paired record yet -- nothing written")

C45, CG = CVS[KEY45], CVS[KEYGRPO]
D45, P45, LO45, HI45 = (C45["delta_vs_sft"], C45["mcnemar"]["p_exact_two_sided"],
                        C45["delta_ci95_rows"][0], C45["delta_ci95_rows"][1])
EX45 = C45["excludes_zero"]
if EX45 != (LO45 > 0 or HI45 < 0):
    sys.exit("the record's excludes_zero disagrees with its own interval; nothing written")
DG, PG = CG["delta_vs_sft"], CG["mcnemar"]["p_exact_two_sided"]

# ------------------------------------------------------------------- 5.2.4, part 1
# Shape 2 wants one line and one "name（value 对 SFT）" per run, named the way the column
# is. Order is by macro-F1 descending, which is the order the checker recomputes.
_LIST = "、".join(f"{n}（{F[n]:.4f} 对 {FSFT:.4f}）" for n in ABOVE)
A_OLD = """六个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 GRPO lr1e-5（0.6197 对 0.6114），
成对检验下与 SFT 完全无法区分（Δ准确率 0.0000、`p` = 1.00）；"""
A_NEW = (f"七个 RL run 里有 {len(ABOVE)} 个 macro-F1 点估计高于 SFT：{_LIST}，\n")
if EX45 and D45 > 0:
    A_NEW += (f"成对检验下 GSPO(G=4) lr1e-5 **高于 SFT 且排除 0**（Δ准确率 {D45:+.4f}、"
              f"`p` = {P45}），而 GRPO lr1e-5 仍与 SFT 完全无法区分（Δ准确率 {DG:+.4f}、"
              f"`p` = {PG}）；")
else:
    A_NEW += (f"成对检验下两个都仍与 SFT 无法区分（GSPO(G=4) lr1e-5 Δ准确率 {D45:+.4f}、"
              f"`p` = {P45}；GRPO lr1e-5 Δ准确率 {DG:+.4f}、`p` = {PG}）；")

# ------------------------------------------------------------------- 5.2.4, part 2
B_OLD = """诚实的表述是：**在本模型规模与本次预算下，RL 阶段没有产生可测的基准增益，
而且在高学习率档上产生了可测的基准损失**；"""
if EX45 and D45 > 0:
    B_NEW = (f"诚实的表述是：**在本模型规模与本次预算下，RL 阶段第一次产生了一个可测的\n"
             f"基准增益 —— GSPO(G=4) lr1e-5 相对 SFT 的成对准确率差 {D45:+.4f}、"
             f"`p` = {P45}，排除 0\n—— 而且在高学习率档上仍然有可测的基准损失**；")
else:
    B_NEW = (f"诚实的表述是：**在本模型规模与本次预算下，RL 阶段仍然没有产生可测的\n"
             f"基准增益，而且在高学习率档上产生了可测的基准损失**；这条在 run 45 落地后\n"
             f"需要说清楚：macro-F1 的**点估计**最高的现在已经是 GSPO(G=4) lr1e-5 的 "
             f"{F[NEW]:.4f}（SFT {FSFT:.4f}），\n"
             f"而**成对检验**对 SFT 仍不排除 0（Δ准确率 {D45:+.4f}、`p` = {P45}）"
             f"—— 点估计动了，可测的增益没有动，\n两者不是同一句话。")

# ------------------------------------------------------------------- 5.2.5
# The counts and values are recomputed for every lr 1e-5 run; the checker verifies the
# count in the anchor and the class-zeroing above it.
LOW = [n for n in RL if n.endswith("lr1e-5")]
CN = "一二三四五六七八九十"
CNT = CN[len(LOW) - 1] if 0 < len(LOW) <= 10 else str(len(LOW))


def values(run, cls):
    return f"{run} {PC[run][cls]['f1-score']:.3f}"


_byclass = "；".join("`%s`：" % c + "、".join(values(n, c) for n in LOW) for c in ("none", "Donut"))
_zeroed = [n for n in LOW if any(PC[n].get(c, 1.0) == 0.0 for c in CLASSES)]
_intact = [n for n in LOW if n not in _zeroed]

C_OLD = """而两个 lr 1e-5 的 run 两个类别都在
（`none` 0.343
与 0.462、
`Donut` 0.421
与 0.462）。"""
if not _zeroed:
    C_NEW = (f"而{CNT}个 lr 1e-5 的 run 两个类别都在\n（{_byclass}"
             f"；run 45 落在其中，它既没有丢类别，macro-F1 又是全对照最高的（见 5.2.4））。")
else:
    C_NEW = (f"而{CNT}个 lr 1e-5 的 run 里，"
             + ("、".join(_intact) + " 两个类别都在，" if _intact else "")
             + "、".join(f"{n} 丢了 " + "、".join(f"`{c}`" for c in CLASSES
                                                if PC[n].get(c, 1.0) == 0.0)
                         for n in _zeroed)
             + f"。\n（{_byclass}）。")

# --------------------------------------------------------------------- the write
EDITS = ((A_OLD, A_NEW, "5.2.4's above-SFT sentence"),
         (B_OLD, B_NEW, "5.2.4's verdict"),
         (C_OLD, C_NEW, "5.2.5's lr 1e-5 enumeration"))
s = DOC.read_text(encoding="utf-8")
shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch98")
before = s.count("**")
for old, new, what in EDITS:
    if s.count(old) != 1:
        sys.exit(f"{what} appears {s.count(old)} times, expected 1; nothing written")
    # Each edit must be bold-balanced on its own: a net-even change could still hide two
    # unbalanced edits that cancel, and an unbalanced pair renders the rest of the
    # section bold -- a formatting fault that reads as emphasis the author chose.
    if (new.count("**") - old.count("**")) % 2:
        sys.exit(f"{what} would leave an unbalanced bold marker; nothing written")
    s = s.replace(old, new, 1)
after = s.count("**")
if (after - before) % 2:
    sys.exit(f"bold markers went {before} -> {after}, an odd change; nothing written")
DOC.write_text(s, encoding="utf-8")

print(f"5.2.4 and 5.2.5 rewritten ({len(s.splitlines())} lines, bold {before} -> {after})")
print(f"  above SFT     : {[(n, round(F[n], 4)) for n in ABOVE]}  (SFT {FSFT:.4f})")
print(f"  run 45 vs SFT : d={D45:+.4f} CI=[{LO45:+.4f},{HI45:+.4f}] p={P45} "
      f"excludes_zero={EX45} -> "
      f"{'a measurable gain' if EX45 and D45 > 0 else 'still not measurable'}")
print(f"  lr 1e-5 runs  : {'、'.join(LOW)}  ({len(LOW)})")
print(f"  zeroed among them: {_zeroed or 'none'}")
