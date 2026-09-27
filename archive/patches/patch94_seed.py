"""Run 42 (a second seed of GRPO at lr 1e-5) lands: write the noise floor down.

Every verdict in this document compares two *configurations*. Significance there says
"this difference is larger than the 252 benchmark rows can resolve" -- a statement
about benchmark size. It says nothing about the question a reader actually asks: if
you trained the same configuration again with a different seed, how far would the score
move? If that distance is comparable to the effects being claimed, the effects are not
established, however small their p-values are.

Run 42 is the one honest estimate of that distance this project can produce. The
threshold was fixed in `tools/seed_variance.py` before the number existed (the seed
difference is compared against half the within-GRPO learning-rate effect), so this
patch reads a decision rather than making one.

The patch is written to be conditional, because the answer is not known when it is
written:

  * if the rule triggers, the learning-rate effect is not separable from seed noise on
    this benchmark, and 5.2.2 has to say so *where it makes the claim*. A finding
    recorded only in a later section, while the claim it overturns stands earlier
    unqualified, is the defect this document records in section 8: every number
    correct, and the sentence about a set that no longer exists. So the edit to 5.2.2
    is part of the patch, not a follow-up;
  * if the rule does not trigger, the same paragraph says the effect is above the floor
    and quotes the ratio.

It also states what two seeds cannot give (a difference, not a spread), because a
lower bound that is read as a measurement would be a worse error than no number.

--- rewritten after run 45 landed -----------------------------------------------

The version of this patch written before run 45 anchored 5.2.4 on the sentence
"六个 RL run 里**唯一** macro-F1 点估计高于 SFT 的是 GRPO lr1e-5（…）". Run 45 replaced
it: there are seven RL runs, two of them clear SFT on the point estimate, and the
sentence is now a list. That version would have matched nothing and refused, writing no
5.2.7 at all -- a silent loss of the section this run exists to produce, at 18:30, with
the log's only symptom being a message about an anchor.

So the caveat now attaches to the *paragraph* rather than to the sentence, found by
locating the one place that says "个 macro-F1 点估计高于 SFT" and appending at that
paragraph's end. A list of N runs has no single "the" run to qualify, and any regex
naming the runs would go stale again the next time one lands. The paragraph end is
where a caveat about the paragraph belongs and it does not depend on how many runs the
list holds or in what order.

The wording of the comparison also changed with it. It used to read "5.2.4 记下的那个
唯一高于 SFT 的 RL run"; it now says which entry in the list is the one run 42 reseeds
(GRPO(G=4) lr1e-5 is), and it quotes that entry's own paired test against SFT, because
a point estimate above SFT that the paired test cannot separate from SFT is a weaker
thing to reseed than the old sentence made it sound. Both are read from the records.

Every value is read from `outputs/reports/seed_variance.json`, the two runs' reports,
and the paired record. Nothing is typed in, and the patch exits without writing if the
seed record is absent.
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
DOC = ROOT / "LIMITATIONS.md"
REP = ROOT / "outputs/reports"
SV = REP / "seed_variance.json"

if not SV.is_file():
    sys.exit(f"no {SV.name}: run 42 has not been scored yet -- nothing written")
SVJ = json.loads(SV.read_text(encoding="utf-8"))


def macro_f1(tag):
    p = REP / f"{tag}__report.json"
    if not p.is_file():
        sys.exit(f"no report for {tag}; nothing written")
    return json.loads(p.read_text(encoding="utf-8"))["classification"]["macro_f1"]


SD = SVJ["seed_difference_accuracy"]
A1, A2 = SVJ["accuracy"]["seed1"], SVJ["accuracy"]["seed2"]
D, LO, HI = SD["delta"], SD["ci95_rows"][0], SD["ci95_rows"][1]
P = SD["mcnemar"]["p_exact_two_sided"]
EXCL = SD["excludes_zero"]
REF = SVJ["reference_lr_effect_within_grpo"]["accuracy_delta"]
RATIO = SVJ["ratio_seed_to_lr_effect"]
HALF = SVJ["pre_registered_rule"]["fraction"]
TRIG = SVJ["rule_triggered"]
REP_SFT = SVJ["replication_vs_sft"]
MR = SVJ["structured_and_caption"]

if EXCL != (LO > 0 or HI < 0):
    sys.exit("the seed record's excludes_zero disagrees with its own interval; "
             "nothing written")

F1_1 = macro_f1("qwen35_9b_grpo")
F1_2 = macro_f1("qwen35_9b_grpo_lr1e5_seed3408")
F1_SFT = macro_f1("qwen35_9b_adapter")

# The reseeded configuration's own paired test against SFT, from the same record the
# rest of the document reads. Quoted so the caveat does not make the entry sound
# stronger than it is; skipped rather than typed if the record is not there yet.
_CG = {}
_psp = REP / "paired_significance.json"
if _psp.is_file():
    _CG = ((json.loads(_psp.read_text(encoding="utf-8")).get("comparisons_vs_SFT") or {})
           .get("GRPO_G4_lr1e5") or {})

# The two seeds, from the training records rather than from this file. If either record
# does not carry one, the sentence drops the parenthetical instead of printing a
# literal that a reader would have no way to check.
def _seed(stem):
    p = REP / f"{stem}_train_result.json"
    if not p.is_file():
        return None
    return (json.loads(p.read_text(encoding="utf-8")).get("config") or {}).get("seed")


_S1 = _seed("grpo_train_result")
_S2 = _seed("qwen35_9b_grpo_lr1e5_seed3408_train_result")
_seed_txt = f"（{_S1} → {_S2}）" if (_S1 is not None and _S2 is not None) else ""

# The design claim -- "the two runs differ only in the seed" -- comes from the record,
# which compares the two configs when it is built rather than asserting it. Which of the
# four cases holds decides how the difference below may be read, so it is branched on
# rather than hedged.
_DES = SVJ.get("design") or {}
_cfg_cmp = _DES.get("config_compared")
_cfg_diff = _DES.get("config_diff") or []
if not _cfg_cmp:
    _design_txt = (f"run 42 把 GRPO(G=4, lr 1e-5) 换种子{_seed_txt}重跑。"
                   f"**「只有种子不同」这一条没有被比对过** —— 两份训练记录没有同时在"
                   f"盘上，所以下面这条差值只能读成「换了一次种子」，不能读成"
                   f"「纯种子效应」。")
elif _cfg_diff == ["seed"]:
    _design_txt = (f"run 42 把 GRPO(G=4, lr 1e-5) 换种子{_seed_txt}重跑，其余完全相同"
                   f"（逐字段比过两份训练记录，只有 `seed` 不同）。")
elif not _cfg_diff:
    _design_txt = ("run 42 与 GRPO(G=4) lr1e-5 的配置逐字段相同，**连种子也一样** —— "
                   "这是一次重复跑，不是换种子。")
else:
    _design_txt = (f"run 42 把 GRPO(G=4, lr 1e-5) 换种子{_seed_txt}重跑，但**这不是"
                   f"单变量对照**：两份训练记录的配置里另有 "
                   + "、".join(f"`{k}`" for k in _cfg_diff)
                   + " 不同，所以下面这条差值里混着它们。")

_d_txt = ("（**排除 0**）" if EXCL else "（包含 0）")
_ratio_txt = (
    f"比值 **{RATIO:.3f}** ≥ 预设阈值 **{HALF}** ⇒ 按**跑之前就定好**的规则，"
    f"**学习率效应与种子噪声在本基准上不可分**：差值 {abs(D):.4f} 已经不小于"
    f"学习率自身效应 {abs(REF):.4f} 的一半。这不是「学习率效应不存在」，"
    f"而是「这个基准 + 一个种子分辨不出它」"
    if TRIG else
    f"比值 **{RATIO:.3f}** < 预设阈值 **{HALF}** ⇒ 按同一条规则，学习率效应"
    f"**大于**种子噪声，5.2.2 的结论不被这一项推翻")
_rep_txt = REP_SFT["reading"] + (
    f"（seed1 Δ = {REP_SFT['seed1']['delta']:+.4f}、`p` = {REP_SFT['seed1']['p_exact_two_sided']:.5f}；"
    f"seed2 Δ = {REP_SFT['seed2']['delta']:+.4f}、`p` = {REP_SFT['seed2']['p_exact_two_sided']:.5f}）")
# The record's own comparison block. macro_f1 is in it, and is quoted separately below
# from the two reports themselves, so it is left out here rather than said twice.
_mrows_txt = "；".join(
    f"{k} Δ = {v['diff']:+.4f}（学习率效应 {v['lr_effect']:+.4f}"
    + (f"、比值 {v['ratio_seed_to_lr']:.3f}" if v.get("ratio_seed_to_lr") is not None
       else "、比值 n/a") + ")"
    for k, v in MR.items() if k != "classification macro_f1")
# Three-way, not two: "低于" in a sentence whose own number is equal would be the
# defect this document exists to record. Equality of two runs' macro-F1 to six places
# is unlikely, which is a reason it would go unnoticed, not a reason to skip it.
if F1_2 > F1_SFT:
    _VERDICT = "above"
    _CMP_DOC = f"仍高于 SFT 的 {F1_SFT:.4f}"
elif F1_2 == F1_SFT:
    _VERDICT = "equal"
    _CMP_DOC = f"与 SFT 的 {F1_SFT:.4f} 完全相同"
else:
    _VERDICT = "below"
    _CMP_DOC = f"低于 SFT 的 {F1_SFT:.4f}"

_f1_txt = (
    f"两次抽样的 macro-F1 分别是 **{F1_1:.4f}** 与 **{F1_2:.4f}**（SFT {F1_SFT:.4f}），"
    f"第二次{_CMP_DOC} —— "
    + ("两次都高于 SFT。但这是**同一个格子的两个种子**，不是这个配置稳定地高于 SFT —— "
       "两个种子都给不出后者；而 5.2.4 把 GRPO(G=4) lr1e-5 列进短名单，依据的正是这个"
       "点估计。"
       if _VERDICT == "above" else
       "第二次**与 SFT 分不出高低**。5.2.4 把 GRPO(G=4) lr1e-5 列进短名单靠的是它的"
       "点估计高于 SFT，那句话没有错，但它所依赖的那个格子是**单次抽样**。"
       if _VERDICT == "equal" else
       "第二次**低于** SFT，而 5.2.4 的短名单里就有它。那句话没有错（它说的是点估计），"
       "但它所依赖的那个格子是**单次抽样**抽出来的，第二次抽样就翻到了另一边。")
)

SECTION = f"""#### 5.2.7 第二个种子：本项目的噪声地板

上面每一条结论都是「跑两个**不同配置**，比较」。显著性回答的是「这个差值超出 252
行基准能分辨的范围了吗」—— 一个关于基准大小的问题。它**不**回答读者会问的那个问题：
同一套配置换个随机种子再跑一次，分数会动多少？如果那个动的幅度和要主张的效应差不多，
那效应就没有被建立起来，`p` 值再小也一样。**基准固定、种子在变**，所以基准上的 `p`
值对种子变异什么也没说。

{_design_txt}

**它不作为本文件里那几张表的一列。** 那些表列的是**配置**，而 run 42 与
GRPO(G=4) lr1e-5 是同一套配置的第二个种子；把它们并列成两列，读者会读成两个配置。
所以上面表格的 run 数不含它，它的数值只在这一节给出（`comparison.md` 里它是单独
一行，名字里带 `_s2`）。

- 准确率：seed1 **{A1:.4f}**、seed2 **{A2:.4f}**，Δ = **{D:+.4f}**，95% 区间
  **[{LO:+.4f}, {HI:+.4f}]**{_d_txt}、McNemar `p` = {P}。
- 参照量：GRPO 内部的学习率效应（lr5e-5 − lr1e-5，seed 1）准确率 Δ = **{REF:+.4f}**。
- {_ratio_txt}。
- 对 SFT 的结论是否复现：{_rep_txt}。
- 记录里另外几项（其 `structured_and_caption` 块）：{_mrows_txt}。这几项记录只做了数值
  比较、没有逐行检验，所以上面的比值只能当**量级**看，不能当检验结论读。
- {_f1_txt}

**两次种子给不出方差。** 两个抽样只给一个差值，给不出这个差值的离散程度；真正的噪声
地板要多个种子，本项目没跑。所以上面这个数是**下界** —— 它可能低估种子变异，不可能高估。
这个不对称是记录里写明的读法，也是这里要写清楚的：{SVJ['what_two_seeds_cannot_give']}

"""

ANCHOR6 = "## 6. 评测与指标解释"
s = DOC.read_text(encoding="utf-8")
shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch94")

if s.count(ANCHOR6) != 1:
    sys.exit(f"the section 6 heading appears {s.count(ANCHOR6)} times; nothing written")
if "#### 5.2.7" in s:
    sys.exit("5.2.7 already exists; nothing written")

before = s.count("**")
s = s.replace(ANCHOR6, SECTION + ANCHOR6, 1)

# ---------------------------------------------------- and the claim it bears on
# 5.2.2 makes the learning-rate claim; the seed floor has to be stated where the claim
# is, not only in a later section. patch92b wrote this sentence when run 45 landed, so
# the anchor is its output, matched loosely on the enumerated cells.
LR_PARA = re.compile(
    r"(算法内部的学习率效应（同算法、同组大小、只差学习率）同样显著：\n.+?。)", re.S)
m = LR_PARA.search(s)
if not m:
    sys.exit("5.2.2's learning-rate sentence is not in the form patch92b writes; "
             "nothing written (the section-5.2.7 text is prepared but not written)")
_qualifier = (
    f"\n  噪声地板的对照（5.2.7）：换种子重跑同一配置的准确率差是 {abs(D):.4f}，"
    f"与学习率效应 {abs(REF):.4f} 的比值是 {RATIO:.3f}"
    + ("，达到预设阈值，所以**这条学习率效应在本基准上与种子噪声不可分** —— "
       "「显著」在这里只意味着大过基准能分辨的尺度，不意味着大过重跑一次能分辨的尺度。"
       if TRIG else
       "，低于预设阈值，所以这条学习率效应**高于**种子噪声地板。")
)
s = s[:m.end(1)] + _qualifier + s[m.end(1):]

# ------------------------------------------------------ and 5.2.4, where it bears
# 5.2.4's list tells a reader that two RL configurations clear the SFT baseline on the
# point estimate. That is true of one draw each. The entry run 42 reseeds is
# GRPO(G=4) lr1e-5, and whether it is still above SFT after the reseed decides how the
# reader may take that entry -- so the second draw goes next to the paragraph, and the
# wording changes with the result rather than being hedged to cover every case.
NEEDLE = "个 macro-F1 点估计高于 SFT"
if s.count(NEEDLE) != 1:
    sys.exit(f"the 5.2.4 list sentence appears {s.count(NEEDLE)} times, expected 1; "
             f"nothing written -- 5.2.7 alone would leave it standing as if its entry "
             f"were not a single draw")
_i = s.find(NEEDLE)
_j = s.find("\n\n", _i)
if _j < 0:
    sys.exit("cannot find the end of the 5.2.4 paragraph; nothing written")
_note = (
    f"  但短名单里 GRPO(G=4) lr1e-5 的那个点估计是**单次抽样**：该配置换种子重跑"
    f"（run 42）后 macro-F1 为 {F1_2:.4f}，**{_CMP_DOC}**。"
    + (f"所以「点估计高于 SFT」在这两个种子上都成立 —— 这是两次抽样，"
       f"不是这个配置稳定地高于 SFT（见 5.2.7）。"
       if _VERDICT == "above" else
       f"所以这句话没有错，它只是不能读成「这个配置高于 SFT」（见 5.2.7）。")
    + (f"它的成对检验本来也分不开：Δ准确率 {_CG['delta_vs_sft']:+.4f}、"
       f"`p` = {_CG['mcnemar']['p_exact_two_sided']}。"
       if _CG else "")
)
s = s[: _j] + "\n" + _note + s[_j:]

after = s.count("**")
if (after - before) % 2:
    sys.exit(f"bold markers went {before} -> {after}, an odd change; nothing written")

DOC.write_text(s, encoding="utf-8")
print(f"5.2.7 written and 5.2.2/5.2.4 qualified ({len(s.splitlines())} lines, "
      f"bold {before} -> {after})")
print(f"  seed acc   : {A1:.4f} -> {A2:.4f} (d={D:+.4f}, p={P}, excludes_zero={EXCL})")
print(f"  seeds      : {_seed_txt or '(not in the records; omitted)'}")
print(f"  lr effect  : {REF:+.4f} ; ratio = {RATIO:.3f} vs {HALF} -> triggered={TRIG}")
print(f"  macro-F1   : seed1 {F1_1:.4f}, seed2 {F1_2:.4f}, SFT {F1_SFT:.4f} "
      f"-> {_VERDICT}")
if _CG:
    print(f"  paired test: d={_CG['delta_vs_sft']:+.4f} "
          f"p={_CG['mcnemar']['p_exact_two_sided']}")
else:
    print("  paired test: not in the record; the clause is omitted")
