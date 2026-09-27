"""The third draw lands: 5.2.8 writes the range, and 5.2.7 stops saying it was not run.

5.2.7 ends by admitting what two draws cannot give: they give one difference and no
measure of spread, so the number above it is a lower bound. The sentence patch94 wrote
for that ends *"真正的噪声地板要多个种子，本项目没跑"*. Every word of it was true when
it was written, and the moment queue 43's evaluation finishes it is false -- the
document would contain, in the section whose whole subject is what was and was not
measured, a statement that the measurement was not made. That is the defect this
project records as section 8 item 9 (a number byte-for-byte correct inside a sentence
whose scope moved) arriving from the direction of too much honesty rather than too
little. So this patch is two edits and they belong together: the caveat is corrected,
and the third draw's numbers are written down.

WHERE THE THIRD DRAW GOES. Not into the tables: it measures the same cell as GRPO(G=4)
lr1e-5, so a column would count one configuration twice. It is a row in comparison.md
(`_s3`) and a section here, exactly as the second seed is.

WHAT IT ADDS. A range: the third draw may land inside the pair's interval or outside it,
and a difference cannot show which. The rule for the range is the pre-registered one --
half the within-GRPO learning-rate effect -- applied to the widest pair, and
`tools/seed_variance.py` records that it was written before the third draw's number
existed. If the range crosses the threshold while the pair did not, then the extra draw
is what moves the reading, and 5.2.2 -- which carries the learning-rate claim and the
qualifier patch94 attached to it -- is amended here rather than left standing. A finding
recorded only in a later section, while the claim it qualifies reads unqualified
earlier, is the failure patch94 itself was written to avoid.

Everything quoted is read from `outputs/reports/seed_variance.json` and the runs'
reports; nothing is typed in. Honours `WAFER_DOC` so it can be dry-run against a copy
(patch101's lesson: a patch that ignores it writes production during a "dry run"). It
refuses, writing nothing, if the record holds fewer than three draws, if 5.2.7 is
absent, or if any anchor it needs is missing.
"""
import json
import os
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
DOC = pathlib.Path(os.environ.get("WAFER_DOC") or (ROOT / "LIMITATIONS.md"))
REP = ROOT / "outputs/reports"
SV = REP / "seed_variance.json"
ANCHOR6 = "## 6. 评测与指标解释"

if not SV.is_file():
    sys.exit(f"no {SV.name}: the seed record has not been written yet -- nothing written")
SVJ = json.loads(SV.read_text(encoding="utf-8"))
SP = SVJ.get("seed_spread") or {}
ES = SVJ.get("extra_seeds") or {}
if SP.get("n_draws", 0) < 3 or not ES.get("per_draw"):
    sys.exit(f"the record holds {SP.get('n_draws', 'no')} draws and "
             f"{len(ES.get('per_draw') or {})} extra draw(s); 5.2.8 is about a third "
             f"one -- nothing written")

A1 = SVJ["accuracy"]["seed1"]
A2 = SVJ["accuracy"]["seed2"]
DRAWS = SP["draws"]                       # label -> accuracy, in run order
LABELS = list(DRAWS)
EXTRA_LABEL = [l for l in DRAWS if l not in (LABELS[0], LABELS[1])]
if len(EXTRA_LABEL) != 1:
    sys.exit(f"expected exactly one extra draw beyond the two seeds, found "
             f"{EXTRA_LABEL}; nothing written")
E3 = EXTRA_LABEL[0]
A3 = DRAWS[E3]
R3 = ES["per_draw"][E3]
RNG = SP["range"]
HALF_LR = SP["threshold_half_lr_effect"]
TRIG = SP["rule_triggered"]
TRIG_PAIR = SP["rule_triggered_by_the_pair"]
AGREE = SP["rules_agree"]
WORST, BEST = SP["widest_pair"]
REF = SVJ["reference_lr_effect_within_grpo"]["accuracy_delta"]
RATIO = SVJ["ratio_seed_to_lr_effect"]
DES3 = ((SVJ.get("design") or {}).get("extra") or {}).get(E3) or {}

# The record's own arithmetic, checked before it is quoted -- a record that disagrees
# with itself must not be written into the document.
if abs(RNG - (max(DRAWS.values()) - min(DRAWS.values()))) > 1e-9:
    sys.exit(f"the record's range {RNG} is not the spread of its draws {DRAWS}; "
             f"nothing written")
if AGREE != (TRIG == TRIG_PAIR):
    sys.exit("the record's rules_agree disagrees with its own two booleans; nothing "
             "written")


def seed_of(stem):
    p = REP / f"{stem}_train_result.json"
    if not p.is_file():
        return None
    return (json.loads(p.read_text(encoding="utf-8")).get("config") or {}).get("seed")


_S3 = seed_of("qwen35_9b_grpo_lr1e5_seed3409")
_seed_txt = f"（{_S3}）" if _S3 is not None else ""

# ------------------------------------------------------------------ the design claim
if not DES3.get("compared"):
    _design_txt = (f"第三次抽样的配置**没有被比对过**：它和第一次抽样的训练记录没有同时在"
                   f"盘上，所以「同一套配置」在这里是假定，不是核对结果。")
elif DES3.get("diff") == ["seed"]:
    _design_txt = (f"逐字段比过第三次抽样与第一次抽样的训练记录，**只有 `seed` 不同**"
                   f"（{DES3.get('identical_except', '')}），所以上面那个范围是换种子换出来"
                   f"的。")
elif not DES3.get("diff"):
    _design_txt = ("第三次抽样与第一次抽样的配置逐字段相同，**连种子也一样** —— 这是一次"
                   "重复跑，不是第三个抽样，它加进范围里的是重复而不是新信息。")
else:
    _design_txt = (f"第三次抽样与第一次抽样**不是单变量对照**：两份训练记录的配置里另有 "
                   + "、".join(f"`{k}`" for k in DES3["diff"])
                   + " 不同，所以它加进范围里的东西混着它们。")

# ------------------------------------------------------------------- the comparison
_d3_txt = ("（**排除 0**）" if R3.get("excludes_zero") else "（包含 0）")
_spread_txt = (
    f"三次抽样里最高与最低相差 **{RNG:.4f}**（{WORST} {DRAWS.get(WORST, float('nan')):.4f} "
    f"… {BEST} {DRAWS.get(BEST, float('nan')):.4f}），预设阈值是学习率自身效应 "
    f"{abs(REF):.4f} 的一半 **{HALF_LR:.4f}** ⇒ "
    + ("**达到阈值**：三个抽样一起看，学习率效应与种子噪声在本基准上**不可分**"
       if TRIG else
       "**未达阈值**：把三次抽样一起看，学习率效应仍然**高于**种子噪声地板"))
if AGREE:
    _agree_txt = ("这一条与 5.2.7 那条只看一对种子的规则的结论一致 —— "
                  "多出来的那个抽样没有改变读法。"
                  if TRIG_PAIR else
                  "这一条与 5.2.7 那条规则的结论一致（两者都未触发）—— "
                  "多出来的抽样没有改变读法。")
else:
    _agree_txt = (
        f"**与 5.2.7 那条规则不一致**：只看 seed1 与 seed2 时，差值是 {abs(A2 - A1):.4f}，"
        f"与学习率效应之比是 {RATIO:.3f}，低于阈值 {SVJ['pre_registered_rule']['fraction']}；"
        f"但把第三个抽样算进来，范围 {RNG:.4f} 已经达到阈值 {HALF_LR:.4f}。"
        f"这不是「学习率效应不存在」，而是**「一对种子看不出的事，三个抽样看得见」** —— "
        f"第三个抽样落在那一对之外，5.2.2 里那条限定必须按这个范围重读。")

SECTION = f"""#### 5.2.8 第三个种子：从差值到一个范围

5.2.7 的收尾自己说了它给不出什么：两个抽样只给一个差值，给不出离散程度，所以那个数是下界。
队列 43 用**同一套配置**又跑了一个抽样（GRPO G=4、lr 1e-5、150 步，只改 `seed`{_seed_txt}）。
三个抽样仍然不是方差估计，但它能显示差值结构上看不见的一件事：第三个抽样落在那一对**里面**
还是**外面**。

**和前两次一样，它不作为本文件表格里的一列。** 它测的是同一个格子，列进去等于把一个配置
数两次，所以上面表格的 run 数不含它；`comparison.md` 里它是单独一行，名字里带 `_s3`。

- 三次抽样的准确率：seed1 **{A1:.4f}**、seed2 **{A2:.4f}**、seed3 **{A3:.4f}**。
- 第三次抽样对第一次抽样：Δ = **{R3['delta_vs_seed1']:+.4f}**，95% 区间
  **[{R3['ci95_rows'][0]:+.4f}, {R3['ci95_rows'][1]:+.4f}]**{_d3_txt}、McNemar
  `p` = {R3['mcnemar_vs_seed1']['p_exact_two_sided']}；对 SFT：Δ =
  {R3['vs_sft']['delta']:+.4f}（`p` = {R3['vs_sft']['p_exact_two_sided']}）。
- 范围与阈值：{_spread_txt}。
- 与 5.2.7 那条规则的关系：{_agree_txt}
- 配置核对：{_design_txt}
- **范围不是方差。** {SP.get('what_a_range_cannot_give', '')} 记录里的读法是：
  「{SP.get('reading', '')}」

"""

s = DOC.read_text(encoding="utf-8")
if "#### 5.2.8" in s:
    sys.exit("5.2.8 already exists; nothing written")
if "#### 5.2.7" not in s:
    sys.exit("5.2.7 is not in the document: the third draw is its continuation, and "
             "nothing is written until the second seed's section is there")
if s.count(ANCHOR6) != 1:
    sys.exit(f"the section 6 heading appears {s.count(ANCHOR6)} times; nothing written")

shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch95")
before_bold = s.count("**")
s = s.replace(ANCHOR6, SECTION + ANCHOR6, 1)

# ------------------------------------------- and the sentence that is no longer true
STALE = re.compile(r"\*\*两次种子给不出方差。\*\*.*?(?=\n\n)", re.S)
m = STALE.search(s)
if not m:
    sys.exit("5.2.7's closing paragraph is not in the form patch94 writes; nothing "
             "written (5.2.8 is prepared but not written)")
caveat = SP.get("what_a_range_cannot_give", "")
two_seed = SVJ.get("what_two_seeds_cannot_give", "")
NEW_PARA = (
    "**两个抽样给不出方差；三个也还不给。** 两个抽样只给一个差值，给不出这个差值的离散"
    "程度，所以 5.2.7 里那个数是**下界** —— 它可能低估种子变异，不可能高估。第三个抽样"
    f"（5.2.8）把差值换成了一个**范围**，这比差值多、比方差少：{caveat}。"
    f"两个抽样自己写明的读法不变：{two_seed}")
s = s[:m.start()] + NEW_PARA + s[m.end():]

# ------------------------------------------------------------ and 5.2.2, if needed
if not AGREE and TRIG:
    NEEDLE = "噪声地板的对照（5.2.7）"
    i = s.find(NEEDLE)
    if i < 0:
        sys.exit("the range crosses the threshold but 5.2.2's noise-floor qualifier is "
                 "not in the form patch94 writes; nothing written -- 5.2.8 alone would "
                 "leave 5.2.2's qualifier understating the floor")
    j = s.find("\n", i)
    if j < 0:
        sys.exit("cannot find the end of 5.2.2's qualifier line; nothing written")
    s = s[:j] + (
        f" 第三个抽样把这条限定加宽了：三次抽样的范围是 {RNG:.4f}，已经达到同一个阈值 "
        f"{HALF_LR:.4f}，所以下面 5.2.8 记下的读法（范围达到阈值）也适用于这条学习率"
        f"效应，而不只是 5.2.7 里那一对种子的差值。") + s[j:]

after_bold = s.count("**")
if (after_bold - before_bold) % 2:
    sys.exit(f"bold markers went {before_bold} -> {after_bold}, an odd change; nothing "
             f"written")

DOC.write_text(s, encoding="utf-8")
print(f"5.2.8 written and 5.2.7's closing caveat corrected ({len(s.splitlines())} lines, "
      f"bold {before_bold} -> {after_bold})")
print(f"  draws      : {LABELS} -> {[round(DRAWS[l], 4) for l in LABELS]}")
print(f"  range      : {RNG:.4f} vs threshold {HALF_LR:.4f} -> triggered={TRIG} "
      f"(pair rule: {TRIG_PAIR}, agree={AGREE})")
print(f"  third draw : acc {A3:.4f} (vs seed1 {R3['delta_vs_seed1']:+.4f}, "
      f"p={R3['mcnemar_vs_seed1']['p_exact_two_sided']})")
print(f"  design     : {DES3.get('identical_except', DES3)}")
print(f"  5.2.2      : {'amended (the range moved the reading)' if (not AGREE and TRIG) else 'left as 5.2.7 wrote it'}")
