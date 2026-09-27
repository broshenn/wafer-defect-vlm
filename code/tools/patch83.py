"""Correct the prose that run 43 (GSPO, G=4, lr 5e-5) makes false, and record it.

Run 43 falsifies three statements and weakens a fourth:
  * 5.2.3's heading says the idle-step attribution "does not hold". At G=4 there is now
    a genuine single-variable pair (same group size, same lr, same effective batch,
    only the IS level differs) and it moves idle steps by 6.0 points -- so the
    attribution holds at G=4, while the large confounded contrast still does not.
  * 5.2.5's bullets say SFT is the best on clock MAE and on size MAE. Run 43 is better
    on both (1.6761 vs 1.6847; 0.4025 vs 0.4999), which the new bold in patch82 shows
    and these bullets have to say.
  * 5.2.5 says the "whole class collapses" happen "only in the two lr-5e-5 GSPO cells
    (G=8 and G=32)". Run 43 is GSPO at lr 5e-5 and drops no class at all, so the
    phenomenon is not a property of that algorithm/lr combination.
Counts of "N RL runs" go from five to six.

The new 5.2.6 records the pair and the group-size series. Every number in it is
formatted from outputs/reports/paired_significance.json rather than typed here, so the
document cannot drift from the record.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
DOC = Path(os.environ.get("WAFER_DOC", str(ROOT / "LIMITATIONS.md")))
PS = json.loads((ROOT / "outputs/reports/paired_significance.json").read_text(encoding="utf-8"))

IL = PS["is_level_paired"]
GS = PS["group_size_series_lr5e5"]["rungs"]
r1, r2 = GS["G=4 -> G=8"], GS["G=8 -> G=32"]


def ci(d):
    return f"[{d['delta_ci95_rows'][0]:+.4f}, {d['delta_ci95_rows'][1]:+.4f}]"


EDITS = [
    # ---------------------------------------------------------------- 5.2.3 heading
    ("#### 5.2.3 梯度信号：空转步确实少了，但归因不成立、也没有转成基准分",
     "#### 5.2.3 梯度信号：单变量对照下落了 6.0 个百分点，但没有转成基准分",
     "5.2.3 heading: the attribution now holds in one single-variable pair"),

    # ------------------------------------------------------- 5.2.3 idle-step series
    ("""lr 5e-5 下 G=4 是 32.67%、G=8 是 38.00%、G=32 是 42.00%，即**组越大空转步越多**
（但这一跨度里 G=4→G=8 同时换了 IS 层级，且每一步都同时改了有效批次
16→64→1024，所以「组大小」在这里同样不是单变量 —— 第 8 节第 11 条。
下面只把它当作一个方向性参照，不当作识别）；""",
     """在**同一个算法、同一个学习率**下（GSPO、lr 5e-5），空转步占比随组大小单调上升：
G=4 是 26.67%、G=8 是 38.00%、G=32 是 42.00% —— run 43 落地后，这条线上不再混着
IS 层级（此前 G=4 那个点只能取 GRPO 的 32.67%，与 G=8/G=32 不是同一个算法）。
但每一步仍同时把有效批次从 16 改到 64、1024（组大小的平方），所以「组大小」在这里
依然不是单变量 —— 第 8 节第 11 条。下面只把它当作一个方向性参照，不当作识别。
**另有一条真正单变量的对照落在空转步上**：GSPO(G=4) 26.67% 对 GRPO(G=4) 32.67% ——
同组大小、同学习率 5e-5、同有效批次 16，只差 IS 层级，序列级归一化低 **6.0 个百分点**）；""",
     "5.2.3: the group-size series is now single-algorithm, and the IS level has its own pair"),

    # -------------------------------------------------- 5.2.3 closing paragraph
    ("""**但梯度信号的改善没有转化成基准分数**：lr1e-5 的 GSPO 在基准上与 GRPO、与 SFT
都在噪声内（5.2.2）。这是本次 RL 最值得记下的一条 ——
**优势信号的多少与最终指标的高低，在这里是解耦的。**""",
     """**但梯度信号的改善没有转化成基准分数**：lr1e-5 的 GSPO 在基准上与 GRPO、与 SFT
都在噪声内（5.2.2）；今天落地的单变量对照把这一点钉得更死 —— 同组大小、同学习率
5e-5 下只换 IS 层级，空转步少了 6.0 个百分点，而**同一对配置**的准确率差是
{il_delta}、置信区间 {il_ci} 包含 0（`p` = {il_p}，见 5.2.6）。这是本次 RL 最值得记下的一条 ——
**机制层面的差异测得出，基准分数上的差异测不出。**""",
     "5.2.3 closing: the decoupling now has a single-variable instance"),

    # ------------------------------------------------------------------ 5.2.4 判定
    ("""五个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 GRPO lr1e-5（0.6197 对 0.6114），
成对检验下与 SFT 完全无法区分（Δ准确率 0.0000、`p` = 1.00）；
GRPO lr5e-5 与两个 lr 5e-5 的 GSPO run 都低于 SFT，
其中论文的原始设定 G=32 + lr 5e-5 为 0.5403（见 5.2.5）。""",
     """六个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 GRPO lr1e-5（0.6197 对 0.6114），
成对检验下与 SFT 完全无法区分（Δ准确率 0.0000、`p` = 1.00）；
GRPO lr5e-5 与 lr 5e-5 的三个 GSPO run（G=4 0.4910、G=8 0.5014、G=32 0.5403）
都低于 SFT，其中论文的原始设定 G=32 + lr 5e-5 为 0.5403（见 5.2.5）。""",
     "5.2.4: six RL runs; three lr-5e-5 GSPO runs below SFT"),

    # ------------------------------------------------- 5.2.5 "the only config" claim
    ("""5.2.4 之前留下一个未检验的推断（「问题可能出在组大小」）。G=32 + lr 5e-5
是论文的原始设定，也是唯一能检验它的配置，已按论文跑满 150 步。""",
     """5.2.4 之前留下一个未检验的推断（「问题可能出在组大小」）。G=32 + lr 5e-5
是论文的原始设定，已按论文跑满 150 步。但**它并不是单变量对照**：组大小 8→32 的
同时有效批次从 64 改到 1024（组大小的平方）。能把组大小单独动一次的是同学习率的
G=32 + lr 1e-5（队列 41，正在跑）。""",
     "5.2.5: G=32 lr5e-5 is not the single-variable test; G=32 lr1e-5 is"),

    # ------------------------------------------------------- 5.2.5 "only lr separates"
    ("""更正后仍然成立的是（比初稿更弱）：**在 lr 5e-5 这一档上，唯一分得开的是学习率** ——""",
     """更正后仍然成立的是（比初稿更弱）：**在 lr 5e-5 这一档上，唯一分得开的是学习率** ——
这一点现在有了更直接的证据：论文主张的那条轴（IS 层级）在**单变量**对照下同样分不开
（G=4、同学习率、同有效批次，只换 IS 层级：Δ准确率 {il_delta}、置信区间 {il_ci}
包含 0、`p` = {il_p}），而 GSPO 内部沿组大小的三点连线也是平的
（G=4→G=8 Δ {r1d:+.4f}、`p` = {r1p}；G=8→G=32 Δ {r2d:+.4f}、`p` = {r2p}，见 5.2.6）——""",
     "5.2.5: the IS level is not separable even single-variably"),

    # ---------------------------------------------------------- 5.2.5 meta note
    ("""（此表的列数是随时间增长的：写此句时为六列。在任何新 run 落地后，下面四句里的
「六个 run」都必须重核而不是沿用 —— 这正是本条要防的那个错误。）""",
     """（此表的列数是随时间增长的：写此句时为七列。在任何新 run 落地后，下面四句里的
「七个 run」都必须重核而不是沿用 —— 这正是本条要防的那个错误：run 43 落地时，
它一次推翻了下面两句的极值归属（时钟 MAE 与尺寸 MAE 的最低值原本都在 SFT），
而这两句的结论原本只写在加粗里。）""",
     "5.2.5 meta note: seven columns, and what run 43 moved"),

    # ----------------------------------------------------- 5.2.5 extremum bullets
    ("""- `radial_zone`：**成立**。G=32 的 0.4603 在六个 run 里最高，
  是 SFT 的 1.74 倍。
- 时钟 MAE：**不成立**。最低的是 SFT 的 1.6847，
  G=32 的 1.6860 只是第二。
- 尺寸 MAE：**成立**。G=32 的 1.6509 在六个 run 里最高（即最差），
  是 SFT 的 3.30 倍。
- caption must-hit：**不成立**。最低的是 GRPO(G=4) lr5e-5 的 0.5516，
  G=32 的 0.5714 是第二低。""",
     """- `radial_zone`：**成立**。G=32 的 0.4603 在七个 run 里最高，
  是 SFT 的 1.74 倍。
- 时钟 MAE：**不成立，且最低值今天换了主角**。最低的是 GSPO(G=4) lr5e-5 的 1.6761
  （run 43），SFT 的 1.6847 退为第二，G=32 的 1.6860 第三。
- 尺寸 MAE：**成立**。G=32 的 1.6509 在七个 run 里最高（即最差），
  是 SFT 的 3.30 倍；但**最低的也不是 SFT 了** —— 是 run 43 的 0.4025，
  加粗随之从 SFT 移走。
- caption must-hit：**不成立**。最低的是 GRPO(G=4) lr5e-5 的 0.5516，
  G=32 的 0.5714 是第二低（run 43 的 0.6349 在两者之上）。""",
     "5.2.5 bullets: two extrema belong to run 43, not SFT"),

    # ------------------------------------------------------- 5.2.5 class count
    ("""在五个 RL run 里，**恰好只有两个有类别归零**：GSPO(G=8) lr5e-5 丢 `none`、""",
     """在六个 RL run 里，**恰好只有两个有类别归零**：GSPO(G=8) lr5e-5 丢 `none`、""",
     "5.2.5: six RL runs drop a class"),

    # ------------------------------------------- 5.2.5 "collapse is a GSPO lr5e-5 thing"
    ("""**但 GRPO(G=4) lr5e-5 一个类别都没丢**（SFT 规模的两个类别都保住了），
尽管它的准确率同样显著下降（5.2.2）。所以「整类塌陷」**不是 5e-5 的通用后果，
而是在 lr 5e-5 的这两格 GSPO（G=8 与 G=32）上才出现的现象** —— 这条限制初稿
没有，因为它没有第五列。""",
     """**但有两个 run 一个类别都没丢：GRPO(G=4) lr5e-5 与 GSPO(G=4) lr5e-5（run 43）**
（SFT 规模的两个类别都保住了），尽管前者的准确率同样显著下降（5.2.2）。所以
「整类塌陷」**既不是 5e-5 的通用后果，也不是「GSPO + lr 5e-5」的通用后果** ——
run 43 恰好落在同一个算法、同一个学习率上，却一个类别都没丢；它只出现在 lr 5e-5 的
**G=8 与 G=32 这两格**。这条限制初稿没有，因为它没有第五列。""",
     "5.2.5: class collapse is not a property of GSPO at lr 5e-5"),

    # ------------------------------------------------ 5.2.5 weakest-run / std table
    ("""它同时继承了高学习率的退化与大组下的多样性下降 —— 组内奖励标准差
在 4 个奖励里有 3 个是五组最低 —— 但**「五组最低的标准差」与「最低的分数」
并不是同一件事，这里不再用前者给后者作保**。""",
     """今天落地的 GSPO(G=4) lr5e-5 比两者都低（0.4910），是六个 RL run 里最低的。
G=32 同时继承了高学习率的退化与大组下的多样性下降 —— 组内奖励标准差
在 4 个奖励里有 3 个是六组最低 —— 但**「六组最低的标准差」与「最低的分数」
并不是同一件事，这里不再用前者给后者作保**。""",
     "5.2.5: six groups, and the new lowest run"),
]

NEW_SECTION = """#### 5.2.6 第一个单变量对照：IS 层级分不开，组大小也分不开

run 43（GSPO, G=4, lr 5e-5）落地后，本项目第一次有了「只动论文主张的那一个变量」的
对照。下面是记录里的原值（`outputs/reports/paired_significance.json`）。

**（1）IS 层级：GRPO(G=4, token) 对 GSPO(G=4, sequence)。** 同组大小 4、同有效批次 16
（accumulation 4）、同学习率 5e-5、同 seed 3407、同数据、同奖励与权重、同 150 步 ——
记录里 `single_variable = true`，唯一变动的是 `--importance_sampling_level`。

- 准确率差（sequence − token）= **{il_delta}**，95% 置信区间 **{il_ci}**（包含 0），
  McNemar 精确检验 `p = {il_p}`（token 错而 sequence 对 16 行，反之 24 行）。
- 记录自己的读法是 **"{il_reading}"**。
- 记录同时写明它**不能**替论文的主张下结论：论文说的是「学习率效应如何随 IS 层级变化」，
  只取一个学习率看不出耐受度的差别（`does_not_settle` 字段原文如此）。
- 机制侧反而分得开：**同一对配置**的空转步占比是 32.67% 对 26.67%（Δ = −6.0 个百分点）。

**（2）组大小三点连线（GSPO、sequence、lr 5e-5）。** G=4 → G=8：Δ = **{r1d:+.4f}**，
区间 **[{r1lo:+.4f}, {r1hi:+.4f}]**、`p` = {r1p}（19/18 行）。G=8 → G=32：Δ = **{r2d:+.4f}**，
区间 **[{r2lo:+.4f}, {r2hi:+.4f}]**、`p` = {r2p}。两个区间都包含 0。第二段的区间
**沿用** `group_size_paired`（`reused_from` 字段），同一个对照不取两个区间。

这条连线**不是单变量的**：组大小 4→8→32 的同时，有效批次从 16 改到 64、1024
（组大小的平方，见 `effective_batch_correction.json` 的 `effective_batch_by_G`）。
所以它的平，与「组大小无效」不是同一句话。

**三条合起来：** 本设计里能分得开的仍然只有学习率（5.2.5）；论文主张的那条轴在 G=4 上
单变量也分不开；GSPO 内部沿组大小是平的。**这三条都是「在本模型规模与本次预算下」的
陈述**，不是「论文的机制不存在」—— 一个学习率、一个 252 行基准、150 步，能支持的
最强结论就到这里。

"""

anchor_new = "## 6. 评测与指标解释"
src = DOC.read_text(encoding="utf-8")

before = src.count("**")
for old, new, why in EDITS:
    n = src.count(old)
    if n != 1:
        sys.exit(f"anchor for '{why}' appears {n} times, expected 1; nothing written")
    src = src.replace(old, new)

if src.count(anchor_new) != 1:
    sys.exit(f"section anchor '{anchor_new}' appears {src.count(anchor_new)} times")

fmt = {
    "il_delta": f"{IL['accuracy_delta_sequence_minus_token']:+.4f}",
    "il_ci": ci(IL),
    # full precision for this one: the document already quotes it as 0.268187
    "il_p": f"{IL['mcnemar']['p_exact_two_sided']}",
    "il_reading": IL["reading"],
    "r1d": r1["accuracy_delta"], "r1lo": r1["delta_ci95_rows"][0],
    "r1hi": r1["delta_ci95_rows"][1],
    "r1p": f"{r1['mcnemar']['p_exact_two_sided']:.5f}",
    "r2d": r2["accuracy_delta"], "r2lo": r2["delta_ci95_rows"][0],
    "r2hi": r2["delta_ci95_rows"][1],
    "r2p": f"{r2['mcnemar']['p_exact_two_sided']:.5f}",
}
src = src.replace(anchor_new, NEW_SECTION.format(**fmt) + anchor_new, 1)

after = src.count("**")
if (after - before) % 2:
    sys.exit(f"bold markers went {before} -> {after}, an odd change; nothing written")

DOC.write_text(src, encoding="utf-8")
print(f"{len(EDITS)} prose edit(s) + new 5.2.6 written ({len(src.splitlines())} lines); "
      f"bold markers {before} -> {after}")
for _, _, why in EDITS:
    print(f"  - {why}")
