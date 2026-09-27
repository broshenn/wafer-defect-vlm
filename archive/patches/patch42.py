"""Close out 5.2.2 now that G=32 has run, and disclose the missing control cell.

The heading asserted "the outlier is the learning rate, not the algorithm". That
is established within GSPO (two group sizes, both lrs) and at lr 1e-5 across
algorithms -- but GRPO was never run at lr 5e-5, so it cannot yet rule out that
GSPO is specifically the fragile one. The heading and the closing paragraph now
say exactly that much and no more.
"""
import sys
from pathlib import Path

path = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
src = path.read_text(encoding="utf-8")
edits = []

# ---- A. the heading overstated what a run set without GRPO@5e-5 can support
old_a = "#### 5.2.2 基准成绩：离群的是学习率，不是算法"
new_a = "#### 5.2.2 基准成绩：离群的是学习率 —— 以及一个尚未补齐的对照格"
edits.append(("A", old_a, new_a))

# ---- B. the table was missing the G=32 column the section's own argument needs
old_b = """| 指标 | SFT | GRPO (G=4, lr1e-5) | GSPO (G=8, **lr5e-5**) | GSPO (G=8, lr1e-5) |
| --- | --- | --- | --- | --- |
| 分类准确率 | 0.6230 | 0.6230 | **0.5397** | 0.6151 |
| macro-F1 | 0.6114 | 0.6197 | **0.5014** | 0.6089 |
| macro-F1 95% CI | [0.5496, 0.6602] | [0.5554, 0.6704] | **[0.4367, 0.5537]** | [0.5472, 0.6613] |
| structured defect_type | 0.5840 | 0.6032 | 0.5635 | 0.5595 |
| structured radial_zone | 0.2640 | 0.2698 | 0.3056 | 0.2460 |
| clock circular MAE | 1.6847 | 1.7121 | **1.9519** | 1.7098 |
| size MAE (R) | 0.4999 | 0.5695 | **0.6927** | 0.6203 |
| caption must-hit | 0.7540 | 0.7222 | **0.6627** | 0.7143 |
| robustness accuracy | 0.5622 | 0.5661 | **0.5146** | 0.5688 |
| flip rate | 0.2474 | 0.2672 | 0.2354 | 0.2487 |
| **零优势步占比（实测）** | — | 36.00% | 38.00% | **15.30%** |"""
new_b = """| 指标 | SFT | GRPO (G=4, lr1e-5) | GSPO (G=8, **lr5e-5**) | GSPO (G=8, lr1e-5) | GSPO (G=32, lr5e-5) |
| --- | --- | --- | --- | --- | --- |
| 分类准确率 | 0.6230 | 0.6230 | **0.5397** | 0.6151 | 0.5675 |
| macro-F1 | 0.6114 | 0.6197 | **0.5014** | 0.6089 | 0.5403 |
| macro-F1 95% CI | [0.5496, 0.6602] | [0.5554, 0.6704] | **[0.4367, 0.5537]** | [0.5472, 0.6613] | [0.4807, 0.5965] |
| structured defect_type | 0.5840 | 0.6032 | 0.5635 | 0.5595 | 0.5595 |
| structured radial_zone | 0.2640 | 0.2698 | 0.3056 | 0.2460 | **0.4603** |
| clock circular MAE | 1.6847 | 1.7121 | **1.9519** | 1.7098 | 1.6860 |
| size MAE (R) | 0.4999 | 0.5695 | **0.6927** | 0.6203 | **1.6509** |
| caption must-hit | 0.7540 | 0.7222 | **0.6627** | 0.7143 | **0.5714** |
| robustness accuracy | 0.5622 | 0.5661 | **0.5146** | 0.5688 | 0.5556 |
| flip rate | 0.2474 | 0.2672 | 0.2354 | 0.2487 | 0.2540 |
| retrieval mAP@10 | 0.3873 | 0.3742 | 0.3607 | 0.3782 | **0.4337** |
| **零优势步占比（实测）** | — | 36.00% | 38.00% | **15.30%** | **42.00%** |"""
edits.append(("B", old_b, new_b))

# ---- C. the G=32 branch was open; it has now resolved
old_c = """G=32 的 run 正是在检验这个解释：**更大的组显著降低优势方差，通常能容忍更高的学习率**
—— 这恰恰是 GSPO 论文主张自己解决的问题。若 G=32 + 5e-5 恢复正常，
则「5e-5 只是对本模型过大」成立；若同样崩坏，则说明该学习率对本模型规模根本不适用。"""
new_c = """G=32 的 run 正是在检验这个解释：**更大的组显著降低优势方差，通常能容忍更高的学习率**
—— 这恰恰是 GSPO 论文主张自己解决的问题。**该分支现已判定：G=32 + lr 5e-5 的
macro-F1 是 0.5403，仍然明显低于 SFT，未恢复正常，因此「该学习率对本模型规模
不适用」这一支成立。** 组大小确实有帮助（同 lr 下 0.5014 → 0.5403），
但量级不到降学习率的 1/3，详见 5.2.5。"""
edits.append(("C", old_c, new_c))

# ---- D. disclose the gap rather than let the heading imply more than it can
old_d = """措辞因此定为**「明显更差」而非「显著更差」**。"""
new_d = """措辞因此定为**「明显更差」而非「显著更差」**。

**一个必须点明的缺口：GRPO 只跑了 lr 1e-5，没跑 lr 5e-5。**
所以「坏的是学习率、不是算法」这句话，证据只覆盖到：
（a）GSPO 内部两个学习率的对照（5e-5 明显更差，且已在 G=8 与 G=32 两个组大小上各见一次）；
（b）学习率对齐到 1e-5 后 GSPO 与 GRPO 无差别。
但**没有** GRPO 在 5e-5 下的读数，因此无法排除「崩的是 GSPO 本身」这一解释 ——
即无法直接验证论文「GSPO 比 GRPO 更耐高学习率」的中心主张。
补上 GRPO(G=4, lr5e-5) 即成为完整 2×2 对照（算法 × 学习率），本轮未跑，
已在 `outputs/reports/provenance.json` 的 `deviations_from_spec` 中登记为未决事项。
在补齐之前，本节结论的适用范围限于上述 (a)(b) 两条。"""
edits.append(("D", old_d, new_d))

for tag, old, new in edits:
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAILED at edit {tag}: {n} occurrences (need exactly 1)")
    src = src.replace(old, new)

path.write_text(src, encoding="utf-8")
print("LIMITATIONS 5.2.2: G=32 column added, branch resolved, control-cell gap disclosed")
print(f"file now {len(src.splitlines())} lines")
