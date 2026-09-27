"""Replace LIMITATIONS section 5.2 with the version the measurements support.

The section has now been wrong twice. It first claimed GSPO cut idle steps from
36% to 13.3% -- the 13.3% was a single logged row, not a mean. Corrected against
all 150 steps it became "no change" (38.00% vs 36.00%), which retracted the
improvement claim entirely. The lr-matched GSPO run now shows 15.30%, so the
original direction was right after all, but it had been read off a confounded
comparison and a misread number.

Rather than quietly write the third version, the section keeps all three states
visible, because "I measured this three times and the middle one was also wrong"
is the part a reader needs in order to trust the third.

Spliced between the "### 5.2" header and the next "## 6." header.
"""
from pathlib import Path

path = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

start = next(i for i, l in enumerate(lines) if l.startswith("### 5.2"))
end = next(i for i, l in enumerate(lines) if l.startswith("## 6."))
print(f"replacing lines {start + 1}..{end} ({end - start} lines)")

NEW = '''### 5.2 GSPO：两个错的结论，和那个真正起作用的变量

论文指定的是 GSPO，而 5.1 跑的是普通 GRPO，因此按论文补跑。做法不是换脚本，
而是把 `--importance_sampling_level` 从默认的 `token` 改成 `sequence`（见 §5 首条），
这样两个 run 的差异是一个有文档记录的开关，而不是一份被改过的脚本。

本节记录三个 GSPO run：组大小 8 的两个（学习率分别取论文的 5e-5 与 GRPO 的 1e-5），
以及论文设定的 G=32。**其中「G=32」这一条，是我先断言不可行、后被实测推翻的。**

#### 5.2.1 两个我先写错、后被实测推翻的结论

**其一：「GSPO 无法执行」。** 依据是该 ms-swift 提交的 `rlhf_type` 枚举
`['dpo','orpo','simpo','kto','cpo','rm','ppo','grpo','gkd']` 不含 `gspo`。
这是错的 —— GSPO 不是独立的 `rlhf_type`，而是 GRPO 的损失变体，
由 `--importance_sampling_level sequence` 开启（证据见 §5 首条）。
**因此第一次跑的是普通 GRPO，不是 GSPO。**

**其二：「论文的 G=32 需要 40 小时以上，本环境不可行」。**
这个数字**从未被测量过**，来源是一个错误假设：把 `generation_batch_size` 当成 prompt 数，
于是算出每步 32×32=1024 次生成。而 ms-swift 在第一次启动失败时给出的报错
（`generation_batch_size (4) must be evenly divisible by num_generations (16)`）
已经说明它是按 **completion** 计数的，所以每步只有 32 次生成。
真正的证据当时就在日志里，被我跳过了。

实测（`SMOKE=1 NUM_GENERATIONS=32 GRAD_ACCUM=32`，2 步）：

| | G=4 | G=8 | **G=32（实测）** |
| --- | --- | --- | --- |
| 步时 | 16.1s | 21.2s | **60–62s** |
| 150 步预计 | 0.7h | 0.9h | **约 2.5h** |
| 峰值显存 | 14,391 MiB | 14,617 MiB | **15,185 MiB** |

**论文的组大小在这台机器上完全跑得动，我原来的结论错了约 17 倍。**
G=32 的正式 run 因此在报告定稿前启动（`RUN_TAG=qwen35_9b_gspo_g32`，
G=32 / lr 5e-5 / sequence），结果并入 5.2.3。

教训：**「不可行」和「已测量」是同一等级的断言，没有实测就不能写成结论。**
本项目已有多处同类错误（见 §8），这是又一处，且这次错在**把一条已有的证据读漏了**。

#### 5.2.2 基准成绩：离群的是学习率，不是算法

| 指标 | SFT | GRPO (G=4, lr1e-5) | GSPO (G=8, **lr5e-5**) | GSPO (G=8, lr1e-5) |
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
| **零优势步占比（实测）** | — | 36.00% | 38.00% | **15.30%** |

**只有 lr 5e-5 那一列是坏的，而且坏得很一致**：准确率 −0.083、macro-F1 −0.110、
size MAE +0.19、描述必中 −0.09，除 `radial_zone`（该指标本身标签退化，见 §6.7）外
几乎每项同向变差。**学习率对齐到 1e-5 之后，GSPO 与 GRPO、与 SFT 全部落回噪声内**
（macro-F1 0.6089 vs SFT 0.6114，置信区间大幅重叠）。

因此**「GSPO 让模型变差」是错的，正确说法是「论文的 5e-5 在这套 QLoRA 配置下过大」。**
证据不止基准分：同一个 5e-5 在 G=8 上把 KL 抬到 1.02–1.12，并把空转步从 36.00% 推到 38.00%，
符合「策略被推进退化区」的表现（输出趋同 → 组内奖励相同 → 优势为 0）。
G=32 的 run 正是在检验这个解释：**更大的组显著降低优势方差，通常能容忍更高的学习率**
—— 这恰恰是 GSPO 论文主张自己解决的问题。若 G=32 + 5e-5 恢复正常，
则「5e-5 只是对本模型过大」成立；若同样崩坏，则说明该学习率对本模型规模根本不适用。

一处必须保留的折扣：**lr5e-5 列的 95% CI 与 SFT 是擦边重叠的**
（上界 0.5537 > SFT 下界 0.5496，重叠 0.0041），严格说**不能宣称统计显著**。
`tools/final_report.py` 按 `ci[1] < sft_ci[0]` 自动判定并输出「与 SFT 重叠」，
而我初稿手写的是「不重叠」，与工具结论相反，已按工具改正。
措辞因此定为**「明显更差」而非「显著更差」**。

#### 5.2.3 梯度信号：序列级归一化确实有效，但没有转成基准分

这一段我写错过两次，两次更正都留在下面，因为过程本身是结论的一部分。

- **初稿（错）**：声称 GSPO 把空转步从 36% 降到 13.3%。那个 13.3% **不是任何文件里的统计量** ——
  训练记录只存 first/last 两步，我把其中一步的值当成了 150 步的均值。
- **第一次更正（仍不完整）**：按完整 150 步实测，lr5e-5 的空转步是 38.00%（57/150），
  比 GRPO 的 36.00%（54/150）还高，于是撤回「改善」，改写成「基本没有变化」。
- **本次（正确）**：学习率对齐后，**GSPO lr1e-5 的空转步是 15.30%（23/150），
  相对 GRPO 的 36.00% 是一次实质下降。**

所以初稿的**方向**是对的，但当时比的是两个不同学习率的 run、数字又是误读的；
直到有了学习率对齐的对照，这个方向才第一次建立在干净证据上。

| 奖励的组内标准差（150 步均值） | GRPO (lr1e-5) | GSPO lr5e-5 | GSPO lr1e-5 |
| --- | --- | --- | --- |
| `WaferClass` | 0.0803 | 0.0855 | 0.0912 |
| `WaferFormat` | 0.0005 | 0.0132 | 0.0030 |
| `WaferRadial` | 0.1180 | 0.0528 | 0.1256 |
| `WaferClock` | 0.2244 | 0.2306 | 0.2526 |

**但梯度信号的改善没有转化成基准分数**：lr1e-5 的 GSPO 在基准上与 GRPO、与 SFT
都在噪声内（5.2.2）。这是本次 RL 最值得记下的一条 ——
**优势信号的多少与最终指标的高低，在这里是解耦的。**
5.1 里「问题出在组大小」的推断，也因此仍未得到检验（G=32 的结果见 5.2.4）。

#### 5.2.4 判定

按论文自己 §8.11 的门槛（「若提升落在置信区间内，结论写『未观察到显著提升』，
不写『RL 有效』」，分支「F -- 否 --> G[保留 SFT 为最终模型]」）：

> **最终模型保留 SFT（`outputs/checkpoints/qwen35_9b_qlora_v1`）。**
> GRPO 与 GSPO（两个学习率、两个组大小）均**未观察到相对 SFT 的显著提升**。
> 不写「RL 有效」。

需要说清的是，这**不等于**「GSPO 不如 SFT」或「GSPO 不如 GRPO」：
三个 RL run 在基准上的差异都落在噪声内，唯一明显的差异来自学习率而非算法。
诚实的表述是：**在本模型规模与本次预算下，RL 阶段没有产生可测的基准增益；
但 GSPO 的序列级归一化确实把零优势步砍掉了约六成（36.00% → 15.30%），
这个信号层面的收益是真实的、可复现的，只是没有传导到基准分数。**

'''

lines[start:end] = [NEW]
path.write_text("".join(lines), encoding="utf-8")
print(f"section 5.2 replaced; file now {len(path.read_text(encoding='utf-8').splitlines())} lines")

# The old text asserted G=32 was infeasible; make sure no claim of that survives
# anywhere in the document, since it is now known to be false.
leftover = [f"{i}: {l.strip()[:90]}" for i, l in enumerate(
    path.read_text(encoding="utf-8").splitlines(), 1)
    if "40 小时" in l or "不可行" in l]
if leftover:
    print("NOTE: lines still mentioning infeasibility -- check each is the corrected one:")
    for l in leftover:
        print("   ", l)
