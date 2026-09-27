"""Record the G=32 verdict in LIMITATIONS 5.2.

Four edits, all anchored on text that must appear exactly once:
  A. the reward-std table in 5.2.3 gains the G=32 column and an idle-step row
  B. the "still untested" sentence in 5.2.3 now points at the run that tested it
  C. a new 5.2.5 carries the verdict
  D. the 5.2.4 conclusion states which RL runs beat SFT and which did not
"""
import ast
import sys
from pathlib import Path

path = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
src = path.read_text(encoding="utf-8")
edits = []

# ---- A. reward-std table: add the G=32 column, and the idle step with it
old_a = """| 奖励的组内标准差（150 步均值） | GRPO (lr1e-5) | GSPO lr5e-5 | GSPO lr1e-5 |
| --- | --- | --- | --- |
| `WaferClass` | 0.0803 | 0.0855 | 0.0912 |
| `WaferFormat` | 0.0005 | 0.0132 | 0.0030 |
| `WaferRadial` | 0.1180 | 0.0528 | 0.1256 |
| `WaferClock` | 0.2244 | 0.2306 | 0.2526 |"""
new_a = """| 奖励的组内标准差（150 步均值） | GRPO (G=4, lr1e-5) | GSPO (G=8, lr5e-5) | GSPO (G=8, lr1e-5) | GSPO (G=32, lr5e-5) |
| --- | --- | --- | --- | --- |
| `WaferClass` | 0.0803 | 0.0855 | 0.0912 | **0.0517** |
| `WaferFormat` | 0.0005 | 0.0132 | 0.0030 | 0.0219 |
| `WaferRadial` | 0.1180 | 0.0528 | 0.1256 | **0.0346** |
| `WaferClock` | 0.2244 | 0.2306 | 0.2526 | **0.1435** |
| **空转步占比** | 36.00% | 38.00% | **15.30%** | 42.00% |"""
edits.append(("A", old_a, new_a))

# ---- B. the inference in 5.1 was testable, and now has been tested
old_b = "5.1 里「问题出在组大小」的推断，也因此仍未得到检验（G=32 的结果见 5.2.4）。"
new_b = ("5.1 里「问题出在组大小」的推断，已由 G=32 + lr 5e-5 实测检验，结论见 5.2.5：\n"
         "**组大小是次要变量，学习率是主导变量。**")
edits.append(("B", old_b, new_b))

# ---- C. the verdict
old_c = "## 6. 评测与指标解释"
new_c = """#### 5.2.5 G=32：论文的组大小救不回论文的学习率

5.2.4 之前留下一个未检验的推断（「问题可能出在组大小」）。G=32 + lr 5e-5
是论文的原始设定，也是唯一能检验它的配置，已按论文跑满 150 步。

| 方向 | 配置变化 | macro-F1 |
| --- | --- | --- |
| **降学习率** | G=8：5e-5 → 1e-5 | 0.5014 → 0.6089（**+0.108**）|
| **加组大小** | lr 5e-5：G=8 → G=32 | 0.5014 → 0.5403（**+0.039**）|

两个方向的量级差约 2.8 倍，所以**组大小确实有效，只是远不足以补上学习率的代价**。
而且即便把论文的两个设定叠在一起（G=32 + lr 5e-5），macro-F1 0.5403 仍够不到
SFT 的 0.6114；其 95% CI [0.4807, 0.5965] 与 SFT 的 [0.5496, 0.6602] **重叠**，
所以这一栏同样只能写「明显更低」，**不能写「显著下降」**。

**空转步与基准分的读数再次相反**：G=32 的空转步 42.00%（63/150）是四组里最差的，
但其 macro-F1 又高于空转步只有 38.00% 的 G=8 lr5e-5。
「零优势步的多少」因此不能当作成绩的代理指标 —— 与 5.2.3 的结论一致。

**G=32 最有信息量的一点是它没有把错误等比例放大，而是重塑了错误画像：**

| 指标 | SFT | GSPO G=8 lr5e-5 | GSPO G=32 lr5e-5 |
| --- | --- | --- | --- |
| `radial_zone` | 0.2640 | 0.3056 | **0.4603** ← 全部 run 最高，SFT 的 1.74 倍 |
| 时钟 MAE | 1.6847 | 1.9519 | **1.6860** ← 全部 run 最好 |
| 尺寸 MAE | 0.4999 | 0.6927 | **1.6509** ← 全部 run 最差，SFT 的 3.3 倍 |
| caption must-hit | 0.7540 | 0.6627 | **0.5714** ← 全部 run 最差 |

按类别看更清楚 —— **两个 lr 5e-5 的 run 各自丢掉了一整个类别，丢的却不是同一个：**

| 类别 | SFT | GSPO G=8 lr5e-5 | GSPO G=8 lr1e-5 | GSPO G=32 lr5e-5 |
| --- | --- | --- | --- | --- |
| Donut | 0.462 | 0.378 | 0.421 | **0.000** |
| none | 0.400 | **0.000** | 0.343 | 0.412 |
| Edge_Ring | 0.870 | 0.778 | 0.892 | 0.524 |
| Scratch | 0.286 | 0.286 | 0.333 | **0.462** |

G=8 丢 `none`，G=32 丢 `Donut`，而 lr 1e-5 两个都恢复。
**整类塌陷在 lr 5e-5 下不是一个稳定的偏差，而是一个不稳定的偏差 —— 换个组大小
就换一个牺牲品。** 这比「分数低」更能支持「策略被推进退化区」的诊断，
因为它同时排除了「这个类别本身就难」这一解释：同类别在 lr 1e-5 下是学得到的。
另一方面 G=32 在 `Scratch` 上拿到全部 run 最高的 0.462，说明它也不是全面退化。

**因此论文的原始设定（G=32 + lr 5e-5）是本项目测过的所有 RL 配置里较弱的组合**：
它同时继承了高学习率的退化与大组下的多样性下降 —— 组内奖励标准差在 4 个奖励里
有 3 个是四组最低。

顺带更正一条我在实测中途给过的说法：曾据前 72 步的数据断言「KL 只由学习率决定、
与组大小无关」。跑满 150 步后 GSPO G=8 lr5e-5 的 KL 是 1.022、G=32 lr5e-5 是 0.959，
并不相等。**学习率是 KL 的主导因素，但不是唯一因素** —— 中途读数不足以支持
「无关」这种强度的断言，这也是本项目第二次由「不足的样本 + 过强的措辞」造成问题
（第一次见 5.2.3 的 13.3%）。

## 6. 评测与指标解释"""
edits.append(("C", old_c, new_c))

# ---- D. 5.2.4 states which runs beat SFT and which did not
old_d = """这个信号层面的收益是真实的、可复现的，只是没有传导到基准分数。**"""
new_d = """这个信号层面的收益是真实的、可复现的，只是没有传导到基准分数。**

四个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 GRPO（0.6197 对 0.6114），
差异远在噪声内；两个 lr 5e-5 的 GSPO run 都低于 SFT，其中论文的原始设定
G=32 + lr 5e-5 为 0.5403，是较弱的一个（见 5.2.5）。"""
edits.append(("D", old_d, new_d))

for tag, old, new in edits:
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAILED at edit {tag}: {n} occurrences (need exactly 1)")
    src = src.replace(old, new)

path.write_text(src, encoding="utf-8")
print("LIMITATIONS 5.2: G=32 verdict recorded (edits A-D)")
print(f"file now {len(src.splitlines())} lines")
