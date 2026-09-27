"""Downgrade the group-size claim in 5.2.5, now that it has a significance test.

5.2.5 argued "group size really does help, just not enough" from a 2.8x RATIO
between two marginal macro-F1 deltas -- a magnitude argument. Magnitude is not
significance, and 5.2.2 is being corrected this same round for exactly that
class of reasoning. Applying the same paired instrument to the same 252 rows:
G=8 -> G=32 at lr 5e-5 gives +0.0278 accuracy, CI [-0.0278, +0.0833], p=0.410.
So the direction survives and the strength claim does not.

What makes this worth stating rather than quietly softening: the two contrasts
around G=32 are +0.0278 (vs G=8) and -0.0278 (vs GRPO G=4), both p=0.41. At
lr 5e-5 the design cannot separate G=4 from G=8 from G=32 AT ALL. The only
variable that separates anything at lr 5e-5 is the learning rate. That is a
stronger and cleaner statement than the one being replaced, and it is what the
completed 2x2 plus the group-size arm actually show.
"""
import sys
from pathlib import Path

p = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
s = p.read_text(encoding="utf-8")

OLD1 = """| 方向 | 配置变化 | macro-F1 |
| --- | --- | --- |
| **降学习率** | G=8：5e-5 → 1e-5 | 0.5014 → 0.6089（**+0.108**）|
| **加组大小** | lr 5e-5：G=8 → G=32 | 0.5014 → 0.5403（**+0.039**）|

两个方向的量级差约 2.8 倍，所以**组大小确实有效，只是远不足以补上学习率的代价**。
而且即便把论文的两个设定叠在一起（G=32 + lr 5e-5），macro-F1 0.5403 仍够不到
SFT 的 0.6114；其 95% CI [0.4807, 0.5965] 与 SFT 的 [0.5496, 0.6602] **重叠**，
所以这一栏同样只能写「明显更低」，**不能写「显著下降」**。"""

NEW1 = """| 方向 | 配置变化 | macro-F1（边际） | 成对准确率检验（同为 252 行）|
| --- | --- | --- | --- |
| **降学习率** | G=8：5e-5 → 1e-5 | 0.5014 → 0.6089（**+0.108**）| +0.0754，`p` = 0.00132 → **显著** |
| **加组大小** | lr 5e-5：G=8 → G=32 | 0.5014 → 0.5403（**+0.039**）| +0.0278，`p` = 0.41010 → **不显著** |

**本节初稿的「组大小确实有效」过强了，这里更正。** 初稿的依据是两个边际
macro-F1 差值之比约 2.8 —— 那是**量级之比，不是显著性之比**，而 5.2.2 这一轮
被更正的正是同一类推理（用点估计的比值代替检验）。补上成对检验后两行分道扬镳：
降学习率显著，加组大小**与 G=8 无法区分**（95% CI [−0.0278, +0.0833] 含 0）。
方向仍然一致（点估计为正），但本设计给不出「有效」这个词所需的证据。

更正后的事实比初稿更干净：**在 lr 5e-5 下，G=32 相对两侧的对照都是 ±0.0278、
`p` 都是 0.41**（对 G=8 是 +0.0278，对 GRPO(G=4) 是 −0.0278，
见第 9 节 `paired_significance.json` 的 `group_size_paired` 与 `cross_algorithm_paired`）。
即**在 lr 5e-5 这一档上，算法与组大小两两都分不开，唯一分得开的是学习率** ——
这比「组大小有效但不足」更强，也更贴合数据。

即便把论文的两个设定叠在一起（G=32 + lr 5e-5），macro-F1 0.5403 仍够不到
SFT 的 0.6114；其 95% CI [0.4807, 0.5965] 与 SFT 的 [0.5496, 0.6602] **重叠**，
成对检验也**未达显著**（Δ准确率 −0.0556、`p` = 0.08695，CI 含 0），
所以这一栏只能写「明显更低」。"""

OLD2 = """**因此论文的原始设定（G=32 + lr 5e-5）是本项目测过的所有 RL 配置里较弱的组合**：
它同时继承了高学习率的退化与大组下的多样性下降 —— 组内奖励标准差在 4 个奖励里
有 3 个是四组最低。"""

NEW2 = """**因此论文的原始设定（G=32 + lr 5e-5）低于 SFT，且低得可以复现**，
但它并非本项目最弱的一次 RL：按 macro-F1 点估计，同学习率、更小分组的
GSPO(G=8) lr5e-5 是 0.5014，比它更低（两者相差 +0.039，成对检验下不显著，
见上表）。它同时继承了高学习率的退化与大组下的多样性下降 —— 组内奖励标准差
在 4 个奖励里有 3 个是四组最低 —— 但**「四组最低的标准差」与「最低的分数」
并不是同一件事，这里不再用前者给后者作保**。"""

for old, new, name in ((OLD1, NEW1, "5.2.5 magnitude block"),
                       (OLD2, NEW2, "5.2.5 weakest-run sentence")):
    n = s.count(old)
    if n != 1:
        sys.exit(f"FAILED: {name} appears {n} times, expected 1")
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("5.2.5: group-size claim downgraded to what the paired test supports")
print(f"lines: {s.count(chr(10)) + 1}")
