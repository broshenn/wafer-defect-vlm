"""Two corrections that follow from what 5.2.2 and 5.2.5 now say.

5.2.3's headline attributes the idle-step drop (36.00% -> 15.30%) to sequence-
level normalization. Learning rate is aligned in that comparison but group size
is not: GSPO is G=8 and GRPO is G=4, so IS level and group size move together.
That is the same confound item 9 of section 8 records, committed in the very
section that reports it. The attribution is withdrawn and what survives is
stated with its assumption named.

5.2.4 still counts three RL runs and says their differences "all fall inside
the noise". There are five now, and the paired test says the two lr 5e-5 runs
are significantly below SFT -- so "no significant gain" understates it: at the
high learning rate there is a significant LOSS. The verdict (keep SFT) does not
change, but the reason gets stronger.
"""
import sys
from pathlib import Path

p = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
s = p.read_text(encoding="utf-8")

OLD_A = """- **本次（正确）**：学习率对齐后，**GSPO lr1e-5 的空转步是 15.30%（23/150），
  相对 GRPO 的 36.00% 是一次实质下降。**

所以初稿的**方向**是对的，但当时比的是两个不同学习率的 run、数字又是误读的；
直到有了学习率对齐的对照，这个方向才第一次建立在干净证据上。"""

NEW_A = """- **本次（数字对了，归因仍不成立）**：学习率对齐后，GSPO lr1e-5 的空转步是
  15.30%（23/150），GRPO lr1e-5 是 36.00%（54/150）。

学习率对齐**消掉了学习率的混淆，但没有消掉组大小** —— GSPO 是 G=8、GRPO 是 G=4，
两者仍然同时变化。所以初稿写的「15.30% 是序列级归一化的功劳」**此处撤回**：
本设计里没有一次对照能把「IS 层级」与「组大小」分开（同 5.2.2 的中心主张，
也是第 8 节第 9 条记的那类混淆）。

能站住的是更弱的一条：**这个下降的方向，与 lr 5e-5 上实测到的组大小效应方向相反**。
lr 5e-5 下 G=4 是 32.67%、G=8 是 38.00%、G=32 是 42.00%，即**组越大空转步越多**；
若该方向在 lr 1e-5 上同样成立，G=8 本该比 G=4 更多空转步，而实测反而少 20.7 个百分点。
这**倾向于**说明下降不来自组大小，但它依赖「组大小效应的方向不随学习率翻转」
这个未检验的假设 —— 是**倾向性证据，不是识别**。真正能识别的那条对照
（G=32 + lr 1e-5：同算法、同学习率、只改组大小）在队列 41 里跑。"""

OLD_B = """5.1 里「问题出在组大小」的推断，已由 G=32 + lr 5e-5 实测检验，结论见 5.2.5：
**组大小是次要变量，学习率是主导变量。**"""

NEW_B = """5.1 里「问题出在组大小」的推断已由 G=32 实测检验，结论见 5.2.5：
**在 lr 5e-5 下，G=32 相对 G=8、相对 GRPO(G=4) 在成对检验下都分不开
（两侧都是 `p` = 0.41），所以「组大小」不是本设计能识别出的变量；
能识别出的只有学习率。** 这比初稿的「组大小是次要变量」更弱也更准确 ——
「次要」暗示它可测而量小，实际是这个量在本设计里测不出来。"""

OLD_C = """需要说清的是，这**不等于**「GSPO 不如 SFT」或「GSPO 不如 GRPO」：
三个 RL run 在基准上的差异都落在噪声内，唯一明显的差异来自学习率而非算法。
诚实的表述是：**在本模型规模与本次预算下，RL 阶段没有产生可测的基准增益；
但 GSPO 的序列级归一化确实把零优势步砍掉了约六成（36.00% → 15.30%），
这个信号层面的收益是真实的、可复现的，只是没有传导到基准分数。**

四个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 GRPO（0.6197 对 0.6114），
差异远在噪声内；两个 lr 5e-5 的 GSPO run 都低于 SFT，其中论文的原始设定
G=32 + lr 5e-5 为 0.5403，是较弱的一个（见 5.2.5）。"""

NEW_C = """需要说清的是，这**不等于**「GSPO 不如 SFT」或「GSPO 不如 GRPO」：
在 lr 1e-5 上 GSPO、GRPO 与 SFT 两两都落在噪声内；在 lr 5e-5 上两个算法都
**显著低于** SFT（5.2.2），但它们彼此之间仍分不开。**算法不是差异的来源。**

诚实的表述是：**在本模型规模与本次预算下，RL 阶段没有产生可测的基准增益，
而且在高学习率档上产生了可测的基准损失**；空转步在 lr 1e-5 的 GSPO 上确实
大幅少于 GRPO（36.00% → 15.30%），但那个对照同时改了组大小，所以它是
**观察到的差异，不是归因给序列级归一化的证据**（5.2.3）。

五个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 GRPO lr1e-5（0.6197 对 0.6114），
成对检验下与 SFT 完全无法区分（Δ准确率 0.0000、`p` = 1.00）；
GRPO lr5e-5 与两个 lr 5e-5 的 GSPO run 都低于 SFT，
其中论文的原始设定 G=32 + lr 5e-5 为 0.5403（见 5.2.5）。"""

for old, new, name in ((OLD_A, NEW_A, "5.2.3 attribution"),
                       (OLD_B, NEW_B, "5.2.3 closing summary"),
                       (OLD_C, NEW_C, "5.2.4 counts and significance")):
    n = s.count(old)
    if n != 1:
        sys.exit(f"FAILED: {name} appears {n} times, expected 1")
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("5.2.3 attribution withdrawn; 5.2.4 counts and significance corrected")
print(f"lines: {s.count(chr(10)) + 1}")
