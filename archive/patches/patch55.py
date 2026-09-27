"""Make FINAL_REPORT.md carry the paired result, and fix three stale claims.

The generator is what writes the report, so fixing the report means fixing the
generator -- otherwise the next run silently restores the old text.

Three things in it are now false:

  * It says GRPO at lr 5e-5 was never run and that therefore "the algorithm
    collapsed" cannot be excluded. That cell has been filled (run A). The
    sentence does not merely age badly, it asserts a gap that no longer exists.
  * It counts four RL runs and says their differences all fall inside the noise.
    There are five, and the paired test says two of them are significantly below
    SFT -- "no significant gain" understates a significant loss.
  * It credits sequence-level normalization with cutting idle steps from 36.00%
    to 15.33%. That comparison also changes group size (G=8 vs G=4), so the
    attribution does not hold; 5.2.3 withdraws it.

And the significance verdict itself: the generator used two marginal 95%
intervals compared for overlap. Paired on the same 252 rows the answer differs,
so the paired table is emitted alongside the marginal rule rather than replacing
it silently -- a reader who saw the old rule should be able to find it and the
reason it was superseded.
"""
import sys
from pathlib import Path

p = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
s = p.read_text(encoding="utf-8")

OLD_A = """                line += ("，**与 SFT 不重叠 → 显著下降，不是噪声**"
                         if ci[1] < sft_ci[0] else "，与 SFT 重叠")
        out.append(line)"""

NEW_A = """                line += ("，边际区间**与 SFT 不重叠**"
                         if ci[1] < sft_ci[0] else "，**边际区间与 SFT 重叠**")
                line += ("（该规则只比较两个边际区间，对配对数据偏保守 —— "
                         "六次 run 回答的是同一批 252 行，结论以紧随其后的成对检验为准）")
        out.append(line)"""

OLD_B = """    out.append(
        "- **判定（按论文 §8.11 的门槛）**："""

NEW_B = """    # Paired test. The marginal-CI rule above is what this generator used to
    # rely on; every run answered the same 252 rows, so the paired structure is
    # where the power is and the marginal rule understates what is there. The
    # numbers come from the record, not from arithmetic done here.
    _ps_file = reports / "paired_significance.json"
    if _ps_file.is_file():
        ps = json.loads(_ps_file.read_text(encoding="utf-8"))
        _lbl = {"GRPO_G4_lr1e5": "GRPO(G=4) lr1e-5", "GRPO_G4_lr5e5": "GRPO(G=4) lr5e-5",
                "GSPO_G8_lr1e5": "GSPO(G=8) lr1e-5", "GSPO_G8_lr5e5": "GSPO(G=8) lr5e-5",
                "GSPO_G32_lr5e5": "GSPO(G=32) lr5e-5"}
        out.append("- **成对显著性检验**（同一批 252 行的准确率，按行 bootstrap + 精确 "
                   "McNemar；macro-F1 无逐样本分解，不适用该检验，仍按边际值阅读）：")
        for _k, _name in _lbl.items():
            c = ps["comparisons_vs_SFT"].get(_k)
            if not c:
                continue
            lo, hi = c["delta_ci95_rows"]
            _v = "**显著下降**" if c["excludes_zero"] else "与 SFT 无法区分"
            out.append(f"  - {_name} vs SFT：Δ准确率 {c['delta_vs_sft']:+.4f}，"
                       f"95% CI [{lo:+.4f}, {hi:+.4f}]，"
                       f"McNemar p = {c['mcnemar']['p_exact_two_sided']:.6g} → {_v}")
        _ctrl = ps.get("positive_control", {})
        out.append(f"  - 正对照 BASE vs SFT：Δ {_ctrl.get('accuracy_delta', 0):+.4f}，"
                   f"p = {_ctrl.get('p_exact_two_sided', float('nan')):.6g}"
                   f"（该检验若看不出这么大的差，脚本直接退出、不输出结论）")
        _x5 = ps.get("cross_algorithm_paired", {}).get("lr5e-5")
        if _x5:
            out.append(f"  - **论文的中心主张**（序列级归一化更耐高学习率）：lr 5e-5 下 "
                       f"GSPO(G=8) 比 GRPO(G=4) 低 "
                       f"{abs(_x5['accuracy_delta_gspo_minus_grpo']):.4f}，"
                       f"p = {_x5['mcnemar']['p_exact_two_sided']:.6g} —— **不构成差异**；"
                       f"且两者组大小不同（8 对 4），该对照无法把「IS 层级」与「组大小」"
                       f"分开。因此这条主张**既未被支持，也未被否定**。")
        _gs = ps.get("group_size_paired")
        if _gs:
            out.append(f"  - 组大小（GSPO lr 5e-5，G=8 → G=32）：Δ准确率 "
                       f"{_gs['accuracy_delta_g32_minus_g8']:+.4f}，"
                       f"p = {_gs['mcnemar']['p_exact_two_sided']:.6g} —— "
                       f"点估计为正但**未达显著**。即在 lr 5e-5 上，算法与组大小两两都"
                       f"分不开，**唯一分得开的是学习率**。")

    out.append(
        "- **判定（按论文 §8.11 的门槛）**："""

OLD_C = """    out.append(
        "- 需要说清的是，这**不等于**「GSPO 不如 SFT」或「GSPO 不如 GRPO」："
        "四个 RL run 在基准上的差异都落在噪声内，最明显的一项差异来自学习率"
        "（论文的 5e-5 在这套 QLoRA 配置下把模型推进了退化区，见 LIMITATIONS 5.2.2）。"
        "**该判断只覆盖 GSPO 内部两个学习率的对照**：GRPO 仅跑了 lr 1e-5，"
        "缺 lr 5e-5 这一格，因此无法排除「崩的是 GSPO 本身」这一解释；"
        "补齐后才是完整的算法 × 学习率 2×2 对照（见 provenance 的未决事项）。")"""

NEW_C = """    out.append(
        "- 需要说清的是，这**不等于**「GSPO 不如 SFT」或「GSPO 不如 GRPO」："
        "在 lr 1e-5 上 GSPO、GRPO 与 SFT 两两都落在噪声内；在 lr 5e-5 上"
        "**两个算法都显著低于 SFT**（见上表），而它们彼此之间仍分不开 —— "
        "**算法不是差异的来源**。完整的算法 × 学习率 2×2 已补齐（GRPO 的 lr 5e-5 "
        "对照格已跑完，见 LIMITATIONS 5.2.2），此前「无法排除崩的是 GSPO 本身」"
        "这一保留意见已由该对照格解除：同一个 5e-5 作用在 GRPO 上同样退化。")"""

OLD_D = """    out.append(
        "- **信号层面的收益是真实的**：把学习率对齐后，GSPO 的零优势步从 GRPO 的 "
        "36.00% 降到 15.30%（见上）。但该收益**没有转化为基准分数**。"
        "这是本次 RL 最值得记下的一条：优势信号的多寡与最终指标的高低，在这里是解耦的。")"""

NEW_D = """    out.append(
        "- **空转步的下降是观察到的，但归因不成立，且没有转化为分数**：lr 1e-5 下 "
        "GSPO 的零优势步是 15.33%（23/150），GRPO 是 36.00%（54/150）。"
        "但该对照**同时改了组大小**（G=8 对 G=4），本设计无法把「IS 层级」与"
        "「组大小」分开，所以这是**观察到的差异，不是归因给序列级归一化的证据**"
        "（见 LIMITATIONS 5.2.3）。无论归因如何，它都没有转化为基准分数 —— "
        "优势信号的多寡与最终指标的高低，在这里是解耦的。")"""

for old, new, name in ((OLD_A, NEW_A, "marginal-CI labelling"),
                       (OLD_B, NEW_B, "paired block"),
                       (OLD_C, NEW_C, "2x2 completeness"),
                       (OLD_D, NEW_D, "idle-step attribution")):
    n = s.count(old)
    if n != 1:
        sys.exit(f"FAILED: {name} appears {n} times, expected 1")
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("final_report.py: paired block added; three stale claims corrected")
print(f"lines: {s.count(chr(10)) + 1}")
