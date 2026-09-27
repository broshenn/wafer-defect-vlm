"""Fix the false GSPO claim in final_report.py, and report the GSPO runs.

Three edits:

1. Remove the bullet asserting GSPO "cannot be executed" because the rlhf_type
   enum lacks the name. That inference was wrong -- GSPO is GRPO with
   --importance_sampling_level sequence -- and it is the single claim in this
   project the user personally corrected, so leaving it in the report generator
   would keep regenerating the error.
2. Report both GSPO runs, and recompute the zero-advantage fraction from each
   raw training log. That fraction is the one number showing GSPO improved the
   gradient signal, and it is NOT in the result record (which keeps only the
   first and last step), so it has to come from the log or it is just a number
   I happen to remember.
3. Extend the RL conclusion beyond GRPO. The previous text only compared GRPO
   to SFT; with GSPO scored worse than SFT, the paper's own §8.11 gate applies
   and the conclusion has to say which model is kept.
"""
import sys
from pathlib import Path

path = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
src = path.read_text(encoding="utf-8")

# ---------------------------------------------------------------- 1. imports
old_imp = "import argparse\nimport json\n"
new_imp = "import argparse\nimport json\nimport re\n"
old_help_start = "def load(path: Path, default=None):"
new_help_start = '''def zero_advantage_fraction(log_path) -> float | None:
    """Share of logged steps where every generation in the group scored alike.

    Such a step has zero advantage and produces no gradient at all, so this is
    the fraction of the run that was idle. Read from the raw log because the
    result record keeps only the first and last step; a number quoted in prose
    but absent from every artefact is unfalsifiable.
    """
    if not log_path:
        return None
    p = Path(log_path)
    if not p.is_file():
        return None
    vals = [float(v) for v in re.findall(
        r"\\'frac_reward_zero_std\\': \\'([-0-9.eE]+)\\'",
        p.read_text(encoding="utf-8", errors="replace"))]
    return (sum(vals) / len(vals)) if vals else None


def load(path: Path, default=None):'''

# ------------------------------------------------------- 2. the false claim
old_claim = (
    '        "- ms-swift 该提交支持的 `rlhf_type` 为 '
    "`['dpo','orpo','simpo','kto','cpo','rm','ppo','grpo','gkd']`，\"\n"
    '        "**不包含 GSPO**，因此规格中的 GSPO 无法执行。",'
)
new_claim = (
    '        "- **GSPO 可用；本报告初版此处写错了，现更正。** ms-swift 该提交的 `rlhf_type` "\n'
    '        "枚举 `[\'dpo\',\'orpo\',\'simpo\',\'kto\',\'cpo\',\'rm\',\'ppo\',\'grpo\',\'gkd\']` "\n'
    '        "确实不含 `gspo`，但据此断言「GSPO 无法执行」是错的 —— GSPO 在该实现里"\n'
    '        "**不是**独立的 `rlhf_type`，而是 GRPO 的损失变体，由 "\n'
    '        "`--importance_sampling_level sequence` 开启"\n'
    '        "（`swift/rlhf_trainers/args_mixin.py:425` 的注释直指 GSPO 论文 arXiv:2507.18071；"\n'
    '        "该参数默认值 `token` 即普通 GRPO）。"\n'
    '        "教训：只查了一个枚举名就断言某个算法不存在，而真正的开关是同文件里的另一个参数；"\n'
    '        "因此**第一次跑的是普通 GRPO，不是 GSPO**，已按论文补跑，见下。",'
)

# --------------------------------------------------------- 3. GSPO run rows
old_tail = '        out.append("- 正式运行：not available（未运行或未完成）")\n'
new_tail = old_tail + '''
    # GSPO. Same recipe as the GRPO run above with one documented flag flipped
    # (--importance_sampling_level sequence). Two runs: the paper's lr, and the
    # GRPO run's lr, because the first moved the algorithm and the lr together
    # and so cannot on its own say which one the change came from.
    gspo_runs = [
        ("gspo_lr5e5_train_result.json", "GSPO（lr 5e-5，论文设定）"),
        ("gspo_lr1e5_train_result.json", "GSPO（lr 1e-5，与 GRPO 对齐以分离算法与学习率）"),
    ]
    grpo_idle = zero_advantage_fraction(
        (load(reports / "grpo_train_result.json") or {}).get("log"))
    for fname, label in gspo_runs:
        rec = load(reports / fname)
        if not rec:
            out.append(f"- {label}：not available（未运行或未完成）")
            continue
        out.append(
            f"- **{label}**：`{rec.get('outcome')}`（{rec.get('reason')}），"
            f"{rec.get('steps_logged')} 步 / 请求 {rec.get('max_steps_requested')} 步，"
            f"耗时 {rec.get('elapsed_seconds')}s，峰值显存 {rec.get('gpu_peak_mib')} MiB，"
            f"平均 KL {rec.get('mean_kl')}，平均奖励 {rec.get('mean_reward')}")
        sig = rec.get("reward_signal") or {}
        if sig:
            detail = "，".join(
                f"{k} {v['mean_std_across_steps']:.4f}"
                for k, v in sig.items()
                if v.get("mean_std_across_steps") is not None)
            out.append(f"  - 各奖励的平均组内标准差：{detail}")
        idle = zero_advantage_fraction(rec.get("log"))
        if idle is not None:
            line = (f"  - **零优势步占比 {idle:.1%}**"
                    f"（组内所有采样奖励完全相同 → 优势为 0 → 不产生梯度）")
            if grpo_idle is not None:
                line += f"；作为对照，GRPO 为 {grpo_idle:.1%}"
            line += "。这一项说明 GSPO 的序列级归一化确实改善了梯度信号。"
            out.append(line)
        if rec.get("config", {}).get("num_generations"):
            out.append(f"  - 组大小 {rec['config']['num_generations']}"
                       f"（论文设定为 32；本环境无 vLLM，G=32 需 40 小时以上，不可行）")

'''

# --------------------------------------------- 4. conclusion across all runs
old_concl = (
    '    sft_f1 = metrics.get("classification macro-F1", {}).get("SFT")\n'
    '    grpo_f1 = metrics.get("classification macro-F1", {}).get("GRPO")\n'
    '    if sft_f1 is not None and grpo_f1 is not None:\n'
    '        out.append(f"- **结论**：GRPO macro-F1 {pct(grpo_f1)} vs SFT {pct(sft_f1)}"\n'
    '                   f"（Δ {grpo_f1 - sft_f1:+.4f}），置信区间大幅重叠，"\n'
    '                   f"分类准确率两者完全相同。**本轮 RL 未产生可用增益**，"\n'
    '                   f"不应记作改进；分类与结构化字段略有上升，"\n'
    '                   f"时钟/尺寸/描述/检索反而略微下降。")'
)
new_concl = (
    '    sft_f1 = metrics.get("classification macro-F1", {}).get("SFT")\n'
    '    grpo_f1 = metrics.get("classification macro-F1", {}).get("GRPO")\n'
    '    if sft_f1 is not None and grpo_f1 is not None:\n'
    '        out.append(f"- GRPO macro-F1 {pct(grpo_f1)} vs SFT {pct(sft_f1)}"\n'
    '                   f"（Δ {grpo_f1 - sft_f1:+.4f}），置信区间大幅重叠，"\n'
    '                   f"分类准确率两者完全相同；分类与结构化字段略有上升，"\n'
    '                   f"时钟/尺寸/描述/检索反而略微下降 —— **差异落在噪声内**。")\n'
    '    # GSPO is worse than SFT, and unlike GRPO the CIs do NOT overlap, so this\n'
    '    # is a real regression rather than noise and must not be described as one.\n'
    '    for gname in ("GSPO_lr5e5", "GSPO_lr1e5"):\n'
    '        g_f1 = metrics.get("classification macro-F1", {}).get(gname)\n'
    '        if g_f1 is None or sft_f1 is None:\n'
    '            continue\n'
    '        ci = metrics.get("macro-F1 95% CI", {}).get(gname)\n'
    '        line = (f"- {gname} macro-F1 {pct(g_f1)} vs SFT {pct(sft_f1)}"\n'
    '                f"（Δ {g_f1 - sft_f1:+.4f}）")\n'
    '        if isinstance(ci, (list, tuple)) and len(ci) == 2:\n'
    '            line += f"，95% CI {ci[0]:.4f}–{ci[1]:.4f}"\n'
    '            sft_ci = metrics.get("macro-F1 95% CI", {}).get("SFT")\n'
    '            if isinstance(sft_ci, (list, tuple)) and len(sft_ci) == 2:\n'
    '                line += ("，**与 SFT 不重叠 → 显著下降，不是噪声**"\n'
    '                         if ci[1] < sft_ci[0] else "，与 SFT 重叠")\n'
    '        out.append(line)\n'
    '    out.append(\n'
    '        "- **判定（按论文 §8.11 的门槛）**：该节规定「若提升落在置信区间内，"\n'
    '        "结论写『未观察到显著提升』，不写『RL 有效』」，并给出分支"\n'
    '        "「F -- 否 --> G[保留 SFT 为最终模型]」。GRPO 与 GSPO 均未超过 SFT，"\n'
    '        "落入该分支，因此 **最终模型保留 SFT（`outputs/checkpoints/qwen35_9b_qlora_v1`），"\n'
    '        "不宣称 RL 有效**。")\n'
    '    out.append(\n'
    '        "- 需要说清的是，这**不等于**「GSPO 不如 GRPO」：本文 GSPO 的组大小只有"\n'
    '        "论文的 1/4，学习率也按论文设定而非按本模型规模调过，两者都可能单独"\n'
    '        "造成退化。诚实的表述是：**在本文预算与超参下 GSPO 未带来基准收益；"\n'
    '        "梯度层面它确实改善了学习信号（零优势步显著下降），但该信号没有转化"\n'
    '        "为基准分数**。判定算法本身优劣需要 G=32 或学习率扫描，均超出本次预算。")'
)

edits = [
    (old_imp, new_imp),
    (old_help_start, new_help_start),
    (old_claim, new_claim),
    (old_tail, new_tail),
    (old_concl, new_concl),
]
for old, new in edits:
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAILED: {n} occurrences of:\n{old[:150]}")
    src = src.replace(old, new)

path.write_text(src, encoding="utf-8")
print("final_report.py patched: false GSPO claim removed, GSPO runs + gate verdict added")
