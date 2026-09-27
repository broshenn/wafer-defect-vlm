"""Correct the GSPO gradient-signal claim in final_report.py.

patch32 wrote "...说明 GSPO 的序列级归一化确实改善了梯度信号", based on a
"zero-advantage 13.3%" figure that turned out to be a single logged row rather
than a mean. Re-measured over all 150 steps the share is 38.00% for GSPO and
36.00% for GRPO -- no improvement at all. The sentence is replaced by a
data-driven comparison so the text cannot assert a direction the numbers do not
show, and the function now returns step counts as well, because "57/150" is
checkable against the log in a way that "38.00%" alone is not.
"""
import sys
from pathlib import Path

path = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
src = path.read_text(encoding="utf-8")

old_fn = '''def zero_advantage_fraction(log_path) -> float | None:
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
'''
new_fn = '''def zero_advantage_stats(log_path) -> dict | None:
    """Steps where every generation in the group scored alike, so advantage was 0.

    Such a step produces no gradient at all, which makes this the share of the
    run that was idle. Read from the raw log rather than the result record: the
    record keeps only the first and last step, and quoting one of those as if it
    were the mean is exactly how this project once reported 13.3% for a run that
    actually idled 38.00% of its steps.

    The per-step value is bimodal -- 1.0 when the whole group scored alike, 0.0
    otherwise -- so the mean is the idle share and the count is meaningful too.
    """
    if not log_path:
        return None
    p = Path(log_path)
    if not p.is_file():
        return None
    vals = [float(v) for v in re.findall(
        r"\\'frac_reward_zero_std\\': \\'([-0-9.eE]+)\\'",
        p.read_text(encoding="utf-8", errors="replace"))]
    if not vals:
        return None
    return {"fraction": sum(vals) / len(vals),
            "steps_idle": sum(1 for v in vals if v >= 1.0),
            "steps_logged": len(vals)}
'''

old_call = '''    grpo_idle = zero_advantage_fraction(
        (load(reports / "grpo_train_result.json") or {}).get("log"))'''
new_call = '''    grpo_idle = zero_advantage_stats(
        (load(reports / "grpo_train_result.json") or {}).get("log"))'''

old_row = '''        idle = zero_advantage_fraction(rec.get("log"))
        if idle is not None:
            line = (f"  - **零优势步占比 {idle:.1%}**"
                    f"（组内所有采样奖励完全相同 → 优势为 0 → 不产生梯度）")
            if grpo_idle is not None:
                line += f"；作为对照，GRPO 为 {grpo_idle:.1%}"
            line += "。这一项说明 GSPO 的序列级归一化确实改善了梯度信号。"
            out.append(line)'''
new_row = '''        idle = zero_advantage_stats(rec.get("log"))
        if idle is not None:
            line = (f"  - **零优势步占比 {idle['fraction']:.2%}**"
                    f"（{idle['steps_idle']}/{idle['steps_logged']} 步："
                    f"组内所有采样奖励完全相同 → 优势为 0 → 不产生梯度）")
            if grpo_idle is not None:
                delta = idle["fraction"] - grpo_idle["fraction"]
                direction = "低于" if delta < 0 else ("高于" if delta > 0 else "等于")
                line += (f"；作为对照 GRPO 为 {grpo_idle['fraction']:.2%}"
                         f"（{grpo_idle['steps_idle']}/{grpo_idle['steps_logged']} 步），"
                         f"本 run {direction} GRPO {abs(delta):.2%}")
            out.append(line)'''

old_concl = '''        "造成退化。诚实的表述是：**在本文预算与超参下 GSPO 未带来基准收益；"
        "梯度层面它确实改善了学习信号（零优势步显著下降），但该信号没有转化"
        "为基准分数**。判定算法本身优劣需要 G=32 或学习率扫描，均超出本次预算。")'''
new_concl = '''        "造成退化。诚实的表述是：**在本文预算与超参下 GSPO 既未带来基准收益，"
        "也未在梯度层面显示优势**（零优势步占比见上，两者基本持平）——"
        "初版曾据一个被误读的单步数字声称 GSPO 改善了学习信号，已撤回。"
        "判定算法本身优劣需要 G=32 或学习率扫描，均超出本次预算。")'''

for old, new in ((old_fn, new_fn), (old_call, new_call),
                 (old_row, new_row), (old_concl, new_concl)):
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAILED: {n} occurrences of:\n{old[:160]}")
    src = src.replace(old, new)

path.write_text(src, encoding="utf-8")
print("final_report.py: gradient-signal claim replaced with a measured comparison")
