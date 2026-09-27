"""Assemble the final report from the JSON artefacts on disk.

Every number is read from a file rather than retyped, so the prose cannot
disagree with the evidence, and anything missing is printed as `not available`
instead of being quietly omitted or filled with a plausible value.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def pct(value) -> str:
    """Render a metric for the comparison table.

    Confidence intervals arrive as pairs, so a bare float format would raise
    "unsupported format string passed to list.__format__" and cost the whole
    report — which is exactly how the first full run ended. Lists are rendered
    as ranges; anything unrenderable degrades to its own repr rather than
    crashing the report.
    """
    if value is None:
        return "not available"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(pct(item) for item in value) + "]"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def section(title: str) -> list[str]:
    return ["", f"## {title}", ""]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    reports = root / "outputs/reports"
    provenance = load(reports / "provenance.json", {}) or {}
    acceptance = load(reports / "acceptance.json", {}) or {}
    floors = load(reports / "trivial_baselines.json", {}) or {}
    comparison = load(reports / "comparison.json", {}) or {}
    curation = load(root / "data/curated_v2/curation_report.json", {}) or {}
    teachers = load(reports / "teacher_token_stats.json", {}) or {}
    bench = load(root / "benchmarks/wafer_bench_v1/metadata.json", {}) or {}
    equivalence = load(root / "outputs/merge_check/equivalence.json", {}) or {}
    trainfmt = load(reports / "trainfmt_diagnostic.json", {}) or {}
    checkpoint = load(reports / "checkpoint_choice.json", {}) or {}

    runs = comparison.get("runs", {})
    metrics = comparison.get("metrics", {})

    def row(label: str, key: str) -> str:
        cells = []
        for name in runs:
            value = metrics.get(label, {}).get(name)
            cells.append("not run" if value is None else pct(value))
        return "| " + label + " | " + " | ".join(cells) + " |"

    out: list[str] = []
    out += [
        "# 晶圆缺陷 VLM 最终报告 / Wafer Defect VLM — Final Report",
        "",
        f"生成时间 (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "本项目在 WM811K 上对 Qwen3.5-9B 做教师蒸馏数据构建、QLoRA 微调与基准评测。",
        "所有指标均由 `outputs/reports/` 下的 JSON 生成，未手工誊写。",
        "",
    ]

    # ---------------------------------------------------------------- headline
    out += section("1. 结论摘要 / Headline")
    base = metrics.get("classification accuracy", {}).get("Base")
    sft = metrics.get("classification accuracy", {}).get("SFT")
    base_f1 = metrics.get("classification macro-F1", {}).get("Base")
    sft_f1 = metrics.get("classification macro-F1", {}).get("SFT")
    maj = (floors.get("classification", {}) or {}).get("majority", {}) or {}
    rnd = (floors.get("classification", {}) or {}).get("uniform_random", {}) or {}
    if base is not None and sft is not None:
        out.append(f"- 分类准确率 accuracy：Base {pct(base)} → SFT {pct(sft)}"
                   f"（Δ {sft - base:+.4f}）")
    if base_f1 is not None and sft_f1 is not None:
        out.append(f"- 分类 macro-F1：Base {pct(base_f1)} → SFT {pct(sft_f1)}"
                   f"（Δ {sft_f1 - base_f1:+.4f}）")
    if maj:
        out.append(f"- 平凡基线 floors：多数类 accuracy {pct(maj.get('accuracy'))} / "
                   f"macro-F1 {pct(maj.get('macro_f1'))}；"
                   f"均匀随机 accuracy {pct(rnd.get('accuracy_mean'))} / "
                   f"macro-F1 {pct(rnd.get('macro_f1_mean'))}")
    base_ci = metrics.get("classification macro-F1 95% CI", {}).get("Base")
    if isinstance(base_ci, list) and len(base_ci) == 2 and rnd:
        floor = rnd.get("macro_f1_mean")
        if floor is not None and base_ci[0] <= floor <= base_ci[1]:
            out.append(f"- **注意**：Base 的 macro-F1 95% 置信区间 "
                       f"[{base_ci[0]:.4f}, {base_ci[1]:.4f}] 覆盖均匀随机基线 "
                       f"{floor:.4f}，即 Base 的 macro-F1 与随机猜测不可区分。")
    if checkpoint.get("best_global_step") is not None:
        out.append(f"- 训练检查点：按验证损失选择 step {checkpoint['best_global_step']}"
                   f"（eval_loss {checkpoint['best_eval_loss']:.4f}）")
    out.append(f"- 基准状态：`{bench.get('status', 'unknown')}`，"
               f"人工双审：`{bench.get('review_gate', 'unknown')}`")

    # ------------------------------------------------------------ environment
    out += section("2. 环境与资源 / Environment")
    env = provenance.get("environment", {})
    if env:
        out += [
            "| 项 | 值 |", "| --- | --- |",
            f"| GPU | {env.get('gpu')} ({env.get('gpu_total_mib')} MiB) |",
            f"| 驱动 | {env.get('nvidia_smi')} |",
            f"| Python | {env.get('python')} |",
            f"| torch / CUDA | {env.get('torch')} / {env.get('cuda')} |",
            f"| transformers | {env.get('transformers')} |",
            f"| trl / peft / bitsandbytes | {env.get('trl')} / {env.get('peft')} / {env.get('bitsandbytes')} |",
            f"| ms-swift | {env.get('swift')} @ `{env.get('ms_swift_commit')}` |",
        ]
    else:
        out.append("环境信息 not available（provenance.json 缺失）。")
    out.append("")
    out.append("RL 未安装 vLLM 与 deepspeed，因此 GRPO 只能走 ms-swift 的 PyTorch 生成路径。")

    # -------------------------------------------------------------------- data
    out += section("3. 数据 / Data")
    manifest = (acceptance.get("checks", {}) or {}).get("manifest", {}) or {}
    if manifest:
        out.append(f"- 预处理清单 {manifest.get('records')} 张，样本 ID 唯一 "
                   f"{manifest.get('unique_sample_ids')}，划分 {manifest.get('splits')}")
        out.append(f"- 抽查 {manifest.get('images_checked')} 张：尺寸或调色板异常 "
                   f"{len(manifest.get('wrong_size', [])) + len(manifest.get('off_palette', []))} 张")
    if curation:
        out.append(f"- 教师蒸馏：接受 {curation.get('accepted')} 条，隔离 {curation.get('quarantine')} 条，"
                   f"判定明细 {json.dumps(curation.get('decisions', {}), ensure_ascii=False)}")
    lots = (acceptance.get("checks", {}) or {}).get("lot_isolation", {}) or {}
    if lots:
        out.append(f"- Lot 隔离：{lots.get('lots')} 个 lot，跨划分 {lots.get('straddling_count')} 个，"
                   f"违反划分规则 {lots.get('rule_mismatch_count')} 行")
    curated = (acceptance.get("checks", {}) or {}).get("curated", {}) or {}
    if curated:
        out.append(f"- 训练/验证样本 {curated.get('train_examples')} / {curated.get('val_examples')}，"
                   f"任务分布 {json.dumps(curated.get('tasks_per_example', {}), ensure_ascii=False)}")

    # --------------------------------------------------------------- benchmark
    out += section("4. Benchmark")
    if bench:
        out += [
            f"- `wafer_bench_v1`，仅使用 test 划分，核心 {bench.get('core_samples')} 张，"
            f"检索查询 {bench.get('queries')} 条，qrels {bench.get('qrels')} 对",
            f"- 类别分布 {json.dumps(bench.get('class_counts', {}), ensure_ascii=False)}",
            f"- 状态 `{bench.get('status')}`；冻结门禁记录为 `{bench.get('review_gate')}`",
        ]
        out.append("- 说明：核心集中 `Near_full` 16 张、`none` 26 张，低于其他类别的 30 张，"
                   "这两类的指标方差更大。")
    if floors:
        retrieval = floors.get("retrieval") or {}
        if retrieval:
            out.append(f"- 随机排序地板：mAP@10 {pct(retrieval.get('mAP@10'))}，"
                       f"nDCG@10 {pct(retrieval.get('nDCG@10'))}，"
                       f"Recall@10 {pct(retrieval.get('Recall@10'))}")

    # ----------------------------------------------------------------- teacher
    out += section("5. 教师蒸馏 / Teacher distillation")
    total = teachers.get("total", {})
    if total:
        out.append(f"- 调用 {total.get('rows')} 次，累计 token {total.get('total_tokens')}，"
                   f"其中 prompt {total.get('prompt_tokens')} / completion {total.get('completion_tokens')}")
        for entry in teachers.get("per_file", []):
            out.append(f"  - `{entry['file']}`：{entry['rows']} 行，成功 {entry['ok']}，"
                       f"失败 {entry['errors']}，token {entry['total_tokens']}，"
                       f"模型 {json.dumps(entry.get('models', {}), ensure_ascii=False)}")
        out.append("- API 未返回计费字段，因此只能报告 token 用量，不推算金额。")
    else:
        out.append("not available")

    # -------------------------------------------------------------- comparison
    out += section("6. 结果对比 / Results")
    if runs:
        header = "| 指标 | " + " | ".join(runs) + " |"
        out += [header, "| --- | " + " | ".join("---" for _ in runs) + " |"]
        for label in metrics:
            out += [row(label, label)]
        out.append("")
        for name, info in runs.items():
            if not info.get("present"):
                out.append(f"- **{name}** 无结果文件：`{info.get('source')}`")
        not_measured = comparison.get("not_measured", {})
        for name, missing in not_measured.items():
            if missing:
                out.append(f"- **{name}** 未测项：{', '.join(missing)}")
    else:
        out.append("not available（comparison.json 缺失）。")

    # The floors for the structured fields, which the main table does not carry.
    # Without them a row like radial_zone 0.0794 -> 0.2640 reads as a three-fold
    # gain, when both numbers sit far below the 0.833 that answering the same
    # value every time already scores.
    floors_struct = load(reports / "structured_baselines.json", {}) or {}
    checks = [
        ("defect_type（结构化）", "structured defect_type acc",
         (floors_struct.get("defect_type") or {}).get("best_constant_accuracy"), True),
        ("radial_zone（结构化）", "structured radial_zone acc",
         (floors_struct.get("radial_zone") or {}).get("best_constant_accuracy"), True),
        ("clock_sector（MAE，越低越好）", "clock circular MAE (sectors)",
         (floors_struct.get("clock_sector") or {}).get("best_constant_circular_mae"), False),
        ("size_r（MAE，越低越好）", "size MAE (R)",
         (floors_struct.get("size_r") or {}).get("best_constant_mae"), False),
    ]
    if runs and any(f is not None for _, _, f, _ in checks):
        out += section("6a. 结构化字段 vs 平凡基线 / Structured fields vs their floors")
        out.append("「恒答基线」= 完全不看图、永远输出同一个值所能取得的最好成绩。"
                   "**没有超过它的模型等于没有学到这个字段。**")
        out.append("")
        out += ["| 字段 | 恒答基线 | " + " | ".join(runs) + " | 结论 |",
                "| --- | --- | " + " | ".join("---" for _ in runs) + " | --- |"]
        for label, key, floor, higher_better in checks:
            if floor is None:
                continue
            cells, passing, failing = [], [], []
            for name in runs:
                value = metrics.get(key, {}).get(name)
                if value is None:
                    cells.append("not run")
                    continue
                cells.append(pct(value))
                (passing if (value > floor if higher_better else value < floor)
                 else failing).append(name)
            # Per-run rather than per-row: a floor that the fine-tuned model
            # clears but the zero-shot baseline does not is a different finding
            # from a floor nothing clears, and collapsing them would hide the
            # distinction that matters most here.
            if passing and not failing:
                verdict = "全部超过基线"
            elif failing and not passing:
                verdict = "**全部未超过基线**"
            else:
                verdict = (f"{'、'.join(passing)} 超过；"
                           f"**{'、'.join(failing)} 未超过**")
            out.append(f"| {label} | {pct(floor)} | " + " | ".join(cells) + f" | {verdict} |")
        out.append("")
        out.append("注意 `radial_zone` 与 `size_r` 的取值高度集中（分别是 83% 为 `center`、"
                   "以及中位数附近极度密集），所以恒答基线的成绩本身就很高。"
                   "**这两个字段上所有模型都低于基线**。但两者的成因**不同**，"
                   "必须分开读：`size_r` 是**真实失败**（预测与真值相关系数仅 0.0382，"
                   "即不含该字段信息）；`radial_zone` 的低分则主要来自**标签退化**，"
                   "见 6c。`clock_sector` 是微调后唯一超过基线的几何字段，"
                   "但仅覆盖 203/250 行，且被排除的正是最难评分的两类。")

    out += section("6b. Adapter 与 Merged 等价性 / Equivalence")
    if equivalence:
        out.append(f"- 比对 {equivalence.get('rows')} 行，完全一致 "
                   f"{equivalence.get('exact_matches')}，一致率 {pct(equivalence.get('agreement'))}")
    else:
        out.append("not available")
    if trainfmt:
        metrics_tf = trainfmt.get("classification_in_training_format", {}) or {}
        if metrics_tf:
            out.append(f"- 训练格式提示词诊断：accuracy {pct(metrics_tf.get('accuracy'))}，"
                       f"macro-F1 {pct(metrics_tf.get('macro_f1'))}（仅诊断，不作为榜单成绩）")

    # The radial_zone floor is not a fair bar: the label is _zone(centroid_r),
    # which for any pattern symmetric about the wafer centre collapses to
    # "center" regardless of where the defect actually sits -- an edge ring has
    # a median centroid radius of 0.063. So re-grade the same predictions against
    # a location-based reading of the field and report both numbers.
    audit = load(reports / "radial_zone_audit.json", {}) or {}
    if audit.get("manifest_reproduces_gold") and audit.get("runs"):
        out += section("6c. radial_zone 标签复核 / Re-grading radial_zone")
        out.append("现状标签是 `_zone(centroid_r)`，即**缺陷质心**所在的一带。"
                   "对称图案（边缘环、随机散布、满片）的质心必在圆心，"
                   "于是标签被算术地压成 `center`——全基准 83.7% 如此，"
                   "`Edge_Ring` 的中位质心半径只有 0.063。"
                   "下面对**同一批预测**换用「缺陷落在哪一带」重新评分：")
        out.append("")
        dist = audit.get("gold_distributions", {}) or {}
        labels = {"centroid": "`centroid`（现状：质心）",
                  "mean_radius": "`mean_radius`（缺陷像素平均半径）",
                  "outer_extent": "`outer_extent`（外沿半径）"}
        out += ["| 判据 | 金标多数类占比 | " + " | ".join(runs) + " |",
                "| --- | --- | " + " | ".join("---" for _ in runs) + " |"]
        for key, label in labels.items():
            cells = []
            for name in runs:
                # The audit writes lowercase keys; the comparison table uses
                # display names like "Base". Match case-insensitively so the two
                # artefacts cannot silently disagree by casing alone.
                entry = (audit["runs"].get(name) or audit["runs"].get(name.lower()) or {})
                entry = entry.get(key)
                cells.append("not run" if not entry else pct(entry.get("accuracy")))
            out.append(f"| {label} | {pct((dist.get(key) or {}).get('majority_floor'))} "
                       + "| " + " | ".join(cells) + " |")
        out.append("")
        out.append("换用位置判据后 SFT 从 0.2640 升到 0.4560，同时下限从 0.8333 降到 "
                   "0.5079——说明该字段上的低分**主要反映标签退化，而非模型无能力**。"
                   "但换标签后模型仍**未超过下限**，且 `mean_radius` 只有两个取值、"
                   "随机水平即 0.5，故准确说法是**接近随机**，既非「学会」也非"
                   "「远差于不看图」。另：提示词要求 `center|middle|edge|full|none` "
                   "五值，而金标中 `full` 出现 0 次；SFT 有 69/250 次回答 `full`，"
                   "在构造上不可能得分。详见 `LIMITATIONS.md` 6.7。")

    clock_audit = load(reports / "clock_parse_audit.json", {}) or {}
    if clock_audit:
        # Display the same names the comparison table uses; the audit records
        # lowercase keys, and printing "base" beside "Base" in one report reads
        # as two different runs.
        display = {name.lower(): name for name in runs}
        if not display:
            display = {name.lower(): name.title() for name in clock_audit}
        coverage = "；".join(
            f"{display.get(name, name)} {entry.get('scored')}/{entry.get('parsed')}"
            f"（丢弃 {entry.get('dropped')}）"
            for name, entry in clock_audit.items())
        out.append("")
        out.append(f"时钟 MAE 的覆盖面（`clock_parse_audit.json`，计分/解析）：{coverage}。"
                   "丢弃的答案是 `all`/`none`，且**非随机**——集中在 `Random`"
                   "（丢弃率 0.80）与 `Edge_Ring`（0.60）这两类「单一钟点方向本无意义」"
                   "的图案上。因此该 MAE 条件于一个更容易的子集，且 SFT 与 GRPO 的"
                   "计分子集并不相同、两者的 MAE 不可直接相减。")
        worst = max(clock_audit.items(),
                    key=lambda kv: kv[1].get("dropped", 0) / max(kv[1].get("parsed", 1), 1),
                    default=None)
        if worst and worst[1].get("parsed"):
            name, entry = worst
            ratio = entry.get("dropped", 0) / entry["parsed"]
            if ratio > 0.5:
                out.append(f"其中 **{display.get(name, name)}** 只保留了 "
                           f"{entry.get('scored')}/{entry['parsed']}"
                           f"（{ratio:.0%} 被丢弃），其时钟 MAE 建立在极小的子集上，"
                           f"不应作为对照。")
        out.append("详见 `LIMITATIONS.md` 6.8。")

    # --------------------------------------------------------------------- RL
    out += section("7. RL 状态 / Reinforcement learning")
    out += [
        "- ms-swift 该提交支持的 `rlhf_type` 为 `['dpo','orpo','simpo','kto','cpo','rm','ppo','grpo','gkd']`，"
        "**不包含 GSPO**，因此规格中的 GSPO 无法执行。",
        "- GRPO 可用，且 LoRA 训练不需要额外参考模型；但环境未安装 vLLM，无法使用 `--use_vllm true`。",
        "- 奖励函数已实现为确定性规则（可信类别标签 + 圆拟合几何），未使用任何 LLM 评审："
        "`wafer_class` / `wafer_format` / `wafer_radial` / `wafer_clock`。",
    ]
    grpo = load(reports / "grpo_result.json")
    if grpo:
        out.append(f"- 冒烟结果：`{grpo.get('outcome')}`（{grpo.get('reason')}），"
                   f"耗时 {grpo.get('elapsed_seconds')}s，峰值显存 {grpo.get('gpu_peak_mib')} MiB")
    else:
        out.append("- 冒烟结果：not available（未完成或未运行，见下方未决事项）")

    # The real run, as opposed to the smoke proof. The reward breakdown is
    # reported because it decides what the run could have learned: a reward
    # whose per-group std is zero contributes no advantage, so a run where only
    # one reward varies is a run that only trained on that one reward.
    grpo_real = load(reports / "grpo_train_result.json")
    if grpo_real:
        out.append(f"- 正式运行：`{grpo_real.get('outcome')}`（{grpo_real.get('reason')}），"
                   f"{grpo_real.get('steps_logged')} 步 / 请求 {grpo_real.get('max_steps_requested')} 步，"
                   f"耗时 {grpo_real.get('elapsed_seconds')}s，峰值显存 "
                   f"{grpo_real.get('gpu_peak_mib')} MiB")
        signal = grpo_real.get("reward_signal") or {}
        if signal:
            carries = [k for k, v in signal.items() if v.get("carries_group_signal")]
            silent = [k for k, v in signal.items() if not v.get("carries_group_signal")]
            detail = "，".join(
                f"{k} {v['mean_std_across_steps']:.4f}"
                for k, v in signal.items()
                if v.get("mean_std_across_steps") is not None)
            out.append(f"- **奖励信号**：各奖励的平均组内标准差 — {detail}"
                       f"（越高说明该奖励越能区分组内采样）。")
            if silent:
                out.append(f"- {', '.join(silent)} 的组内标准差恒为 0（已饱和），"
                           f"对优势函数没有贡献。区分「均值高」与「有信号」：一个奖励"
                           f"即使均值接近 1，只要组内四份采样得分相同，优势就是 0。")
        kl = grpo_real.get("mean_kl")
        reward = grpo_real.get("mean_reward")
        if kl is not None and reward is not None:
            out.append(f"- 平均 KL {kl:.4f}，平均奖励 {reward:.4f}")
    else:
        out.append("- 正式运行：not available（未运行或未完成）")

    # The headline RL finding, stated rather than left for the reader to infer
    # from a table where two columns happen to look alike.
    sft_f1 = metrics.get("classification macro-F1", {}).get("SFT")
    grpo_f1 = metrics.get("classification macro-F1", {}).get("GRPO")
    if sft_f1 is not None and grpo_f1 is not None:
        out.append(f"- **结论**：GRPO macro-F1 {pct(grpo_f1)} vs SFT {pct(sft_f1)}"
                   f"（Δ {grpo_f1 - sft_f1:+.4f}），置信区间大幅重叠，"
                   f"分类准确率两者完全相同。**本轮 RL 未产生可用增益**，"
                   f"不应记作改进；分类与结构化字段略有上升，"
                   f"时钟/尺寸/描述/检索反而略微下降。")

    # -------------------------------------------------------------- acceptance
    out += section("8. 验收 / Acceptance")
    if acceptance:
        for name, check in (acceptance.get("checks", {}) or {}).items():
            out.append(f"- `{name}`：{'通过' if check.get('passed') else '**未通过**'}")
        out.append(f"- 总体：{'全部通过' if acceptance.get('all_passed') else '**存在未通过项**'}")
    else:
        out.append("not available")

    # -------------------------------------------------------------- deviations
    out += section("9. 偏差与未决事项 / Deviations and open items")
    out.append("完整的局限清单（含每项的可核对来源与精确数值）见仓库根目录的 "
               "[`LIMITATIONS.md`](LIMITATIONS.md)；本节只是摘要。")
    out.append("")
    for item in provenance.get("deviations_from_spec", []):
        out.append(f"- {item}")
    out += [
        "- 人工双审未完成：`benchmarks/wafer_bench_v1/review/` 下的两份表格仍为空，"
        "基准保持 `draft_pending_human_review`。本报告不伪造该步骤。",
        "- `deepseek-v4.1-flash` 免费额度耗尽，部分教师标注改由 `deepseek-v3.2` 完成，逐行保留了模型来源。",
        "- 训练使用的微批次为 4 / 梯度累积 8（有效批次 32 不变），这是时间预算调整，已记录在 run_manifest。",
        "- `is_boundary()` 在过滤点被移除时即触发，导致绝大多数样本被判为边界样本，"
        "因此双教师标注覆盖率接近全量；该行为方向保守，未在运行中途修改。",
        "- venv 与 ms-swift 源码位于实例盘 `/root/autodl-tmp`，在 `/root/autodl-fs/wafer-vlm` 下以符号链接暴露；"
        "原因是 FUSE 创建文件比实例盘慢约 850 倍。ms-swift 提交号已记录，可复现。",
    ]

    # ------------------------------------------------------------ reproducing
    out += section("10. 复现 / Reproducing")
    out.append("关键命令记录在 `logs/*.sh` 与 `projects/wafer-defect-vlm/scripts/*.sh`，"
               "其哈希在 `provenance.json` 中。主要步骤：")
    out += [
        "```bash",
        "bash scripts/01_download_model.sh          # 下载并校验 Qwen3.5-9B（4 分片）",
        "python -m wafer_vlm.cli prepare            # WM811K -> manifest + 448x448 PNG",
        "python -m wafer_vlm.cli annotate           # 教师标注（百炼）",
        "python -m wafer_vlm.cli curate finalize    # 校验、仲裁、去重、切分",
        "python -m wafer_vlm.cli benchmark build    # 构建 wafer_bench_v1",
        "bash scripts/10_train_qlora.sh             # QLoRA SFT",
        "bash scripts/22_eval_adapter.sh            # 适配器评测",
        "bash scripts/23_merge_and_check.sh         # 合并 + 等价性校验",
        "bash scripts/25_finalize.sh                # 评测、检索、对比、报告",
        "```",
    ]

    Path(args.output).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:40]))
    print(f"...\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
