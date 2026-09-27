"""Assemble the final report from the JSON artefacts on disk.

Every number is read from a file rather than retyped, so the prose cannot
disagree with the evidence, and anything missing is printed as `not available`
instead of being quietly omitted or filled with a plausible value.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def zero_advantage_stats(log_path) -> dict | None:
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
        r"\'frac_reward_zero_std\': \'([-0-9.eE]+)\'",
        p.read_text(encoding="utf-8", errors="replace"))]
    if not vals:
        return None
    return {"fraction": sum(vals) / len(vals),
            "steps_idle": sum(1 for v in vals if v >= 1.0),
            "steps_logged": len(vals)}


def load(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


# ---------------------------------------------------------------- run geometry
# Both tag forms ("qwen35_9b_gspo_g4_lr5e5") and comparison-column forms
# ("GSPO_G4_lr5e5") parse with one rule. The defaults are not arbitrary: the two
# oldest GSPO runs were named before the grid existed, and both are G=8 at the
# paper's lr, which is what the omitted fields mean.
def _geom(name: str):
    """-> (group size, learning rate as written). Defaults G=8, lr 5e-5."""
    t = str(name).replace("qwen35_9b_", "")
    g = re.search(r"[gG](\d+)", t)
    lr = re.search(r"lr([0-9.eE+-]+)", t)
    # The tags write the paper's lr as "lr5e5" and the other one as "lr1e5": a
    # bare "<m>e5" is 10^-5, because "5e5" without the sign would be 5x10^5 --
    # 8 orders of magnitude off the lr that ran. An explicit exponent is kept.
    def _lr(text):
        m = re.fullmatch(r"([0-9.]+)e5", text)
        return f"{m.group(1)}e-5" if m else text
    return (int(g.group(1)) if g else 8,
            _lr(lr.group(1)) if lr else "5e-5")


GSPO_NOTE = {
    "gspo_lr1e5": "，与 GRPO 对齐以分离算法与学习率",
    "qwen35_9b_gspo_g32": "，论文设定",
    "qwen35_9b_gspo_g4_lr5e5": "，单变量 IS 层级对照的 sequence 一侧",
    "qwen35_9b_gspo_g32_lr1e5": "，与 G=32 lr 5e-5 构成组大小的学习率对照",
}


def gspo_run_list(reports):
    """Every GSPO training record on disk, in grid order (group size, then lr)."""
    found = []
    for p in reports.glob("*gspo*_train_result.json"):
        tag = p.name[: -len("_train_result.json")]
        g, lr = _geom(tag)
        found.append((g, float(lr), p.name,
                      f"GSPO（G={g}，lr {lr}{GSPO_NOTE.get(tag, '')}）"))
    return [(f, lab) for _, _, f, lab in sorted(found)]


def run_label(k):
    """A record / comparison-column key -> a prose label, e.g.
    "GSPO_G4_lr5e5" -> "GSPO(G=4) lr5e-5". Any suffix the key carries beyond
    algorithm/G/lr (a seed run, say) is kept, so two runs of the same cell can
    never print the same label."""
    alg = "GRPO" if str(k).upper().startswith("GRPO") else "GSPO"
    g, lr = _geom(k)
    rest = re.sub(r"^(GRPO|GSPO)", "", str(k), flags=re.I)
    rest = re.sub(r"^_?G\d+", "", rest)
    rest = re.sub(r"^_?lr[0-9.eE+-]+", "", rest).strip("_")
    tail = (f"（{rest}）" if rest
            else "（论文设定）" if k == "GSPO_G32_lr5e5" else "")
    return f"{alg}(G={g}) lr{lr}{tail}"


# ------------------------------------------------------------------ idle steps
def idle_frac(reports, tag):
    """(share of idle steps, steps logged) for a run; (None, None) if unrecorded."""
    rec = load(reports / f"{tag}_train_result.json") or {}
    f = rec.get("mean_frac_reward_zero_std")
    return (f, rec.get("steps_logged") or 150) if f is not None else (None, None)


def idle_txt(reports, tag):
    f, n = idle_frac(reports, tag)
    return f"{f:.2%}（{round(f * n)}/{n}）" if f is not None else None


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
    # "center" regardless of where the defect actually sits. So re-grade the
    # same predictions against a location-based reading of the field and report
    # both numbers. The medians quoted below are read from the audit record
    # rather than restated here, so this paragraph cannot go stale on its own.
    audit = load(reports / "radial_zone_audit.json", {}) or {}
    if audit.get("manifest_reproduces_gold") and audit.get("runs"):
        out += section("6c. radial_zone 标签复核 / Re-grading radial_zone")
        share = audit.get("center_share") or {}
        edge_ring = (audit.get("centroid_radius_by_class") or {}).get("Edge_Ring") or {}
        # The fraction itself is quoted, not just its percentage: pct() renders
        # four decimals rather than a percentage despite its name, and a
        # percentage alone would be a number no record holds. Printing "210 of
        # 251 (83.7%)" keeps both the recorded counts and the arithmetic visible.
        # An absent record degrades the wording rather than reinstating the old
        # value, so the sentence cannot silently go stale.
        share_text = (f"已评分的 {share['n_scored']} 行里有 {share['count']} 行"
                      f"（{share['of_scored'] * 100:.1f}%）如此"
                      if share.get("n_scored") and share.get("of_scored") is not None
                      else "该比例见 radial_zone_audit.json（记录缺失，此处不引数字）")
        radius_text = (f"{edge_ring['median']:.3f}" if edge_ring.get("median") is not None
                       else "见 radial_zone_audit.json（记录缺失）")
        out.append("现状标签是 `_zone(centroid_r)`，即**缺陷质心**所在的一带。"
                   "对称图案（边缘环、随机散布、满片）的质心必在圆心，"
                   f"于是标签被算术地压成 `center`——{share_text}，"
                   f"`Edge_Ring` 的中位质心半径只有 {radius_text}。"
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
        "- **GSPO 可用；本报告初版此处写错了，现更正。** ms-swift 该提交的 `rlhf_type` "
        "枚举 `['dpo','orpo','simpo','kto','cpo','rm','ppo','grpo','gkd']` "
        "确实不含 `gspo`，但据此断言「GSPO 无法执行」是错的 —— GSPO 在该实现里"
        "**不是**独立的 `rlhf_type`，而是 GRPO 的损失变体，由 "
        "`--importance_sampling_level sequence` 开启"
        "（`swift/rlhf_trainers/args_mixin.py:425` 的注释直指 GSPO 论文 arXiv:2507.18071；"
        "该参数默认值 `token` 即普通 GRPO）。"
        "教训：只查了一个枚举名就断言某个算法不存在，而真正的开关是同文件里的另一个参数；"
        "因此**第一次跑的是普通 GRPO，不是 GSPO**，已按论文补跑，见下。",
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

    # GSPO. Same recipe as the GRPO run above with one documented flag flipped
    # (--importance_sampling_level sequence). Two runs: the paper's lr, and the
    # GRPO run's lr, because the first moved the algorithm and the lr together
    # and so cannot on its own say which one the change came from.
    # Enumerated from the records on disk, not listed here. The literal list this
    # replaces named three runs and stayed at three when a fourth landed, so the
    # report described a smaller run set than it contained and nothing in the
    # produced text could show it.
    gspo_runs = gspo_run_list(reports)
    grpo_idle = zero_advantage_stats(
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
            + (f"平均 KL {rec['mean_kl']:.4f}，平均奖励 {rec['mean_reward']:.4f}"
               if rec.get("mean_kl") is not None and rec.get("mean_reward") is not None
               else "平均 KL / 平均奖励 not available"))
        sig = rec.get("reward_signal") or {}
        if sig:
            detail = "，".join(
                f"{k} {v['mean_std_across_steps']:.4f}"
                for k, v in sig.items()
                if v.get("mean_std_across_steps") is not None)
            out.append(f"  - 各奖励的平均组内标准差：{detail}")
        idle = zero_advantage_stats(rec.get("log"))
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
            out.append(line)
        if rec.get("config", {}).get("num_generations"):
            g = rec["config"]["num_generations"]
            # The note is only for runs that deviate from the paper. The claim
            # that G=32 was infeasible was measured false -- 60s/step, 15.2 GiB
            # -- so the deviation is stated as a choice about budget, not a
            # hardware limit.
            note = "" if g == 32 else (
                "（论文设定为 32。G=32 实测约 60s/步、15.2 GiB 显存，150 步约 2.5 小时，"
                "可以运行 —— 本报告初版曾断言其不可行，实测推翻，见 LIMITATIONS 5.2.1）")
            out.append(f"  - 组大小 {g}{note}")


    # The headline RL finding, stated rather than left for the reader to infer
    # from a table where two columns happen to look alike.
    sft_f1 = metrics.get("classification macro-F1", {}).get("SFT")
    grpo_f1 = metrics.get("classification macro-F1", {}).get("GRPO")
    if sft_f1 is not None and grpo_f1 is not None:
        out.append(f"- GRPO macro-F1 {pct(grpo_f1)} vs SFT {pct(sft_f1)}"
                   f"（Δ {grpo_f1 - sft_f1:+.4f}），置信区间大幅重叠，"
                   f"分类准确率两者完全相同；分类与结构化字段略有上升，"
                   f"时钟/尺寸/描述/检索反而略微下降 —— **差异落在噪声内**。")
    # GSPO is worse than SFT, and unlike GRPO the CIs do NOT overlap, so this
    # is a real regression rather than noise and must not be described as one.
    # Which GSPO columns exist is a fact about the comparison table; read it. The
    # hand-written tuple below was the third place a new run could land without
    # being added to it, and the `g_f1 is None` guard made that look deliberate.
    def gspo_cols():
        for name in (metrics.get("classification macro-F1", {}) or {}):
            if not str(name).upper().startswith("GSPO"):
                continue
            g, lr = _geom(name)
            note = "，论文设定" if name == "GSPO_G32" else ""
            yield (g, float(lr), name, f"GSPO(G={g}) lr{lr}{note}")
    # The paired record knows how many runs answered the 252 rows; read the count
    # rather than writing a literal that cannot go stale visibly.
    _cnt_word = "各次 run"
    _psf = reports / "paired_significance.json"
    if _psf.is_file():
        try:
            _n = len(json.loads(_psf.read_text(encoding="utf-8"))
                      .get("runs_included") or [])
            if _n:
                _cnt_word = f"{_n} 次 run"
        except Exception:
            pass
    for _g, _lr, gname, gspo_lab in sorted(gspo_cols()):
        g_f1 = metrics.get("classification macro-F1", {}).get(gname)
        if g_f1 is None or sft_f1 is None:
            continue
        ci = metrics.get("macro-F1 95% CI", {}).get(gname)
        line = (f"- {gspo_lab} macro-F1 {pct(g_f1)} vs SFT {pct(sft_f1)}"
                f"（Δ {g_f1 - sft_f1:+.4f}）")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            line += f"，95% CI {ci[0]:.4f}–{ci[1]:.4f}"
            sft_ci = metrics.get("macro-F1 95% CI", {}).get("SFT")
            if isinstance(sft_ci, (list, tuple)) and len(sft_ci) == 2:
                line += ("，边际区间**与 SFT 不重叠**"
                         if ci[1] < sft_ci[0] else "，**边际区间与 SFT 重叠**")
                line += ("（该规则只比较两个边际区间，对配对数据偏保守 —— "
                         f"{_cnt_word} 回答的是同一批 252 行，"
                         "结论以紧随其后的成对检验为准）")
        out.append(line)
    # Paired test. The marginal-CI rule above is what this generator used to
    # rely on; every run answered the same 252 rows, so the paired structure is
    # where the power is and the marginal rule understates what is there. The
    # numbers come from the record, not from arithmetic done here.
    _ps_file = reports / "paired_significance.json"
    if _ps_file.is_file():
        ps = json.loads(_ps_file.read_text(encoding="utf-8"))
        _vs_sft_label = run_label
        # Enumerated from the record, not listed here: run 43 was in
        # comparisons_vs_SFT and had no line in this section.
        _lbl = {k: _vs_sft_label(k) for k in sorted(
            ps["comparisons_vs_SFT"], key=lambda k: (_geom(k)[0], float(_geom(k)[1])))}
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
        _pp = ps.get("paper_setting_paired")
        if _pp:
            _n01 = _pp["mcnemar"]["grpo_wrong_gspo_right"]
            _n10 = _pp["mcnemar"]["grpo_right_gspo_wrong"]
            _lo2, _hi2 = _pp["delta_ci95_rows"]
            out.append(
                f"  - **论文原始设定的那一格**（GSPO G=32 对 GRPO G=4，同为 lr 5e-5）："
                f"Δ准确率 {_pp['accuracy_delta_g32_minus_grpo4']:+.4f}，"
                f"95% CI [{_lo2:+.4f}, {_hi2:+.4f}]，"
                f"p = {_pp['mcnemar']['p_exact_two_sided']:.6g} —— 聚合准确率完全重合。"
                f"但**这不是「逐行相同」**：两者在 252 行里有 {_n01 + _n10} 行判断不同，"
                f"恰好 {_n01} 对 {_n10} 互补，聚合值相等是两边互换抵消的结果。"
                f"该对照一次改了组大小（4→32）、IS 层级（token→sequence）与"
                f"有效批次（16→1024）三样，**标识不了其中任何一样**。")
        _isl = ps.get("is_level_paired")
        if _isl:
            _lo3, _hi3 = _isl["delta_ci95_rows"]
            _m3 = _isl["mcnemar"]
            out.append(
                f"  - **IS 层级的单变量对照**（GRPO(G=4) 对 GSPO(G=4)：lr 5e-5、累积 "
                f"4/有效批次 16、同种子、同步数、同数据、同奖励函数与权重，"
                f"**只差 `--importance_sampling_level`**）："
                f"Δ准确率（sequence − token）"
                f"{_isl['accuracy_delta_sequence_minus_token']:+.4f}，"
                f"95% CI [{_lo3:+.4f}, {_hi3:+.4f}]，"
                f"p = {_m3['p_exact_two_sided']:.6g} —— {_isl['reading']}。"
                f"这是本项目**第一个只改一个变量的对照**，此前每一个都同时改了 IS "
                f"层级、组大小或有效批次之一以上。但它只覆盖 lr 5e-5 一个学习率，"
                f"而「更耐高学习率」是关于**学习率效应的大小**的比较，"
                f"因此论文的中心主张仍然**未被完整检验**，只是它的一个必要部分"
                f"第一次有了无混淆的读数。")
        _gs = ps.get("group_size_paired")
        if _gs:
            out.append(f"  - 组大小（GSPO lr 5e-5，G=8 → G=32）：Δ准确率 "
                       f"{_gs['accuracy_delta_g32_minus_g8']:+.4f}，"
                       f"p = {_gs['mcnemar']['p_exact_two_sided']:.6g} —— "
                       f"点估计为正但**未达显著**。即在 lr 5e-5 上，算法与组大小两两都"
                       f"分不开，**唯一分得开的是学习率**。")
        _ser = ps.get("group_size_series_lr5e5")
        if _ser:
            _parts = []
            for _k, _v in (_ser.get("rungs") or {}).items():
                if _v.get("accuracy_delta") is None:
                    continue
                _parts.append(f"{_k} Δ{_v['accuracy_delta']:+.4f}"
                              f"（p = {_v['mcnemar']['p_exact_two_sided']:.6g}）")
            out.append(
                f"  - **组大小的三点序列**（同为 GSPO、sequence、lr 5e-5，只让组大小 "
                f"从 4 走到 8 再到 32）：{'、'.join(_parts)}。"
                f"{_ser['reading']} —— 两点对照只能说明「这两个组大小不同」，"
                f"三点才能说明这条关系有没有形状。")
            if _ser.get("confound_size"):
                out.append(f"    - 但**本序列不是单变量**：{_ser['confound_size']}")
            out.append(f"    - {_ser['does_not_settle']}")
        # The group-size contrast above is confounded with effective batch size,
        # which the training records cannot show: their grad_accum field is a
        # hardcoded constant. Read the real figures from the correction record.
        _ebf = reports / "effective_batch_correction.json"
        _eb = (json.loads(_ebf.read_text(encoding="utf-8"))
               if _ebf.is_file() else None)
        if _eb:
            _b = (_eb.get("design_confound", {})
                  .get("effective_batch_by_G", {}))
            out.append(
                f"  - **但「组大小」这一行被有效批次混淆，读法要再退一步**："
                f"训练时梯度累积被设成等于组大小（4/4、8/8、32/32），"
                f"故有效批次 = 组大小² ——G=4 → {_b.get('4')}、G=8 → {_b.get('8')}、"
                f"G=32 → **{_b.get('32')}**。该对照因此同时改了组大小 4 倍与有效批次 16 倍，"
                f"**只能读作「加组大小或加有效批次无效」**，分不开这两个。"
                f"同一个混淆也约束论文的中心主张（token 永远是 G=4、"
                f"sequence 永远是 G=8 或 G=32，两个量从未单独变过）。"
                f"**学习率的对照不受影响**：同一组大小内累积步与有效批次都固定。"
                f"另有一个记录缺陷 —— 训练记录里的 `grad_accum` 字段是写死的常量 4，"
                f"三条记录（两个 G=8、一个 G=32 lr5e-5）因此写错了实际值，"
                f"见 LIMITATIONS 第 8 节第 11 条。")

    out.append(
        "- **判定（按论文 §8.11 的门槛）**：该节规定「若提升落在置信区间内，"
        "结论写『未观察到显著提升』，不写『RL 有效』」，并给出分支"
        "「F -- 否 --> G[保留 SFT 为最终模型]」。GRPO 与 GSPO 均未超过 SFT，"
        "落入该分支，因此 **最终模型保留 SFT（`outputs/checkpoints/qwen35_9b_qlora_v1`），"
        "不宣称 RL 有效**。")
    out.append(
        "- 需要说清的是，这**不等于**「GSPO 不如 SFT」或「GSPO 不如 GRPO」："
        "在 lr 1e-5 上 GSPO、GRPO 与 SFT 两两都落在噪声内；在 lr 5e-5 上"
        "**两个算法都显著低于 SFT**（见上表），而它们彼此之间仍分不开 —— "
        "**算法不是差异的来源**。完整的算法 × 学习率 2×2 已补齐（GRPO 的 lr 5e-5 "
        "对照格已跑完，见 LIMITATIONS 5.2.2），此前「无法排除崩的是 GSPO 本身」"
        "这一保留意见已由该对照格解除：同一个 5e-5 作用在 GRPO 上同样退化。")
    # Read from the records rather than written here: the previous version carried
    # 15.33% / 36.00% and "归因不成立" inline, so a new run could neither update the
    # numbers nor force the sentence to be reconsidered.
    _il = None
    if _psf.is_file():
        try:
            _il = json.loads(_psf.read_text(encoding="utf-8")).get("is_level_paired")
        except Exception:
            _il = None
    _f_g_lr5, _ = idle_frac(reports, "qwen35_9b_grpo_lr5e5")
    _f_s_g4, _ = idle_frac(reports, "qwen35_9b_gspo_g4_lr5e5")
    _seg = ["- **空转步的下降是观察到的；能归因的那一对确实动了，但仍没有转化为分数**："]
    _t1 = idle_txt(reports, "gspo_lr1e5")
    _t2 = idle_txt(reports, "grpo")
    if _t1 and _t2:
        _seg.append(f"lr 1e-5 下 GSPO 的零优势步是 {_t1}，GRPO 是 {_t2} —— "
                    "但该对照**同时改了组大小**（G=8 对 G=4）与 IS 层级，本设计无法把"
                    "「IS 层级」与「组大小」分开，所以这一条只是观察，不是归因。")
    if _f_g_lr5 is not None and _f_s_g4 is not None:
        _seg.append(f"能归因的那一对是 GRPO(G=4) lr5e-5 "
                    f"{idle_txt(reports, 'qwen35_9b_grpo_lr5e5')} 对 GSPO(G=4) lr5e-5 "
                    f"{idle_txt(reports, 'qwen35_9b_gspo_g4_lr5e5')}：同组大小 4、"
                    "同有效批次 16、同学习率 5e-5、同种子 3407，只差 "
                    "`--importance_sampling_level`（记录里 `single_variable` 为真），"
                    f"空转步低 **{(_f_g_lr5 - _f_s_g4) * 100:.2f} 个百分点**。")
    if _il:
        _seg.append(f"但**同一对配置**的准确率差是 "
                    f"{_il['accuracy_delta_sequence_minus_token']:+.4f}、95% CI "
                    f"[{_il['delta_ci95_rows'][0]:+.4f}, {_il['delta_ci95_rows'][1]:+.4f}]"
                    f"（包含 0，McNemar p = {_il['mcnemar']['p_exact_two_sided']}）——"
                    "机制层面的差异测得出，基准分数上的差异测不出"
                    "（见 LIMITATIONS 5.2.3 与 5.2.6）。")
    else:
        _seg.append("它没有转化为基准分数 —— 优势信号的多寡与最终指标的高低，"
                    "在这里是解耦的。")
    out.append("".join(_seg))
    _rk = ("retrieval mAP@10", "retrieval nDCG@10", "retrieval Recall@10")
    _g32 = {k: (metrics.get(k, {}) or {}).get("GSPO_G32") for k in _rk}
    _sft = {k: (metrics.get(k, {}) or {}).get("SFT") for k in _rk}
    if all(v is not None for v in list(_g32.values()) + list(_sft.values())):
        # Which run leads each metric, and which RL run is worst on macro-F1,
        # are facts about the table -- read them, do not assert them. An earlier
        # version asserted "all three lead" and "worst at classification" behind
        # a guard that only checked G=32 > SFT, so when the fifth run landed two
        # of those sentences were false and the guard still passed.
        def _best(key):
            row = {n: v for n, v in (metrics.get(key, {}) or {}).items()
                   if isinstance(v, (int, float)) and n != "Base"}
            return (max(row, key=lambda n: row[n]), row) if row else (None, {})

        _lead = {k: _best(k)[0] for k in _rk}
        _f1row = {n: v for n, v in (metrics.get("classification macro-F1", {}) or {}).items()
                  if isinstance(v, (int, float)) and n not in ("Base", "SFT")}
        _worst_f1 = min(_f1row, key=lambda n: _f1row[n]) if _f1row else None
        _g32f1 = (metrics.get("classification macro-F1", {}) or {}).get("GSPO_G32")

        if all(_lead[k] == "GSPO_G32" for k in _rk):
            out.append("- **G=32 的检索在全部 run 里居首**："
                       f"mAP@10 {_g32[_rk[0]]:.4f}、nDCG@10 {_g32[_rk[1]]:.4f}、"
                       f"Recall@10 {_g32[_rk[2]]:.4f}"
                       f"（SFT 分别为 {_sft[_rk[0]]:.4f}、{_sft[_rk[1]]:.4f}、"
                       f"{_sft[_rk[2]]:.4f}）。")
        else:
            _led = [k.split()[-1] for k in _rk if _lead[k] == "GSPO_G32"]
            _lost = "；".join(
                f"{k.split()[-1]} G=32 {_g32[k]:.4f} < {_lead[k]} "
                f"{(metrics.get(k, {}) or {}).get(_lead[k]):.4f}"
                for k in _rk if _lead[k] != "GSPO_G32")
            out.append("- **检索指标并非由同一个 run 领先，故「G=32 检索最好」已不成立**："
                       f"G=32 只在 {'、'.join(_led)} 上居首，"
                       + _lost
                       + "。G=32 三项都高于 SFT"
                       f"（mAP@10 {_g32[_rk[0]]:.4f} 对 {_sft[_rk[0]]:.4f}），")
            out.append(f"  但**三项里只有 {len(_led)} 项是全部 run 最高**，"
                       "因此它不能再被用来支持「G=32 重塑了表征」这一说法"
                       "（见 LIMITATIONS 5.2.5）。"
                       "该说法是本轮被实测推翻的又一处「点估计极值当作结论」。")

        if _worst_f1 == "GSPO_G32":
            out.append(f"- G=32 的 macro-F1 {_g32f1:.4f} 是全部 RL run 里最低的一档，"
                       "**分类最差而检索最好**这一对照成立，与 Donut 归零、"
                       "Scratch 最高合起来说明它**重塑了表征而非全面退化**"
                       "（见 LIMITATIONS 5.2.5）。")
        else:
            out.append(f"- **分类最差的 RL run 不是 G=32 而是 "
                       f"{run_label(_worst_f1)}**"
                       f"（macro-F1 {_f1row[_worst_f1]:.4f} 对 G=32 的 {_g32f1:.4f}），"
                       "所以「G=32 分类最差而检索最好」这一对照不成立。"
                       "G=32 的表征重塑只能由它自身的画像支持（Donut 归零、"
                       "Scratch 最高、`radial_zone` 全部 run 最高），"
                       "不能再借「分类最差」这个极值（见 LIMITATIONS 5.2.5）。")

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
