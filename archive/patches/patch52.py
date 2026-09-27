"""Rewrite 5.2.2 now that the missing control cell is filled.

Three things changed and all three bear on this section:

  * GRPO at lr 5e-5 has been run, so the 2x2 the section calls incomplete is
    complete. Both algorithms now have a reading at both learning rates.
  * A paired test replaces the marginal-overlap rule the section relied on for
    its significance claim. It reaches the opposite verdict, so the sentence
    saying the drop cannot be called significant has to go -- and the reason it
    was wrong has to be stated, or the next reader will re-derive it.
  * The mechanism the section proposed (KL rising, idle steps rising) does not
    survive checking. The KL claim is not merely backwards; it is unmeasurable
    in this design, because the completions differ in length. So it is removed
    rather than reversed, and the evidence that does hold is put in its place.

The table is built from the __report.json files and the train records at
runtime, not typed in. A hand-typed table is a second copy of every number in
it, free to disagree with the first -- which is item 7 of section 8.
"""
import json
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
REPORTS = ROOT / "outputs/reports"
DOC = ROOT / "LIMITATIONS.md"

COLS = [("SFT", "qwen35_9b_adapter", None),
        ("GRPO (G=4, lr1e-5)", "qwen35_9b_grpo", "grpo"),
        ("GRPO (G=4, lr5e-5)", "qwen35_9b_grpo_lr5e5", "qwen35_9b_grpo_lr5e5"),
        ("GSPO (G=8, lr5e-5)", "qwen35_9b_gspo_v1", "qwen35_9b_gspo_v1"),
        ("GSPO (G=8, lr1e-5)", "gspo_lr1e5", "gspo_lr1e5"),
        ("GSPO (G=32, lr5e-5)", "qwen35_9b_gspo_g32", "qwen35_9b_gspo_g32")]

rep, idle = {}, {}
for label, tag, train_tag in COLS:
    p = REPORTS / f"{tag}__report.json"
    if not p.is_file():
        sys.exit(f"missing report for {label}: {p}")
    rep[label] = json.loads(p.read_text(encoding="utf-8"))
    if train_tag:
        t = json.loads((REPORTS / f"{train_tag}_train_result.json").read_text(encoding="utf-8"))
        idle[label] = t.get("mean_frac_reward_zero_std")

missing = [c for c, d in rep.items() if d.get("retrieval") is None]
if missing:
    print(f"WARNING: retrieval is null for {missing}; the table will show it as "
          f"unmeasured. Driver 40 step 3b fills it.")


def val(label, fn):
    try:
        v = fn(rep[label])
    except (KeyError, TypeError):
        return None
    return v


def f4(v):
    return "—" if v is None else f"{v:.4f}"


def ci(label):
    c = rep[label]["classification"].get("macro_f1_95ci")
    return "—" if not c else f"[{c[0]:.4f}, {c[1]:.4f}]"


def rt(label):
    r = rep[label].get("retrieval")
    if not r:
        return "—"
    for k in ("mAP@10", "mAP@10_mean", "map@10"):
        if k in r:
            return f"{r[k]:.4f}"
    return "—"


ROWS = [
    ("分类准确率", lambda L: f4(val(L, lambda d: d["classification"]["accuracy"]))),
    ("macro-F1", lambda L: f4(val(L, lambda d: d["classification"]["macro_f1"]))),
    ("macro-F1 95% CI", ci),
    ("structured defect_type", lambda L: f4(val(L, lambda d: d["structured"]["field_accuracy"]["defect_type"]))),
    ("structured radial_zone", lambda L: f4(val(L, lambda d: d["structured"]["field_accuracy"]["radial_zone"]))),
    ("clock circular MAE", lambda L: f4(val(L, lambda d: d["structured"]["clock_circular_mae_sectors"]))),
    ("size MAE (R)", lambda L: f4(val(L, lambda d: d["structured"]["size_mae_r"]))),
    ("caption must-hit", lambda L: f4(val(L, lambda d: d["caption"]["must_hit_rate"]))),
    ("robustness accuracy", lambda L: f4(val(L, lambda d: d["robustness"]["accuracy"]))),
    ("flip rate", lambda L: f4(val(L, lambda d: d["robustness"]["flip_rate_vs_clean"]))),
    ("retrieval mAP@10", rt),
    ("**零优势步占比（实测）**", lambda L: "—" if idle.get(L) is None else f"{idle[L]:.2%}"),
]

# The paired results are read from their record rather than typed here. Prose
# that restates a number is a second copy of it, free to drift from the first --
# item 7 of section 8, and this document has already lost time to it once.
PS = json.loads((REPORTS / "paired_significance.json").read_text(encoding="utf-8"))
KC = json.loads((REPORTS / "kl_length_confound.json").read_text(encoding="utf-8"))

def cmp_row(key):
    c = PS["comparisons_vs_SFT"][key]
    lo, hi = c["delta_ci95_rows"]
    return (f"| {'**' if abs(c['delta_vs_sft']) > 0.03 else ''}"
            f"{c['label']}{'**' if abs(c['delta_vs_sft']) > 0.03 else ''} vs SFT | "
            f"{'**' if abs(c['delta_vs_sft']) > 0.03 else ''}{c['delta_vs_sft']:+.4f}"
            f"{'**' if abs(c['delta_vs_sft']) > 0.03 else ''} | "
            f"{'**' if c['excludes_zero'] else ''}[{lo:+.4f}, {hi:+.4f}]"
            f"{'**' if c['excludes_zero'] else ''} | "
            f"{'**' if c['mcnemar']['p_exact_two_sided'] < 0.05 else ''}"
            f"{c['mcnemar']['p_exact_two_sided']:.5f}"
            f"{'**' if c['mcnemar']['p_exact_two_sided'] < 0.05 else ''} |")

LABELS = {"GRPO_G4_lr1e5": "GRPO(G=4) lr1e-5",
          "GRPO_G4_lr5e5": "GRPO(G=4) lr5e-5",
          "GSPO_G8_lr1e5": "GSPO(G=8) lr1e-5",
          "GSPO_G8_lr5e5": "GSPO(G=8) lr5e-5",
          "GSPO_G32_lr5e5": "GSPO(G=32) lr5e-5"}
for k, v in LABELS.items():
    PS["comparisons_vs_SFT"][k]["label"] = v
paired_tbl = "\n".join(cmp_row(k) for k in LABELS)

def fp(x):
    return f"{x:.5f}" if x >= 0.01 else f"{x:.5g}"

lr_g, lr_s = PS["lr_effect_paired"]["GRPO G=4"], PS["lr_effect_paired"]["GSPO G=8"]
xa5 = PS["cross_algorithm_paired"]["lr5e-5"]
xa1 = PS["cross_algorithm_paired"]["lr1e-5"]
v_sft = PS["accuracy"]["SFT"]
acc = PS["accuracy"]
_idle = KC["runs"]
_len = {k: KC["runs"][k]["mean_completion_length"] for k in KC["runs"]}
_kl = {k: KC["runs"][k]["mean_kl"] for k in KC["runs"]}
_idl = {k: KC["runs"][k]["mean_frac_reward_zero_std"] for k in KC["runs"]}
IDLE_TAG = {"grpo": "grpo", "qwen35_9b_grpo_lr5e5": "qwen35_9b_grpo_lr5e5",
            "gspo_lr1e5": "gspo_lr1e5", "qwen35_9b_gspo_v1": "qwen35_9b_gspo_v1"}

labels = [c[0] for c in COLS]
tbl = ["| 指标 | " + " | ".join(labels) + " |",
       "| --- | " + " | ".join("---" for _ in labels) + " |"]
for name, fn in ROWS:
    tbl.append(f"| {name} | " + " | ".join(fn(L) for L in labels) + " |")
table = "\n".join(tbl)


def mf1(L):
    return rep[L]["classification"]["macro_f1"]


g1, g5 = mf1("GRPO (G=4, lr1e-5)"), mf1("GRPO (G=4, lr5e-5)")
s1, s5 = mf1("GSPO (G=8, lr1e-5)"), mf1("GSPO (G=8, lr5e-5)")
sft = mf1("SFT")

NEW = f"""#### 5.2.2 基准成绩：离群的是学习率（对照格已补齐）

{table}

**对照格已补齐，结论比补齐前更强。** GRPO 在 5e-5 下同样变差：macro-F1
{g1:.4f} → {g5:.4f}（{g5 - g1:+.4f}），GSPO 为 {s1:.4f} → {s5:.4f}（{s5 - s1:+.4f}）。
两个算法在同一学习率下**同向、同量级**地变差，所以「坏的是学习率、不是算法」
不再只由 GSPO 一列支撑，而有了一次跨算法的复现。
**但论文的中心主张——序列级归一化让 GSPO 更耐高学习率——既未被支持也未被否定**：
lr5e-5 下 GSPO(G=8) 比 GRPO(G=4) 低 {abs(xa5['accuracy_delta_gspo_minus_grpo']):.4f}，
成对 McNemar `p` = {xa5['mcnemar']['p_exact_two_sided']:.2f}，不构成差异；
且两者组大小不同（8 与 4），这个对照无法把「IS 层级」与「组大小」分开
（见第 9 节 `paired_significance.json` 的 `cross_algorithm_paired`）。
lr1e-5 下两者同样无法区分（差
{abs(xa1['accuracy_delta_gspo_minus_grpo']):.4f}，`p` = {xa1['mcnemar']['p_exact_two_sided']:.2f}）。

**必须更正原稿在这里的一处保守说法。** 原稿按 `tools/final_report.py` 的规则报告
「lr5e-5 列的 95% CI 与 SFT 擦边重叠，严格说不能宣称统计显著」，措辞因此定为
「明显更差」而非「显著更差」。该规则比较两个**边际**置信区间；但六次 run 回答的是
**同一批 252 行、同一批提示词**，配对结构正是这个设计里承载检验力的部分，
比较边际区间把它丢掉了。改用成对检验（按行 bootstrap + 精确 McNemar）：

| 对比（同为 252 行） | Δ准确率 | 95% CI（按行重采样） | McNemar `p` |
| --- | --- | --- | --- |
{paired_tbl}

结论改写为：**lr5e-5 的下降在两个算法上都达到显著**
（GRPO `p` = {PS['comparisons_vs_SFT']['GRPO_G4_lr5e5']['mcnemar']['p_exact_two_sided']:.4f}，
GSPO `p` = {fp(PS['comparisons_vs_SFT']['GSPO_G8_lr5e5']['mcnemar']['p_exact_two_sided'])}），
而 lr1e-5 的两次与 SFT **无法区分**（GRPO 差
{acc['GRPO_G4_lr1e5'] - v_sft:+.4f}、`p` = {PS['comparisons_vs_SFT']['GRPO_G4_lr1e5']['mcnemar']['p_exact_two_sided']:.2f}；
GSPO 差 {acc['GSPO_G8_lr1e5'] - v_sft:+.4f}、`p` = {PS['comparisons_vs_SFT']['GSPO_G8_lr1e5']['mcnemar']['p_exact_two_sided']:.2f}）。
算法内部的学习率效应同样显著：GRPO
{lr_g['accuracy_delta']:+.4f}（`p` = {lr_g['mcnemar']['p_exact_two_sided']:.4f}）、
GSPO {lr_s['accuracy_delta']:+.4f}（`p` = {fp(lr_s['mcnemar']['p_exact_two_sided'])}）。
即**方向未变，变的是显著性**：边际重叠规则低估了它本应看到的差异。
该检验只覆盖准确率——macro-F1 没有逐样本分解，无法配对，仍按边际值阅读。
正对照（BASE vs SFT，差 {PS['positive_control']['accuracy_delta']:+.4f}、
`p` = {fp(PS['positive_control']['p_exact_two_sided'])}）不通过则脚本拒绝输出，见第 9 节。

**原稿给这个现象配的机制解释，经核对不成立，已删除。** 原稿用两条证据说明
「策略被推进退化区」：KL 被抬到 1.02–1.12，以及空转步从 36.00% 推到 38.00%。
两条都不成立，且**不是方向反了那么简单**：

- KL：那两个数字属实，但 {_kl['gspo_lr1e5']:.4f} 属于 lr1e-5、{_kl['qwen35_9b_gspo_v1']:.4f}
  属于 lr5e-5，区间横跨了两个学习率，恰好掩盖了「学习率升高时 KL 下降」。
  同算法同组大小的两个对照里 KL 都是下降的
  （GRPO {_kl['grpo']:.4f} → {_kl['qwen35_9b_grpo_lr5e5']:.4f}，
  GSPO {_kl['gspo_lr1e5']:.4f} → {_kl['qwen35_9b_gspo_v1']:.4f}）。
  而 `kl` 是**逐 token** 均值，5e-5 的补全却明显更短
  （GRPO {_len['grpo']:.1f} → {_len['qwen35_9b_grpo_lr5e5']:.1f} token、
  GSPO {_len['gspo_lr1e5']:.1f} → {_len['qwen35_9b_gspo_v1']:.1f}），
  均值取在不同的 token 跨度上，**这个测量无法判定方向** ——
  所以该句删除，而不是反向重写。
- 空转步：那句拿 GRPO(G=4, lr1e-5) 比 GSPO(G=8, lr5e-5)，算法与组大小同时变了。
  同条件对照的方向相反：GRPO(G=4)
  {_idl['grpo']:.2%} → {_idl['qwen35_9b_grpo_lr5e5']:.2%}，
  GSPO(G=8) {_idl['gspo_lr1e5']:.2%} → {_idl['qwen35_9b_gspo_v1']:.2%}。
  即「学习率推高空转步」在 G=8 上成立、在 G=4 上不成立。

**5e-5 真正一致的签名是另外两条**：两个算法的奖励都下降、补全都变短 ——
策略收敛到更短、得分更低的输出，而不是「走得太远」。详见第 8 节第 9 条，
数值集中在 `outputs/reports/kl_length_confound.json`。

**G=32 那一列的强度需要下调。** 它比分低于 SFT，但成对检验 `p` =
{PS['comparisons_vs_SFT']['GSPO_G32_lr5e5']['mcnemar']['p_exact_two_sided']:.3f}、
置信区间**包含 0**，即**未达显著**。这与 5.2.5「组大小有帮助但不足」一致，
同时说明该列的证据强度弱于 GSPO(G=8) lr5e-5 那一列。
"""

doc = DOC.read_text(encoding="utf-8")
start = doc.index("#### 5.2.2")
end = doc.index("#### 5.2.3")
old = doc[start:end]
if "尚未补齐的对照格" not in old:
    sys.exit("REFUSING: the 5.2.2 block does not look like the one to replace")
doc = doc[:start] + NEW + "\n" + doc[end:]
DOC.write_text(doc, encoding="utf-8")
print(f"5.2.2 replaced: {len(old)} -> {len(NEW)} chars")
print(f"lines: {doc.count(chr(10)) + 1}")
