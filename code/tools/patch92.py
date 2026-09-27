"""Run 45 (GSPO, G=4, lr 1e-5) lands: write the lr axis at G=4.

Run 45 completes the G=4 cell of the grid. Two things become true that were not:

  * the lr effect inside GSPO is now measurable at a second group size (G=4), so
    "the lr effect is not an artefact of one group size" stops being untested;
  * at G=4 there is now a second single-variable contrast -- same algorithm, same
    group size, same IS level, same seed and steps, only the learning rate differs
    -- alongside the IS-level pair that run 43 created.

What it does NOT do is settle the paper's proposition, which is about how the lr
effect differs *by IS level*; that needs the interaction, and the record says so in
its own `does_not_settle` field. The text below says that where it belongs.

Every number is formatted from the records: the lr pair from
`paired_significance.json`, the idle rates from the two runs' train_results. The
idle-step direction sentence is *counted* from the three same-algorithm,
same-group-size pairs rather than asserted, because the third pair can point either
way and a hand-written "the direction is X" would be a coin flip.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
DOC = Path(os.environ.get("WAFER_DOC", str(ROOT / "LIMITATIONS.md")))
REP = ROOT / "outputs/reports"

PS = json.loads((REP / "paired_significance.json").read_text(encoding="utf-8"))
try:
    LR = PS["lr_effect_both_group_sizes"]
    B4 = LR["blocks"]["G=4"]
except KeyError as e:
    sys.exit(f"the paired record has no G=4 lr block ({e}); run 45 has not been "
             f"folded in yet -- nothing written")


def rec(tag):
    p = REP / f"{tag}_train_result.json"
    if not p.is_file():
        sys.exit(f"no train_result for {tag} at {p}; nothing written")
    return json.loads(p.read_text(encoding="utf-8"))


def idle(tag):
    d = rec(tag)
    f = d["mean_frac_reward_zero_std"]
    n = d.get("steps_logged") or 150
    return f, f"{f:.2%}（{round(f * n)}/{n}）"


# the three same-algorithm, same-group-size lr pairs, lr 1e-5 -> lr 5e-5
_f_g1, _t_g1 = idle("grpo")
_f_g5, _t_g5 = idle("qwen35_9b_grpo_lr5e5")
_f_s1, _t_s1 = idle("gspo_lr1e5")
_f_s5, _t_s5 = idle("qwen35_9b_gspo_v1")
_f_x1, _t_x1 = idle("qwen35_9b_gspo_g4_lr1e5")
_f_x5, _t_x5 = idle("qwen35_9b_gspo_g4_lr5e5")

PAIRS = [("GRPO(G=4)", _f_g1, _f_g5), ("GSPO(G=8)", _f_s1, _f_s5),
         ("GSPO(G=4)", _f_x1, _f_x5)]
_pt = [(n, (b - a) * 100) for n, a, b in PAIRS]
_up = [n for n, d in _pt if d > 0]
_down = [n for n, d in _pt if d < 0]
_dir_txt = (f"三条**同算法、同组大小**的对照里，lr 5e-5 把空转步推高的有 "
            f"{len(_up)} 条（{'、'.join(_up) if _up else '无'}）、压低的 "
            f"{len(_down)} 条（{'、'.join(_down) if _down else '无'}）—— "
            f"方向随算法与组大小改变，**不是学习率单方面的效应**"
            if _up and _down else
            f"三条**同算法、同组大小**的对照里方向一致（全部"
            f"{'上升' if _up else '下降'}）—— 但对照的样本是三条，且 GRPO 一侧"
            f"的绝对值最小，这仍不足以把它读成学习率的普遍效应")

EDITS = [
    # ------------------------------------------------ 5.2.2 the lr effect, now twice
    ("""算法内部的学习率效应同样显著：GRPO
-0.0556（`p` = 0.0243）、
GSPO -0.0754（`p` = 0.0013187）。""",
     f"""算法内部的学习率效应同样显著：GRPO
-0.0556（`p` = 0.0243）、
GSPO -0.0754（`p` = 0.0013187）。
run 45 落地后这条在 **GSPO 内部有了第二个组大小**：G=4 上
{B4['accuracy_delta']:+.4f}（`p` = {B4['mcnemar']['p_exact_two_sided']}，与 -0.0754
同向），记录自己的读法是 "{LR['reading']}"。""",
     "5.2.2: the lr effect now has two group sizes inside GSPO"),

    # ------------------------------------------- 5.2.2 the idle-step direction
    ("""  同条件对照的方向相反：GRPO(G=4)
  36.00% → 32.67%，
  GSPO(G=8) 15.33% → 38.00%。
  即「学习率推高空转步」在 G=8 上成立、在 G=4 上不成立。""",
     f"""  同算法同组大小的三条对照：GRPO(G=4)
  {_t_g1} → {_t_g5}、GSPO(G=8)
  {_t_s1} → {_t_s5}、GSPO(G=4)
  {_t_x1} → {_t_x5}。
  {_dir_txt}。""",
     "5.2.2: the third same-algorithm pair, and the direction counted"),
]

# ------------------------------------------------- 5.2.6 a second single-variable pair
A3_OLD = "**三条合起来：**"
A3_NEW = f"""**（3）学习率：在 G=4 上的第二个单变量对照。** run 45 与 run 43 是同算法（GSPO）、
同组大小（G=4）、同 IS 层级（sequence）、同种子、同 150 步，**只差学习率**（1e-5 对
5e-5）：Δ准确率 = **{B4['accuracy_delta']:+.4f}**，95% 置信区间
**[{B4['delta_ci95_rows'][0]:+.4f}, {B4['delta_ci95_rows'][1]:+.4f}]**、McNemar
`p` = {B4['mcnemar']['p_exact_two_sided']}，
{"**排除 0**，与 G=8 上的学习率效应同向 —— 因此「学习率效应只在某一个组大小上出现」"
 "这条解释被排除" if B4['excludes_zero'] else
 "**包含 0**，即在这个组大小上学习率**未达显著** —— 因此它**不能**排除「学习率效应"
 "只在某一个组大小上出现」这条解释，与 G=8 上的显著结果并不矛盾，但也不支持它"}。
两个对照都不触及论文的主张 —— 后者说的是学习率效应**如何随 IS 层级变化**，要的是两者
的交互项，记录在 `does_not_settle` 字段里写明了这一点。

**三条合起来：**"""
if B4["excludes_zero"] != (B4["delta_ci95_rows"][0] > 0 or B4["delta_ci95_rows"][1] < 0):
    sys.exit("the record's excludes_zero disagrees with its own interval; nothing written")

EDITS.append((A3_OLD, A3_NEW, "5.2.6: the lr pair at G=4, as the second single-variable contrast"))

# --------------------------------------------- 5.2.5 the lr sentence, hardened
EDITS.append((
    "**在 lr 5e-5 这一档上，唯一分得开的是学习率**",
    ("**在 lr 5e-5 这一档上，唯一分得开的是学习率**（run 45 落地后这条更硬："
     "GSPO 内部在 G=4 与 G=8 两个组大小上学习率都分得开 —— G=4 "
     f"{B4['accuracy_delta']:+.4f}、`p` = {B4['mcnemar']['p_exact_two_sided']}，"
     "G=8 -0.0754、`p` = 0.0013187，见 5.2.6）"
     if B4["excludes_zero"] else
     "**在 lr 5e-5 这一档上，唯一分得开的是学习率**（run 45 落地后**没有**变成两处："
     "GSPO 内部的 lr 效应在 G=8 上显著、在 G=4 上未达显著（"
     f"{B4['accuracy_delta']:+.4f}、`p` = {B4['mcnemar']['p_exact_two_sided']}），"
     "所以这一条仍只在一个组大小上成立，见 5.2.6）"),
    "5.2.5: whether the lr effect reaches a second group size"))

s = DOC.read_text(encoding="utf-8")
before = s.count("**")
for old, new, why in EDITS:
    n = s.count(old)
    if n != 1:
        sys.exit(f"anchor for '{why}' appears {n} times, expected 1; nothing written")
    s = s.replace(old, new, 1)
after = s.count("**")
if (after - before) % 2:
    sys.exit(f"bold markers went {before} -> {after}, an odd change; nothing written")

DOC.write_text(s, encoding="utf-8")
print(f"{len(EDITS)} edit(s) applied ({len(s.splitlines())} lines), bold {before} -> {after}")
for _, _, why in EDITS:
    print(f"  - {why}")
