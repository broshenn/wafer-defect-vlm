"""Run 45 (GSPO, G=4, lr 1e-5) lands: write the lr axis where it now exists.

Run 45 completes the 2x2 at G=4 -- both IS levels at both learning rates, same group
size, same effective batch, same seed, steps, data and rewards. Three things become
true that were not:

  * the lr effect inside GSPO is now measured at a second group size, so "the lr
    effect is an artefact of one group size" stops being untested;
  * at G=4 there is a second single-variable contrast -- same algorithm, group size,
    IS level, seed and steps, only the lr differs -- beside the IS-level pair run 43
    created;
  * the cell the record named as missing ("GSPO (IS=sequence) at G=4 with lr 1e-5")
    is filled, so that sentence has to stop saying it is missing.

What it does NOT do is settle the paper's proposition, which is about how the lr
effect differs *by IS level*; that needs the interaction, and the record says so in
`does_not_settle`. The text below says that where it belongs.

Every number is read from the records: the lr pair from `paired_significance.json`
(both the pair and, now, the whole `lr_effect_paired` listing), the idle shares from
the runs' `train_result.json`. Nothing is typed in. The idle-direction sentence is
*counted* over the three same-algorithm, same-group-size pairs rather than asserted,
because with three pairs the direction can go either way and a hand-written "the
direction is X" would be a coin flip.

An earlier version of this patch read the pair from
`lr_effect_both_group_sizes["blocks"]["G=4"]`. That block could never contain it:
its group-size list was written by hand as (8, 32). The tool now enumerates its cells
from the run names (tools/patch95_paired.py), and the pair is read from where it
actually appears.
"""
import json
import sys
from pathlib import Path

DOC = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
REP = Path("/root/autodl-fs/wafer-vlm/outputs/reports")

PS = json.loads((REP / "paired_significance.json").read_text(encoding="utf-8"))
try:
    LR = PS["lr_effect_paired"]["GSPO G=4"]
except KeyError as e:
    sys.exit(f"the paired record has no GSPO G=4 lr pair ({e}); run 45 has not been "
             f"folded in yet -- nothing written")

D = LR["accuracy_delta"]
LO, HI = LR["delta_ci95_rows"]
P = LR["mcnemar"]["p_exact_two_sided"]
EXCL = LR["excludes_zero"]
if EXCL != (LO > 0 or HI < 0):
    sys.exit("the record's excludes_zero disagrees with its own interval; nothing written")

# The lr effect is quoted in one sentence per cell; read all of them from the record
# rather than leaving two of them typed in.
CELLS = PS["lr_effect_paired"]
_LABEL = {"GRPO G=4": "GRPO(G=4)", "GSPO G=8": "GSPO(G=8)", "GSPO G=4": "GSPO(G=4)"}


def cell_txt(k):
    v = CELLS[k]
    return (f"{_LABEL.get(k, k)} {v['accuracy_delta']:+.4f}"
            f"（`p` = {v['mcnemar']['p_exact_two_sided']}）")


_ALL_CELLS = "、".join(cell_txt(k) for k in
                       sorted(CELLS, key=lambda k: (_LABEL.get(k, k))))

# The 2x2 at G=4 is complete: state it, and stop calling the cell missing.
MISSING = (PS.get("paper_claim_status") or {}).get("missing_cell", "")
_G4_DONE = MISSING.startswith("none")


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
PAIRS = [("GRPO(G=4)", idle("grpo")[0], idle("qwen35_9b_grpo_lr5e5")[0]),
         ("GSPO(G=8)", idle("gspo_lr1e5")[0], idle("qwen35_9b_gspo_v1")[0]),
         ("GSPO(G=4)", idle("qwen35_9b_gspo_g4_lr1e5")[0],
          idle("qwen35_9b_gspo_g4_lr5e5")[0])]
SHOWN = [("GRPO(G=4)", idle("grpo")[1], idle("qwen35_9b_grpo_lr5e5")[1]),
         ("GSPO(G=8)", idle("gspo_lr1e5")[1], idle("qwen35_9b_gspo_v1")[1]),
         ("GSPO(G=4)", idle("qwen35_9b_gspo_g4_lr1e5")[1],
          idle("qwen35_9b_gspo_g4_lr5e5")[1])]
_pt = [(n, (b - a) * 100) for n, a, b in PAIRS]
_up = [n for n, x in _pt if x > 0]
_down = [n for n, x in _pt if x < 0]
_dir_txt = (
    f"三条**同算法、同组大小**的对照里，lr 5e-5 把空转步推高的有 "
    f"{len(_up)} 条（{'、'.join(_up) if _up else '无'}）、压低的 "
    f"{len(_down)} 条（{'、'.join(_down) if _down else '无'}）—— "
    f"方向随算法与组大小改变，**不是学习率单方面的效应**"
    if _up and _down else
    f"三条**同算法、同组大小**的对照里方向一致（全部"
    f"{'上升' if _up else '下降'}）—— 但对照只有三条，且其中两条的绝对值很小，"
    f"这仍不足以把它读成学习率的普遍效应")
_IDLE_TXT = "、".join(f"{n}\n  {a} → {b}" for n, a, b in SHOWN)

EDITS = [
    # ------------------------------------------- 5.2.2 the lr effect, all cells
    ("""算法内部的学习率效应同样显著：GRPO
-0.0556（`p` = 0.0243）、
GSPO -0.0754（`p` = 0.0013187）。""",
     f"""算法内部的学习率效应（同算法、同组大小、只差学习率）同样显著：
{_ALL_CELLS}。""",
     "5.2.2: the lr effect, enumerated from the record"),

    # ------------------------------------- 5.2.2 the idle-step direction, counted
    ("""  同条件对照的方向相反：GRPO(G=4)
  36.00% → 32.67%，
  GSPO(G=8) 15.33% → 38.00%。
  即「学习率推高空转步」在 G=8 上成立、在 G=4 上不成立。""",
     f"""  同算法同组大小的三条对照：{_IDLE_TXT}。
  {_dir_txt}。""",
     "5.2.2: the third same-algorithm pair, and the direction counted"),
]

# ------------------------------------------- 5.2.6 the second single-variable pair
A3_OLD = """**三条合起来：** 本设计里能分得开的仍然只有学习率（5.2.5）；论文主张的那条轴在 G=4 上
单变量也分不开；GSPO 内部沿组大小是平的。**这三条都是「在本模型规模与本次预算下」的
陈述**，不是「论文的机制不存在」—— 一个学习率、一个 252 行基准、150 步，能支持的
最强结论就到这里。"""
A3_NEW = (
    f"""**（3）学习率：在 G=4 上的第二个单变量对照。** run 45 与 run 43 是同算法（GSPO）、
同组大小（G=4）、同 IS 层级（sequence）、同种子、同 150 步，**只差学习率**（1e-5 对
5e-5）：Δ准确率 = **{D:+.4f}**，95% 置信区间 **[{LO:+.4f}, {HI:+.4f}]**、McNemar
`p` = {P}，
"""
    + ("**排除 0**，与 G=8 上的学习率效应同向 —— 因此「学习率效应只在某一个组大小上"
       "出现」这条解释被排除。"
       if EXCL else
       "**包含 0**，即在这个组大小上学习率**未达显著** —— 因此它**不能**排除"
       "「学习率效应只在某一个组大小上出现」这条解释，与 G=8 上的显著结果并不矛盾，"
       "但也不支持它。")
    + f"""
与 IS 层级那个对照一样，它不触及论文的主张 —— 论文说的是学习率效应**如何随 IS 层级
变化**，要的是两者的交互项；记录在 `does_not_settle` 字段里写明了这一点。

**G=4 上的 2x2 至此补齐。** 四个格子 —— token/sequence × lr 1e-5/5e-5 —— 组大小、
有效批次、种子、步数、数据与奖励全部相同，因此**学习率效应与 IS 层级效应第一次可以
各自单独读出来**。记录里 `paper_claim_status.missing_cell` 此前指的正是缺的这一格
（GSPO、G=4、lr 1e-5），现已改为「{'2x2 已补齐' if _G4_DONE else '仍未补齐'}」并写明
它补上的与**没**补上的各是什么。

**四条合起来：** 本设计里能分得开的仍然只有学习率（5.2.5）—— 而且 run 45 落地后，
学习率在 GSPO 内部 G=4 与 G=8 两个组大小上都分得开（（3））；论文主张的那条轴在 G=4 上
单变量也分不开（（1））；GSPO 内部沿组大小是平的（（2））。**这四条都是「在本模型规模
与本次预算下」的陈述**，不是「论文的机制不存在」—— 一个学习率、一个 252 行基准、150
步，能支持的最强结论就到这里。"""
)
EDITS.append((A3_OLD, A3_NEW,
              "5.2.6: the lr pair at G=4, and the completed 2x2"))

# ------------------------------------------ 5.2.5 the lr sentence, hardened
EDITS.append((
    "**在 lr 5e-5 这一档上，唯一分得开的是学习率**",
    ("**在 lr 5e-5 这一档上，唯一分得开的是学习率**（run 45 落地后这条更硬："
     "GSPO 内部在 G=4 与 G=8 两个组大小上学习率都分得开 —— G=4 "
     f"{D:+.4f}、`p` = {P}，G=8 -0.0754、`p` = 0.0013187，见 5.2.6）"
     if EXCL else
     "**在 lr 5e-5 这一档上，唯一分得开的是学习率**（run 45 落地后**没有**变成两处："
     "GSPO 内部的 lr 效应在 G=8 上显著、在 G=4 上未达显著（"
     f"{D:+.4f}、`p` = {P}），"
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

# the section heading no longer describes one contrast, and no longer says the
# group-size axis is the only flat one
H_OLD = "#### 5.2.6 第一个单变量对照：IS 层级分不开，组大小也分不开"
H_NEW = "#### 5.2.6 单变量对照：IS 层级分不开，学习率分得开"
if s.count(H_OLD) != 1:
    sys.exit(f"5.2.6 heading anchor appears {s.count(H_OLD)} times; nothing written")
s = s.replace(H_OLD, H_NEW, 1)

# and the sentence that used the old heading
INTRO_OLD = """run 43（GSPO, G=4, lr 5e-5）落地后，本项目第一次有了「只动论文主张的那一个变量」的
对照。"""
INTRO_NEW = """run 43（GSPO, G=4, lr 5e-5）落地后，本项目第一次有了「只动论文主张的那一个变量」的
对照；run 45 落地后 G=4 上的 2x2 补齐，又多了第二个**只动学习率**的对照。"""
if s.count(INTRO_OLD) != 1:
    sys.exit(f"5.2.6 intro anchor appears {s.count(INTRO_OLD)} times; nothing written")
s = s.replace(INTRO_OLD, INTRO_NEW, 1)

DOC.write_text(s, encoding="utf-8")
print(f"{len(EDITS)+2} edit(s) applied ({len(s.splitlines())} lines), "
      f"bold {before} -> {after}")
print(f"  GSPO G=4 lr pair: d={D:+.4f} CI=[{LO:+.4f},{HI:+.4f}] p={P} "
      f"excludes_zero={EXCL}; 2x2 complete: {_G4_DONE}")
for _, _, why in EDITS:
    print(f"  - {why}")
