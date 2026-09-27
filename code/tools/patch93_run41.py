"""Run 41 (GSPO, G=32, lr 1e-5) lands: the group-size axis gets a second learning rate.

Until now the group-size line was drawn once, at lr 5e-5: G=8 -> G=32, flat. That is
weak evidence on its own, because the lr effect is the one thing this project *can*
separate, and a flat group-size line at a single lr cannot tell "group size does not
matter" from "group size does not matter at lr 5e-5, where the policy is already in
the regime the high lr puts it in". Run 41 draws the same rung at lr 1e-5, which is
where the policy is *not* in that regime (15.33% idle steps against 38.00%).

Run 41 also completes the third cell of `lr_effect_both_group_sizes`, so the lr effect
is now measured at G=4, G=8 and G=32.

Two things it does not do, and the text says so where it belongs:
  * it does not make the group-size line a three-point line at lr 1e-5 -- run 45 gives
    G=4 at that lr, but the record's `group_size_paired_lr1e5` contrast is the G=8 ->
    G=32 rung, and this patch quotes what the record has rather than a line it does not
    draw;
  * it does not turn a flat pair at two lrs into "group size is irrelevant" -- the
    confound is the same at both (effective batch moves with group size), and the
    record's own caveat says a non-monotone effect between the two sizes would not be
    seen.

Every number is read from `outputs/reports/paired_significance.json`; nothing is typed
in. The patch refuses while run 41 is absent, and refuses again if a field it quotes
does not exist, so it cannot half-write a sentence about a contrast that is not in the
record.

This replaces an earlier version of the same patch, which had four defects of the kind
the document itself catalogues (section 8). It said "有了两个格子" and then listed every
block the record held -- and the record holds three once run 41 lands, so the sentence
would have contradicted the list printed inside it. It sorted the block names
lexicographically, printing "G=32、G=4、G=8". It asserted "方向一致" -- a direction claim
nothing read from the record. And its closing paragraph asserted the group-size line was
flat at both learning rates, which its own （4） block contradicts whenever run 41's rung
separates. All four are computed here, and the closing paragraph branches on the
computed result rather than on the one that was expected.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
DOC = ROOT / "LIMITATIONS.md"
PS = json.loads((ROOT / "outputs/reports/paired_significance.json")
                .read_text(encoding="utf-8"))

for key in ("group_size_paired_lr1e5", "group_size_verdict", "lr_effect_both_group_sizes"):
    if not PS.get(key):
        sys.exit(f"the paired record has no {key}; run 41 has not been folded in yet "
                 f"-- nothing written")

GS = PS["group_size_paired_lr1e5"]
VD = PS["group_size_verdict"]
LR = PS["lr_effect_both_group_sizes"]

D = GS["accuracy_delta_g32_minus_g8"]
LO, HI = GS["delta_ci95_rows"]
P = GS["mcnemar"]["p_exact_two_sided"]
N01 = GS["mcnemar"]["g8_wrong_g32_right"]
N10 = GS["mcnemar"]["g8_right_g32_wrong"]
EXCL = GS["excludes_zero"]
if EXCL != (LO > 0 or HI < 0):
    sys.exit("the record's excludes_zero disagrees with its own interval; nothing written")

# The rung at the other lr, for the comparison the paragraph makes. Read, not typed.
G5 = PS["group_size_series_lr5e5"]["rungs"]["G=8 -> G=32"]
D5, P5 = G5["accuracy_delta"], G5["mcnemar"]["p_exact_two_sided"]
FLAT5 = not PS["group_size_series_lr5e5"]["significant_rungs"]

# --- the lr effect per group size -------------------------------------------------
# Numeric, not lexicographic: sorted() over "G=4"/"G=8"/"G=32" puts G=32 first.
def gno(k):
    return int(k.split("=")[1])


CN = "一二三四五六七八九十"
BLK = sorted(LR["blocks"], key=gno)
SIG = sorted(LR["significant_at"], key=gno)
if not BLK:
    sys.exit("the record's lr_effect_both_group_sizes holds no blocks; nothing written")
CNT = CN[len(BLK) - 1] if 0 < len(BLK) <= 10 else str(len(BLK))
BLOCKLIST = "、".join(BLK)
DELTAS = "、".join(f"{k} {LR['blocks'][k]['accuracy_delta']:+.4f}" for k in BLK)
_sig_txt = "、".join(SIG) if SIG else "无一"

_up = [k for k in BLK if LR["blocks"][k]["accuracy_delta"] > 0]
_dn = [k for k in BLK if LR["blocks"][k]["accuracy_delta"] < 0]
if _up and _dn:
    _dir_txt = f"这{CNT}个格子方向**不一致**：{DELTAS}"
elif _up or _dn:
    _dir_txt = f"这{CNT}个格子方向一致（都{'为正' if _up else '为负'}：{DELTAS}）"
else:
    _dir_txt = f"这{CNT}个格子的差都是 0（{DELTAS}）"

if not SIG:
    _lr_claim = f"**在测过的{CNT}个组大小（{BLOCKLIST}）上一个都没分得开**"
elif len(SIG) == len(BLK):
    _lr_claim = f"**在 {BLOCKLIST} 这{CNT}个组大小上都分得开**"
else:
    _lr_claim = (f"**只在 {_sig_txt} 上分得开**（测了{CNT}个组大小：{BLOCKLIST}）")

_f5 = "是平的" if FLAT5 else "不平"
_f1 = "是平的" if not EXCL else "不平"
_flat_claim = f"在 lr 5e-5 上{_f5}、在 lr 1e-5 上{_f1}"

_shown = [b for b in VD["contrasts"] if b.get("label")]
_cmp_txt = "；".join(
    f"{b['label']} Δ = {b['delta']:+.4f}、`p` = {b['p']:.5f}" for b in _shown)

BLOCK4 = f"""**（4）组大小：换一个学习率再走一遍同一级台阶（GSPO、sequence、lr 1e-5）。**
（2）的台阶只在 lr 5e-5 上走过一遍。run 41 把同一级台阶放到 lr 1e-5 上：G=8 → G=32，
Δ = **{D:+.4f}**，区间 **[{LO:+.4f}, {HI:+.4f}]**、McNemar `p` = {P}（{N01}/{N10} 行）
—— """
BLOCK4 += ("**排除 0**，即在这个学习率上组大小分得开。" if EXCL else
           "**包含 0**，与 lr 5e-5 上同一级台阶的结论一致。")
BLOCK4 += f"""
记录把两档合并成 `group_size_verdict`：{_cmp_txt}，其中显著的 **{VD['significant_count']}** 条。
"""
BLOCK4 += f"""这条与（2）一样**不是单变量的**：G=8 → G=32 的同时有效批次从 64 改到 1024
（组大小的平方）。所以它平的，与「组大小无效」仍然不是同一句话；记录自己的
`caveat` 也写明只有两个组大小、且非单调的效应看不出来。

**学习率效应在每个测量到的组大小上都被重新测了一遍。** run 41 落地后
`lr_effect_both_group_sizes` 有了 {CNT} 个格子（{BLOCKLIST}）；{_dir_txt}；
记录判定显著的为 **{_sig_txt}**。记录自己的读法是
"{LR['reading']}"；它同时写明这**不**触及论文的主张（`does_not_settle` 字段），
因为论文要的是学习率效应**随 IS 层级**的变化。

"""

TAIL_OLD = """**四条合起来：** 本设计里能分得开的仍然只有学习率（5.2.5）—— 而且 run 45 落地后，
学习率在 GSPO 内部 G=4 与 G=8 两个组大小上都分得开（（3））；论文主张的那条轴在 G=4 上
单变量也分不开（（1））；GSPO 内部沿组大小是平的（（2））。**这四条都是「在本模型规模
与本次预算下」的陈述**，不是「论文的机制不存在」—— 一个学习率、一个 252 行基准、150
步，能支持的最强结论就到这里。"""

TAIL_NEW = f"""**合起来（（1）–（4）与 `lr_effect_both_group_sizes`）：** 本设计里能分得开的仍然只有学习率（5.2.5）—— 而且 run 41 与
run 45 落地后，学习率在 GSPO 内部{_lr_claim}（（3）、（4）与
`lr_effect_both_group_sizes`）；论文主张的那条轴（IS 层级）在 G=4 上单变量也分不开（（1））；
GSPO 内部沿组大小{_flat_claim}（（2）lr 5e-5：Δ {D5:+.4f}、`p` = {P5:.5f}；
（4）lr 1e-5：Δ {D:+.4f}、`p` = {P:.5f}）。**这些都是「在本模型规模与本次预算下」的陈述**，
不是「论文的机制不存在」—— 一个学习率、一个 252 行基准、150 步，能支持的最强结论就到这里。"""

s = DOC.read_text(encoding="utf-8")
shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch93")

TAIL_ANCHOR = "**四条合起来：**"
if s.count(TAIL_ANCHOR) != 1:
    sys.exit(f"the '四条合起来' anchor appears {s.count(TAIL_ANCHOR)} times, expected 1 "
             f"(patch92b writes it when run 45 lands); nothing written")
if s.count(TAIL_OLD) != 1:
    sys.exit(f"the paragraph to replace appears {s.count(TAIL_OLD)} times, expected 1; "
             f"nothing written -- it is patch92b's output, so patch92b has to have run")

before = s.count("**")
s = s.replace(TAIL_OLD, BLOCK4 + TAIL_NEW, 1)

# The section's heading says what it now shows. patch92b left it as "单变量对照：IS 层级
# 分不开，学习率分得开". After run 41 the group-size axis has been walked at both learning
# rates, so the heading names that -- but only if it is still flat, which is why the
# heading is chosen from FLAT5/EXCL rather than written once.
H_OLD2 = "#### 5.2.6 单变量对照：IS 层级分不开，学习率分得开"
H_NEW2 = ("#### 5.2.6 单变量对照：IS 层级分不开，学习率分得开，组大小两个 lr 上都平"
          if FLAT5 and not EXCL else
          "#### 5.2.6 单变量对照：IS 层级分不开，学习率分得开，组大小在 lr 1e-5 上分得开")
if s.count(H_OLD2) == 1:
    s = s.replace(H_OLD2, H_NEW2, 1)
    print(f"heading updated: {H_NEW2}")
else:
    print(f"note: the 5.2.6 heading is not in the patch92b form "
          f"({s.count(H_OLD2)} hits); left alone")

after = s.count("**")
if (after - before) % 2:
    sys.exit(f"bold markers went {before} -> {after}, an odd change; nothing written")

DOC.write_text(s, encoding="utf-8")
print(f"5.2.6: the （4） block and the summary written "
      f"({len(s.splitlines())} lines, bold {before} -> {after})")
print(f"  group size @ lr1e-5 : d={D:+.4f} CI=[{LO:+.4f},{HI:+.4f}] p={P} "
      f"excludes_zero={EXCL}")
print(f"  group size @ lr5e-5 : d={D5:+.4f} p={P5:.5f} flat={FLAT5}")
print(f"  lr effect at        : {BLOCKLIST} (significant: {_sig_txt}; "
      f"all={len(SIG) == len(BLK)})")
print(f"  direction           : {_dir_txt}")
print(f"  verdict             : {VD['reading']}")
