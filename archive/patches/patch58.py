"""Fix two defects the rebuilt 5.2.5 table introduced.

1. Bolding marked both the best and the worst cell in each row. On a row where
   the direction is "lower is better", bold-then-bold gives the reader no way to
   tell which is which. Bold the best cell only and label the direction.

2. The caption bullet read "最低的是 SFT 的 0.7540". Higher is better for
   must-hit, so the lowest value is the WORST one, and it is not SFT's -- it is
   GRPO(G=4) lr5e-5's 0.5516. The guard passed because it tested which run held
   the *best* value, while the sentence quoted the *lowest*: a check and a claim
   about different quantities, which is how a guard can be satisfied and the
   sentence still wrong.

Also fixes the class-table lead-in, which says "两个 lr 5e-5 的 run" when only the
two GSPO ones collapse a class; GRPO lr5e-5 collapses none.
"""
import json
import sys
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
REPORTS = R / "outputs/reports"
DOC = R / "LIMITATIONS.md"

COLS = [("SFT", "qwen35_9b_adapter"), ("GRPO(G=4) lr1e-5", "qwen35_9b_grpo"),
        ("GRPO(G=4) lr5e-5", "qwen35_9b_grpo_lr5e5"), ("GSPO(G=8) lr1e-5", "gspo_lr1e5"),
        ("GSPO(G=8) lr5e-5", "qwen35_9b_gspo_v1"), ("GSPO(G=32) lr5e-5", "qwen35_9b_gspo_g32")]
rep = {n: json.loads((REPORTS / f"{t}__report.json").read_text(encoding="utf-8"))
       for n, t in COLS}
names = [n for n, _ in COLS]
G32 = "GSPO(G=32) lr5e-5"

PROFILE = [("`radial_zone`", "越高越好", ("structured", "field_accuracy", "radial_zone"), True),
           ("时钟 MAE", "越低越好", ("structured", "clock_circular_mae_sectors"), False),
           ("尺寸 MAE", "越低越好", ("structured", "size_mae_r"), False),
           ("caption must-hit", "越高越好", ("caption", "must_hit_rate"), True)]
rows, best_of, worst_of, vals_of = [], {}, {}, {}
for label, direction, path, higher in PROFILE:
    v = {n: rep[n][path[0]][path[1]] if len(path) == 2
         else rep[n][path[0]][path[1]][path[2]] for n in names}
    best = max(v, key=lambda n: v[n]) if higher else min(v, key=lambda n: v[n])
    worst = min(v, key=lambda n: v[n]) if higher else max(v, key=lambda n: v[n])
    best_of[label], worst_of[label], vals_of[label] = best, worst, v
    rows.append(f"| {label}（{direction}）| " + " | ".join(
        f"**{v[n]:.4f}**" if n == best else f"{v[n]:.4f}" for n in names) + " |")

if best_of["`radial_zone`"] != G32 or worst_of["尺寸 MAE"] != G32:
    sys.exit("G=32 is no longer radial-best / size-worst; prose below needs rewriting")
if best_of["时钟 MAE"] != "SFT":
    sys.exit(f"clock MAE best is now {best_of['时钟 MAE']}, not SFT")
_caption_worst = worst_of["caption must-hit"]
if _caption_worst == G32:
    sys.exit("G=32 is now the caption worst; the corrected claim would be wrong")

TABLE = ("| 指标 | " + " | ".join(names) + " |\n"
         "| --- | " + " | ".join("---" for _ in names) + " |\n"
         + "\n".join(rows) + "\n")

_c, _s, _m = vals_of["时钟 MAE"], vals_of["尺寸 MAE"], vals_of["caption must-hit"]
PROSE = f"""**这张表初稿只有三列，因此「全部 run 最高/最好/最差」四句话里有两句是错的。**
它们是按四个 run 写的，第五个 run（GRPO(G=4) lr5e-5）落地后失效 ——
而表里没有这一列，读表的人看不见它已经失效（第 8 节第 9 条那一类：数字都对，
「全部 run」这个量词的范围变了，而产物本身显示不出来）。补全六列后逐条核对，
极值按各行的方向标注（加粗 = 该行最好的一档）：

- `radial_zone`：**成立**。G=32 的 {vals_of['`radial_zone`'][G32]:.4f} 在六个 run 里最高，
  是 SFT 的 {vals_of['`radial_zone`'][G32] / vals_of['`radial_zone`']['SFT']:.2f} 倍。
- 时钟 MAE：**不成立**。最低的是 {best_of['时钟 MAE']} 的 {_c[best_of['时钟 MAE']]:.4f}，
  G=32 的 {_c[G32]:.4f} 只是第二。
- 尺寸 MAE：**成立**。G=32 的 {_s[G32]:.4f} 在六个 run 里最高（即最差），
  是 SFT 的 {_s[G32] / _s['SFT']:.2f} 倍。
- caption must-hit：**不成立**。最低的是 {_caption_worst} 的 {_m[_caption_worst]:.4f}，
  G=32 的 {_m[G32]:.4f} 是第二低。

两处失效的不是数字（每一个都能回溯到记录），是**量词**。
本节更正时还犯过一次方向错误并已修掉：第一次改写把 caption 那行写成了
「最低的是 SFT 的 0.7540」—— 该指标越高越好，最低值是最**差**的一档，而 SFT 是最高。
守卫当时通过了，因为它查的是「哪一档最好」，句子引用的却是「最低值」：
**校验与陈述落在两个不同的量上，是守卫能通过而句子仍然错的常见方式。**
"""

s = DOC.read_text(encoding="utf-8")

start = s.index("| 指标 | SFT | GRPO(G=4) lr1e-5")
end = s.index("按类别看更清楚")
old_block = s[start:end]
if "两处失效的不是数字" not in old_block:
    sys.exit("REFUSING: the 5.2.5 profile block does not look like the one to replace")
s = s[:start] + TABLE + "\n" + PROSE + "\n" + s[end:]

OLD_LEAD = "按类别看更清楚 —— **两个 lr 5e-5 的 run 各自丢掉了一整个类别，丢的却不是同一个：**"
NEW_LEAD = ("按类别看更清楚 —— **两个 GSPO lr 5e-5 的 run 各自丢掉了一整个类别，"
            "丢的却不是同一个**（同学习率的 GRPO 一个都没丢，见下）：")
if s.count(OLD_LEAD) != 1:
    sys.exit("class-table lead-in not found")
s = s.replace(OLD_LEAD, NEW_LEAD)

DOC.write_text(s, encoding="utf-8")
print("5.2.5: direction-aware bolding, caption direction fixed, lead-in narrowed")
print(f"lines: {s.count(chr(10)) + 1}")
