"""Rebuild three tables from the records, with every run in them.

5.2.3's reward-std table has four columns and five RL runs exist; 5.2.5's two
tables have three and four. The missing column is not a cosmetic gap: 5.2.5 read
"全部 run 最高/最好/最差" off those tables, and twice the claim was false once
the fifth run existed, invisibly -- the table did not contain the run that
falsified it. That is item 9 of section 8 again: the numbers were right, the
quantifier's scope moved, and nothing in the artefact could show it.

So the tables are built here from the reports, with the extreme marked per row
rather than asserted in prose. Table A also fixes a second thing: the std table
bolded G=32's values unconditionally, which was a visual claim that they were
the lowest; the minimum is computed per row instead.

Claims that were checked and DID survive are left standing, with the check
stated, rather than being softened along with the ones that failed.
"""
import json
import sys
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
REPORTS = R / "outputs/reports"
DOC = R / "LIMITATIONS.md"

COLS = [("SFT", "qwen35_9b_adapter"),
        ("GRPO(G=4) lr1e-5", "qwen35_9b_grpo"),
        ("GRPO(G=4) lr5e-5", "qwen35_9b_grpo_lr5e5"),
        ("GSPO(G=8) lr1e-5", "gspo_lr1e5"),
        ("GSPO(G=8) lr5e-5", "qwen35_9b_gspo_v1"),
        ("GSPO(G=32) lr5e-5", "qwen35_9b_gspo_g32")]
RL = [c for c in COLS if c[0] != "SFT"]
rep = {n: json.loads((REPORTS / f"{t}__report.json").read_text(encoding="utf-8"))
       for n, t in COLS}
# The first GRPO run's record is grpo_train_result.json, not <tag>_train_result.json,
# because its launcher shared a tag with the GSPO run that overwrote it.
TRAIN_FILE = {"GRPO(G=4) lr1e-5": "grpo_train_result.json"}
train = {n: json.loads((REPORTS / TRAIN_FILE.get(
    n, f"{t}_train_result.json")).read_text(encoding="utf-8")) for n, t in RL}
names = [n for n, _ in COLS]
rl_names = [n for n, _ in RL]


def field(n, path):
    d = rep[n]
    for k in path:
        d = d[k]
    return d


# ---------------------------------------------------------------- table A (5.2.3)
REWARDS = ["WaferClass", "WaferFormat", "WaferRadial", "WaferClock"]
rowsA = []
for r in REWARDS:
    v = {n: (train[n].get("reward_signal") or {}).get(r, {}).get("mean_std_across_steps")
         for n in rl_names}
    known = {k: x for k, x in v.items() if x is not None}
    lo = min(known, key=lambda k: known[k]) if known else None
    cells = []
    for n in rl_names:
        s = "n/a" if v[n] is None else f"{v[n]:.4f}"
        cells.append(f"**{s}**" if n == lo else s)
    rowsA.append(f"| `{r}` | " + " | ".join(cells) + " |")
idle = {n: train[n]["mean_frac_reward_zero_std"] for n in rl_names}
lo_idle = min(idle, key=lambda k: idle[k])
rowsA.append("| **空转步占比** | " + " | ".join(
    f"**{idle[n]:.2%}**" if n == lo_idle else f"{idle[n]:.2%}" for n in rl_names) + " |")
if len(rowsA) != len(REWARDS) + 1:
    sys.exit("table A row count wrong")
TABLE_A = ("| 奖励的组内标准差（150 步均值） | " + " | ".join(rl_names) + " |\n"
           "| --- | " + " | ".join("---" for _ in rl_names) + " |\n"
           + "\n".join(rowsA) + "\n")

OLD_A = """| 奖励的组内标准差（150 步均值） | GRPO (G=4, lr1e-5) | GSPO (G=8, lr5e-5) | GSPO (G=8, lr1e-5) | GSPO (G=32, lr5e-5) |
| --- | --- | --- | --- | --- |
| `WaferClass` | 0.0803 | 0.0855 | 0.0912 | **0.0517** |
| `WaferFormat` | 0.0005 | 0.0132 | 0.0030 | 0.0219 |
| `WaferRadial` | 0.1180 | 0.0528 | 0.1256 | **0.0346** |
| `WaferClock` | 0.2244 | 0.2306 | 0.2526 | **0.1435** |
| **空转步占比** | 36.00% | 38.00% | **15.33%** | 42.00% |
"""

# ---------------------------------------------------------------- table B (5.2.5)
PROFILE = [("`radial_zone`", ("structured", "field_accuracy", "radial_zone"), True),
           ("时钟 MAE", ("structured", "clock_circular_mae_sectors"), False),
           ("尺寸 MAE", ("structured", "size_mae_r"), False),
           ("caption must-hit", ("caption", "must_hit_rate"), True)]
# best = the endpoint that is good; worst = the other one. Size MAE's extreme is
# its WORST value, so collapsing these two into one "extreme" per row hides the
# direction -- which is how the guard below first failed.
rowsB, best_of, worst_of, vals_of = [], {}, {}, {}
for label, path, higher in PROFILE:
    v = {n: field(n, path) for n in names}
    best = max(v, key=lambda n: v[n]) if higher else min(v, key=lambda n: v[n])
    worst = min(v, key=lambda n: v[n]) if higher else max(v, key=lambda n: v[n])
    best_of[label], worst_of[label], vals_of[label] = best, worst, v
    cells = [f"**{v[n]:.4f}**" if n in (best, worst) else f"{v[n]:.4f}" for n in names]
    rowsB.append(f"| {label} | " + " | ".join(cells) + " |")
TABLE_B = ("| 指标 | " + " | ".join(names) + " |\n"
           "| --- | " + " | ".join("---" for _ in names) + " |\n"
           + "\n".join(rowsB) + "\n")

G32 = "GSPO(G=32) lr5e-5"
_r_over, _c_over = vals_of["`radial_zone`"], vals_of["时钟 MAE"]
_s_over, _m_over = vals_of["尺寸 MAE"], vals_of["caption must-hit"]
if best_of["`radial_zone`"] != G32 or worst_of["尺寸 MAE"] != G32:
    sys.exit("G=32 is no longer the radial best / size worst; rewrite the prose below")
if best_of["时钟 MAE"] == G32 or best_of["caption must-hit"] == G32:
    sys.exit("a claim previously false is now true; rewrite the prose below")
if best_of["时钟 MAE"] != "SFT" or best_of["caption must-hit"] != "SFT":
    sys.exit("the clock/caption best changed; rewrite the prose below")
PROSE_B = f"""**这张表初稿只有三列，因此「全部 run 最高/最好/最差」四句话里有两句是错的。**
它们是按四个 run 写的，第五个 run（GRPO(G=4) lr5e-5）落地后失效 ——
而表里没有这一列，读表的人看不见它已经失效（第 8 节第 9 条那一类：数字都对，
「全部 run」这个量词的范围变了，而产物本身显示不出来）。补全六列后逐条核对：

- `radial_zone`：**成立**。G=32 的 {_r_over[G32]:.4f} 在六个 run 里最高，
  是 SFT 的 {_r_over[G32] / _r_over['SFT']:.2f} 倍。
- 时钟 MAE：**不成立**。最低的是 {best_of['时钟 MAE']} 的 {_c_over[best_of['时钟 MAE']]:.4f}，
  G=32 的 {_c_over[G32]:.4f} 只是第二。
- 尺寸 MAE：**成立**。G=32 的 {_s_over[G32]:.4f} 在六个 run 里最高，
  是 SFT 的 {_s_over[G32] / _s_over['SFT']:.2f} 倍。
- caption must-hit：**不成立**。最低的是 {best_of['caption must-hit']} 的
  {_m_over[best_of['caption must-hit']]:.4f}，G=32 的 {_m_over[G32]:.4f} 是第二低。

两处失效的不是数字（每一个都能回溯到记录），是**量词**。
"""

OLD_B = """| 指标 | SFT | GSPO G=8 lr5e-5 | GSPO G=32 lr5e-5 |
| --- | --- | --- | --- |
| `radial_zone` | 0.2640 | 0.3056 | **0.4603** ← 全部 run 最高，SFT 的 1.74 倍 |
| 时钟 MAE | 1.6847 | 1.9519 | **1.6860** ← 全部 run 最好 |
| 尺寸 MAE | 0.4999 | 0.6927 | **1.6509** ← 全部 run 最差，SFT 的 3.3 倍 |
| caption must-hit | 0.7540 | 0.6627 | **0.5714** ← 全部 run 最差 |
"""

# ---------------------------------------------------------------- table C (5.2.5)
CLASSES = ["Donut", "none", "Edge_Ring", "Scratch"]
rowsC, zeros = [], []
for c in CLASSES:
    v = {n: rep[n]["classification"]["per_class"][c]["f1-score"] for n in names}
    ext = max(v, key=lambda n: v[n])
    cells = []
    for n in names:
        s = f"{v[n]:.3f}"
        if n == ext:
            s = f"**{s}**"
        cells.append(s)
        if v[n] == 0.0:
            zeros.append((c, n))
    rowsC.append(f"| {c} | " + " | ".join(cells) + " |")
TABLE_C = ("| 类别（F1）| " + " | ".join(names) + " |\n"
           "| --- | " + " | ".join("---" for _ in names) + " |\n"
           + "\n".join(rowsC) + "\n")

if sorted(zeros) != sorted([("Donut", G32), ("none", "GSPO(G=8) lr5e-5")]):
    sys.exit(f"the zero-collapse pattern changed: {zeros}; rewrite the prose below")
_scr = {n: rep[n]["classification"]["per_class"]["Scratch"]["f1-score"] for n in names}
_scr_ext = max(_scr, key=lambda n: _scr[n])
if _scr_ext != G32:
    sys.exit("G=32 no longer tops Scratch; rewrite the prose below")

PROSE_C = f"""在五个 RL run 里，**恰好只有两个有类别归零**：GSPO(G=8) lr5e-5 丢 `none`、
GSPO(G=32) lr5e-5 丢 `Donut`，而两个 lr 1e-5 的 run 两个类别都在
（`none` {rep['GSPO(G=8) lr1e-5']['classification']['per_class']['none']['f1-score']:.3f}
与 {rep['GRPO(G=4) lr1e-5']['classification']['per_class']['none']['f1-score']:.3f}、
`Donut` {rep['GSPO(G=8) lr1e-5']['classification']['per_class']['Donut']['f1-score']:.3f}
与 {rep['GRPO(G=4) lr1e-5']['classification']['per_class']['Donut']['f1-score']:.3f}）。

**但 GRPO(G=4) lr5e-5 一个类别都没丢**（SFT 规模的两个类别都保住了），
尽管它的准确率同样显著下降（5.2.2）。所以「整类塌陷」**不是 5e-5 的通用后果，
而是 GSPO 在 5e-5 下才出现的现象** —— 这条限制初稿没有，因为它没有第五列。
「不确定性」与「只在 GSPO 上出现」合起来，比初稿的「策略被推进退化区」更窄也更准确：
后者作为通用解释已被 GRPO lr5e-5 那一列否掉。
另一方面 G=32 在 `Scratch` 上拿到全部 run 最高的 {_scr[G32]:.3f}
（SFT {_scr['SFT']:.3f}），说明它也不是全面退化。
"""

OLD_C = """| 类别 | SFT | GSPO G=8 lr5e-5 | GSPO G=8 lr1e-5 | GSPO G=32 lr5e-5 |
| --- | --- | --- | --- | --- |
| Donut | 0.462 | 0.378 | 0.421 | **0.000** |
| none | 0.400 | **0.000** | 0.343 | 0.412 |
| Edge_Ring | 0.870 | 0.778 | 0.892 | 0.524 |
| Scratch | 0.286 | 0.286 | 0.333 | **0.462** |

G=8 丢 `none`，G=32 丢 `Donut`，而 lr 1e-5 两个都恢复。
**整类塌陷在 lr 5e-5 下不是一个稳定的偏差，而是一个不稳定的偏差 —— 换个组大小
就换一个牺牲品。** 这比「分数低」更能支持「策略被推进退化区」的诊断，
因为它同时排除了「这个类别本身就难」这一解释：同类别在 lr 1e-5 下是学得到的。
另一方面 G=32 在 `Scratch` 上拿到全部 run 最高的 0.462，说明它也不是全面退化。
"""

s = DOC.read_text(encoding="utf-8")
for old, new, tag in ((OLD_A, TABLE_A, "5.2.3 std table"),
                      (OLD_B, TABLE_B + "\n" + PROSE_B, "5.2.5 profile table"),
                      (OLD_C, TABLE_C + "\n" + PROSE_C, "5.2.5 class table")):
    if s.count(old) != 1:
        sys.exit(f"FAILED: {tag} appears {s.count(old)} times")
    s = s.replace(old, new)

# "四组" counts were right when four runs existed; there are five.
s = s.replace("42.00%（63/150）是四组里最差的", "42.00%（63/150）是五组里最差的")
s = s.replace("在 4 个奖励里有 3 个是四组最低", "在 4 个奖励里有 3 个是五组最低")

DOC.write_text(s, encoding="utf-8")
print("three tables rebuilt from records, all runs included")
print(f"lines: {s.count(chr(10)) + 1}")
