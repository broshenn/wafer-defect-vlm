"""Apply the verified corrections to LIMITATIONS.md.

Every anchor below was printed from the live file and every replacement number was
re-derived from a record (see tools/derive2.py). Each anchor must appear exactly
once: a silently non-unique anchor edits the wrong place, which is a failure mode
this project has already hit twice.

Bold markers ("**") are validated per edit as well as in total. A whole-file parity
check is necessary but not sufficient -- it passes when one replacement drops a
marker and another adds one. The per-edit table makes the culprit nameable.
"""
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")

EDITS = [
    # ------------------------------------------------------------ A. the delta
    # The sentence welded two coincidentally-equal numbers from different metric
    # families, and attached the cross_algorithm_paired row to the wrong pair.
    ("A-g32-two-contrasts",
     """更正后的事实比初稿更干净：**在 lr 5e-5 下，G=32 相对两侧的对照都是 ±0.0278、
`p` 都是 0.41**（对 G=8 是 +0.0278，对 GRPO(G=4) 是 −0.0278，
见第 9 节 `paired_significance.json` 的 `group_size_paired` 与 `cross_algorithm_paired`）。
即**在 lr 5e-5 这一档上，算法与组大小两两都分不开，唯一分得开的是学习率** ——
这比「组大小有效但不足」更强，也更贴合数据。""",
     """**上面这一段此前写错了一处，这里更正。** 原文说「在 lr 5e-5 下，G=32 相对两侧的
对照都是 ±0.0278、`p` 都是 0.41」—— 核对记录后**只有一侧成立**：

- **G=32 对 G=8（同为 GSPO、同为 lr 5e-5）**：准确率 Δ = **+0.0278**，
  95% CI [−0.0278, +0.0833]（含 0），`p` = 0.41010 —— 不显著。这一侧属实
  （`group_size_paired`）。
- **G=32 对 GRPO(G=4)（同为 lr 5e-5）**：原文把 −0.0278 记在这一侧，但
  `cross_algorithm_paired.lr5e-5` 的两个成员是 **GSPO(G=8) 与 GRPO(G=4)**，
  不是 G=32 —— 那个 −0.0278 是「G=8 对 G=4」的准确率差（见 5.2.2）。G=32 与
  GRPO(G=4) 的准确率**完全相同，都是 0.567460（Δ = 0.0000）**；成对检验见
  第 9 节新增的 `paper_setting_paired`：Δ = 0.0000、95% CI [−0.0635, +0.0635]、
  `p` = 1.00。**但两者在 252 行里有 64 行判断不同，恰好 32 对 32 互补** ——
  「聚合值相等」与「逐行相同」是两件事，这正是必须真去算而不是从相等推出来的
  理由（若按聚合值推断 `p`，会漏掉这 64 行的互换）。macro-F1 上 G=32 低
  0.0132（0.5403 对 0.5535），该差值没有配对检验。

错误的成因值得记下来：两个数**在数值上恰好互为相反数，却来自不同的度量与不同的
对照组**（`group_size_paired` 的 +0.0278 是准确率、对照组是 G=8→G=32；被误引的
−0.0278 也是准确率、对照组却是 G=8→G=4）。成员写错之后，句子就从「组大小与算法
各有一侧对照」变成了「G=32 两侧都是 ±0.0278」。

更正后仍然成立的是（比初稿更弱）：**在 lr 5e-5 这一档上，唯一分得开的是学习率** ——
G=32 对 G=8 分不开（`p` = 0.41），对 GRPO(G=4) 的**准确率完全重合**。但后一个对照
一次改了组大小（4→32）、IS 层级（token→sequence）与有效批次（16→1024）三样，
**它既不能支持也不能否定论文的主张**（第 8 节第 11 条）。这也正是要把
`paper_setting_paired` 写进记录的原因：论文的中心主张就架在这一对照上，
而在此之前**项目从未对它做过配对检验**。"""),

    # ------------------------------------------- B. group size was never alone
    ("B-idle-direction-confound",
     """lr 5e-5 下 G=4 是 32.67%、G=8 是 38.00%、G=32 是 42.00%，即**组越大空转步越多**；""",
     """lr 5e-5 下 G=4 是 32.67%、G=8 是 38.00%、G=32 是 42.00%，即**组越大空转步越多**
（但这一跨度里 G=4→G=8 同时换了 IS 层级，且每一步都同时改了有效批次
16→64→1024，所以「组大小」在这里同样不是单变量 —— 第 8 节第 11 条。
下面只把它当作一个方向性参照，不当作识别）；"""),

    ("C-queue41-not-single-variable",
     """（G=32 + lr 1e-5：同算法、同学习率、只改组大小）在队列 41 里跑。""",
     """（G=32 + lr 1e-5：同算法、同学习率，组大小由 8 改到 32 —— 但有效批次随之从
64 改到 1024，所以它仍不是单变量）在队列 41 里跑。"""),

    # ------------------------------------------ D. what the collapse belongs to
    ("D-collapse-attribution",
     """**但 GRPO(G=4) lr5e-5 一个类别都没丢**（SFT 规模的两个类别都保住了），
尽管它的准确率同样显著下降（5.2.2）。所以「整类塌陷」**不是 5e-5 的通用后果，
而是 GSPO 在 5e-5 下才出现的现象** —— 这条限制初稿没有，因为它没有第五列。""",
     """**但 GRPO(G=4) lr5e-5 一个类别都没丢**（SFT 规模的两个类别都保住了），
尽管它的准确率同样显著下降（5.2.2）。所以「整类塌陷」**不是 5e-5 的通用后果，
而是在 lr 5e-5 的这两格 GSPO（G=8 与 G=32）上才出现的现象** —— 这条限制初稿
没有，因为它没有第五列。**归因还要再退一步**：这两格与 GRPO(G=4) lr5e-5 差的不止
是算法，组大小（8/32 对 4）与有效批次（64/1024 对 16）也同时变了，所以只能说
**与该格的三样之一或之几有关，不能单独归给 IS 层级**（第 8 节第 11 条）。"""),

    # ---------------------------------------------------- E. the class drop rates
    ("E-base-drop-rates",
     """Base 更极端：**丢弃 251/252 = 99.6%，仅剩 1 行计分**，且九个类别的丢弃率全是 1.00 ——""",
     """Base 更极端：**丢弃 251/252 = 99.6%，仅剩 1 行计分**。初稿写「九个类别的丢弃率
全是 1.00」，这句既**不成立**也**自相矛盾**：九个里八个确为 1.00（`Center`/`Donut`/
`Edge_Loc`/`Edge_Ring`/`Loc`/`Random`/`Scratch` 各 30/30，`none` 26/26），
**只有 `Near_full` 是 15/16 = 0.9375**；而若九个全是 1.00，就不该剩下任何一行。
支撑数取自冻结基准 `benchmarks/wafer_bench_v1/classification.jsonl` 的自身标签，
丢弃数取自 `clock_parse_audit.json` 的 `dropped_defect_types`（复核脚本
`tools/verify_droprate.py`）。由这两个事实可直接推出：**Base 唯一计分的那一行，
正是 `Near_full` 的幸存者** ——"""),

    # ---------------------------------------------------------------- F/G. the sum
    ("F-idle-sum-inline",
     """五个 run 的全部 271 个空转步均满足，""",
     """五个 run 的全部 246 个空转步均满足（逐 run 实测为 54+49+23+57+63；初稿写 271，
是一个加错了的旧数。逐 run 计数见 `kl_length_confound.json` 的
`idle_steps_tested`），"""),

    ("G-idle-sum-inventory",
     """五个 run 全部 271 个空转步上的逐值核对。""",
     """五个 run 全部 246 个空转步上的逐值核对（逐 run 计数见该记录的
   `idle_steps_tested`；文档他处曾写 271，已更正）。"""),

    # ---------------------------------------------- H/I. quantifiers about run count
    ("H-reproduction-count",
     """（a）解析必须**复现**项目自身 scorer 写下的准确率，六个 run 逐一相等，否则直接退出；""",
     """（a）解析必须**复现**项目自身 scorer 写下的准确率，**记录在案的 run 逐一相等**
（该清单随新 run 增长，写此句时为 7 个：BASE、SFT 与 5 个 RL run；复核时以记录里
`runs_included` 的长度为准，不要沿用这里写的数），否则直接退出；"""),

    ("I-marginal-ci-quantifier",
     """该规则比较两个**边际**置信区间；但六次 run 回答的是""",
     """该规则比较两个**边际**置信区间；但各次 run 回答的是"""),

    ("J-published-columns-scope",
     """是因为改动评测定义会使已公布的 Base/SFT/GRPO 三列不再可比；""",
     """是因为改动评测定义会使当时已公布的 Base/SFT/GRPO 三列不再可比；"""),

    # ---------------------------------- K. warn on the table's count before it rots
    ("K-column-count-warning",
     """极值按各行的方向标注（加粗 = 该行最好的一档）：""",
     """极值按各行的方向标注（加粗 = 该行最好的一档）。
（此表的列数是随时间增长的：写此句时为六列。在任何新 run 落地后，下面四句里的
「六个 run」都必须重核而不是沿用 —— 这正是本条要防的那个错误。）"""),
]

s = P.read_text(encoding="utf-8")
before_lines = s.count("\n") + 1
print(f"file: {before_lines} lines, '**' count {s.count('**')} "
      f"(parity {'even' if s.count('**') % 2 == 0 else 'ODD'})")
print()
print(f"{'edit':<28} {'old **':>7} {'new **':>7} {'delta':>6}  count")
print("-" * 62)

ok = True
total_delta = 0
for tag, old, new in EDITS:
    n = s.count(old)
    o, w = old.count("**"), new.count("**")
    total_delta += w - o
    flag = "" if n == 1 else "  <-- ANCHOR NOT UNIQUE"
    if n != 1:
        ok = False
    parity = "" if (w - o) % 2 == 0 else "  <-- flips parity"
    print(f"{tag:<28} {o:>7} {w:>7} {w - o:>+6}  {n}{flag}{parity}")

print("-" * 62)
print(f"total bold delta: {total_delta:+d}")
after_parity = (s.count("**") + total_delta) % 2
print(f"resulting parity: {'even' if after_parity == 0 else 'ODD'}")
if not ok:
    sys.exit("\nrefusing to write: an anchor is not unique")
if after_parity != 0:
    sys.exit("\nrefusing to write: bold markers would be unbalanced. A per-edit "
             "odd delta is fine only if another edit compensates.")

for tag, old, new in EDITS:
    s = s.replace(old, new)

after_lines = s.count("\n") + 1
P.write_text(s, encoding="utf-8")
print(f"\nLIMITATIONS.md: {len(EDITS)} edits applied, {before_lines} -> {after_lines} lines")
