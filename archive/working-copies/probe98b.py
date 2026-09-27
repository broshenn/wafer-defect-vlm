# Does every anchor patch98 will look for exist exactly once in the live document,
# and does patch92b (land45's step 4) leave them alone? Booleans out, so this prints
# nothing that a terminal can mangle.
from pathlib import Path

s = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md").read_text(encoding="utf-8")

A_OLD = """六个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 GRPO lr1e-5（0.6197 对 0.6114），
成对检验下与 SFT 完全无法区分（Δ准确率 0.0000、`p` = 1.00）；"""

B_OLD = """诚实的表述是：**在本模型规模与本次预算下，RL 阶段没有产生可测的基准增益，
而且在高学习率档上产生了可测的基准损失**；"""

C_OLD = """而两个 lr 1e-5 的 run 两个类别都在
（`none` 0.343
与 0.462、
`Donut` 0.421
与 0.462）。"""

# What patch92b writes, which must not overlap any of the above.
P92B = ["**三条合起来：**", "**四条合起来：**",
        "#### 5.2.6 单变量对照：IS 层级分不开，学习率分得开",
        "**在 lr 5e-5 这一档上，唯一分得开的是学习率**"]

for label, anchor in (("A above-SFT", A_OLD), ("B verdict", B_OLD), ("C lr1e5 enum", C_OLD)):
    print("%-14s hits=%d" % (label, s.count(anchor)))

print("--- patch92b's own anchors (land45 step 4) ---")
for x in P92B:
    print("  hits=%d  %s" % (s.count(x), ascii(x)[:60]))

# Overlap test: none of patch98's anchors may contain, or be contained by, patch92b's.
overlap = [x for x in P92B if x in A_OLD or x in B_OLD or x in C_OLD
           or A_OLD in x or B_OLD in x or C_OLD in x]
print("overlap between patch92b and patch98 anchors:", overlap or "NONE")

# Where each anchor sits, as a line number, so the patch's print can be cross-checked.
for label, anchor in (("A", A_OLD), ("B", B_OLD), ("C", C_OLD)):
    i = s.find(anchor)
    print("%s at line %d" % (label, s[:i].count("\n") + 1 if i >= 0 else -1))
print("document lines:", len(s.splitlines()), " bold markers:", s.count("**"))
