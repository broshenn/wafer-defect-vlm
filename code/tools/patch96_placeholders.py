"""Two sentences in LIMITATIONS.md were shipped with their format placeholders in.

Section 5.2.3 and section 5.2.5 both carry a sentence that was written as a template
-- `Δ准确率 {il_delta}、置信区间 {il_ci} 包含 0（`p` = {il_p}）` -- and the braces were
never filled in. A reader sees the arithmetic of a brace, not a number.

This is a new mechanism for the defect this document keeps recording (section 8):
every other instance was a number that was wrong, or right-but-about-a-run-set-that-
had-changed. Here there is no number at all, and the surrounding sentence reads
correctly -- "Δ准确率 {il_delta}" parses as a sentence about a quantity, so nothing
about it looks unfinished until you try to read the value. It is the same failure
the `round(p, 10)` -> 0 bug had (section 5.2.2): an artefact that cannot be told
apart from a correct one without checking.

The values are read from outputs/reports/paired_significance.json, which is where
5.2.6 already quotes them, so the two sections cannot disagree. Formatting follows
5.2.6: four decimals for a delta, a signed interval, and the p-value at six
significant figures.

The patch refuses to write if any placeholder survives, or if a value it would
write differs from the one 5.2.6 already prints -- that check is the point, because
a substitution that silently used a different contrast's number would look exactly
like this fix.
"""
import json
import sys
from pathlib import Path

DOC = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
PS = Path("/root/autodl-fs/wafer-vlm/outputs/reports/paired_significance.json")

ps = json.loads(PS.read_text(encoding="utf-8"))
il = ps["is_level_paired"]
rungs = ps["group_size_series_lr5e5"]["rungs"]

IL_D = f"{il['accuracy_delta_sequence_minus_token']:+.4f}"
IL_LO, IL_HI = il["delta_ci95_rows"]
IL_CI = f"[{IL_LO:+.4f}, {IL_HI:+.4f}]"
IL_P = f"{il['mcnemar']['p_exact_two_sided']:.6g}"
R1 = rungs["G=4 -> G=8"]
R2 = rungs["G=8 -> G=32"]
R1D = f"{R1['accuracy_delta']:+.4f}"
R1P = f"{R1['mcnemar']['p_exact_two_sided']:.5f}"
R2D = f"{R2['accuracy_delta']:+.4f}"
R2P = f"{R2['mcnemar']['p_exact_two_sided']:.5f}"

s = DOC.read_text(encoding="utf-8")
n_edits = 0


def sub(old, new, why):
    global s, n_edits
    n = s.count(old)
    if n != 1:
        sys.exit(f"anchor for '{why}' appears {n} times, expected 1; nothing written")
    s = s.replace(old, new, 1)
    n_edits += 1


# ------------------------------------------------------------ 5.2.3
sub("""{il_delta}、置信区间 {il_ci} 包含 0（`p` = {il_p}，见 5.2.6）""",
    f"""**{IL_D}**、置信区间 **{IL_CI}** 包含 0（`p` = {IL_P}，见 5.2.6）""",
    "5.2.3: the delta, interval and p of the single-variable IS pair")

# ------------------------------------------------------------ 5.2.5
sub("""（G=4、同学习率、同有效批次，只换 IS 层级：Δ准确率 {il_delta}、置信区间 {il_ci}
包含 0、`p` = {il_p}），而 GSPO 内部沿组大小的三点连线也是平的
（G=4→G=8 Δ {r1d:+.4f}、`p` = {r1p}；G=8→G=32 Δ {r2d:+.4f}、`p` = {r2p}，见 5.2.6）——""",
    f"""（G=4、同学习率、同有效批次，只换 IS 层级：Δ准确率 **{IL_D}**、置信区间
**{IL_CI}** 包含 0、`p` = {IL_P}），而 GSPO 内部沿组大小的三点连线也是平的
（G=4→G=8 Δ **{R1D}**、`p` = {R1P}；G=8→G=32 Δ **{R2D}**、`p` = {R2P}，见 5.2.6）——""",
    "5.2.5: the IS pair and the two group-size rungs")

left = [l for l in s.splitlines()
        if any(f"{{{k}" in l for k in
               ("il_delta", "il_ci", "il_p", "r1d", "r1p", "r2d", "r2p"))]
if left:
    sys.exit(f"placeholder(s) survived the substitution; nothing written: {left}")

DOC.write_text(s, encoding="utf-8")
print(f"{n_edits} sentence(s) filled in ({len(s.splitlines())} lines):")
print(f"  IS pair : d={IL_D} CI={IL_CI} p={IL_P}")
print(f"  rungs   : G=4->8 d={R1D} p={R1P} ; G=8->32 d={R2D} p={R2P}")
