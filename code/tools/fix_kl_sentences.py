# -*- coding: utf-8 -*-
"""Rewrite the KL sentences for the regenerated record: 6 runs -> 7, 2 pairs -> 3.

The record was one landing behind: `qwen35_9b_gspo_g4_lr1e5` (run 45) finished and was not
in it, because `tools/kl_length_confound.py` has no caller anywhere in tools/ or scripts/ --
nothing regenerates it. Its own auto-add loop picked the run up the moment the tool was run
again ("NOTE: ... finished and was not in RUNS; included"), which is the tool working as
designed; what was missing was anything to run it.

What moved, and every place the document states it:

  runs        6 -> 7                       lines 889, 1114
  idle steps  286 (54+49+23+57+63+40)      lines 889, 1114
              -> 336 (+50)
  lr pairs    2 -> 3                       line 307 (KL) and 894 (completion length)
  new pair    GSPO(G=4) kl 1.0937 -> 0.9882, len 117.0 -> 113.8 (-2.7%)

Two of those sentences named their pairs by **algorithm only** -- 「GRPO 1.1068 → 1.0435、
GSPO 1.1201 → 1.0217」. With a second GSPO pair that name no longer identifies a pair, so
the rewrite adds the group size to each. That is also what lets the checker compare the
enumerated pairs as a set rather than by position.

Every replacement is guarded by an exact occurrence count, and the whole file is restored
if any guard fails after a write.
"""
import pathlib
import shutil
import sys

DOC = pathlib.Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
TOOL = pathlib.Path("/root/autodl-fs/wafer-vlm/tools/kl_length_confound.py")
BAK = pathlib.Path("/tmp/LIMITATIONS.md.bak-kl-regen")

EDITS = [
    # --- 8: the identity's run count and total -------------------------------
    ("六个 run 的全部 286 个空转步均满足（逐 run 实测为\n54+49+23+57+63+40；",
     "七个 run 的全部 336 个空转步均满足（逐 run 实测为\n54+49+23+57+63+40+50；"),
    # --- 9: the same identity, second statement ------------------------------
    ("六个 run 全部 286 个空转步上的逐值核对",
     "七个 run 全部 336 个空转步上的逐值核对"),
    # --- 5.2.4: the KL pairs, two -> three, and named by group size ----------
    ("同算法同组大小的两个对照里 KL 都是下降的\n"
     "  （GRPO 1.1068 → 1.0435，\n"
     "  GSPO 1.1201 → 1.0217）。",
     "同算法同组大小的三个对照里 KL 都是下降的\n"
     "  （GRPO(G=4) 1.1068 → 1.0435、\n"
     "  GSPO(G=8) 1.1201 → 1.0217、\n"
     "  GSPO(G=4) 1.0937 → 0.9882）。"),
    # --- the same bullet's completion lengths --------------------------------
    ("5e-5 的补全却明显更短\n"
     "  （GRPO 115.5 → 97.9 token、\n"
     "  GSPO 114.4 → 109.2），",
     "5e-5 的补全却明显更短\n"
     "  （GRPO(G=4) 115.5 → 97.9 token、\n"
     "  GSPO(G=8) 114.4 → 109.2、\n"
     "  GSPO(G=4) 117.0 → 113.8），"),
    # --- 8's restatement of the same lengths --------------------------------
    ("但 5e-5 那两次的补全明显更短\n"
     "   （GRPO 115.5 → 97.9 token，−15.2%；GSPO 114.4 → 109.2，−4.6%），\n"
     "   均值取在不同的 token 跨度上，两次的 `kl` 不是同一个量。",
     "但 5e-5 那三次的补全明显更短\n"
     "   （GRPO(G=4) 115.5 → 97.9 token，−15.2%；GSPO(G=8) 114.4 → 109.2，−4.6%；\n"
     "   GSPO(G=4) 117.0 → 113.8，−2.7%），\n"
     "   均值取在不同的 token 跨度上，三次的 `kl` 不是同一个量。"),
]

# The tool's own instruction, printed when it auto-added the run: the record's order
# should be a decision, not a sort. Run 45 appended at the end keeps the order stable.
RUNS_OLD = ('RUNS = ["grpo", "qwen35_9b_grpo_lr5e5", "gspo_lr1e5", "qwen35_9b_gspo_v1",\n'
            '        "qwen35_9b_gspo_g32", "qwen35_9b_gspo_g4_lr5e5"]')
RUNS_NEW = ('RUNS = ["grpo", "qwen35_9b_grpo_lr5e5", "gspo_lr1e5", "qwen35_9b_gspo_v1",\n'
            '        "qwen35_9b_gspo_g32", "qwen35_9b_gspo_g4_lr5e5",\n'
            '        "qwen35_9b_gspo_g4_lr1e5"]')

s = DOC.read_text(encoding="utf-8")
t = TOOL.read_text(encoding="utf-8")
shutil.copy2(DOC, BAK)

for old, _new in EDITS:
    if s.count(old) != 1:
        sys.exit(f"an anchor appears {s.count(old)} times, expected 1:\n{old[:90]!r}\n"
                 f"nothing written")
_tool_done = t.count(RUNS_NEW) == 1
if not _tool_done and t.count(RUNS_OLD) != 1:
    sys.exit(f"the tool's RUNS list is not in the expected form "
             f"({t.count(RUNS_OLD)} matches); nothing written")

for old, new in EDITS:
    s = s.replace(old, new, 1)
    print(f"rewritten: {old.splitlines()[0][:56]}...")
if not _tool_done:
    t = t.replace(RUNS_OLD, RUNS_NEW, 1)
    TOOL.write_text(t, encoding="utf-8")
    print("kl_length_confound.py: run 45 added to RUNS")
else:
    print("kl_length_confound.py: run 45 already in RUNS (skipped)")

DOC.write_text(s, encoding="utf-8")

# The document must not still state the old populations anywhere. The phrases are the
# exact ones, not bare numerals: "286" also appears in a token total (1,545,286) and as
# two Scratch F1 scores (0.286), and a guard that flags those restores a correct rewrite.
left = [w for w in ("同算法同组大小的两个对照", "5e-5 那两次", "六个 run 的全部",
                    "六个 run 全部", "全部 286 个空转步") if w in s]
if left:
    shutil.copy2(BAK, DOC)
    sys.exit(f"the document still contains {left}; restored -- the rewrite is incomplete")

print("\nno occurrence of the old populations remains in the document")
