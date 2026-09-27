"""Bring the remaining "how many runs" counts up to six, and fix a field pointer.

Two of these are pointers into a record, and one of them is wrong in a way that cost
real time today: the document says the per-run idle counts are in `idle_steps_tested`,
which is not a field of the run -- it is nested at
`runs.<tag>.kl_identity.idle_steps_tested`. Looking it up at the top of the run raises
nothing and returns nothing, which reads as "the record does not have this", so the
pointer now names the path.

The rest are quantifier counts: 5 runs / five groups / six columns, all of which were
true when written and are now six runs / six groups / seven columns.
"""
import os
import sys
from pathlib import Path

DOC = Path(os.environ.get("WAFER_DOC",
                          "/root/autodl-fs/wafer-vlm/LIMITATIONS.md"))
s = DOC.read_text(encoding="utf-8")

EDITS = [
    ("**空转步与基准分的读数再次相反**：G=32 的空转步 42.00%（63/150）是五组里最差的，",
     "**空转步与基准分的读数再次相反**：G=32 的空转步 42.00%（63/150）是六组里最差的，",
     "5.2.5: six groups, not five"),

    ("「全部 run」这个量词的范围变了，而产物本身显示不出来）。补全六列后逐条核对，",
     "「全部 run」这个量词的范围变了，而产物本身显示不出来）。补全七列后逐条核对，",
     "5.2.5: seven columns, not six"),

    ("""   `loss == beta * kl` 严格成立 —— 五个 run 的全部 246 个空转步均满足（逐 run 实测为 54+49+23+57+63；初稿写 271，
是一个加错了的旧数。逐 run 计数见 `kl_length_confound.json` 的
`idle_steps_tested`），""",
     """   `loss == beta * kl` 严格成立 —— 六个 run 的全部 286 个空转步均满足（逐 run 实测为
54+49+23+57+63+40；初稿写 271，是一个加错了的旧数。逐 run 计数见
`kl_length_confound.json` 的 `runs.<tag>.kl_identity.idle_steps_tested` ——
不是 run 顶层的 `idle_steps_tested`，按后者查会什么都查不到，看起来像记录里没有这个数），""",
     "section 8: six runs, 286 idle steps, and the nested field path"),

    ("""- 记录与日志的对应关系核对 `tools/audit_log_provenance.py`：5 份 RL 记录，每一份""",
     """- 记录与日志的对应关系核对 `tools/audit_log_provenance.py`：6 份 RL 记录，每一份""",
     "section 9: six RL records"),

    ("""三者必须一致（当前 5 个 run 全部一致：
  36.00%、32.67%、38.00%、15.33%、42.00%）。日志名在各 run 之间并不统一 ——""",
     """三者必须一致（当前 6 个 run 全部一致，按该工具 `RUNS` 的顺序：
  36.00%、32.67%、38.00%、15.33%、42.00%、26.67%）。日志名在各 run 之间并不统一 ——""",
     "section 9: six idle rates verified three ways"),

    ("""  五个 run 全部 246 个空转步上的逐值核对（逐 run 计数见该记录的
   `idle_steps_tested`；文档他处曾写 271，已更正）。""",
     """  六个 run 全部 286 个空转步上的逐值核对（逐 run 计数见该记录的
   `runs.<tag>.kl_identity.idle_steps_tested`；文档他处曾写 271，已更正）。""",
     "section 9: the KL bullet's counts and field path"),
]

before = s.count("**")
for old, new, why in EDITS:
    n = s.count(old)
    if n != 1:
        sys.exit(f"anchor for '{why}' appears {n} times, expected 1; nothing written")
    s = s.replace(old, new)
after = s.count("**")
if (after - before) % 2:
    sys.exit(f"bold markers went {before} -> {after}, an odd change; nothing written")

DOC.write_text(s, encoding="utf-8")
print(f"{len(EDITS)} edit(s) applied ({len(s.splitlines())} lines), bold {before} -> {after}")
for _, _, why in EDITS:
    print(f"  - {why}")
