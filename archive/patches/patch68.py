"""Make the generator's run count a reading of the record instead of a literal.

The string "六次 run" was hardcoded while the record already knew the number: the
paired tool writes `runs_included`, and its length is the count. A literal here
cannot disagree with anything, so it cannot disclose that it has gone stale -- the
same defect class as the training records' hardcoded grad_accum.
"""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
s = P.read_text(encoding="utf-8")

ANCHOR = """    for gname in ("GSPO_lr5e5", "GSPO_lr1e5", "GSPO_G32"):
"""

PRE = """    # The paired record knows how many runs answered the 252 rows; read the count
    # rather than writing a literal that cannot go stale visibly.
    _cnt_word = "各次 run"
    _psf = reports / "paired_significance.json"
    if _psf.is_file():
        try:
            _n = len(json.loads(_psf.read_text(encoding="utf-8"))
                      .get("runs_included") or [])
            if _n:
                _cnt_word = f"{_n} 次 run"
        except Exception:
            pass
"""

OLD_LINE = """                line += ("（该规则只比较两个边际区间，对配对数据偏保守 —— "
                         "六次 run 回答的是同一批 252 行，结论以紧随其后的成对检验为准）")"""

NEW_LINE = """                line += ("（该规则只比较两个边际区间，对配对数据偏保守 —— "
                         f"{_cnt_word}回答的是同一批 252 行，"
                         "结论以紧随其后的成对检验为准）")"""

if s.count(ANCHOR) != 1:
    sys.exit(f"loop anchor appears {s.count(ANCHOR)} times, expected 1")
s = s.replace(ANCHOR, PRE + ANCHOR)

if s.count(OLD_LINE) != 1:
    sys.exit(f"string anchor appears {s.count(OLD_LINE)} times, expected 1")
s = s.replace(OLD_LINE, NEW_LINE)

ast.parse(s)
if "六次 run" in s:
    sys.exit("hardcoded count still present")
P.write_text(s, encoding="utf-8")
print("final_report.py: run count read from runs_included")
