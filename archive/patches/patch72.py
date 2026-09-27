"""Space before the verb: "{count} 次 run回答的是" reads as one token."""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
s = P.read_text(encoding="utf-8")

OLD = '''                         f"{_cnt_word}回答的是同一批 252 行，"'''
NEW = '''                         f"{_cnt_word} 回答的是同一批 252 行，"'''

if s.count(OLD) != 1:
    sys.exit(f"anchor appears {s.count(OLD)} times, expected 1")
s = s.replace(OLD, NEW)
ast.parse(s)
P.write_text(s, encoding="utf-8")
print("final_report.py: spacing fixed")
