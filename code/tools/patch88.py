"""Normalise the learning rate the way the tags actually write it.

_geom() mapped "1e5" -> "1e-5" because the GSPO lr-1e-5 tag drops the minus. The
same tag family writes the paper's lr as "lr5e5", so the rebuild printed
"GSPO（G=4，lr 5e5，...）" and "GRPO(G=4) lr5e5" -- 5e5 is 5x10^5, five hundred
thousand times the learning rate that actually ran. A label in a report that names
the wrong hyperparameter is the same defect class as a wrong number, so it is fixed
in the parser rather than at the three places it showed up.

Rule: "<mantissa>e5" means the mantissa times 10^-5. An explicit exponent keeps
whatever it says.
"""
import ast
import re
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
s = P.read_text(encoding="utf-8")

OLD = '''    return (int(g.group(1)) if g else 8,
            {"1e5": "1e-5"}.get(lr.group(1), lr.group(1)) if lr else "5e-5")'''
NEW = '''    # The tags write the paper's lr as "lr5e5" and the other one as "lr1e5": a
    # bare "<m>e5" is 10^-5, because "5e5" without the sign would be 5x10^5 --
    # 8 orders of magnitude off the lr that ran. An explicit exponent is kept.
    def _lr(text):
        m = re.fullmatch(r"([0-9.]+)e5", text)
        return f"{m.group(1)}e-5" if m else text
    return (int(g.group(1)) if g else 8,
            _lr(lr.group(1)) if lr else "5e-5")'''
if s.count(OLD) != 1:
    sys.exit(f"anchor appears {s.count(OLD)} times")

s = s.replace(OLD, NEW)
try:
    ast.parse(s)
except SyntaxError as e:
    sys.exit(f"edited file does not parse: {e}; nothing written")

if 're.fullmatch(r"([0-9.]+)e5", text)' not in s:
    sys.exit("the normaliser did not land; nothing written")

P.write_text(s, encoding="utf-8")
print(f"final_report.py: lr normaliser fixed ({len(s.splitlines())} lines, parses)")
