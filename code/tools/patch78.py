"""Emit the three-point group-size series as a bullet in FINAL_REPORT.

Placed directly after the existing two-point group-size bullet, and it says what
that bullet cannot: whether the relation between group size and accuracy has a
shape at all. It also states the confound's size, because a series that moves
effective batch by a factor of 64 is not evidence about group size alone.
"""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
s = P.read_text(encoding="utf-8")

ANCHOR = '        # The group-size contrast above is confounded with effective batch size,'

BLOCK = '''        _ser = ps.get("group_size_series_lr5e5")
        if _ser:
            _parts = []
            for _k, _v in (_ser.get("rungs") or {}).items():
                if _v.get("accuracy_delta") is None:
                    continue
                _parts.append(f"{_k} Δ{_v['accuracy_delta']:+.4f}"
                              f"（p = {_v['mcnemar']['p_exact_two_sided']:.6g}）")
            out.append(
                f"  - **组大小的三点序列**（同为 GSPO、sequence、lr 5e-5，只让组大小 "
                f"从 4 走到 8 再到 32）：{'、'.join(_parts)}。"
                f"{_ser['reading']} —— 两点对照只能说明「这两个组大小不同」，"
                f"三点才能说明这条关系有没有形状。")
            if _ser.get("confound_size"):
                out.append(f"    - 但**本序列不是单变量**：{_ser['confound_size']}")
            out.append(f"    - {_ser['does_not_settle']}")
'''

if s.count(ANCHOR) != 1:
    sys.exit(f"anchor appears {s.count(ANCHOR)} times, expected 1")
if "group_size_series_lr5e5" in s:
    print("already applied; nothing to do")
    sys.exit(0)

s = s.replace(ANCHOR, BLOCK + ANCHOR)
ast.parse(s)
P.write_text(s, encoding="utf-8")
print("final_report.py: group-size series bullet added")
