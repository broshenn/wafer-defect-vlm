# What patch94 anchors on, as it now reads after patch92b and patch98.
import json
import re
from pathlib import Path

s = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md").read_text(encoding="utf-8")


def show(label, needle, back=0, fwd=520):
    i = s.find(needle)
    print("==", label, "hits=%d" % s.count(needle), "line=%d" % (s[:i].count("\n") + 1 if i >= 0 else -1))
    if i >= 0:
        print(json.dumps(s[i - back:i + fwd], ensure_ascii=False))


# patch94's LR_PARA, verbatim
LR_PARA = re.compile(r"(算法内部的学习率效应（同算法、同组大小、只差学习率）同样显著：\n.+?。)", re.S)
m = LR_PARA.search(s)
print("LR_PARA matches:", bool(m))
if m:
    print(json.dumps(m.group(1), ensure_ascii=False))

# patch94's ABOVE, verbatim
ABOVE = re.compile(
    r"((?:[一二三四五六七八九十两]+|\d+)\s*个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 "
    r"[^（\n]+（[\d.]+ 对 [\d.]+）)")
print("ABOVE (shape 1) matches:", bool(ABOVE.search(s)))

show("5.2.2 lr sentence", "算法内部的学习率效应")
show("5.2.4 above-SFT sentence", "个 macro-F1 点估计")
show("5.2.4 verdict", "诚实的表述是", fwd=700)
