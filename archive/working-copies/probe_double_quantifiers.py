"""Sentences that state a population twice -- where one half can move and the other not.

The finding this generalizes: 5.2.2 had one sentence saying 「3 个是七组最低 —— 但
「六组最低的标准差」…」. Run 45 added the seventh reward-table column, the first numeral
became 七, the second stayed 六, and the claims checker anchored on the first. A stale
numeral inside a sentence the tool reads is strictly worse than one in a sentence it
does not: the green light covers it.

So: find every sentence that carries two or more quantifiers over the same unit. Those
are the places where a scope word can move in one half only, and where reading one half
certifies the other. The check is mechanical; the judgement of whether both halves name
the same population is not, so this reports candidates rather than verdicts.
"""
import pathlib
import re

DOC = pathlib.Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
text = DOC.read_text(encoding="utf-8")

UNIT = {"个": "个", "组": "组", "列": "列", "句": "句", "条": "条", "张": "张", "行": "行"}
QUANT = re.compile(r"([0-9一二三四五六七八九十两]+)\s*(个|组|列|句|条|张|行)")

# Sentences, keeping the line numbers they came from. Split on the sentence-final marks
# and on newlines; the document writes one sentence per line mostly, but not always.
lines = text.split("\n")
sentences = []
for i, l in enumerate(lines):
    for part in re.split(r"(?<=[。！？；])", l):
        if part.strip():
            sentences.append((i + 1, part))

found = 0
for n, s in sentences:
    hits = QUANT.findall(s)
    if len(hits) < 2:
        continue
    by_unit = {}
    for num, unit in hits:
        by_unit.setdefault(unit, []).append(num)
    dup = {u: v for u, v in by_unit.items() if len(v) >= 2}
    if not dup:
        continue
    found += 1
    print(f"--- line {n}: {len(hits)} quantifier(s), repeated unit(s) {list(dup)}")
    print(f"    {s.strip()[:230]}")
    for u, nums in dup.items():
        print(f"      {u}: {nums}")
    print()

print(f"{found} sentence(s) state a population twice over the same unit.")
print()
print("Not all are defects: 「四个 run」 beside 「第五个 run」 is a count and an index, and")
print("「两个 run」 beside 「两个 run」 may be two different pairs. What makes one a defect")
print("is that the two halves name the SAME population and only one of them moved -- which")
print("is what happened at 5.2.2, and what nothing was in a position to notice.")
