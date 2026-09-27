import re
from pathlib import Path

t = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md").read_text(encoding="utf-8")
NUM = r"(?:[一二三四五六七八九十两]+|\d+)"

for name, p in [
    ("A zeroed", r"在(" + NUM + r")个 RL run 里，\*\*恰好只有(" + NUM + r")有类别归零\*\*"),
    ("A1", r"恰好只有(" + NUM + r")有类别归零\*\*"),
    ("A2", r"在" + NUM + r"个 RL run 里，"),
    ("B intact", r"但(" + NUM + r")个 run 一个类别都没丢：([^（\n]+)（run 43）"),
    ("B1", r"但(" + NUM + r")个 run 一个类别都没丢："),
    ("B2", r"([^（\n]+)（run 43）"),
]:
    m = re.search(p, t, re.S)
    print(f"{name:10s} -> {m.groups() if m else 'NO MATCH'}")

i = t.find("恰好只有")
print("\naround 恰好只有:")
print(repr(t[i - 30:i + 30]))
print(t[i - 30:i + 30].encode("utf-8").hex())

j = t.find("但有两个 run 一个类别都没丢")
print("\naround 但有两个:")
print(repr(t[j - 6:j + 60]))
print(t[j:j + 12].encode("utf-8").hex())
