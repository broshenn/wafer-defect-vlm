import re
from pathlib import Path

t = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md").read_text(encoding="utf-8")
NUM = r"(?:[一二三四五六七八九十两]+|\d+)"

probes = [
    ("literal 恰好只有两个有类别归零", "恰好只有两个有类别归零"),
    ("literal 恰好只有", "恰好只有"),
    ("literal 有类别归零", "有类别归零"),
    ("literal 归零**", "归零**"),
    ("literal 恰好只有两个", "恰好只有两个"),
    ("literal 但有两个 run 一个类别都没丢：", "但有两个 run 一个类别都没丢："),
    ("literal 但有两个", "但有两个"),
    ("literal 一个类别都没丢", "一个类别都没丢"),
]
for name, s in probes:
    print(f"{name:34s} in text: {s in t}   pattern-match: "
          f"{bool(re.search(re.escape(s), t))}")

print()
for name, p in [
    ("恰好只有(NUM)有类别归零", r"恰好只有(" + NUM + r")有类别归零"),
    ("恰好只有(NUM)", r"恰好只有(" + NUM + r")"),
    ("但(NUM)个", r"但(" + NUM + r")个"),
    ("但(NUM)个 run", r"但(" + NUM + r")个 run"),
    ("但(NUM)个 run 一个", r"但(" + NUM + r")个 run 一个"),
]:
    m = re.search(p, t)
    print(f"{name:34s} -> {m.groups() if m else 'NO MATCH'}")

i = t.find("恰好只有")
print("\nhex of 恰好只有两个有类别归零**:",
      t[i:i + 12].encode("utf-8").hex())
print("hex of the literal         :",
      "恰好只有两个有类别归零".encode("utf-8").hex())
j = t.find("但有两个")
print("\nhex of 但有两个 run       :", t[j:j + 11].encode("utf-8").hex())
print("hex of the literal         :", "但有两个 run".encode("utf-8").hex())
