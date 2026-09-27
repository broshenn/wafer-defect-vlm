import re
from pathlib import Path

t = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md").read_text(encoding="utf-8")
NUM = r"([一二三四五六七八九十两]+|\d+)"

print("NUM hex:", NUM.encode("utf-8").hex())
for probe in ("是六个 RL run 里最低的", "在六个 RL run 里", "在 4 个奖励里有 3 个"):
    i = t.find(probe)
    print(f"\nprobe {probe!r}: index {i}")
    if i >= 0:
        seg = t[i:i + len(probe)]
        print("  file   :", seg.encode("utf-8").hex())
        print("  probe  :", probe.encode("utf-8").hex())
        print("  equal  :", seg == probe)

for name, p in [
    ("NUM alone", NUM),
    ("lowest", r"是 (" + NUM + r") 个 RL run 里最低的"),
    ("lowest-nospace", r"是(" + NUM + r") 个 RL run 里最低的"),
    ("reward", r"在 4 个奖励里有 (" + NUM + r") 个是 (" + NUM + r") 组最低"),
]:
    m = re.search(p, t, re.S)
    print(f"{name:16s} -> {m.groups() if m else 'NO MATCH'}")
