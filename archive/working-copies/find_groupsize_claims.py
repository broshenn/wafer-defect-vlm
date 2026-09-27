"""Locate every place group size is treated as a cause, so the new confound can
be attached to the right sentences rather than dumped in one paragraph."""
from pathlib import Path

D = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
lines = D.read_text(encoding="utf-8").split("\n")

PAT = ["组大小", "G=32", "G=8", "组大小确实", "分组大小", "grad_accum",
       "有效批", "梯度累积", "accum"]
hits = {}
for i, ln in enumerate(lines, 1):
    for p in PAT:
        if p in ln:
            hits.setdefault(p, []).append(i)

for p, ls in hits.items():
    print(f"{p!r}: {len(ls)} lines -- {ls}")

print()
print("=== section 5.2 neighbourhood, every line mentioning 组大小 ===")
for i, ln in enumerate(lines, 1):
    if "组大小" in ln:
        print(f"{i}: {ln[:150]}")
