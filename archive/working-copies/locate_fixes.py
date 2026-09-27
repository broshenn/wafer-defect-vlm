"""Print the exact current text around each confirmed finding, so the patch can
anchor on real strings instead of line numbers (the file changed since the audit)."""
from pathlib import Path

D = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
lines = D.read_text(encoding="utf-8").split("\n")

PROBES = ["相对两侧的对照都是", "回两侧的对照", "组越大空转步越多",
          "而是 GSPO 在 5e-5 下才出现的现象", "271", "丢弃率全是 1.00",
          "六个 run", "六次 run", "472 个指标数字", "三列不可比", "三列"]

for p in PROBES:
    hits = [i + 1 for i, ln in enumerate(lines) if p in ln]
    print(f"### {p!r} -> lines {hits}")
    for h in hits:
        for j in range(max(1, h - 3), min(len(lines), h + 3) + 1):
            mark = ">>" if j == h else "  "
            print(f"{mark}{j}: {lines[j-1]}")
        print()
