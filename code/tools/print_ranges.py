"""Print the line ranges the pending corrections live in, so each edit anchors on
text that is verified present rather than on the auditor's line numbers."""
import sys
from pathlib import Path

D = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
lines = D.read_text(encoding="utf-8").split("\n")

for spec in sys.argv[1:]:
    a, _, b = spec.partition("-")
    a, b = int(a), int(b or a)
    print(f"----- {a}-{b} -----")
    for j in range(a, min(b, len(lines)) + 1):
        print(f"{j}: {lines[j-1]}")
    print()
