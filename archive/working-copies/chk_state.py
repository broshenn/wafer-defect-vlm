"""State check: did patch62 land, how many stale scope-words remain, where is queue 41.

The Chinese-inline-over-ssh failures recur, so this runs server-side from a
base64 push rather than being typed into a shell string.
"""
import json
import re
import subprocess
import time
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")

print("=== git-free mtimes ===")
for f in ("LIMITATIONS.md", "FINAL_REPORT.md", "outputs/reports/comparison.md",
          "outputs/reports/paired_significance.json"):
    p = R / f
    print(f"  {f}: {time.strftime('%H:%M:%S', time.localtime(p.stat().st_mtime))}"
          f"  {p.stat().st_size}B" if p.exists() else f"  {f}: MISSING")

s = (R / "LIMITATIONS.md").read_text(encoding="utf-8")
lines = s.split("\n")

print("\n=== scope words still present in LIMITATIONS.md ===")
for pat in ("四个 run", "四个 RL", "四组", "三个 RL", "四个 GSPO", "五组", "五个 RL"):
    hits = [i + 1 for i, ln in enumerate(lines) if pat in ln]
    if hits:
        print(f"  {pat!r}: lines {hits}")

print("\n=== patch62 markers ===")
for probe in ("五组最低的标准差", "组大小 × 学习率网格"):
    print(f"  {probe!r}: {'PRESENT' if probe in s else 'ABSENT'}")

print("\n=== bold balance ===")
for f in ("LIMITATIONS.md", "FINAL_REPORT.md"):
    t = (R / f).read_text(encoding="utf-8")
    print(f"  {f}: ** = {t.count('**')} ({'even' if t.count('**') % 2 == 0 else 'ODD'})")

print("\n=== GPU ===")
print(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())

print("\n=== queue 41 outer log tail ===")
for lg in ("logs/41_queue_outer.log", "logs/42_queue_outer.log"):
    p = R / lg
    if p.exists():
        tail = p.read_text(encoding="utf-8", errors="replace").strip().split("\n")[-6:]
        print(f"  --- {lg} (mtime "
              f"{time.strftime('%H:%M:%S', time.localtime(p.stat().st_mtime))}) ---")
        for ln in tail:
            print(f"    {ln[:160]}")
    else:
        print(f"  {lg}: not created yet")

print("\n=== running processes of interest ===")
ps = subprocess.run(["pgrep", "-af", "swift|41_queue|42_queue|torchrun"],
                    capture_output=True, text=True).stdout.strip()
for ln in ps.split("\n")[-12:]:
    if ln:
        print(f"  {ln[:170]}")
