"""Per-class F1 across every run, to see whether a collapsed class is run-specific.

G=32 scored 0.0 F1 on Donut. If SFT scores 0.0 there too, that is a hard class
and not evidence about this run; if only G=32 does, it is a real regression and
belongs in the report.
"""
import json
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm/outputs/reports")
NAMES = [
    ("Base", "qwen35_9b_zero_shot__report.json"),
    ("SFT", "qwen35_9b_adapter__report.json"),
    ("GRPO G=4 lr1e-5", "qwen35_9b_grpo__report.json"),
    ("GSPO G=8 lr5e-5", "qwen35_9b_gspo_v1__report.json"),
    ("GSPO G=8 lr1e-5", "gspo_lr1e5__report.json"),
    ("GSPO G=32 lr5e-5", "qwen35_9b_gspo_g32__report.json"),
]

tables = []
classes = []
for label, fn in NAMES:
    p = R / fn
    if not p.is_file():
        continue
    pc = json.loads(p.read_text(encoding="utf-8")).get("classification", {}).get("per_class", {})
    tables.append((label, pc))
    for c in pc:
        if c not in classes:
            classes.append(c)

short = {label: label.split()[0] + ("G32" if "G=32" in label else
                                    "G8" if "G=8" in label else
                                    "G4" if "G=4" in label else "")
         for label, _ in tables}
print("class".ljust(12) + "".join(f"{short[l]:>9}" for l, _ in tables) + "   support")
for c in classes:
    supp = next((t[c].get("support") for _, t in tables if c in t), None)
    cells = "".join(f"{t[c]['f1-score']:>9.3f}" if c in t else f"{'-':>9}" for _, t in tables)
    flag = "  <-- collapsed" if all((c not in t or t[c]["f1-score"] < 0.01) for _, t in tables) else ""
    print(c.ljust(12) + cells + f"   {supp}{flag}")
