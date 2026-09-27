"""Print the six-column benchmark table straight from the report files.

Reads the nested report schema (classification / structured / caption /
robustness) rather than guessing at top-level keys, which is the mistake that
made an earlier inspection print blanks for a row that was actually fine.
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
KEYS = ["acc", "f1", "dtype", "zone", "clk", "size", "cap", "rob"]
HDR = ["accuracy", "macro-F1", "def_type", "radial", "clkMAE", "sizeMAE", "caption", "robust"]

rows = []
for label, fn in NAMES:
    p = R / fn
    if not p.is_file():
        rows.append((label, None))
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    c = d.get("classification", {})
    s = d.get("structured", {})
    fa = s.get("field_accuracy", {})
    rows.append((label, {
        "acc": c.get("accuracy"), "f1": c.get("macro_f1"), "ci": c.get("macro_f1_95ci"),
        "dtype": fa.get("defect_type"), "zone": fa.get("radial_zone"),
        "clk": s.get("clock_circular_mae_sectors"), "size": s.get("size_mae_r"),
        "cap": d.get("caption", {}).get("must_hit_rate"),
        "rob": d.get("robustness", {}).get("accuracy"),
    }))

print("run".ljust(18) + "".join(f"{h:>10}" for h in HDR))
for label, r in rows:
    if r is None:
        print(label.ljust(18) + "   (no report yet)")
        continue
    cells = "".join(
        f"{r[k]:>10.4f}" if isinstance(r[k], (int, float)) else f"{'-':>10}" for k in KEYS)
    print(label.ljust(18) + cells)

print()
for label, r in rows:
    if r and r.get("ci"):
        print(f"  {label:18} macro-F1 95% CI: [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]")
