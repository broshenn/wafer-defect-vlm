import json
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
tags = [("SFT", "qwen35_9b_adapter"), ("GRPO1e5", "qwen35_9b_grpo"),
        ("GRPO5e5", "qwen35_9b_grpo_lr5e5"), ("GSPO8_5e5", "qwen35_9b_gspo_v1"),
        ("GSPO8_1e5", "gspo_lr1e5"), ("GSPO32_5e5", "qwen35_9b_gspo_g32")]
D = {n: json.loads((R / f"outputs/reports/{t}__report.json").read_text(encoding="utf-8"))
     for n, t in tags}
names = [n for n, _ in tags]
W = 12

rows = [("radial_zone", lambda d: d["structured"]["field_accuracy"]["radial_zone"], True),
        ("clock MAE", lambda d: d["structured"]["clock_circular_mae_sectors"], False),
        ("size MAE", lambda d: d["structured"]["size_mae_r"], False),
        ("caption must-hit", lambda d: d["caption"]["must_hit_rate"], True)]
print("metric".ljust(20) + "".join(n.rjust(W) for n in names) + "   extreme")
for label, fn, higher in rows:
    v = {n: fn(D[n]) for n in names}
    best = max(v, key=lambda n: v[n]) if higher else min(v, key=lambda n: v[n])
    print(label.ljust(20) + "".join(f"{v[n]:>{W}.4f}" for n in names)
          + f"   {'max' if higher else 'min'}={best}")

print("\nper-class F1:")
key = "per_class"
classes = sorted(D[names[0]]["classification"][key])
print("class".ljust(20) + "".join(n.rjust(W) for n in names) + "   best/zero")
for c in classes:
    v = {n: D[n]["classification"][key][c]["f1-score"] for n in names}
    best = max(v, key=lambda n: v[n])
    zeros = [n for n in names if v[n] == 0.0]
    print(c.ljust(20) + "".join(f"{v[n]:>{W}.4f}" for n in names)
          + f"   best={best}" + (f"  ZERO={zeros}" if zeros else ""))

print("\nretrieval (best across all runs):")
for m in ("mAP@10", "nDCG@10", "Recall@10"):
    v = {n: D[n]["retrieval"][m] for n in names}
    best = max(v, key=lambda n: v[n])
    print(f"  {m:10s} best={best:11s} "
          + " ".join(f"{n}={v[n]:.4f}" for n in names))

print("\nmacro-F1 (worst RL run):")
v = {n: D[n]["classification"]["macro_f1"] for n in names}
print("  " + " ".join(f"{n}={v[n]:.4f}" for n in names))
print("  worst excluding SFT:", min((n for n in names if n != "SFT"), key=lambda n: v[n]))
