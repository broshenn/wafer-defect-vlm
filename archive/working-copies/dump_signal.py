import json
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm/outputs/reports")
TAGS = [("GRPO(G=4) lr1e-5", "grpo"), ("GRPO(G=4) lr5e-5", "qwen35_9b_grpo_lr5e5"),
        ("GSPO(G=8) lr1e-5", "gspo_lr1e5"), ("GSPO(G=8) lr5e-5", "qwen35_9b_gspo_v1"),
        ("GSPO(G=32) lr5e-5", "qwen35_9b_gspo_g32")]
D = {n: json.loads((R / f"{t}_train_result.json").read_text(encoding="utf-8")) for n, t in TAGS}
names = [n for n, _ in TAGS]
W = 18

rewards = sorted({k for d in D.values() for k in (d.get("reward_signal") or {})})
print("reward".ljust(16) + "".join(n.rjust(W) for n in names) + "   min")
for r in rewards:
    v = {}
    for n in names:
        s = (D[n].get("reward_signal") or {}).get(r) or {}
        v[n] = s.get("mean_std_across_steps")
    known = {k: x for k, x in v.items() if x is not None}
    lo = min(known, key=lambda k: known[k]) if known else None
    print(r.ljust(16) + "".join(
        ("n/a" if v[n] is None else f"{v[n]:.4f}").rjust(W) for n in names) + f"   {lo}")

print("\nidle steps:")
for n in names:
    f = D[n].get("mean_frac_reward_zero_std")
    print(f"  {n:18s} {f:.4%}  ({round(f * 150)}/150)")

print("\nfor G=32: lowest of how many rewards?")
g = D["GSPO(G=32) lr5e-5"].get("reward_signal") or {}
cnt = 0
for r in rewards:
    v = {n: (D[n].get("reward_signal") or {}).get(r, {}).get("mean_std_across_steps")
         for n in names}
    known = {k: x for k, x in v.items() if x is not None}
    if known and min(known, key=lambda k: known[k]) == "GSPO(G=32) lr5e-5":
        cnt += 1
print(f"  G=32 has the lowest std in {cnt} of {len(rewards)} rewards")
