"""Everything the pending corrections need, with the tag->run mapping taken from
each run's own recorded config (the filenames are not self-describing).
"""
import json
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
REP = R / "outputs/reports"

# tag -> (label, lr, G, IS) all read from the train_result config in map_runs.py output
TAGS = {
    "qwen35_9b_zero_shot": "BASE",
    "qwen35_9b_adapter": "SFT",
    "qwen35_9b_grpo": "GRPO G=4 lr1e-5",
    "qwen35_9b_grpo_lr5e5": "GRPO G=4 lr5e-5",
    "qwen35_9b_gspo_v1": "GSPO G=8 lr5e-5",
    "gspo_lr1e5": "GSPO G=8 lr1e-5",
    "qwen35_9b_gspo_g32": "GSPO G=32 lr5e-5",
}


def rep(tag):
    return json.loads((REP / f"{tag}__report.json").read_text(encoding="utf-8"))


print("=== 1. the accuracy deltas the sentence quotes ===")
a = {t: rep(t)["classification"]["accuracy"] for t in TAGS}
f = {t: rep(t)["classification"]["macro_f1"] for t in TAGS}
for t, lab in TAGS.items():
    print(f"  {lab:<20} acc={a[t]:.6f}  macro_f1={f[t]:.6f}")
print()
print(f"  G32 - GRPO(G=4) lr5e-5   [accuracy] = {a['qwen35_9b_gspo_g32'] - a['qwen35_9b_grpo_lr5e5']:+.6f}")
print(f"  G8  - GRPO(G=4) lr5e-5   [accuracy] = {a['qwen35_9b_gspo_v1'] - a['qwen35_9b_grpo_lr5e5']:+.6f}")
print(f"  G32 - G8 lr5e-5      [macro_f1] = {f['qwen35_9b_gspo_g32'] - f['qwen35_9b_gspo_v1']:+.6f}")
print(f"  G32 - GRPO(G=4) lr5e-5   [macro_f1] = {f['qwen35_9b_gspo_g32'] - f['qwen35_9b_grpo_lr5e5']:+.6f}")
print("  (1/36 = %.6f)" % (1 / 36))

print()
print("=== 2. paired_significance.json: every contrast, verbatim ===")
ps = json.loads((REP / "paired_significance.json").read_text(encoding="utf-8"))
for key in sorted(ps):
    v = ps[key]
    if not isinstance(v, dict):
        continue
    if "mcnemar" in v:
        mc = v["mcnemar"]
        dk = [k for k in v if "delta" in k]
        print(f"  {key}")
        print(f"      from={v.get('from')!r}  to={v.get('to')!r}")
        for k in dk:
            print(f"      {k} = {v[k]}")
        print(f"      metric={mc.get('metric', 'ABSENT')} p={mc.get('p_exact_two_sided')}")
        ci = {k: v[k] for k in v if "ci" in k.lower() or "bootstrap" in k.lower()}
        if ci:
            print(f"      ci: {json.dumps(ci, ensure_ascii=False)[:300]}")

print()
print("=== 3. reproduction check coverage ===")
print(f"  accuracy keys ({len(ps.get('accuracy', {}))}): {list(ps.get('accuracy', {}))}")
print(f"  runs_included ({len(ps.get('runs_included', []))}): {ps.get('runs_included')}")
print(f"  runs_optional_absent: {ps.get('runs_optional_absent')}")

print()
print("=== 4. idle steps, summed ===")
kl = json.loads((REP / "kl_length_confound.json").read_text(encoding="utf-8"))
tot = 0
for tag, v in (kl.get("runs") or {}).items():
    ki = v.get("kl_identity") or {}
    print(f"  {tag:<34} idle={ki.get('idle_steps_tested')}")
    tot += ki.get("idle_steps_tested") or 0
print(f"  SUM = {tot}")

print()
print("=== 5. Base clock-parse survival ===")
b = json.loads((REP / "clock_parse_audit.json").read_text(encoding="utf-8"))["base"]
print(f"  keys: {sorted(b.keys())}")
for k, v in b.items():
    if isinstance(v, (int, float, str)):
        print(f"  {k} = {v}")
dd = b.get("dropped_defect_types") or {}
print(f"  dropped_defect_types = {dd}")
print(f"  sum dropped = {sum(dd.values())}")
