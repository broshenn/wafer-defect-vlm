"""Re-derive every number the pending LIMITATIONS corrections depend on.

The document mixes two metric families (classification accuracy and classification
macro_f1), so a delta quoted without its metric cannot be checked. Everything here
is printed with its metric name and its source record attached.
"""
import json
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
REP = R / "outputs/reports"


def load(n):
    return json.loads((REP / n).read_text(encoding="utf-8"))


RUNS = [
    ("GRPO(G=4) lr1e-5", "qwen35_9b_grpo_lr1e5"),
    ("GRPO(G=4) lr5e-5", "qwen35_9b_grpo_lr5e5"),
    ("GSPO(G=8) lr1e-5", "qwen35_9b_gspo_v1"),
    ("GSPO(G=8) lr5e-5", "qwen35_9b_gspo_lr5e5"),
    ("GSPO(G=32) lr5e-5", "qwen35_9b_gspo_g32"),
]

print("=== per-run scores, both metric families, straight from each report ===")
print(f"{'run':<20} {'accuracy':>10} {'macro_f1':>10}")
vals = {}
for name, tag in RUNS:
    f = REP / f"{tag}__report.json"
    if not f.is_file():
        print(f"{name:<20} {'MISSING':>10}")
        continue
    d = load(f"{tag}__report.json")["classification"]
    vals[name] = d
    print(f"{name:<20} {d['accuracy']:>10.6f} {d['macro_f1']:>10.6f}")

print()
print("=== the two sentences under repair, checked against the above ===")
g32 = vals["GSPO(G=32) lr5e-5"]
g8 = vals["GSPO(G=8) lr5e-5"]
gr4 = vals["GRPO(G=4) lr5e-5"]
for m in ("accuracy", "macro_f1"):
    print(f"  [{m}] G32 - G8  = {g32[m] - g8[m]:+.6f}")
    print(f"  [{m}] G32 - GR4 = {g32[m] - gr4[m]:+.6f}")

print()
print("=== paired_significance.json: which pairs exist, with what deltas ===")
ps = load("paired_significance.json")
for key in sorted(ps):
    v = ps[key]
    if isinstance(v, dict) and "mcnemar" in v:
        print(f"  {key}: {v.get('from')} -> {v.get('to')}")
        for kk, vv in v.items():
            if kk.startswith("accuracy_delta") or kk.startswith("macro_f1_delta"):
                print(f"      {kk} = {vv:+.6f}")
        mc = v["mcnemar"]
        print(f"      mcnemar p = {mc.get('p_exact_two_sided')}"
              f"  discordant b/c = {mc.get('b')}/{mc.get('c')}"
              f"  metric = {mc.get('metric', 'n/a')}")

print()
print("=== cross_algorithm_paired, every field, verbatim ===")
print(json.dumps(ps.get("cross_algorithm_paired"), ensure_ascii=False, indent=2)[:2000])

print()
print("=== how many runs does the reproduction check cover? ===")
acc = ps.get("accuracy", {})
print(f"  paired_significance.accuracy  : {len(acc)} runs")
for k, v in acc.items():
    print(f"    {k:<34} {v}")
print(f"  runs_included ({len(ps.get('runs_included', []))}): {ps.get('runs_included')}")
print(f"  runs_optional_absent: {ps.get('runs_optional_absent')}")

print()
print("=== idle steps: sum over all runs in the record ===")
kl = load("kl_length_confound.json")
tot = 0
for tag, v in (kl.get("runs") or {}).items():
    n = v.get("kl_identity", {}).get("idle_steps_tested")
    print(f"  {tag:<34} idle={n}  total_steps={v.get('kl_identity', {}).get('steps_tested')}")
    tot += n or 0
print(f"  SUM idle_steps_tested = {tot}")

print()
print("=== Base clock-parse survival, per class ===")
cp = load("clock_parse_audit.json")
b = cp["base"]
dd = b.get("dropped_defect_types") or {}
print(f"  scored rows remaining: {b.get('n_scored')}  (dropped {b.get('n_dropped')})")
for k in sorted(dd):
    print(f"    {k:<12} dropped={dd[k]}")
print(f"  total dropped = {sum(dd.values())}")

print()
print("=== the report-generation tool's hardcoded run count ===")
print(f"  final_report.py exists: {(R / 'tools/final_report.py').is_file()}")
print(f"  tool run count string appears in N reports:")
