"""Verify the auditor's claims against the records before acting on any of them.

Each of these is a claim that a document sentence contradicts a record. A wrong
finding costs more than a missed one, because someone edits the document on the
strength of it -- so every one is re-derived here from the record itself.
"""
import json
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
REP = R / "outputs/reports"


def load(n):
    return json.loads((REP / n).read_text(encoding="utf-8"))


print("=== 1. is there a G=32 vs GRPO(G=4) lr5e-5 comparison, and what is its delta? ===")
g32 = load("qwen35_9b_gspo_g32__report.json")["classification"]["accuracy"]
gr4 = load("qwen35_9b_grpo_lr5e5__report.json")["classification"]["accuracy"]
g8 = load("qwen35_9b_gspo_v1__report.json")["classification"]["accuracy"]
print(f"  GSPO G=32 lr5e-5 acc = {g32:.6f}")
print(f"  GRPO G=4  lr5e-5 acc = {gr4:.6f}   -> delta {g32 - gr4:+.6f}")
print(f"  GSPO G=8  lr5e-5 acc = {g8:.6f}   -> G32-G8 {g32 - g8:+.6f} / G8-GRPO4 {g8 - gr4:+.6f}")
ps = load("paired_significance.json")
print("  paired_significance.cross_algorithm_paired keys:",
      list(ps.get("cross_algorithm_paired", {}).keys()))
for k, v in (ps.get("cross_algorithm_paired") or {}).items():
    print(f"    {k}: {v['grpo_token']} -> {v['gspo_sequence']}  "
          f"delta={v['accuracy_delta_gspo_minus_grpo']:+.6f} p={v['mcnemar']['p_exact_two_sided']}")
print("  group_size_paired:", ps.get("group_size_paired", {}).get("from"), "->",
      ps.get("group_size_paired", {}).get("to"),
      ps.get("group_size_paired", {}).get("accuracy_delta_g32_minus_g8"))
print("  => a G=32-vs-GRPO(G=4) row exists?",
      any("G32" in str(v.get("from", "")) and "GRPO" in str(v.get("from", ""))
          for v in (ps.get("cross_algorithm_paired") or {}).values())
      or "GSPO_G32" in str(ps.get("cross_algorithm_paired")))

print()
print("=== 2. idle steps in kl_length_confound.json ===")
kl = load("kl_length_confound.json")
tot = 0
for tag, v in kl.get("runs", {}).items():
    n = v.get("kl_identity", {}).get("idle_steps_tested")
    ok = v.get("kl_identity", {}).get("holds")
    print(f"  {tag:<34} idle={n} holds={ok}")
    tot += n or 0
print(f"  SUM = {tot}   (document says 271)")

print()
print("=== 3. Base drop rates by class in clock_parse_audit.json ===")
cp = load("clock_parse_audit.json")
base = cp.get("base", {})
dd = base.get("dropped_defect_types") or {}
sup = base.get("support_by_defect_type") or cp.get("support_by_defect_type") or {}
print("  dropped:", dd)
print("  support:", sup)
for k, v in dd.items():
    s = sup.get(k)
    print(f"    {k:<14} dropped={v} support={s} rate="
          + (f"{v / s:.4f}" if s else "?"))

print()
print("=== 4. how many runs does the paired tool's reproduction check cover? ===")
print("  paired_significance.accuracy keys:", len(ps.get("accuracy", {})),
      list(ps.get("accuracy", {}).keys()))
print("  runs_included:", len(ps.get("runs_included", [])), ps.get("runs_included"))
