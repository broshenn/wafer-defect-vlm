import json, os

R = "/root/autodl-fs/wafer-vlm/outputs/reports"
names = sorted(os.listdir(R))
print("== reports ==")
for n in names:
    if n.endswith("__report.json"):
        print("  REPORT", n)
for n in names:
    if n.endswith("_train_result.json"):
        print("  TRAIN ", n)

print("== paired_significance ==")
d = json.load(open(os.path.join(R, "paired_significance.json")))
print("  top keys :", sorted(d.keys()))
print("  cvs keys :", sorted(d.get("comparisons_vs_SFT", {}).keys()))
for k in ("group_size_paired_lr1e5", "group_size_verdict", "lr_effect_both_group_sizes"):
    print("  has %-28s %s" % (k, k in d))
lre = d.get("lr_effect_both_group_sizes") or {}
print("  lre blocks:", sorted((lre.get("blocks") or {}).keys()))
print("  lre sig   :", lre.get("significant_at"))
vd = d.get("group_size_verdict") or {}
print("  gs verdict:", json.dumps(vd, ensure_ascii=True)[:500])
c45 = (d.get("comparisons_vs_SFT") or {}).get("GSPO_G4_lr1e5")
print("  cvs[GSPO_G4_lr1e5]:", json.dumps(c45, ensure_ascii=True) if c45 else "ABSENT")

print("== one report shape ==")
for tag in ("qwen35_9b_gspo_g4_lr1e5", "qwen35_9b_grpo", "qwen35_9b_adapter",
            "gspo_lr1e5"):
    p = os.path.join(R, tag + "__report.json")
    if not os.path.isfile(p):
        print("  %-24s MISSING" % tag)
        continue
    r = json.load(open(p))
    c = r.get("classification") or {}
    pc = c.get("per_class") or {}
    first = sorted(pc.keys())[0] if pc else None
    print("  %-24s macro=%s classes=%s inner=%s" % (
        tag, c.get("macro_f1"), sorted(pc.keys())[:6],
        sorted((pc.get(first) or {}).keys()) if first else None))
    for nm in ("none", "Donut"):
        if pc and first:
            try:
                print("      %-8s %s" % (nm, pc[first][nm]))
            except Exception as e:
                print("      %-8s <no such class> %s" % (nm, str(e)[:60]))
    break

print("== run 45 per-class over all columns ==")
r = json.load(open(os.path.join(R, "qwen35_9b_gspo_g4_lr1e5__report.json")))
pc = r["classification"]["per_class"]
print("  per_class keys:", sorted(pc.keys()))
