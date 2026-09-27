"""Four run-set counts the coverage audit flagged as read by nothing.

Each is a numeral over the *current* run set that no checker item anchors on. Verified
against the records here before anything is written, because "unread" means "unverified",
not "wrong".
"""
import json
import pathlib

REP = pathlib.Path("/root/autodl-fs/wafer-vlm/outputs/reports")
DOC = pathlib.Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
text = DOC.read_text(encoding="utf-8")

RUNS = {
    "SFT": "qwen35_9b_adapter",
    "GRPO(G=4) lr1e-5": "qwen35_9b_grpo",
    "GRPO(G=4) lr5e-5": "qwen35_9b_grpo_lr5e5",
    "GSPO(G=4) lr5e-5": "qwen35_9b_gspo_g4_lr5e5",
    "GSPO(G=8) lr1e-5": "gspo_lr1e5",
    "GSPO(G=8) lr5e-5": "qwen35_9b_gspo_v1",
    "GSPO(G=32) lr5e-5": "qwen35_9b_gspo_g32",
    "GSPO(G=4) lr1e-5": "qwen35_9b_gspo_g4_lr1e5",
    "GSPO(G=32) lr1e-5": "qwen35_9b_gspo_g32_lr1e5",
}
R = {}
for n, t in RUNS.items():
    p = REP / f"{t}__report.json"
    if p.is_file():
        R[n] = json.loads(p.read_text(encoding="utf-8"))

print("=" * 72)
print("B: line 889/1114 -- 「六个 run 的全部 286 个空转步」")
k = REP / "kl_length_confound.json"
if not k.is_file():
    print("  kl_length_confound.json is absent")
else:
    d = json.loads(k.read_text(encoding="utf-8"))
    runs = d.get("runs") or {}
    print(f"  runs in the record: {len(runs)}")
    tot = 0
    for tag, v in runs.items():
        st = (v.get("kl_identity") or {}).get("idle_steps_tested")
        print(f"    {tag:<44} idle_steps_tested={st}")
        if isinstance(st, int):
            tot += st
    print(f"  sum = {tot}")
    print(f"  the document says: six runs, per-run 54+49+23+57+63+40, total 286")
    print(f"  -- with today's runs on disk this must be recomputed, not inherited")

print()
print("=" * 72)
print("A: line 531 -- 「G=32 在 Scratch 上拿到全部 run 最高的 0.462（SFT 0.286）」")
sc = {n: d["classification"]["per_class"].get("Scratch", {}).get("f1-score")
      for n, d in R.items()}
for n, v in sorted(sc.items(), key=lambda kv: -(kv[1] or 0)):
    print(f"    {n:<22} {v}")
worst = max(sc, key=lambda n: sc[n])
print(f"  maximum: {worst} {sc[worst]:.4f}   (document says GSPO(G=32) lr5e-5 0.462)")
print(f"  SFT:     {sc['SFT']:.4f}            (document says 0.286)")

print()
print("=" * 72)
print("A2: line 539/540 -- 「在 4 个奖励里有 3 个是七组最低」/「六组最低的标准差」")
FR = ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock")
print("  the reward table's population (RL runs with a training record):")
tr = REP / "qwen35_9b_gspo_g32_train_result.json"
d0 = json.loads(tr.read_text(encoding="utf-8")) if tr.is_file() else {}
print(f"    reward rows in the record: {[k for k in d0 if k.startswith('rewards/') or 'reward' in k][:8]}")
lowest_in = 0
for f in FR:
    vals = {}
    for n, t in RUNS.items():
        if n == "SFT":
            continue
        p = REP / f"{t}_train_result.json"
        if p.stem in ("grpo_train_result", "sft_train_result"):
            p = REP / f"{t}_train_result.json"
        if not p.is_file():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        v = (rec.get("rewards") or {}).get(f, {}).get("std")
        if isinstance(v, (int, float)):
            vals[n] = v
    if not vals:
        continue
    lo = min(vals, key=vals.get)
    hit = lo == "GSPO(G=32) lr5e-5"
    lowest_in += hit
    print(f"    {f:<12} n={len(vals)}  lowest={lo} ({vals[lo]:.4f})"
          f"{'   <- G=32' if hit else ''}")
print(f"  G=32 is the lowest in {lowest_in} of {len(FR)} rows "
      f"(document says 3 of 4)")
print()
print("  the reward table's column count (what 「七组」 counts):")
lines = text.split("\n")
for i, l in enumerate(lines):
    if l.startswith("| 奖励的组内标准差"):
        n = len([c for c in l.strip().strip("|").split("|")]) - 1
        print(f"    line {i + 1}: {n} run columns -> {n} 组")
        break
print()
print("  every 「N 组」 in the document:")
import re
for i, l in enumerate(lines):
    for m in re.finditer(r"[0-9一二三四五六七八九十两]+\s*组", l):
        print(f"    {i + 1:>4}| ...{l[max(0, m.start() - 40):m.end() + 25].strip()}...")
