"""Read every row of LIMITATIONS.md's main comparison table against its own header.

Standalone probe: same accessors patch107 will install in the checker, run here first so
the patch is known to be green (or known to find a real defect) before it is written.

The point of reading the header by walking up from the row rather than by sentinel: the
document has two nine-column tables and they list the two GSPO(G=8) runs in opposite
orders, so a row read through the other table's header puts every value in the wrong
run's column while each number stays byte-for-byte correct.
"""
import json
import re
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
TRAIN_RESULT = {"GRPO(G=4) lr1e-5": "grpo_train_result", "SFT": "sft_train_result"}

P = {}
for n, t in RUNS.items():
    p = REP / f"{t}__report.json"
    if not p.is_file():
        print(f"  (no report on disk: {n} -> {t})")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    P[n] = dict(
        accuracy=d["classification"].get("accuracy"),
        macro_f1=d["classification"]["macro_f1"],
        f1_ci=d["classification"].get("macro_f1_95ci"),
        defect_type=d["structured"]["field_accuracy"].get("defect_type"),
        radial=d["structured"]["field_accuracy"]["radial_zone"],
        clock=d["structured"]["clock_circular_mae_sectors"],
        size=d["structured"]["size_mae_r"],
        must_hit=d["caption"]["must_hit_rate"],
        rob=(d.get("robustness") or {}).get("accuracy"),
        flip=(d.get("robustness") or {}).get("flip_rate_vs_clean"),
        ret=(d.get("retrieval") or {}).get("mAP@10"),
    )
print(f"reports read: {len(P)}")
print(f"macro_f1_95ci sample: {P['SFT']['f1_ci']!r}  (type {type(P['SFT']['f1_ci']).__name__})")


def col_key(c):
    c = re.sub(r"\s+", " ", c).strip()
    m = re.fullmatch(r"([A-Za-z]+)\s*\(G=(\d+),\s*(lr[\d.eE+-]+)\)", c)
    return f"{m.group(1)}(G={m.group(2)}) {m.group(3)}" if m else c


def row_label(c):
    return c.strip().strip("*").strip().strip("`").strip()


lines = text.split("\n")
head, rows, at = None, {}, -1
for i, l in enumerate(lines):
    if not (l.startswith("| 指标 | SFT |") and l.count("|") == 10):
        continue
    j, r = i + 2, {}
    while j < len(lines) and lines[j].startswith("|"):
        c = [x.strip() for x in lines[j].strip().strip("|").split("|")]
        r[row_label(c[0])] = c[1:]
        j += 1
    if "retrieval mAP@10" in r and "flip rate" in r:
        head = [c.strip() for c in l.strip().strip("|").split("|")][1:]
        rows, at = r, i
        break

cols = [col_key(c) for c in head]
print(f"\nmain table: line {at + 1}, {len(cols)} columns, {len(rows)} rows")
print("columns:", cols)
print("set == records:", sorted(cols) == sorted(P))
print("row labels:", list(rows))
print()

FIELDS = {
    "分类准确率": "accuracy",
    "macro-F1": "macro_f1",
    "structured defect_type": "defect_type",
    "structured radial_zone": "radial",
    "clock circular MAE": "clock",
    "size MAE (R)": "size",
    "caption must-hit": "must_hit",
    "robustness accuracy": "rob",
    "flip rate": "flip",
    "retrieval mAP@10": "ret",
}
bad = 0


def num(v):
    v = v.strip().strip("*").strip()
    return None if v in ("", "—", "-", "n/a") else round(float(v), 4)


for lab, f in FIELDS.items():
    if lab not in rows:
        print(f"{lab:<24} ROW ABSENT")
        bad += 1
        continue
    got = [num(v) for v in rows[lab]]
    exp = [None if P[c][f] is None else round(P[c][f], 4) for c in cols]
    if got == exp:
        print(f"{lab:<24} ok   ({len(got)} cells)")
    else:
        bad += 1
        print(f"{lab:<24} MISMATCH")
        for c, g, e in zip(cols, got, exp):
            if g != e:
                print(f"      {c:<22} doc={g} record={e}")

CI = re.compile(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]")
if "macro-F1 95% CI" in rows:
    got = [[round(float(x), 4) for x in CI.fullmatch(v.strip()).groups()]
           for v in rows["macro-F1 95% CI"]]
    exp = [[round(P[c]["f1_ci"][0], 4), round(P[c]["f1_ci"][1], 4)] for c in cols]
    if got == exp:
        print(f"{'macro-F1 95% CI':<24} ok   ({len(got)} cells)")
    else:
        bad += 1
        print(f"{'macro-F1 95% CI':<24} MISMATCH")
        for c, g, e in zip(cols, got, exp):
            if g != e:
                print(f"      {c:<22} doc={g} record={e}")
else:
    print(f"{'macro-F1 95% CI':<24} ROW ABSENT")
    bad += 1

IDLE_LAB = "零优势步占比（实测）"


def idle_rate(name):
    stem = TRAIN_RESULT.get(name, RUNS[name] + "_train_result")
    p = REP / f"{stem}.json"
    if not p.is_file():
        return ("no-record", stem)
    v = json.loads(p.read_text(encoding="utf-8")).get("mean_frac_reward_zero_std")
    return ("ok", v) if isinstance(v, (int, float)) else ("no-field", stem)


if IDLE_LAB in rows:
    got, exp = [], []
    for v in rows[IDLE_LAB]:
        v = v.strip().strip("*").strip()
        got.append(None if v in ("—", "") else round(float(v.rstrip("%")), 2))
    for c in cols:
        state, v = idle_rate(c)
        if state != "ok":
            print(f"      {c:<22} {state}: {v}")
        exp.append(None if state != "ok" else round(100 * v, 2))
    if got == exp:
        print(f"{IDLE_LAB:<24} ok   ({len(got)} cells)")
    else:
        bad += 1
        print(f"{IDLE_LAB:<24} MISMATCH")
        for c, g, e in zip(cols, got, exp):
            if g != e:
                print(f"      {c:<22} doc={g} record={e}")
else:
    print(f"{IDLE_LAB:<24} ROW ABSENT")
    bad += 1

covered = set(FIELDS) | {"macro-F1 95% CI", IDLE_LAB}
print(f"\nrows with no item: {[k for k in rows if k not in covered]}")
print(f"row widths: {sorted({len(v) for v in rows.values()})} (header has {len(head)})")
print(f"TOTAL MISMATCHES: {bad}")
