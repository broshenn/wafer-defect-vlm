"""Per-class drop rate for Base's clock parse, from the benchmark's own labels.

clock_parse_audit.json records how many rows were dropped per defect type, but not
the support those were drawn from. The support is a property of the frozen
benchmark, so it is computed here from classification.jsonl rather than assumed.
"""
import json
from collections import Counter
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
B = R / "benchmarks/wafer_bench_v1"

rows = [json.loads(l) for l in (B / "classification.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]
support = Counter(r["gold_label"] for r in rows)
print(f"benchmark classification rows: {len(rows)}   classes: {len(support)}")
print()

aud = json.loads((R / "outputs/reports/clock_parse_audit.json").read_text(
    encoding="utf-8"))["base"]
dropped = aud["dropped_defect_types"]

print(f"{'class':<12} {'support':>8} {'dropped':>8} {'rate':>8}")
print("-" * 40)
notfull = []
for cls in sorted(support):
    s, d = support[cls], dropped.get(cls, 0)
    rate = d / s if s else float("nan")
    if d != s:
        notfull.append((cls, s, d, rate))
    print(f"{cls:<12} {s:>8} {d:>8} {rate:>8.4f}")

print()
print(f"sum support = {sum(support.values())}, sum dropped = {sum(dropped.values())}")
print(f"classes not fully dropped: {len(notfull)}")
for cls, s, d, rate in notfull:
    print(f"  {cls}: dropped {d}/{s} = {rate:.4f}, {s - d} row(s) survived")
