#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""score_v5 测试（B 卡 7）：4 条手写原答（全对、格式错、类别错、扇区差一）+1 条缺失，与手算一致。"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = HERE.parent
sys.path.insert(0, str(B))
from common_v5 import FIELDS  # noqa: E402

FIX = HERE / "fixtures"
FIX.mkdir(exist_ok=True)


def six(cls="Edge_Loc", band="20_to_under_50pct", zones=("middle", "edge"),
        ang="single", secs=(3,)):
    return {"defect_class": cls, "coverage_band": band, "red_mass_zones": list(zones),
            "angular_type": ang, "clock_sectors": list(secs), "uncertainty": ""}


def ref_row(sid, lot, sixd, ratio=0.27, zf=(0.0, 0.3, 0.7), r1=0.6, r2=0.4):
    ref = {**sixd,
           "measure": {"red_ratio": round(ratio, 2), "zone_frac": [round(z, 1) for z in zf],
                       "R1": round(r1, 1), "R2": round(r2, 1)},
           "measure_exact": {"red_ratio": ratio, "zone_frac": list(zf), "R1": r1, "R2": r2},
           "split_src": "test", "lot_name": lot}
    return {"messages": [{"role": "user", "content": "<image>\n题面"}],
            "images": ["__IMG__/x.png"], "sample_id": sid,
            "reference_json": json.dumps(ref, ensure_ascii=False, separators=(",", ":"))}


def ans(sixd, ratio=0.27, zf=(0.0, 0.3, 0.7), r1=0.6, r2=0.4):
    line = (f"<measure>red_ratio={ratio:.2f}; zone_frac={zf[0]:.1f},{zf[1]:.1f},{zf[2]:.1f};"
            f" R1={r1:.1f}; R2={r2:.1f}</measure>")
    return line + "\n" + json.dumps(sixd, ensure_ascii=False, separators=(",", ":"))


def main():
    refs = [
        ref_row("s1", "lotA", six()),
        ref_row("s2", "lotB", six()),
        ref_row("s3", "lotC", six()),
        ref_row("s4", "lotD", six()),
        ref_row("s5_missing", "lotE", six()),
    ]
    raws = [
        {"sample_id": "s1", "raw": ans(six())},
        {"sample_id": "s2", "raw": "我不知道，图片太模糊了。"},
        {"sample_id": "s3", "raw": ans(six(cls="Loc"))},
        {"sample_id": "s4", "raw": ans(six(secs=(4,)))},
    ]
    (FIX / "score_ref.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in refs) + "\n", encoding="utf-8")
    (FIX / "score_raw.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in raws) + "\n", encoding="utf-8")

    out = FIX / "score_metrics.json"
    r = subprocess.run([sys.executable, "-I", str(B / "score_v5.py"), "--raw",
                        str(FIX / "score_raw.jsonl"), "--ref", str(FIX / "score_ref.jsonl"),
                        "--kind", "main", "--out", str(out),
                        "--per-sample", str(FIX / "score_per_sample.jsonl")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        return 1
    m = json.loads(out.read_text(encoding="utf-8"))
    results = []
    add = lambda n, c, x="": results.append((n, bool(c), x))
    add("GeoScore=0.75", m["geoscore"] == 0.75, m["geoscore"])
    add("coverage exact=0.75", m["coverage"]["exact_rate"] == 0.75, m["coverage"])
    add("zones exact=0.75 且 Jaccard=0.75", m["zones"]["exact_rate"] == 0.75
        and m["zones"]["jaccard_mean"] == 0.75, m["zones"])
    add("angular acc=0.75", m["angular_type"]["acc"] == 0.75, m["angular_type"])
    add("sector acc=0.5 (n=4)", m["sector"]["acc"] == 0.5 and m["sector"]["n"] == 4, m["sector"])
    add("class acc=0.5", m["class"]["acc"] == 0.5, m["class"]["acc"])
    add("macro_f1=0.07407", m["class"]["macro_f1"] == 0.07407, m["class"]["macro_f1"])
    add("format stripped=0.75", m["format"]["stripped_legal_rate"] == 0.75, m["format"])
    add("consistency=1.0", m["consistency_mean"] == 1.0, m["consistency_mean"])
    add("measure MAE 全 0", all(v == 0.0 for v in m["measure_mae"].values()), m["measure_mae"])
    add("missing=1 且不入分母", m["n_missing"] == 1 and m["class"]["acc"] == 0.5, m["n_missing"])
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, extra in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name} {extra if not ok else ''}")
    print(f"{n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
