#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bootstrap_v5 测试（B 卡 8）：相同输入 → 差值 0、区间 [0,0]；B 的 coverage 全改错 → 差值为正且区间不含 0。"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = HERE.parent
FIX = HERE / "fixtures"


def write_per(p, rows):
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                 encoding="utf-8", newline="\n")


def main():
    FIX.mkdir(exist_ok=True)
    base = [{"sample_id": f"s{i}", "lot_name": f"lot{i % 3}", "present": True,
             "ok": True, "cov_exact": True, "zones_exact": True, "ang_ok": True,
             "class_ok": True, "class_pred": "Center", "true_class": "Center"} for i in range(30)]
    a = FIX / "bs_a.jsonl"
    b = FIX / "bs_b_same.jsonl"
    c = FIX / "bs_b_bad.jsonl"
    write_per(a, base)
    write_per(b, base)
    bad = []
    for r in base:
        r2 = dict(r)
        r2["cov_exact"] = False
        bad.append(r2)
    write_per(c, bad)
    results = []
    add = lambda n, cond, x="": results.append((n, bool(cond), x))  # noqa: E731

    r = subprocess.run([sys.executable, "-I", str(B / "bootstrap_v5.py"), "--a", str(a),
                        "--b", str(b), "--metric", "geoscore", "--n", "2000"],
                       capture_output=True, text=True)
    out = json.loads(r.stdout.strip().splitlines()[-1])
    add("相同输入：点估计 0 且区间 [0,0]", out["point_estimate"] == 0.0 and out["ci95"] == [0.0, 0.0], out)
    add("相同输入：不含 0 标志为 False", out["excludes_zero"] is False)

    r = subprocess.run([sys.executable, "-I", str(B / "bootstrap_v5.py"), "--a", str(a),
                        "--b", str(c), "--metric", "geoscore", "--n", "2000"],
                       capture_output=True, text=True)
    out = json.loads(r.stdout.strip().splitlines()[-1])
    add("B 全错：差值为正且区间不含 0",
        out["point_estimate"] > 0 and out["excludes_zero"] is True and out["ci95"][0] > 0, out)

    r = subprocess.run([sys.executable, "-I", str(B / "bootstrap_v5.py"), "--a", str(a),
                        "--b", str(b), "--metric", "class_macro_f1", "--n", "1000"],
                       capture_output=True, text=True)
    out = json.loads(r.stdout.strip().splitlines()[-1])
    add("macro_f1 相同输入差 0", out["point_estimate"] == 0.0, out)

    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, extra in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name} {extra if not ok else ''}")
    print(f"{n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
