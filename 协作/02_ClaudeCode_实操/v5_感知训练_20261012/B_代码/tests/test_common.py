#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""common_v5 测试（合同 B 卡 1：10 项全 PASS）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common_v5 import band_of_ratio, parse_v5, zones_of_frac  # noqa: E402

GOOD = ('<measure>red_ratio=0.27; zone_frac=0.0,0.3,0.7; R1=0.6; R2=0.4</measure>\n'
        '{"defect_class":"Edge_Loc","coverage_band":"20_to_under_50pct",'
        '"red_mass_zones":["middle","edge"],"angular_type":"single","clock_sectors":[1],'
        '"uncertainty":""}')


def main():
    results = []

    def add(name, cond, extra=""):
        results.append((name, bool(cond), extra))

    r = parse_v5(GOOD)
    add("1 样例答案 ok/raw_ok", r["ok"] and r["raw_ok"], r["errors"])

    r = parse_v5('<think>\n\n</think>\n\n' + GOOD)
    add("2 空think前缀 ok=True/raw_ok=False", r["ok"] and not r["raw_ok"], r["errors"])

    r = parse_v5('<think>abc</think>\n' + GOOD)
    add("3 非空think ok=False", not r["ok"], r["errors"])

    r = parse_v5(GOOD.split("\n", 1)[1])
    add("4 少测量行 ok=False", not r["ok"], r["errors"])

    r = parse_v5(GOOD.replace("red_ratio=0.27", "red_ratio=0.3"))
    add("5 red_ratio 一位小数 ok=False", not r["ok"], r["errors"])

    dup = GOOD.split("\n")[1].replace('{"defect_class":"Edge_Loc",',
                                      '{"defect_class":"Edge_Loc","defect_class":"Loc",')
    r = parse_v5(GOOD.split("\n")[0] + "\n" + dup)
    add("6 JSON 重复键 ok=False", not r["ok"], r["errors"])

    import json as _json
    obj = _json.loads(GOOD.split("\n")[1])
    del obj["uncertainty"]
    r = parse_v5(GOOD.split("\n")[0] + "\n" + _json.dumps(obj, ensure_ascii=False))
    add("7 JSON 少字段 ok=False", not r["ok"], r["errors"])

    r = parse_v5(GOOD + "\n多余一行")
    add("8 JSON 后多一行 ok=False", not r["ok"], r["errors"])

    want = {0.0: "zero", 0.0499: "under_5pct", 0.05: "5_to_under_20pct", 0.1999: "5_to_under_20pct",
            0.2: "20_to_under_50pct", 0.4999: "20_to_under_50pct", 0.5: "half_or_more", 1.0: "half_or_more"}
    ok = all(band_of_ratio(k) == v for k, v in want.items())
    add("9 band_of_ratio 边界", ok, {k: band_of_ratio(k) for k in want})

    add("10 zones_of_frac([0,0.1,0.9])", zones_of_frac([0.0, 0.1, 0.9]) == ["middle", "edge"])

    n_pass = sum(1 for _, ok2, _ in results if ok2)
    for name, ok2, extra in results:
        print(f"  {'PASS' if ok2 else 'FAIL'} {name} {extra if not ok2 else ''}")
    print(f"{n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
