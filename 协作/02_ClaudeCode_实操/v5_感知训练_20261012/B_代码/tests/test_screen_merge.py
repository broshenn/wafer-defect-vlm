#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""screen_rl_pool merge 逻辑测试（假分片）。
设计：c0 全对（零方差）、c1 [错,对,错,对]（不全同 → 入选，pass@1 错）、c2 [对,错,对,错]（入选）、
c3 全错（零方差）、c4 全对（零方差）。入选=c1,c2 + 按顺序补足 c0。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from screen_rl_pool import merge  # noqa: E402


def ref_row(sid):
    six = {"defect_class": "Loc", "coverage_band": "20_to_under_50pct",
           "red_mass_zones": ["middle", "edge"], "angular_type": "single",
           "clock_sectors": [3], "uncertainty": ""}
    return {"sample_id": sid, "reference_json": json.dumps(six, ensure_ascii=False)}


GOOD = ('<measure>red_ratio=0.27; zone_frac=0.0,0.3,0.7; R1=0.6; R2=0.4</measure>\n'
        '{"defect_class":"Loc","coverage_band":"20_to_under_50pct",'
        '"red_mass_zones":["middle","edge"],"angular_type":"single","clock_sectors":[3],"uncertainty":""}')
BAD = "我不会。"


def mk(raw):
    ok = raw.startswith("<measure>")
    return {"raw": raw, "reward": 1.0 if ok else -1.0, "format_ok": ok}


def main():
    results = []
    add = lambda n, c, x="": results.append((n, bool(c), x))  # noqa: E731
    cands = [ref_row(f"c{i}") for i in range(5)]
    shards = [
        [{"sample_id": "c0", "samples": [mk(GOOD)] * 4},
         {"sample_id": "c1", "samples": [mk(BAD), mk(GOOD), mk(BAD), mk(GOOD)]}],
        [{"sample_id": "c2", "samples": [mk(GOOD), mk(BAD), mk(GOOD), mk(BAD)]},
         {"sample_id": "c3", "samples": [mk(BAD)] * 4}],
        [{"sample_id": "c4", "samples": [mk(GOOD)] * 4}],
    ]
    sel, passk, scored = merge(shards, cands, None, n_select=3)
    add("组内不全同入选 + 按顺序补足", sel == ["c1", "c2", "c0"], sel)
    add("补足计数=1", passk["n_filled_from_remainder"] == 1, passk)
    add("零方差组=3（c0,c3,c4）", passk["zero_variance"] == 3, passk)
    add("pass@1=0.6", passk["pass@1"] == 0.6, passk["pass@1"])
    add("pass@4=0.8", passk["pass@4"] == 0.8, passk["pass@4"])
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, extra in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name} {extra if not ok else ''}")
    print(f"{n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
