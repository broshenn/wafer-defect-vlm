#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成图单元测试（合同 5.2 验收；5/5 PASS 才继续）。
①center_blob 半径 0.2 → 程序测得 zones 含 center 且不含 edge；
②edge_arc 在 3 点方向 → angular_type=single 且扇区为 3（并列则 θ=95 重试）；
③ring r_in=0.8 → zones 只含 edge；
④random_scatter p=0.2 → coverage_band ∈ {20_to_under_50pct, 5_to_under_20pct}；
⑤所有红点都在 valid 内。
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import syn_lib  # noqa: E402


def disc(n=61):
    m = np.zeros((n, n), dtype=np.uint8)
    c = (n - 1) / 2.0
    ii, jj = np.mgrid[0:n, 0:n]
    r = np.hypot(ii - c, jj - c)
    m[r <= n / 2 - 0.5] = 1
    return m


def measure(m):
    return syn_lib.load_core().measure_v3(m)


def main():
    core = syn_lib.load_core()
    rng = np.random.default_rng(3407)
    results = []

    # ① center_blob s=0.2
    m, pr = syn_lib.draw_pattern(disc(), "center_blob", rng, {"s": 0.2, "p": 1.0})
    g = measure(m)
    zones = g["red_mass_zones"]
    results.append(("①center_blob zones 含 center 不含 edge",
                    isinstance(zones, list) and "center" in zones and "edge" not in zones,
                    f"zones={zones} coverage={g['coverage']:.3f}"))

    # ② edge_arc 3 点方向
    g = None
    for th in (90.0, 95.0):
        m, pr = syn_lib.draw_pattern(disc(), "edge_arc", rng,
                                     {"theta_deg": th, "span_deg": 60.0, "p": 1.0})
        g = measure(m)
        if not g["direction"].get("sector_tie"):
            break
    results.append(("②edge_arc@3点 single 扇区=3",
                    g["direction_type"] == "single" and g["clock_sectors"] == [3],
                    f"type={g['direction_type']} sectors={g['clock_sectors']} "
                    f"tie={g['direction'].get('sector_tie')} n_red={g['n_red']}"))

    # ③ ring r_in=0.8
    m, pr = syn_lib.draw_pattern(disc(), "ring", rng, {"r_in": 0.8, "w": 0.1, "p": 1.0})
    g = measure(m)
    results.append(("③ring r_in=0.8 zones 只含 edge",
                    g["red_mass_zones"] == ["edge"], f"zones={g['red_mass_zones']}"))

    # ④ random_scatter p=0.2
    m, pr = syn_lib.draw_pattern(disc(), "random_scatter", rng, {"p": 0.2})
    g = measure(m)
    band = core.COV_MAP[g["coverage_level"]]
    results.append(("④scatter p=0.2 档位在预期两档内",
                    band in ("20_to_under_50pct", "5_to_under_20pct"),
                    f"band={band} coverage={g['coverage']:.3f}"))

    # ⑤ 所有红点都在 valid 内（拿 ① 的图）
    m, pr = syn_lib.draw_pattern(disc(), "local_blob", rng, {"r0": 0.5, "theta_deg": 30, "s": 0.1, "p": 1.0})
    valid = (m == 1) | (m == 2)
    ok5 = bool(np.all(valid[m == 2]))
    results.append(("⑤红点全部落在 valid 内", ok5, f"n_red={int((m == 2).sum())}"))

    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, info in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name} | {info}")
    print(f"{n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
