#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reward_v5 测试（合同 B 卡 2：15 项全 PASS；每条打印名字和实际奖励值）。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import os  # noqa: E402

from reward_v5 import WaferV5Reward, score_v5  # noqa: E402


def ref_of(defect_class="Edge_Loc", coverage_band="20_to_under_50pct",
           zones=("middle", "edge"), angular="single", sectors=(3,),
           ratio=0.27, zf=(0.0, 0.3, 0.7), r1=0.6, r2=0.4, uncertainty=""):
    return {"defect_class": defect_class, "coverage_band": coverage_band,
            "red_mass_zones": list(zones) if zones != "unknown" else "unknown",
            "angular_type": angular,
            "clock_sectors": list(sectors),
            "uncertainty": uncertainty,
            "measure": {"red_ratio": round(ratio, 2), "zone_frac": [round(z, 1) for z in zf],
                        "R1": round(r1, 1), "R2": round(r2, 1)},
            "measure_exact": {"red_ratio": ratio, "zone_frac": list(zf), "R1": r1, "R2": r2}}


def ans(ratio=0.27, zf=(0.0, 0.3, 0.7), r1=0.6, r2=0.4, defect_class="Edge_Loc",
        coverage_band="20_to_under_50pct", zones=("middle", "edge"), angular="single",
        sectors=(3,), uncertainty="", wrap_empty_think=False, mz=None):
    z = list(zones) if zones != "unknown" else "unknown"
    obj = {"defect_class": defect_class, "coverage_band": coverage_band,
           "red_mass_zones": z, "angular_type": angular,
           "clock_sectors": list(sectors), "uncertainty": uncertainty}
    mz = mz or f"{zf[0]:.1f},{zf[1]:.1f},{zf[2]:.1f}"
    text = (f"<measure>red_ratio={ratio:.2f}; zone_frac={mz}; R1={r1:.1f}; R2={r2:.1f}</measure>\n"
            + json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    if wrap_empty_think:
        text = "<think>\n\n</think>\n\n" + text
    return text


def main():
    results = []

    def add(name, cond, extra=""):
        results.append((name, bool(cond), extra))

    ref = ref_of()
    # 1 完全正确
    for mode in ("full", "binary", "noproc"):
        r = score_v5(ans(), ref, mode)
        add(f"1 [{mode}] 全对=1.0", abs(r["reward"] - 1.0) < 1e-9, r["reward"])
    # 2 格式不合法
    for mode in ("full", "binary", "noproc"):
        r = score_v5("抱歉我不知道", ref, mode)
        add(f"2 [{mode}] 格式错=-1", r["reward"] == -1.0 and not r["ok"], r["reward"])
    # 3 只有类别错
    r = score_v5(ans(defect_class="Loc"), ref, "full")
    add("3 类别错 full=0.40", abs(r["reward"] - 0.40) < 1e-9, r["reward"])
    # 4 类别 unknown
    r = score_v5(ans(defect_class="unknown"), ref, "full")
    add("4 类别unknown full=0.76", abs(r["reward"] - 0.76) < 1e-9, r["reward"])
    # 5 coverage 差一档
    r = score_v5(ans(coverage_band="5_to_under_20pct"), ref, "full")
    add("5a full coverage=0.5", r["parts"]["coverage"] == 0.5, r["parts"])
    r = score_v5(ans(coverage_band="5_to_under_20pct"), ref, "binary")
    add("5b binary coverage=0", r["parts"]["coverage"] == 0.0, r["parts"])
    # 6 zones [middle,edge] 答 [edge]
    r = score_v5(ans(zones=("edge",)), ref, "full")
    add("6a full zones=0.5", r["parts"]["zones"] == 0.5, r["parts"])
    r = score_v5(ans(zones=("edge",)), ref, "binary")
    add("6b binary zones=0", r["parts"]["zones"] == 0.0, r["parts"])
    # 7 single[3] 答 single[4]
    r = score_v5(ans(sectors=(4,)), ref, "full")
    add("7a full angular=0.8", abs(r["parts"]["angular"] - 0.8) < 1e-9, r["parts"])
    r = score_v5(ans(sectors=(4,)), ref, "binary")
    add("7b binary angular=0", r["parts"]["angular"] == 0.0, r["parts"])
    # 8 opposed [3,9] 答 [4,10]
    ref8 = ref_of(angular="opposed", sectors=(3, 9))
    r = score_v5(ans(angular="opposed", sectors=(4, 10)), ref8, "full")
    add("8 opposed 差1 full angular=0.8", abs(r["parts"]["angular"] - 0.8) < 1e-9, r["parts"])
    # 9 nondirectional
    ref9 = ref_of(angular="nondirectional", sectors=())
    r = score_v5(ans(angular="nondirectional", sectors=()), ref9, "full")
    add("9a nondirectional 对=1.0", r["parts"]["angular"] == 1.0, r["parts"])
    r = score_v5(ans(angular="single", sectors=(3,)), ref9, "full")
    add("9b nondirectional 答single=0", r["parts"]["angular"] == 0.0, r["parts"])
    # 10 参考 angular unknown → 分量移除
    ref10 = ref_of(angular="unknown", sectors=())
    r = score_v5(ans(angular="unknown", sectors=()), ref10, "full")
    add("10 angular 不在 weights_used", "angular" not in r["weights_used"]
        and "angular" not in r["parts"], r["weights_used"])
    # 11 全 unknown 刷分攻击
    attack = ans(ratio=0.0, zf=(0.0, 0.0, 0.0), r1=0.0, r2=0.0, defect_class="unknown",
                 coverage_band="unknown", zones="unknown", angular="unknown", sectors=())
    r = score_v5(attack, ref, "full")
    add("11 全unknown攻击 <0.25", r["reward"] < 0.25, round(r["reward"], 4))
    # 12 自相矛盾：red_ratio=0.03 但答 20_to_under_50pct
    ref12 = ref_of(coverage_band="under_5pct", ratio=0.03)
    r = score_v5(ans(ratio=0.03, coverage_band="20_to_under_50pct"), ref12, "full")
    add("12 矛盾 consistency=0.5", abs(r["parts"]["consistency"] - 0.5) < 1e-9, r["parts"])
    # 13 noproc 无 measure/consistency
    r = score_v5(ans(), ref, "noproc")
    add("13 noproc 无 measure/consistency", "measure" not in r["weights_used"]
        and "consistency" not in r["weights_used"], r["weights_used"])
    # 14 奖励类 raise
    old = os.environ.pop("WAFER_REWARD_MODE", None)
    raised = 0
    try:
        WaferV5Reward()
    except RuntimeError:
        raised += 1
    os.environ["WAFER_REWARD_MODE"] = "full"
    rw = WaferV5Reward()
    try:
        rw(["x"])
    except RuntimeError:
        raised += 1
    try:
        rw(["x"], reference_json=[json.dumps(ref)], sample_id=["a", "b"])
    except RuntimeError:
        raised += 1
    add("14 未设模式/缺列/长度不齐 raise", raised == 3, raised)
    # 15 审计字段 + 空 think 前缀
    rw2 = WaferV5Reward()
    rw2([ans(wrap_empty_think=True)], reference_json=[json.dumps(ref)], sample_id=["s1"])
    a = rw2.audit[0]
    add("15 审计含 had_empty_think_prefix/raw_ok",
        ("had_empty_think_prefix" in a and "raw_ok" in a
         and a["had_empty_think_prefix"] is True and a["raw_ok"] is False), a)

    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, extra in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name} | 实际={extra if 'reward' in name or name.startswith(('3', '4', '11')) else ''}")
    print(f"{n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
