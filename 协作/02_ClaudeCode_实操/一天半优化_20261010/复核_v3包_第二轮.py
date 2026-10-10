#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 包复核·第二轮：修掉上一轮的空检查，并做**独立钟点复算**。

上一轮的错（留痕）：我用 `features.get('n_red', 0)` 统计"none 类的红格数"，
但池里的 `features` 用的是 `defect_count` 键，**没有 n_red**，
于是那条检查全部落到默认值 0，得出"none 类红格>0 的有 0 个"这种**看似通过的空结论**。
本版改用 `defect_count`，并把检查写成"键不存在即报错"，不再静默取默认值。

本轮新增的**独立**验证：不信池里存的扇区，自己从 PNG 反解矩阵、
按 v3 小时中心（12/3/6/9）重算方向，与池里 reference 逐条比对。
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

CODEX = Path("D:/pycode/晶圆图研究/协作/01_Codex_指挥/无卡CPU整备_20261010")
REPO = Path("D:/pycode/晶圆图研究")
MYGEOM = REPO / "协作/02_ClaudeCode_实操/多字段RL_v2_CPU准备_20261009"
sys.path.insert(0, str(CODEX))
sys.path.insert(0, str(MYGEOM))


def jl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


pools = {n: jl(CODEX / "pools" / f"{n}.jsonl")
         for n in ("adapt", "rl", "rule_dev", "confirmation")}
allrows = [(k, r) for k, v in pools.items() for r in v]
rep: dict = {}

# ── 1. 修掉上一轮的空检查：用 defect_count，且缺键即报错 ────────────────
missing_key = [r["sample_id"] for _, r in allrows if "defect_count" not in r["features"]]
none_rows = [(k, r) for k, r in allrows if r["failure_type"] == "none"]
zero_cov = [(k, r) for k, r in allrows if r["reference"]["coverage_band"] == "zero"]
rep["none与红点"] = {
    "features缺defect_count的行": len(missing_key),
    "none类行数": len(none_rows),
    "none类中 defect_count==0 的": sum(1 for _, r in none_rows
                                      if r["features"]["defect_count"] == 0),
    "none类中 defect_count>0 的（这才是 none≠零红点的证据）":
        sum(1 for _, r in none_rows if r["features"]["defect_count"] > 0),
    "none类的 defect_count 中位/最大":
        [int(sorted(r["features"]["defect_count"] for _, r in none_rows)[len(none_rows) // 2]),
         max(r["features"]["defect_count"] for _, r in none_rows)] if none_rows else None,
    "覆盖为 zero 的行数": len(zero_cov),
    "zero 行的失败类别分布": dict(Counter(r["failure_type"] for _, r in zero_cov)),
    "若有zero行，其defect_count是否全为0":
        all(r["features"]["defect_count"] == 0 for _, r in zero_cov),
}

# ── 2. half_or_more 命名与分布 ──────────────────────────────────────────
half = [(k, r) for k, r in allrows if r["reference"]["coverage_band"] == "half_or_more"]
rep["half_or_more"] = {
    "行数": len(half),
    "样例覆盖比": [round(r["features"]["defect_ratio"], 4) for _, r in half[:6]],
    "是否还有旧名 near_all": any(r["reference"]["coverage_band"] == "near_all"
                                 for _, r in allrows),
}

# ── 3. 独立钟点复算：不信池里的扇区，自己从 PNG 算 ──────────────────────
import 几何参考_v2 as G  # noqa: E402
import numpy as np  # noqa: E402

def hour_sector_v3(deg: float) -> int:
    return int(math.floor(((deg + 15) % 360) / 30)) % 12 or 12

def measure_with_v3_hours(matrix):
    """用我自己的实现重算，只把扇区函数换成小时中心；其余阈值不变。"""
    G.sector_of_deg = hour_sector_v3
    G._sector_boundary_margin_deg = lambda d: min((d + 15) % 30, 30 - (d + 15) % 30)
    r = G.measure_matrix(matrix)
    if r.get("direction", {}).get("sector_tie"):
        r["direction_type"], r["clock_sectors"] = "unknown", []
    return r

pos = [ (k, r) for k, r in allrows if r["reference"]["angular_type"] in ("single", "opposed") ]
agree_type, agree_sectors, mism = 0, 0, []
for k, r in pos:
    sid = r["sample_id"]
    rec = G.recover_matrix(REPO / "data" / "images" / f"{sid}.png",
                           hint_hw=tuple(r["matrix_shape"]))
    m = measure_with_v3_hours(rec["matrix"])
    want_type = {"single": "single", "axis": "opposed", "none": "nondirectional",
                 "unknown": "unknown"}[m["direction_type"]]
    same_type = want_type == r["reference"]["angular_type"]
    same_sec = sorted(m["clock_sectors"]) == sorted(r["reference"]["clock_sectors"])
    agree_type += same_type
    agree_sectors += same_sec
    if not (same_type and same_sec):
        mism.append({"pool": k, "sample_id": sid,
                     "池内": [r["reference"]["angular_type"], r["reference"]["clock_sectors"]],
                     "我重算": [want_type, m["clock_sectors"]]})
rep["独立钟点复算"] = {
    "正向参考行数": len(pos),
    "方向类型一致": agree_type,
    "扇区一致": agree_sectors,
    "不一致样例": mism[:6],
}

# ── 4. 训练目标六字段与参考逐条一致 ─────────────────────────────────────
TGT = {"adapt": "data_v3/adapt512_sft.jsonl", "rl": "data_v3/group256_sft4.jsonl"}
bad = []
for pool, fp in TGT.items():
    rows = jl(CODEX / fp)
    ref = {r["sample_id"]: r["reference"] for r in pools[pool]}
    for row in rows:
        sid = row.get("sample_id")
        if sid not in ref:
            bad.append((fp, sid, "不在池里")); continue
        try:
            ans = json.loads(row["messages"][-1]["content"])
        except Exception as e:
            bad.append((fp, sid, f"目标不是JSON:{e}")); continue
        want = ref[sid]
        if set(ans) != set(want):
            bad.append((fp, sid, f"字段集不同 {sorted(ans)} vs {sorted(want)}")); continue
        for k in want:
            if ans[k] != want[k]:
                bad.append((fp, sid, f"{k}: {ans[k]!r} != {want[k]!r}")); break
rep["训练目标与参考一致"] = {"检查文件": list(TGT.values()), "总行数":
                       sum(len(jl(CODEX / f)) for f in TGT.values()),
                       "不一致": len(bad), "样例": bad[:5]}

print(json.dumps(rep, ensure_ascii=False, indent=1))
Path("D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/一天半优化_20261010/复核_v3包_第二轮.json").write_text(
    json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
