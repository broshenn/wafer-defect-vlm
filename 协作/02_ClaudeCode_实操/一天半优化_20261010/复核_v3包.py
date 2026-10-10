#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立复核 Codex 的 v3 CPU 包（只读；不改 Codex 任何文件）。

检查项：
  A. 池规模与 lot / PNG / sample 隔离
  B. 逐图 hash 与 manifest / 实际 PNG 文件是否一致
  C. 参考（reference）字段是否全部通过 v3 schema，且方向字段带 v3 钟点标识
  D. 方向分母：正向 / 非方向 / unknown 各多少；确认池正向是否为 0
  E. 语义修正是否真的落在参考里：none≠零红点、half_or_more≠近全片、opposed≠穿中心线
  F. 训练候选的正向参考数（21/16 之说）与旋转配方是否物化
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CODEX = Path("D:/pycode/晶圆图研究/协作/01_Codex_指挥/无卡CPU整备_20261010")
REPO = Path("D:/pycode/晶圆图研究")
sys.path.insert(0, str(CODEX))
import core_v3 as C  # noqa: E402

def jl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

out = {}
pools = {n: jl(CODEX / "pools" / f"{n}.jsonl")
         for n in ("adapt", "rl", "rule_dev", "confirmation")}
out["池规模"] = {k: len(v) for k, v in pools.items()}

# ── A. 隔离 ──────────────────────────────────────────────────────────────
ids = {k: {r["sample_id"] for r in v} for k, v in pools.items()}
lots = {k: {r["lot_name"] for r in v} for k, v in pools.items()}
pngs = {k: {r["image_sha256"] for r in v} for k, v in pools.items()}
iso = {}
names = list(pools)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        iso[f"{a}×{b}"] = {"sample": len(ids[a] & ids[b]),
                           "lot": len(lots[a] & lots[b]),
                           "png_sha": len(pngs[a] & pngs[b])}
out["隔离"] = iso
out["池内唯一性"] = {k: {"样本": len(ids[k]) == len(pools[k]),
                     "PNGsha": len(pngs[k]) == len(pools[k])} for k in pools}

# ── B. 逐图 hash 与 manifest ────────────────────────────────────────────
man = {json.loads(l)["sample_id"]: json.loads(l)
       for l in (REPO / "data" / "manifest.jsonl").open(encoding="utf-8")}
mismatch, missing = [], []
for k, rows in pools.items():
    for r in rows:
        sid = r["sample_id"]
        m = man.get(sid)
        if m is None:
            missing.append(sid); continue
        p = REPO / "data" / "images" / f"{sid}.png"
        if not p.exists():
            missing.append(sid); continue
        if sha(p) != r["image_sha256"]:
            mismatch.append(sid)
out["hash核对"] = {"与manifest缺失": len(missing), "与本地PNG不符": len(mismatch),
                 "不符样例": mismatch[:5]}

# ── C. 参考合法性 + v3 钟点标识 ─────────────────────────────────────────
bad_ref, dir_refs, v3_tagged = [], 0, 0
dir_by_pool = defaultdict(Counter)
zone_unknown = Counter()
cov = defaultdict(Counter)
for k, rows in pools.items():
    for r in rows:
        ref = r["reference"]
        errs = C.schema_errors(ref)
        if errs:
            bad_ref.append((k, r["sample_id"], errs))
        dir_by_pool[k][ref["angular_type"]] += 1
        cov[k][ref["coverage_band"]] += 1
        if ref["angular_type"] in ("single", "opposed"):
            dir_refs += 1
            # 方向参考必须来自 v3 钟点重算
            if r.get("features", {}).get("clock_contract") == "hour_centers_v3" \
               or r.get("geometry_source") == "v3_hour_recomputed":
                v3_tagged += 1
        if ref["red_mass_zones"] == "unknown":
            zone_unknown[k] += 1
out["参考"] = {"schema不合法": len(bad_ref), "样例": bad_ref[:3],
             "方向参考(含池名)": {k: dict(v) for k, v in dir_by_pool.items()},
             "覆盖分布": {k: dict(v) for k, v in cov.items()},
             "zones=unknown": dict(zone_unknown)}

# ── D/E. 语义抽查：none / half_or_more / opposed ────────────────────────
allrows = [(k, r) for k, rows in pools.items() for r in rows]
none_rows = [(k, r) for k, r in allrows if r["failure_type"] == "none"]
half_rows = [(k, r) for k, r in allrows
             if r["reference"]["coverage_band"] == "half_or_more"]
opp_rows = [(k, r) for k, r in allrows
            if r["reference"]["angular_type"] == "opposed"]
out["语义抽查"] = {
    "none类样本数": len(none_rows),
    "none类里覆盖为zero的": sum(1 for _, r in none_rows
                                if r["reference"]["coverage_band"] == "zero"),
    "none类里红格>0的（说明none≠零红点被遵守）":
        sum(1 for _, r in none_rows if r["features"].get("n_red", 0) > 0),
    "half_or_more样本数": len(half_rows),
    "opposed样本数": len(opp_rows),
    "opposed的扇区是否都成对相反":
        all((a - b) % 12 == 6 for _, r in opp_rows
            for a, b in [tuple(r["reference"]["clock_sectors"])] if len(r["reference"]["clock_sectors"]) == 2),
}
out["语义抽查"]["opposed方向样例"] = [
    (k, r["sample_id"], r["reference"]["clock_sectors"], r["features"].get("direction_type"))
    for k, r in opp_rows[:5]]

# ── F. 旋转配方是否物化 ─────────────────────────────────────────────────
rot = jl(CODEX / "rotation_recipes.jsonl")
uniq_train = {r["sample_id"] for r in pools["adapt"]} | {r["sample_id"] for r in pools["rl"]}
out["旋转"] = {"配方数": len(rot), "配方样例": rot[0] if rot else None,
             "训练侧唯一原图数": len(uniq_train),
             "配方是否已变成训练行":
                 all(not r["sample_id"].endswith(("_r90", "_r180", "_r270")) for r in pools["adapt"] + pools["rl"])}

print(json.dumps(out, ensure_ascii=False, indent=1))
Path("D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/一天半优化_20261010/复核_v3包.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
