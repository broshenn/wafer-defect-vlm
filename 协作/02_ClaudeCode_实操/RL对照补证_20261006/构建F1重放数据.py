#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 F2 的**真实 20 组 ID 与顺序**构造 F1 replay 数据。

为什么必须这样做：原 F1 用的是 pool **前 20 张**（Center×10 + Donut×10），
而 F2 实际训练用的是 reward_audit 里记录的另外 20 张（类别混合）——
两者**不是同一批图**，原 F1 不能作为公平对照。

本文件同时输出：F2 实际组序列、与原 F1 图集的重叠情况。
"""
from __future__ import annotations
import io, json, hashlib
from pathlib import Path

R = Path("D:/pycode/.ssh-tmp/rlfix")
POOL = Path("D:/pycode/.ssh-tmp/rl_pool90")
OUT = R
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]


def load(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


# F2 真实组序列（来自 reward_audit.jsonl，每 4 条一组）
audit = load(R / "f2_run20_reward_audit.jsonl")
groups = []
for i in range(0, len(audit), 4):
    g = audit[i:i + 4]
    assert len({x["sample_id"] for x in g}) == 1, f"组 {i//4} 不是同一张图: {[x['sample_id'] for x in g]}"
    assert len({x["gold"] for x in g}) == 1, f"组 {i//4} gold 不一致"
    groups.append({"idx": i // 4, "sample_id": g[0]["sample_id"], "gold": g[0]["gold"],
                   "rewards": [x["reward"] for x in g],
                   "same_group": len(set(x["reward"] for x in g)) == 1,
                   "responses": [x["response"] for x in g]})
print(f"F2 实际组数 {len(groups)}")
print(f"  组内同 ID: {sum(1 for g in groups if True)}/{len(groups)}（已断言）")
print(f"  同分组: {sum(1 for g in groups if g['same_group'])}/{len(groups)}")
print(f"  全程平均奖励: {sum(x for g in groups for x in g['rewards'])/(len(groups)*4):.4f}")
print(f"  末步 reward/std: {groups[-1]['rewards']}")
print(f"  实际序列: {[g['sample_id'] for g in groups]}")

pool = {r["sample_id"]: r for r in load(POOL / "rl_pool90.jsonl")}
first20 = [r["sample_id"] for r in load(POOL / "rl_pool90.jsonl")][:20]
actual20 = [g["sample_id"] for g in groups]
print()
print(f"pool 前 20 张: {first20[:6]} …  类别 {[pool[i]['label'] for i in first20[:6]]}…")
from collections import Counter
print(f"  前 20 类别分布: {dict(Counter(pool[i]['label'] for i in first20))}")
print(f"F2 实际 20 张类别分布: {dict(Counter(g['gold'] for g in groups))}")
print(f"两者交集: {len(set(first20) & set(actual20))} 张")
print(f"  只在原 F1 里的: {sorted(set(first20) - set(actual20))}")
print(f"  只在 F2 里的  : {sorted(set(actual20) - set(first20))}")

# 构造 replay：按 F2 顺序，每 ID 连出现 4 次
sft = {r["sample_id"]: r for r in load(POOL / "rl_pool90_sft_f1.jsonl")}
rows = []
for g in groups:
    sid = g["sample_id"]
    assert sid in sft, sid
    assert sft[sid]["messages"][1]["content"] == json.dumps({"defect_class": g["gold"]},
                                                            ensure_ascii=False,
                                                            separators=(",", ":")), sid
    for _ in range(4):
        rows.append(sft[sid])
p = OUT / "f1_replay_f2order.jsonl"
with io.open(p, "w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\n写出 {p}  {len(rows)} 行  sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")

meta = {
    "来源": "F2 reward_audit.jsonl 的真实 20 组 ID 与顺序",
    "audit_sha256": hashlib.sha256((R / "f2_run20_reward_audit.jsonl").read_bytes()).hexdigest(),
    "n_groups": len(groups), "n_rows": len(rows),
    "groups": [{k: v for k, v in g.items() if k != "responses"} for g in groups],
    "pool_first20": first20,
    "pool_first20_class_dist": dict(Counter(pool[i]["label"] for i in first20)),
    "f2_actual_class_dist": dict(Counter(g["gold"] for g in groups)),
    "overlap_with_pool_first20": len(set(first20) & set(actual20)),
    "mean_reward_all": round(sum(x for g in groups for x in g["rewards"]) / (len(groups) * 4), 4),
    "n_same_group": sum(1 for g in groups if g["same_group"]),
    "replay_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    "note": "replay 按 F2 顺序每 ID 连续 4 次；训练时**显式关闭 dataset/dataloader 打乱**，"
            "否则 4 份会被拆开、每步不再是同一张图。",
}
(OUT / "f1_replay_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print("写出 f1_replay_meta.json")
