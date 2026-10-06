#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩容准备 第 5 步：上传量核算 + GPU 预算估算。"""
from __future__ import annotations
import hashlib, io, json
from pathlib import Path

REPO = Path("D:/pycode/晶圆图研究")
OUT = Path("D:/pycode/.ssh-tmp/exp")
IMAGES = REPO / "data" / "images"

SCHOOL_UNIQUE = 372          # 学校去重后 PNG 数（现场实测）
SCHOOL_DIRS = {"datasets/images": 198, "datasets/pkg_20261003/images": 198,
               "datasets/val90/images": 84, "datasets/rl_pool90/images": 90,
               "datasets/expand/images": 0}
# 学校已有的图集（去重后 372）= 198 张子集 ∪ val90 84 ∪ rl_pool90 90
# 其中 198 含 train180 的 180 张。
STEP_SEC = 5.357             # 单卡实测 s/it（train_a1_seed3407）
LOAD_MIN = 1.5               # 模型加载 + 首次 eval 的墙钟经验值
EVAL_MIN = 2.6               # dev18 + val84 一次完整评测


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rd(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


print("=" * 78)
print("上传量核算")
print("=" * 78)
known = set()
for p in (REPO / "协作" / "02_ClaudeCode_实操" / "学校训练准备_20261002-025848" / "train_180.jsonl",
          Path("D:/pycode/.ssh-tmp/val90/val90.jsonl"),
          REPO / "协作" / "02_ClaudeCode_实操" / "今晚类别RL_20261006_0325" / "rl_pool90.jsonl"):
    known |= {r["sample_id"] for r in rd(p)}
print(f"  学校确定已有的图片集合: {len(known)} 张")

up = {}
total_bytes = 0
for lv in (360, 720, 1440):
    ids = [x["sample_id"] for x in rd(OUT / f"scale_{lv}.jsonl")]
    need = [i for i in ids if i not in known]
    b = sum((IMAGES / f"{i}.png").stat().st_size for i in need)
    total_bytes += 0
    up[lv] = {"level": lv, "total": len(ids), "already_on_school": len(ids) - len(need),
              "need_upload": len(need), "bytes": b, "MiB": round(b / 2**20, 2)}
    print(f"  scale_{lv}: {len(ids)} 条，学校已有 {len(ids)-len(need)}，**需补传 {len(need)} 张** "
          f"（{b/2**20:.2f} MiB）")
need1440 = up[1440]["need_upload"]

print()
print("=" * 78)
print("GPU 预算估算（基于单卡实测，不是成绩）")
print("=" * 78)
print(f"  实测步速 {STEP_SEC} s/it（A1 seed3407，单卡，pdb=1/GA=4，seq≈320）")
print(f"  加载 {LOAD_MIN} 分钟；一次 dev18+val84 评测 {EVAL_MIN} 分钟")
print()
print(f"  {'规模':>6s} {'1 epoch 步数':>12s} {'训练分钟':>9s} {'+加载':>7s} {'+评测':>7s} {'合计':>8s}")
est = {}
for lv in (180, 360, 720, 1440):
    steps_ep = lv // 4                      # 有效 batch 4
    train = steps_ep * STEP_SEC / 60
    total = train + LOAD_MIN + EVAL_MIN
    est[lv] = {"epoch_steps": steps_ep, "train_min": round(train, 1),
               "load_min": LOAD_MIN, "eval_min": EVAL_MIN, "total_min": round(total, 1)}
    print(f"  {lv:6d} {steps_ep:12d} {train:9.1f} {LOAD_MIN:7.1f} {EVAL_MIN:7.1f} {total:8.1f}")

print()
print("  两种比较方案（回答不同问题）:")
fixed = 60 * STEP_SEC / 60 + LOAD_MIN + EVAL_MIN
print(f"   A) 固定呈现预算 60 步（= 240 次样本呈现），数据量加大也**不增步数**: "
      f"每档约 {fixed:.1f} 设备分钟")
print(f"      → 回答'同样训练量下，给更多数据有没有用'；**不能说 1440 条都被学到了**")
print(f"   B) 充分覆盖 1 epoch: 每档步数 = 规模/4，合计 "
      f"{sum(v['total_min'] for v in est.values()):.1f} 设备分钟")
print(f"      → 回答'把数据全过一遍能到什么水平'；但训练量随规模变化，**不能与 A 混比**")

meta = {
    "school_dirs": SCHOOL_DIRS, "school_unique_png": SCHOOL_UNIQUE,
    "upload_need": up, "upload_1440_need_png": need1440,
    "step_sec": STEP_SEC, "load_min": LOAD_MIN, "eval_min": EVAL_MIN,
    "estimates": est,
    "方案A_固定60步_每档分钟": round(fixed, 1),
    "方案B_充分1epoch_合计分钟": round(sum(v["total_min"] for v in est.values()), 1),
    "本任务GPU授权": 0,
    "边界": "估算是**基于单卡实测步速的外推**，不是成绩；" +
            "当前累计账 157.70/180、剩 22.30 需在执行后写入 ledger 并保留区间依据。",
}
(OUT / "budget_estimate.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"\n写出 budget_estimate.json")
print(f"\n**扩容所需 GPU 预算（供裁决）**:")
print(f"  方案A 固定 60 步 × 4 档 = {fixed*4:.0f} 设备分钟")
print(f"  方案B 充分 1 epoch × 4 档 = {sum(v['total_min'] for v in est.values()):.0f} 设备分钟")
print(f"  仅跑 1440 单档 1 epoch = {est[1440]['total_min']:.0f} 设备分钟")
print(f"  当前剩余 = 22.30 设备分钟 → **不足以启动任何扩容训练，需追加授权**")
