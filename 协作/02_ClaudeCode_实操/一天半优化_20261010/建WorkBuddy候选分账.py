#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WorkBuddy 候选分账（CPU，只读；不追加训练集、不改原件）。

输入：
  · `协作/01_Codex_指挥/接手续验收_20261011/WorkBuddy30_独立核验.json`
      —— followup_allocation（21 usable / 9 review_or_reject / 54 unreviewed）+ per_item（30 条）
  · `协作/04_WorkBuddy_复核/方向形态84_v3_20261010_200950/wbv3_NNN.raw.txt` —— 旧 84 首次原答
  · `协作/04_WorkBuddy_复核/内容复核30_v3_20261010_233400/rv30_NNN.*.json` —— 本轮 30 的观察/复核/台账
  · 六字段池（adapt/rl/rule_dev/confirmation）与 D_N3072 名单 —— 做 ID/lot/PNG 分账

输出（本目录）：
  · `候选清单.jsonl`      84 行，一行一条，含来源/哈希/裁决/隔离/审核状态
  · `来源与哈希.json`      只读输入的指纹 + 取回时间
  · `原件副本/`            旧 84 原答 + 本轮 30 的观察/复核，**按来源分目录**，一字不改
  · `候选分账.md`          汇总

纪律：
  · **不按多数票定真值**、**不从 suggested_revision 自动生成答案**；
  · 未复核的 54 条保持"未复核"，不因格式或抽检放行；
  · Kimi（用户确认的界面选择）与 GLM（会话声明型号）**分别记账**；
  · 只产清单与状态，**不写入任何训练集**。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

ROOT = Path("D:/pycode/晶圆图研究")
HERE = Path(__file__).resolve().parent
CODEX = ROOT / "协作/01_Codex_指挥/接手续验收_20261011"
OLD84 = ROOT / "协作/04_WorkBuddy_复核/方向形态84_v3_20261010_200950"
NEW30 = ROOT / "协作/04_WorkBuddy_复核/内容复核30_v3_20261010_233400"
POOLS = ROOT / "协作/01_Codex_指挥/无卡CPU整备_20261010/pools"
OUT = HERE / "WorkBuddy候选分账_20261011"
STATE = OUT / "_断点状态.json"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _manifest_sha(d: Path, item: str):
    p = d / "raw_sha256_manifest.txt"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.endswith(f"*{item}.raw.txt"):
            return line.split()[0]
    return None


def jl(p: Path):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="只处理前 N 条（先 20 条冒烟）")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    seed = 3407  # 本任务无随机过程；显式记录，保证与全项目一致

    ver = json.loads((CODEX / "WorkBuddy30_独立核验.json").read_text(encoding="utf-8"))
    fa = ver["followup_allocation"]
    usable = list(fa["model_usable_candidates"])
    review_reject = list(fa["review_or_reject"])
    unreviewed = list(fa["remaining_unreviewed"])
    hold = set(fa["semantic_hold_in_usable"])
    all_items = sorted(usable + review_reject + unreviewed)
    assert len(all_items) == 84 == len(set(all_items)), "分账三组必须恰好覆盖 84 条且不重复"

    per_item = {r["item_id"]: r for r in ver["per_item"]}
    BLIND = {r["item_id"]: r for r in jl(
        ROOT / "协作/01_Codex_指挥/无卡CPU整备_20261010/WorkBuddy84/blind84.jsonl")}

    # 本轮 30 的观察/复核文件（rv30_NNN ↔ item_id 不是同号，按内容里的 item_id 对齐）
    obs, rev, meta = {}, {}, {}
    for f in NEW30.glob("rv30_*.observation_raw.json"):
        rid = f.name.split(".")[0]
        revf = NEW30 / f"{rid}.review.json"
        if revf.exists():
            body = json.loads(revf.read_text(encoding="utf-8"))
            item = body.get("item_id")
            if item:
                obs[item] = f
                rev[item] = revf
        mf = NEW30 / f"{rid}.observation_meta.json"
        if mf.exists():
            meta[rid] = mf

    # 隔离账：六字段四池 + D_N3072
    pool_of = {}
    for name in ("adapt", "rl", "rule_dev", "confirmation"):
        for r in jl(POOLS / f"{name}.jsonl"):
            pool_of.setdefault(r["sample_id"], set()).add(name)
    d3072 = set()
    dv2 = ROOT / "协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/数据/D_N3072.jsonl"
    if dv2.exists():
        d3072 = {json.loads(l)["sample_id"] for l in dv2.open(encoding="utf-8") if l.strip()}

    GREEN_LEAD = re.compile(r"^\s*绿色")
    rows = []
    for i, item in enumerate(all_items, 1):
        if a.smoke and i > a.smoke:
            break
        if a.limit and i > a.limit:
            break
        raw_f = OLD84 / f"{item}.raw.txt"
        raw_txt = raw_f.read_text(encoding="utf-8").strip() if raw_f.exists() else None
        raw_obj = json.loads(raw_txt) if raw_txt else {}
        b = BLIND.get(item, {})
        sid = b.get("sample_id") or per_item.get(item, {}).get("sample_id")
        # 定位对象判定（**只看定位字段**，且只看是否把绿色当作被定位主体）
        loc = str(raw_obj.get("main_location_zh", ""))
        if item in hold:
            target = "绿色合格区（Codex 已判定，另挂起）"
        elif GREEN_LEAD.match(loc) or "绿色主要结构" in loc or "绿色团块位于" in loc:
            target = "绿色合格区（本轮初筛命中，待复核）"
        elif "全局" in loc and "散布" in loc:
            target = "全局分布"
        elif loc.strip():
            target = "红色失效结构"
        else:
            target = "不确定"

        rv = per_item.get(item)
        rows.append({
            "item_id": item,
            "sample_id": sid,
            "image_path": (b.get("image_path") or "").replace("\\", "/"),
            "image_sha256": b.get("image_sha256"),
            "allocation": ("candidate_suspend" if item in hold else
                           "candidate_pending_review" if item in usable else
                           "review_or_reject_isolated" if item in review_reject else
                           "unreviewed_keep"),
            "target_object": target,
            "old_raw_sha256": sha(raw_f) if raw_f.exists() else None,
            "old_raw_sha256_in_manifest": _manifest_sha(OLD84, item),
            "observation_raw_sha256": sha(obs[item]) if item in obs else None,
            "review_id": rv.get("review_id") if rv else None,
            "model_overall": rv.get("overall") if rv else None,
            "contradicted_fields": rv.get("contradicted_fields") if rv else None,
            "unverifiable_fields": rv.get("unverifiable_fields") if rv else None,
            "has_suggested_revision": bool((rv or {}).get("suggested_revision")),
            "time_sequence_consistent": rv.get("time_sequence_consistent") if rv else None,
            "provenance_original": "Kimi（用户确认的界面选择；真实后端与采样参数未提供）",
            "provenance_review": ("GLM-5.3-Flash（会话声明型号；真实后端与采样参数未提供）"
                                  if rv else None),
            "gold_eligible": False,
            "notes": ("原答把定位对象写成绿色合格区，未改成红色位置；先挂起"
                      if item in hold else None),
            "isolation": {
                "in_v3_pools": sorted(pool_of.get(sid, [])) if sid else [],
                "seen_by_D_N3072": (sid in d3072) if sid else None,
                "treat_as": "训练侧已接触图，**不能当新留出测试**",
            },
        })

    summary = {
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed, "smoke": a.smoke, "dry_run": a.dry_run,
        "总数": len(rows),
        "分账计数": dict(Counter(r["allocation"] for r in rows)),
        "target_object 计数": dict(Counter(r["target_object"] for r in rows)),
        "来源": {"输入指纹": {
            "WorkBuddy30_独立核验.json": sha(CODEX / "WorkBuddy30_独立核验.json"),
            "旧84目录": str(OLD84), "本轮30目录": str(NEW30)}},
    }
    if a.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        print("dry-run：未写任何文件"); return 0

    OUT.mkdir(exist_ok=True)
    with (OUT / "候选清单.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    # 原件副本：按来源分目录，一字不改
    dst_old = OUT / "原件副本/旧84_原答"; dst_new = OUT / "原件副本/本轮30_观察与复核"
    dst_old.mkdir(parents=True, exist_ok=True); dst_new.mkdir(parents=True, exist_ok=True)
    for r in rows:
        f = OLD84 / f"{r['item_id']}.raw.txt"
        if f.exists():
            shutil.copy2(f, dst_old / f.name)
        for suffix in ("observation_raw.json", "review.json", "observation_meta.json",
                       "started.json", "done.json"):
            rid = r.get("review_id")
            if rid:
                g = NEW30 / f"{rid}.{suffix}"
                if g.exists():
                    shutil.copy2(g, dst_new / g.name)
    (OUT / "来源与哈希.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    STATE.write_text(json.dumps({"done": [r["item_id"] for r in rows],
                                 "seed": seed}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
