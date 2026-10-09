#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""建 8 模型 × 36 图的**匿名描述复核材料**（按冻结枚举 schema）。

匿名规则：
  · 每图的盲号顺序用 `seed:{sample_id}` 逐图打乱；
  · 材料里**只有** 图号 / 图片路径 / 盲号 + 四个描述字段；
  · **无**模型名、**无**公开类别、**无**任何成绩或来源。

输出：`复核材料/分片{1..6}_共6图.json`，另存 `盲号对照_key.json`（**复核期间不得查看**）。
"""
from __future__ import annotations
import io, json, random, hashlib
from pathlib import Path

D = Path(__file__).resolve().parent
REPO = D.parent.parent.parent
DEP = REPO / "协作/02_ClaudeCode_实操/部署原型_20261009_0120"
BLIND = REPO / "协作/01_Codex_指挥/百炼多模型对照_20261009/blind36.jsonl"
KEEP = ["morphology", "radial_zone", "clock_direction", "caption_zh"]
SEED = 3407
BLIND_LABELS = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛"]


def parse_any(raw):
    """把各家原答解析成 dict。剥空 think、剥完整外层围栏（诊断口径）。"""
    import re
    if not isinstance(raw, str):
        return None
    s = re.sub(r"^\s*<think\b[^>]*>\s*</think\s*>\s*", "", raw.strip(), flags=re.S)
    m = re.match(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n?\s*```\s*$", s, re.S)
    if m:
        s = m.group(1).strip()
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else None
    except Exception:
        return None


def main():
    blind = [json.loads(l) for l in io.open(BLIND, encoding="utf-8") if l.strip()]
    labels = {s["sample_id"]: s["公开类别"]
              for s in json.loads((DEP / "样本清单.json").read_text(encoding="utf-8"))["样本"]}

    ans = {}   # model -> {sample_id: dict}
    # ---- 本地 GPU 两组 ----
    for tag, fn in (("Base", "Base_raw.jsonl"), ("L-N3072-3407", "L-N3072-3407_raw.jsonl")):
        f = D / "原答_GPU" / fn
        ans[tag] = {json.loads(l)["sample_id"]: parse_any(json.loads(l)["raw"])
                    for l in io.open(f, encoding="utf-8") if l.strip()}
    # ---- D（部署服务 HTTP）----
    dep = DEP / "性能与自测/http_replay.jsonl"
    ans["D-N3072-3407"] = {}
    for l in io.open(dep, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            ans["D-N3072-3407"][d["sample_id"]] = parse_any(
                (d.get("response") or {}).get("raw_answer"))
    # ---- 百炼四个 ----
    for fn in sorted((D / "原答_API").glob("*_raw.jsonl")):
        tag = fn.stem.replace("_raw", "")
        ans[tag] = {}
        for l in io.open(fn, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                ans[tag][d["sample_id"]] = parse_any(d.get("content"))
    # ---- GLM（外部 102 的子集）----
    ext = REPO / "协作/04_WorkBuddy_复核/外部评测102_20261008_201229"
    ans["GLM-WorkBuddy"] = {}
    for l in io.open(ext / "原答清单.jsonl", encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            f = ext / "raw" / (d["item_id"] + ".txt")
            if f.exists():
                ans["GLM-WorkBuddy"][d["sample_id"]] = parse_any(f.read_text(encoding="utf-8"))

    models = sorted(ans)
    print(f"模型 {len(models)} 个：{models}")

    items, key = [], []
    for b in blind:
        sid = b["sample_id"]
        sets = []
        for m in models:
            o = ans[m].get(sid)
            if isinstance(o, dict):
                sets.append((m, {k: o.get(k) for k in KEEP}))
            else:
                sets.append((m, {k: None for k in KEEP}))
        rnd = random.Random(f"{SEED}:{sid}")
        rnd.shuffle(sets)
        a = [{"盲号": lab, **f} for lab, (_, f) in zip(BLIND_LABELS, sets)]
        items.append({"图号": b["sample_id"], "sample_id": sid,
                      "图片": str((DEP / "图" / f"{sid}.png").resolve()),
                      "答案集": a})
        key.append({"sample_id": sid, "公开类别": labels.get(sid),
                    "盲号对照": {lab: m for lab, (m, _) in zip(BLIND_LABELS, sets)}})

    OUT = D / "复核材料"; OUT.mkdir(exist_ok=True)
    for k in range(6):
        ch = items[k * 6:(k + 1) * 6]
        (OUT / f"分片{k+1}_共{len(ch)}图.json").write_text(json.dumps(
            {"说明": "两个以上匿名答案集的描述事实复核材料。每项含图号、图片路径与"
                     "若干答案集；每个答案集只有盲号与四个描述字段。",
             "枚举口径": "每项六维只接受 SUPPORTED/PARTIAL/CONTRADICTED/UNCERTAIN；"
                         "方向另填 applicable(true/false/unknown) 与 verdict(含 NO_ANSWER)；"
                         "判不了填 UNCERTAIN，缺值填 MISSING，**不得默认放行**",
             "条目": ch}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # 泄漏自检
        data = json.dumps(ch, ensure_ascii=False)
        bad = [x for x in ["公开类别", "来源", "GLM", "Base", "Kimi", "kimi", "qwen",
                           "Center", "Donut", "Edge_", "Random", "Scratch"] if x in data]
        print(f"  分片{k+1}: {len(ch)} 图 × {len(models)} 答案  泄漏检查: {bad or '干净'}")
    (D / "盲号对照_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"写出 复核材料/ 与 盲号对照_key.json（{len(items)} 图 × {len(models)} 模型）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
