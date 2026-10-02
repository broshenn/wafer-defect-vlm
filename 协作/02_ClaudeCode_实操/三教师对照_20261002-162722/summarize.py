#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总三教师对照结果：逐候选一致率、互相分歧、用量、格式。"""
import collections
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs"
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]

# 真值（audit 表，仅本地使用，不给模型）
audit = {}
for l in open(HERE.parent / "口径修正与准备_20261002-021207/audit_selected.jsonl",
              encoding="utf-8"):
    if l.strip():
        o = json.loads(l)
        audit[o["item_id"]] = o["failure_type_normalized"]


def parse_cls(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            v = json.loads(m.group(0)).get("defect_class")
            if isinstance(v, str):
                return v if v in CLASSES + ["unknown"] else None
        except json.JSONDecodeError:
            pass
    for c in CLASSES + ["unknown"]:
        if c in text:
            return c
    return None


# 收集：同一 (candidate, item_id) 只取**最新**一次成功的
best = {}
fails = []
for d in sorted(RUNS.iterdir()):
    if not d.is_dir():
        continue
    for f in sorted(d.glob("*__*.json")):
        o = json.loads(f.read_text(encoding="utf-8"))
        k = (o["candidate"], o["item_id"])
        if o.get("http_status") == 200:
            best[k] = o                      # 后写的覆盖先写的
        else:
            fails.append((o["candidate"], o["item_id"], o.get("http_status")))

cands = sorted({k[0] for k in best})
items = sorted({k[1] for k in best}, key=lambda x: int(x.split("_")[1]))
print("=" * 78)
print("三教师对照 · 汇总")
print("=" * 78)
print(f"覆盖：{len(items)} 张 × {len(cands)} 候选；成功记录 {len(best)} 条")

# 1 逐候选
print("\n[1] 逐候选（对照真值）")
print(f"    {'候选':10s} {'成功':>5s} {'正确':>5s} {'一致率':>8s} {'格式通过':>9s} "
      f"{'总token':>9s} {'中位耗时':>9s}")
per_cand = {}
for c in cands:
    rows = [best[(c, it)] for it in items if (c, it) in best]
    ok = sum(1 for r in rows if parse_cls(r.get("answer_text")) == audit.get(r["item_id"]))
    fmt = sum(1 for r in rows if parse_cls(r.get("answer_text")) in CLASSES)
    tok = sum((r.get("usage") or {}).get("total_tokens") or 0 for r in rows)
    el = sorted(r["elapsed_seconds"] for r in rows)
    med = el[len(el) // 2] if el else 0
    per_cand[c] = {"ok": ok, "n": len(rows), "tok": tok}
    print(f"    {c:10s} {len(rows):5d} {ok:5d} {ok/max(len(rows),1):8.1%} {fmt:9d} "
          f"{tok:9d} {med:8.1f}s")

# 2 逐类
print("\n[2] 逐类一致情况")
print(f"    {'类别':10s} {'真值数':>6s} " + " ".join(f"{c[:8]:>9s}" for c in cands))
for cls in CLASSES:
    its = [it for it in items if audit.get(it) == cls]
    line = f"    {cls:10s} {len(its):6d} "
    for c in cands:
        r = [best[(c, it)] for it in its if (c, it) in best]
        hit = sum(1 for x in r if parse_cls(x.get("answer_text")) == cls)
        line += f"{hit}/{len(its):<6d} "
    print(line)

# 3 候选间分歧
print("\n[3] 候选之间的分歧")
dis = collections.Counter()
rows_dis = []
for it in items:
    preds = {c: parse_cls(best[(c, it)].get("answer_text")) for c in cands if (c, it) in best}
    if len(preds) < 2:
        continue
    if len(set(preds.values())) > 1:
        dis["分歧"] += 1
        rows_dis.append((it, audit.get(it), preds))
    else:
        dis["一致"] += 1
print(f"    三候选完全一致 {dis['一致']} 张；有分歧 {dis['分歧']} 张")
for it, gt, preds in rows_dis:
    print(f"      {it}  真值={gt:10s} " + " ".join(f"{c}={v}" for c, v in sorted(preds.items())))

# 4 用量
print("\n[4] 用量")
for c in cands:
    rows = [best[(c, it)] for it in items if (c, it) in best]
    img = [((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("image_tokens")
           for r in rows]
    img = [x for x in img if x]
    pt = sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in rows)
    ct = sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in rows)
    rt = sum(((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
             or 0 for r in rows)
    print(f"    {c:10s} prompt {pt:6d}  completion {ct:6d}（其中推理 {rt:6d}）  "
          f"图片token {('直接给出 ' + str(img[0])) if img else '未拆分'}")

# 5 失败
print("\n[5] 失败记录")
if fails:
    for c, it, st in fails:
        print(f"    {c:10s} {it}  HTTP={st}")
else:
    print("    无")
