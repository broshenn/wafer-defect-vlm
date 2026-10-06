#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩容 v2：**全量**校验（不再抽样）+ 描述候选 v2 + 盲清单分片。"""
from __future__ import annotations
import hashlib, io, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("D:/pycode/晶圆图研究")
V1 = Path("D:/pycode/.ssh-tmp/exp")
OUT = Path("D:/pycode/.ssh-tmp/exp2")
MANIFEST = REPO / "data" / "manifest.jsonl"
BENCH = REPO / "benchmark"
SRC180 = REPO / "协作" / "02_ClaudeCode_实操" / "学校训练准备_20261002-025848"
RLPOOL = REPO / "协作" / "02_ClaudeCode_实操" / "今晚类别RL_20261006_0325" / "rl_pool90.jsonl"
VAL90 = Path("D:/pycode/.ssh-tmp/val90/val90.jsonl")
SRC = REPO / "协作" / "01_Codex_指挥" / "训练数据就绪_20261003"
DERIVED = re.compile(r"^(wafer_\d{8}_\d{3})__.+$")
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
QUESTION = ("这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。"
            "请从 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_full、Random、Scratch、none "
            "中选一个最符合主要图案的类别。只输出一个 JSON 对象，形如 "
            '{"defect_class":"类别"}；无法可靠判断时填 {"defect_class":"unknown"}。')
BLIND_KEYS = ["item_id", "sample_id", "image_path", "image_sha256"]
LEAK = ["label", "failure_type", "class", "defect_class", "lot_name", "lot",
        "split", "caption", "morphology", "answer", "prediction", "geometry"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rd(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def root_of(v):
    m = DERIVED.match(v)
    return m.group(1) if m else v


def leak(a, b, nm):
    if not b:
        return True, f"{nm} lot 集合为空 —— 检查无效，不能当通过"
    i = a & b
    return bool(i), (f"{nm} 交集 {sorted(i)[:3]}" if i else "")


rows = rd(MANIFEST)
by = {r["sample_id"]: r for r in rows}
splits = {"train": set(), "val": set(), "test": set()}
for r in rows:
    splits.setdefault(r["split"], set()).add(r["lot_name"])

hj = json.load(io.open(OUT / "historical_ids_v2.json", encoding="utf-8"))
hist = {i for i in hj["historical_ids"] if i in by}
e = {}
for n, p in (("train180", SRC180 / "train_180.jsonl"), ("dev18", SRC180 / "dev_18.jsonl"),
             ("val90", VAL90)):
    e[n] = {r["sample_id"] for r in rd(p)}
e["rl_pool90"] = {x["sample_id"] for x in rd(RLPOOL)}
e["provenance180"] = {r["sample_id"] for r in rd(SRC / "provenance_180.jsonl")}
bm = set()
for f in sorted(BENCH.glob("*.jsonl")):
    for o in rd(f):
        if isinstance(o, dict):
            for k in ("sample_id", "query_id", "gallery_id", "variant_id"):
                v = o.get(k)
                if isinstance(v, str) and v.startswith("wafer_"):
                    bm.add(root_of(v))
e["benchmark根"] = bm
t180 = e["train180"]
t180_lots = {by[i]["lot_name"] for i in t180 if i in by}
hist_lots_eff = {by[i]["lot_name"] for i in hist} - t180_lots
excl = set().union(*e.values()) | hist
excl_lots = ({by[i]["lot_name"] for i in set().union(*e.values()) if i in by} | hist_lots_eff)

print("=" * 78)
print("v2 全量独立校验")
print("=" * 78)
res = []


def chk(n, ok, d=""):
    res.append((n, bool(ok), d))
    print(f"  {'OK  ' if ok else 'FAIL'} {n}  {d}")


chk("完整 manifest 的 val lot 非空", bool(splits["val"]), f"{len(splits['val'])}")
chk("完整 manifest 的 test lot 非空", bool(splits["test"]), f"{len(splits['test'])}")
unknown = sorted(i for i in excl if i not in by)
chk("v2 排除集 ID 全部在 manifest 里", not unknown, f"未知 {len(unknown)}")
# 派生 ID 映射根
nroot = len({root_of(v) for f in sorted(BENCH.glob("*.jsonl")) for o in rd(f)
             if isinstance(o, dict) for k in ("sample_id", "query_id", "gallery_id", "variant_id")
             for v in [o.get(k)] if isinstance(v, str) and v.startswith("wafer_")})
chk("benchmark 派生 ID 映射回根晶圆", nroot > 0, f"{nroot} 根")

prev = set()
for lv in (180, 360, 720, 1440):
    rs = rd(OUT / f"scale_{lv}_v2.jsonl")
    ids = {x["sample_id"] for x in rs}
    lots = {x["lot_name"] for x in rs}
    chk(f"{lv}: 条数 == lot 数", len(ids) == len(lots) == lv, f"{len(ids)}/{len(lots)}")
    new = ids - t180
    nl = {by[i]["lot_name"] for i in new}
    chk(f"{lv}: 新增 {len(new)} 条与 v2 排除集零重叠", not (new & excl),
        f"交集 {len(new & excl)}")
    chk(f"{lv}: 新增 lot 与 v2 排除 lot 零重叠", not (nl & excl_lots),
        f"交集 {len(nl & excl_lots)}")
    if lv == 180:
        chk("180 档 == 旧 train180（逐条相同）", ids == t180)
    chk(f"{lv}: 全部 train / ground_truth",
        all(by[i]["split"] == "train" and by[i]["label_source"] == "ground_truth" for i in ids))
    # **全量**图片 hash（不再 rs[:200]）
    mism = [x["sample_id"] for x in rs
            if not (REPO / "data" / "images" / f"{x['sample_id']}.png").exists()
            or sha(REPO / "data" / "images" / f"{x['sample_id']}.png") != x["image_sha256"]]
    chk(f"{lv}: **全部 {len(rs)} 张图逐条 hash 一致**", not mism, f"不符 {len(mism)}")
    if prev:
        chk(f"嵌套 {len(prev)} ⊂ {lv}", prev <= ids)
    prev = ids
    for nm in ("val", "test"):
        lk, why = leak(lots, splits[nm], nm)
        chk(f"{lv}: 与 {nm} 无 lot 交集", not lk, why)
    lk, why = leak(lots, {by[i]["lot_name"] for i in bm if i in by}, "benchmark")
    chk(f"{lv}: 与 benchmark lot 无交集", not lk, why)
    # 与历史接触
    lk, why = leak(lots, {by[i]["lot_name"] for i in hist if i in by}, "历史接触")
    chk(f"{lv}: 新增部分与历史接触 lot 无交集",
        not (nl & {by[i]["lot_name"] for i in hist if i in by}),
        f"交集 {len(nl & {by[i]['lot_name'] for i in hist if i in by})}")

# SFT 题面与类别
print("\n--- SFT 数据 ---")
for lv in (360, 720, 1440):
    rs = rd(OUT / f"scale_{lv}_v2.jsonl")
    recs = []
    for x in rs:
        recs.append({"messages": [
            {"role": "user", "content": "<image>" + QUESTION},
            {"role": "assistant", "content": json.dumps({"defect_class": x["label"]},
                                                        ensure_ascii=False,
                                                        separators=(",", ":"))}],
            "images": [f"/WS/datasets/expand_v2/images/{x['sample_id']}.png"],
            "sample_id": x["sample_id"]})
    p = OUT / f"sft_scale_{lv}_v2.jsonl"
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok_u = all(r["messages"][0]["content"] == "<image>" + QUESTION for r in recs)
    ok_a = all(r["messages"][1]["content"] ==
               json.dumps({"defect_class": by[r["sample_id"]]["failure_type"]},
                          ensure_ascii=False, separators=(",", ":")) for r in recs)
    chk(f"{lv}: 全部 {len(recs)} 行题面逐字一致", ok_u)
    chk(f"{lv}: 全部 {len(recs)} 行类别与源标签一致", ok_a)
    chk(f"{lv}: 无答案性字段泄漏到 user 侧",
        not any(k in r["messages"][0]["content"] for r in recs for k in ("ground_truth", "lot_name")))
    print(f"       sha256 {sha(p)[:16]}")

# 失败测试
print("\n--- 失败测试 ---")
chk("合成同 lot 跨 split → 判为泄漏", leak({"a", "S"}, {"S"}, "test")[0] is True)
chk("空集合 → 显式报警", leak({"a"}, set(), "test")[0] is True)
chk("未知 ID 被识别", "wafer_99999999_999" not in by)

print()
fails = [x for x in res if not x[1]]
print(f"通过 {len(res)-len(fails)}/{len(res)}")
json.dump({"passed": len(res) - len(fails), "total": len(res),
           "fails": [x[0] for x in fails], "checks": res},
          io.open(OUT / "verify_v2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ── 描述候选 v2 ─────────────────────────────────────────
print()
print("=" * 78)
print("描述候选 360 v2")
print("=" * 78)
cap = rd(SRC / "caption_train_158.jsonl")
stc = rd(SRC / "structured_candidates_90.jsonl")
qua = rd(SRC / "caption_quarantine_22.jsonl")
cap_ids = {x.get("sample_id") for x in cap}
stc_ids = {x.get("sample_id") for x in stc}
qua_ids = {x.get("sample_id") for x in qua}
mode = rd(OUT / "scale_1440_v2.jsonl")
pool = [x for x in mode]
by_c = defaultdict(list)
for x in pool:
    by_c[x["label"]].append(x)
rng = random.Random(3407)
for c in by_c:
    head = [x for x in by_c[c] if x["sample_id"] in cap_ids]
    tail = [x for x in by_c[c] if x["sample_id"] not in cap_ids]
    rng.shuffle(head); rng.shuffle(tail)
    by_c[c] = head + tail
sel, idx = [], {c: 0 for c in CLASSES}
while len(sel) < 360:
    for c in CLASSES:
        if idx[c] < len(by_c[c]) and len(sel) < 360:
            sel.append(by_c[c][idx[c]]); idx[c] += 1
rows_d = []
for k, x in enumerate(sel, 1):
    sid = x["sample_id"]
    rows_d.append({"item_id": f"desc_{k:03d}", "sample_id": sid, "label": x["label"],
                   "lot_name": x["lot_name"], "image": f"data/images/{sid}.png",
                   "image_sha256": x["image_sha256"],
                   "has_caption": sid in cap_ids, "has_structured": sid in stc_ids,
                   "was_quarantined": sid in qua_ids,
                   "was_hist_contacted": sid in hist})
p = OUT / "desc_candidates_360_v2.jsonl"
with io.open(p, "w", encoding="utf-8", newline="\n") as f:
    for x in rows_d:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
hc = sum(1 for x in rows_d if x["has_caption"])
hq = sum(1 for x in rows_d if x["was_quarantined"])
hh = sum(1 for x in rows_d if x["was_hist_contacted"])
print(f"  选出 {len(rows_d)} / {len({x['lot_name'] for x in rows_d})} lot")
print(f"  已有描述 {hc}  曾隔离 {hq}  历史接触过 {hh}")
# 拆分
first = [x for x in rows_d if not x["has_caption"] and not x["was_quarantined"]
         and not x["was_hist_contacted"]]
quar = [x for x in rows_d if x["was_quarantined"]]
histc = [x for x in rows_d if x["was_hist_contacted"] and not x["was_quarantined"]]
print(f"  **真正首次标注 {len(first)}** / 原 180 隔离件 {len(quar)} / 历史已作答 {len(histc)}")
print(f"  （不是 202 —— 202 里含已有 raw 的，不能全叫首次）")

blind = [{"item_id": f"bz_{k:03d}", "sample_id": x["sample_id"],
          "image_path": f"/WS/datasets/expand_v2/images/{x['sample_id']}.png",
          "image_sha256": x["image_sha256"]} for k, x in enumerate(first, 1)]
bp = OUT / "zcode_blind_v2_首次.jsonl"
with io.open(bp, "w", encoding="utf-8", newline="\n") as f:
    for x in blind:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
bad = [x for x in blind if set(x) != set(BLIND_KEYS)]
lks = [k for k in LEAK if any(k in x for x in blind)]
print(f"  盲清单 {len(blind)} 条；四字段 {not bad}；无泄漏 {not lks} {lks}")
# 分片：首批 20，其余每份最多 7 图
sh = [blind[i:i + 20] for i in range(0, len(blind), 20)]
fp = OUT / "zcode_blind_v2_批次01.jsonl"
with io.open(fp, "w", encoding="utf-8", newline="\n") as f:
    for x in sh[0]:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
rest = blind[20:]
parts = [rest[i:i + 7] for i in range(0, len(rest), 7)]
print(f"  首批 20 -> {fp}  sha256 {sha(fp)[:16]}")
print(f"  其余 {len(rest)} 张分成 {len(parts)} 份（每份最多 7 图，建议最多 3 并发）")
man = {"blind_sha256": sha(bp), "batch01_sha256": sha(fp),
       "n_first": len(first), "n_quarantine": len(quar), "n_hist": len(histc),
       "n_rest": len(rest), "n_shards": len(parts),
       "prompt": "协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt",
       "checker": "协作/01_Codex_指挥/验收_20261003_GLM180/check_answer_v4.py",
       "note": "只给真正尚无作答记录的新图；旧隔离件交 Codex 内容处理或保持隔离，不自动重试。"}
json.dump(man, io.open(OUT / "blind_meta_v2.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"  写出 blind_meta_v2.json")
sys.exit(0 if not fails else 1)
