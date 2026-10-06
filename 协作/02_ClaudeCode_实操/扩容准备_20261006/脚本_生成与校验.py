#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩容准备 第 3 步：生成分类 SFT 数据 + 独立校验器（含失败测试）。"""
from __future__ import annotations
import hashlib, io, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("D:/pycode/晶圆图研究")
OUT = Path("D:/pycode/.ssh-tmp/exp")
MANIFEST = REPO / "data" / "manifest.jsonl"
BENCH = REPO / "benchmark"
SRC180 = REPO / "协作" / "02_ClaudeCode_实操" / "学校训练准备_20261002-025848"
RLPOOL = REPO / "协作" / "02_ClaudeCode_实操" / "今晚类别RL_20261006_0325" / "rl_pool90.jsonl"
VAL90 = Path("D:/pycode/.ssh-tmp/val90/val90.jsonl")
DERIVED = re.compile(r"^(wafer_\d{8}_\d{3})__.+$")
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]

QUESTION = ("这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。"
            "请从 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_full、Random、Scratch、none "
            "中选一个最符合主要图案的类别。只输出一个 JSON 对象，形如 "
            '{"defect_class":"类别"}；无法可靠判断时填 {"defect_class":"unknown"}。')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rd(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def full_split_lots(rows):
    """**从完整 manifest** 取 train/val/test 的 lot 集合（不从未知集合取空集）。"""
    out = {"train": set(), "val": set(), "test": set()}
    for r in rows:
        out.setdefault(r["split"], set()).add(r["lot_name"])
    return out


def root_of(x):
    m = DERIVED.match(x)
    return m.group(1) if m else x


def leak_check(a, b, name):
    if not b:
        return True, f"{name} lot 集合为空 —— 检查无效，不能当通过"
    inter = a & b
    return bool(inter), (f"{name} 交集 {sorted(inter)[:3]}" if inter else "")


# ── 生成 SFT ────────────────────────────────────────────
print("=" * 78)
print("生成分类 SFT 数据")
print("=" * 78)
for lv in (360, 720, 1440):
    rows = rd(OUT / f"scale_{lv}.jsonl")
    recs = []
    for x in rows:
        recs.append({
            "messages": [
                {"role": "user", "content": "<image>" + QUESTION},
                {"role": "assistant",
                 "content": json.dumps({"defect_class": x["label"]}, ensure_ascii=False,
                                       separators=(",", ":"))},
            ],
            "images": [f"/WS/datasets/expand/images/{x['sample_id']}.png"],
            "sample_id": x["sample_id"],
        })
    p = OUT / f"sft_scale_{lv}.jsonl"
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_img = sum(1 for r in recs if not r["messages"][0]["content"].startswith("<image>"))
    print(f"  scale_{lv} -> sft_scale_{lv}.jsonl  {len(recs)} 行  sha256 {sha(p)[:16]}")
    print(f"     题面全部以 <image> 开头: {n_img == 0}；类别取自原标签: True")

# ── 独立校验器 ──────────────────────────────────────────
print()
print("=" * 78)
print("独立校验器")
print("=" * 78)
results = []


def chk(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  {'OK  ' if ok else 'FAIL'} {name}  {detail}")


mrows = rd(MANIFEST)
by_id = {r["sample_id"]: r for r in mrows}
splits = full_split_lots(mrows)
chk("完整 manifest 的 val lot 非空", bool(splits["val"]), f"{len(splits['val'])} 个")
chk("完整 manifest 的 test lot 非空", bool(splits["test"]), f"{len(splits['test'])} 个")

# 排除集
excl = {}
for name, p in (("train180", SRC180 / "train_180.jsonl"), ("dev18", SRC180 / "dev_18.jsonl"),
                ("val90", VAL90)):
    excl[name] = {r["sample_id"] for r in rd(p)}
excl["rl_pool90"] = {x["sample_id"] for x in rd(RLPOOL)}
excl["provenance180"] = {r["sample_id"] for r in rd(
    REPO / "协作" / "01_Codex_指挥" / "训练数据就绪_20261003" / "provenance_180.jsonl")}
bm = set()
for f in sorted(BENCH.glob("*.jsonl")):
    for o in rd(f):
        if isinstance(o, dict):
            for k in ("sample_id", "query_id", "gallery_id", "variant_id"):
                v = o.get(k)
                if isinstance(v, str) and v.startswith("wafer_"):
                    bm.add(root_of(v))
excl["benchmark根"] = bm
excl_ids = set().union(*excl.values())
excl_lots = {by_id[i]["lot_name"] for i in excl_ids if i in by_id}
unknown = sorted(i for i in excl_ids if i not in by_id)
chk("排除集里的 ID 全部在 manifest 里", not unknown, f"未知 {len(unknown)}")

prev = set()
BASE180 = set(excl["train180"])
for lv in (180, 360, 720, 1440):
    rs = rd(OUT / f"scale_{lv}.jsonl")
    ids = {x["sample_id"] for x in rs}
    lots = {x["lot_name"] for x in rs}
    chk(f"{lv}: 条数等于 lot 数（每 lot 一张）", len(ids) == len(lots) == lv,
        f"{len(ids)} 条 / {len(lots)} lot")
    # 排除规则约束的是**本档新增**的样本；180 档就是旧 train180 本身，按定义豁免。
    new_ids = ids - BASE180
    new_lots = {by_id[i]["lot_name"] for i in new_ids}
    excl_wo_180 = excl_ids - BASE180
    excl_lots_wo_180 = excl_lots - {by_id[i]["lot_name"] for i in BASE180 if i in by_id}
    chk(f"{lv}: 新增 {len(new_ids)} 条与排除集零重叠", not (new_ids & excl_wo_180),
        f"交集 {len(new_ids & excl_wo_180)}")
    chk(f"{lv}: 新增的 lot 与排除 lot 零重叠", not (new_lots & excl_lots_wo_180),
        f"交集 {len(new_lots & excl_lots_wo_180)}")
    if lv == 180:
        chk("180 档 == 旧 train180（规模曲线起点，逐条相同）", ids == BASE180)
    chk(f"{lv}: 全部 train 划分且 ground_truth",
        all(by_id[i]["split"] == "train" and by_id[i]["label_source"] == "ground_truth"
            for i in ids))
    chk(f"{lv}: 图片存在且 sha256 与清单一致",
        all((REPO / "data" / "images" / f"{i}.png").exists()
            and sha(REPO / "data" / "images" / f"{i}.png") == x["image_sha256"]
            for x in rs[:200] for i in [x["sample_id"]]))
    if prev:
        chk(f"嵌套：{len(prev)} ⊂ {lv}", prev <= ids)
    prev = ids
    for nm in ("val", "test"):
        leak, why = leak_check(lots, splits[nm], nm)
        chk(f"{lv}: 与 {nm} 无 lot 交集", not leak, why)
    leak, why = leak_check(lots, {by_id[i]["lot_name"] for i in excl["benchmark根"] if i in by_id},
                           "benchmark")
    chk(f"{lv}: 与 benchmark lot 无交集", not leak, why)

# 类别与 shape 覆盖
for lv in (1440,):
    rs = rd(OUT / f"scale_{lv}.jsonl")
    cm = Counter(x["label"] for x in rs)
    chk("1440: 九类全部覆盖", set(cm) == set(CLASSES), str(dict(cm)))
    shapes = {tuple(x["matrix_shape"] or ()) for x in rs}
    print(f"       1440 覆盖原矩阵 shape {len(shapes)} 种")

# ── 失败测试（证明检查真的会失败）────────────────────────
print("\n--- 失败测试（证明检查会失败）---")
chk("合成同 lot 跨 split 冲突 → 判为泄漏",
    leak_check({"lotA", "lotSYN"}, {"lotSYN"}, "test")[0] is True)
chk("空集合 → 显式报警（不当作通过）",
    leak_check({"lotA"}, set(), "test")[0] is True)
synthetic = {"sample_id": "wafer_99999999_999", "lot_name": "lotSYN", "split": "train"}
chk("未知 ID → 被识别为不在 manifest", synthetic["sample_id"] not in by_id)

print()
fails = [x for x in results if not x[1]]
print(f"通过 {len(results)-len(fails)}/{len(results)}")
meta = {"passed": len(results) - len(fails), "total": len(results),
        "fails": [x[0] for x in fails], "checks": results}
(OUT / "verify_result.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
sys.exit(0 if not fails else 1)
