#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""九类试标盲清单的独立校验器。

刻意**不 import** prepare_blind_set.py —— 抽样、排除、lot 封禁都在这里重新实现一遍。
同一个 bug 不会在两份代码里犯两次，所以「两份代码得出同一结果」才算证据。

全程只读：不改任何文件、不联网、不调用模型。

用法：
    python verify_blind_set.py [--dir <准备产物目录>]
退出码：0 = 全部通过；1 = 有 FAIL；2 = 输入缺失无法校验。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
from pathlib import Path

SEED = 3407
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
BLIND_KEYS = {"item_id", "sample_id", "image_path", "image_sha256"}
PROMPT_REL = "协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt"
PROMPT_SHA = "7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a"

# 与准备脚本独立的第二份排除表：从「哪些文件里有调用记录」反推，写法不同。
CONFIRMED_SENT_FILES = [
    "kimi_k3_cost_test/batches/20261001-232051/results",
    "kimi_k3_cost_test/results",
    "zcode_glm53flash_test",
]
CONFIRMED_SENT_EXTRA = "D:/pycode/workbuddy/wafer-cost-test/20261001-211719"
# 该目录里只进过 plan、没有 call 记录的那 4 张（不得自动排除）
NEVER_SENT = {"wafer_00017102_001", "wafer_00043893_019",
              "wafer_00047160_025", "wafer_00044523_017"}

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    return bool(ok)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(p.read_bytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()
    d = Path(args.dir).resolve()
    repo = Path(__file__).resolve().parents[3]

    print("=" * 78)
    print(f"独立校验：{d}")
    print("=" * 78)

    for fn in ("blind_inputs.jsonl", "audit_selected.jsonl", "plan.json"):
        if not (d / fn).exists():
            print(f"[无法校验] 缺文件 {fn}")
            return 2

    blind = [json.loads(l) for l in open(d / "blind_inputs.jsonl", encoding="utf-8") if l.strip()]
    audit = [json.loads(l) for l in open(d / "audit_selected.jsonl", encoding="utf-8") if l.strip()]
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    blind_raw = (d / "blind_inputs.jsonl").read_text(encoding="utf-8")

    # ── 1. 盲清单结构与保密 ────────────────────────────────
    print("\n[1] 盲清单结构与保密")
    check("每行恰好四个字段", all(set(b) == BLIND_KEYS for b in blind),
          f"出现过的字段并集={set().union(*[set(b) for b in blind])}")
    check("无任何标签/lot/split/难度字段",
          not (set().union(*[set(b) for b in blind]) & {"failure_type", "label", "lot_name",
                                                        "split", "difficulty", "class",
                                                        "defect_class", "geometry"}))
    leaked = [c for c in CLASSES if re.search(re.escape(c), blind_raw, re.I)]
    check("盲清单原文不含任何类别名（大小写不敏感）", not leaked, f"命中={leaked}")
    check("盲清单不含 lot 编号", not re.search(r"lot\d+", blind_raw, re.I))
    check("item_id 连续且格式正确",
          [b["item_id"] for b in blind] == [f"item_{i:03d}" for i in range(1, len(blind) + 1)])
    check("sample_id 互不重复", len({b["sample_id"] for b in blind}) == len(blind))
    check("图片文件名不含类别名",
          not any(c.lower() in Path(b["image_path"]).stem.lower() for b in blind for c in CLASSES))

    # ── 2. 清单 vs 数据 ────────────────────────────────────
    print("\n[2] 与 data/manifest.jsonl 对照")
    rows = [json.loads(l) for l in open(repo / "data" / "manifest.jsonl", encoding="utf-8") if l.strip()]
    by_id = {r["sample_id"]: r for r in rows}
    check("九个样本都在清单里", all(b["sample_id"] in by_id for b in blind))
    check("全部 split=train",
          all(by_id[b["sample_id"]]["split"] == "train" for b in blind))
    check("全部 label_source=ground_truth",
          all(by_id[b["sample_id"]]["label_source"] == "ground_truth" for b in blind))

    cls_by_item = {b["item_id"]: by_id[b["sample_id"]]["failure_type"] for b in blind}
    check("九类各一张（覆盖全部九类且不重复）",
          sorted(cls_by_item.values()) == sorted(CLASSES),
          f"实得={sorted(cls_by_item.values())}")

    lots = [by_id[b["sample_id"]]["lot_name"] for b in blind]
    check("九张来自不同 lot", len(set(lots)) == len(lots), f"{lots}")

    # ── 3. 与 benchmark / val / test 的隔离 ────────────────
    print("\n[3] 与 benchmark、val、test 的隔离")
    bm_ids, bm_lots = set(), set()
    for f in sorted((repo / "benchmark").glob("*.jsonl")):
        for l in open(f, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                o = json.loads(l)
            except json.JSONDecodeError:
                continue
            for k in ("sample_id", "query_id", "gallery_id"):
                v = o.get(k) if isinstance(o, dict) else None
                if isinstance(v, str) and v.startswith("wafer_"):
                    bm_ids.add(v)
    bm_lots = {by_id[i]["lot_name"] for i in bm_ids if i in by_id}
    val_lots = {r["lot_name"] for r in rows if r["split"] == "val"}
    test_lots = {r["lot_name"] for r in rows if r["split"] == "test"}
    check("无样本落在 benchmark 里", not ({b["sample_id"] for b in blind} & bm_ids))
    check("无 lot 落在 benchmark", not (set(lots) & bm_lots))
    check("无 lot 落在 val", not (set(lots) & val_lots))
    check("无 lot 落在 test", not (set(lots) & test_lots))

    # 全清单 lot 交集（任务书要求的那项检查）
    tr_lots = {r["lot_name"] for r in rows if r["split"] == "train"}
    check("全清单 train∩val 为空", not (tr_lots & val_lots))
    check("全清单 train∩test 为空", not (tr_lots & test_lots))
    check("全清单 val∩test 为空", not (val_lots & test_lots))

    # ── 4. 历史试标排除（独立重扫） ────────────────────────
    print("\n[4] 历史试标排除（本脚本独立重扫）")
    seen_ids = set()
    for rel in CONFIRMED_SENT_FILES:
        root = repo / rel
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.json")):
            seen_ids |= set(re.findall(r"wafer_\d{8}_\d{3}", p.read_text(encoding="utf-8", errors="replace")))
    wb = Path(CONFIRMED_SENT_EXTRA)
    if wb.exists():
        for p in sorted(wb.glob("call-*.json")):          # 有 call 记录 = 有 HTTP 往返
            seen_ids |= set(re.findall(r"wafer_\d{8}_\d{3}", p.read_text(encoding="utf-8", errors="replace")))
        for p in sorted(wb.glob("started-*.json")):        # 已发起、无结果 —— 保守计入
            seen_ids |= set(re.findall(r"wafer_\d{8}_\d{3}", p.read_text(encoding="utf-8", errors="replace")))
    check("盲清单未使用任何已试标样本", not ({b["sample_id"] for b in blind} & seen_ids),
          f"重叠={sorted({b['sample_id'] for b in blind} & seen_ids)}")
    seen_lots = {by_id[i]["lot_name"] for i in seen_ids if i in by_id}
    check("盲清单未使用任何已试标 lot", not (set(lots) & seen_lots),
          f"重叠={sorted(set(lots) & seen_lots)}")

    never_sent_hit = {b["sample_id"] for b in blind} & NEVER_SENT
    print(f"    [提示] 命中「只进过计划、无发送证据」的：{sorted(never_sent_hit) or '无'}"
          "（这些**不算**违反本次任务书口径）")

    # ── 5. 图像 ────────────────────────────────────────────
    print("\n[5] 图像实体")
    try:
        from PIL import Image
        import numpy as np
        have_pil = True
    except ImportError:
        have_pil = False
    if have_pil:
        shas = []
        for b in blind:
            p = Path(b["image_path"])
            if not p.exists():
                check(f"{b['item_id']} 图片存在", False, str(p))
                continue
            raw = p.read_bytes()
            got = sha_bytes(raw)
            shas.append(got)
            ok_sha = got == b["image_sha256"]
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                fmt, size = im.format, im.size
                arr = np.asarray(im.convert("RGB"))
            cols = {tuple(int(x) for x in c) for c in np.unique(arr.reshape(-1, 3), axis=0)}
            check(f"{b['item_id']} {b['sample_id']} 指纹/PNG/448/三色",
                  ok_sha and fmt == "PNG" and size == (448, 448) and cols <= {(0, 0, 0), (0, 255, 0), (255, 0, 0)},
                  f"sha_ok={ok_sha} fmt={fmt} size={size} colors={len(cols)}")
        check("九张指纹互不重复", len(set(shas)) == len(shas))
    else:
        check("PIL 可用", False, "未安装 Pillow，无法做图像实体检查")

    # ── 6. audit 与 blind 一致 ─────────────────────────────
    print("\n[6] audit 与 blind 的一致性")
    check("item_id 集合一致",
          [a["item_id"] for a in audit] == [b["item_id"] for b in blind])
    check("sample_id 逐项一致",
          [a["sample_id"] for a in audit] == [b["sample_id"] for b in blind])
    check("audit 标签与 manifest 一致",
          all(a["failure_type_original"] == by_id[a["sample_id"]]["failure_type"] for a in audit))
    check("audit manifest_line 指向正确的行",
          all(rows[a["manifest_line"] - 1]["sample_id"] == a["sample_id"] for a in audit))

    # ── 7. plan ────────────────────────────────────────────
    print("\n[7] plan.json")
    check("seed 为 3407", plan.get("seed") == SEED, str(plan.get("seed")))
    check("plan 记的数量等于实际行数",
          plan.get("planned_count") == len(blind), f"{plan.get('planned_count')} vs {len(blind)}")
    check("blind_inputs 指纹一致", plan.get("blind_inputs_sha256") == sha_file(d / "blind_inputs.jsonl"))
    check("audit_selected 指纹一致", plan.get("audit_selected_sha256") == sha_file(d / "audit_selected.jsonl"))
    check("manifest 指纹一致",
          plan.get("data_files", {}).get("data/manifest.jsonl") == sha_file(repo / "data" / "manifest.jsonl"))
    pp = repo / PROMPT_REL
    check("提示词文件存在", pp.exists())
    if pp.exists():
        check("提示词字节指纹 == 任务书签发值", sha_file(pp) == PROMPT_SHA, sha_file(pp))
        check("plan 里记的提示词指纹与实际一致",
              plan.get("prompt", {}).get("byte_sha256") == sha_file(pp))

    # ── 8. 复现抽样（本脚本独立重实现） ───────────────────
    print("\n[8] 复现抽样：本脚本用另一份实现重跑一遍")
    bm_lots_now = {by_id[i]["lot_name"] for i in bm_ids if i in by_id}
    # 复现准备脚本的**默认口径**：只排除有发送证据的 seen_ids，NEVER_SENT 保留在池内。
    banned = val_lots | test_lots | bm_lots_now | {by_id[i]["lot_name"] for i in seen_ids if i in by_id}
    pool = {c: [] for c in CLASSES}
    for r in rows:
        if r["split"] != "train" or r["label_source"] != "ground_truth":
            continue
        if r["sample_id"] in seen_ids:
            continue
        if r["lot_name"] in banned:
            continue
        if r["failure_type"] in pool:
            pool[r["failure_type"]].append(r)
    for c in pool:
        pool[c].sort(key=lambda r: r["sample_id"])
    rng = random.Random(SEED)
    used, picks = set(), []
    for c in CLASSES:
        cands = [x for x in pool[c] if x["lot_name"] not in used]
        if not cands:
            continue
        x = cands[rng.randrange(len(cands))]
        used.add(x["lot_name"])
        picks.append(x)
    rng2 = random.Random(SEED)
    rng2.shuffle(picks)
    mine = [x["sample_id"] for x in picks]
    got = [b["sample_id"] for b in blind]
    check("独立重实现得出的样本集与盲清单一致", mine == got,
          f"重算={mine}\n        交付={got}")

    # ── 汇总 ───────────────────────────────────────────────
    print("\n" + "=" * 78)
    fails = [r for r in results if not r[1]]
    for name, ok, detail in results:
        if not ok:
            print(f"  FAIL  {name}   {detail}")
    print(f"\n通过 {len(results) - len(fails)}/{len(results)}")
    print("结论：" + ("全部通过" if not fails else f"{len(fails)} 项未通过"))
    print("=" * 78)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
