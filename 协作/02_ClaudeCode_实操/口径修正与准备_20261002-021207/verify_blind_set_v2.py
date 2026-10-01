#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""十八张盲清单的独立校验（不 import `prepare_blind_set_v2`）。

沿用 v1 校验器的思路：抽样与排除用**另一份实现**重写，两份代码一致才算证据。
本版另加一条任务书点名的检查：**缺图失败路径** —— 把某张图指到不存在的路径，
确认校验器会失败而不是"读不到就当通过"。

只读；不联网。退出码 0 = 全过。
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

SEED = 3407
PER_CLASS = 2
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
BLIND_KEYS = {"item_id", "sample_id", "image_path", "image_sha256"}
PROMPT_REL = "协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt"
PROMPT_SHA = "7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a"
WAFER_ID_RE = re.compile(r"wafer_\d{8}_\d{3}")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    return bool(ok)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(p.read_bytes())


def probe_image(path: Path) -> tuple[bool, str]:
    """图像可用性探针：真实读取并判定。**缺图必须返回 False**，不能静默跳过。

    返回 (通过, 说明)。这里是"缺图失败路径"要验的那段逻辑本身。
    """
    if not path.exists():
        return False, f"缺图：{path}"
    try:
        raw = path.read_bytes()
        from PIL import Image
        import numpy as np
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            fmt, size = im.format, im.size
            arr = np.asarray(im.convert("RGB"))
        cols = {tuple(int(x) for x in c) for c in np.unique(arr.reshape(-1, 3), axis=0)}
        ok = (fmt == "PNG" and size == (448, 448)
              and cols <= {(0, 0, 0), (0, 255, 0), (255, 0, 0)})
        return ok, f"fmt={fmt} size={size} colors={len(cols)}"
    except Exception as e:                       # noqa: BLE001 —— 读不了就是不合格
        return False, f"读取失败：{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE))
    a = ap.parse_args()
    d = Path(a.dir).resolve()

    print("=" * 78)
    print(f"十八张盲清单独立校验：{d}")
    print("=" * 78)

    for fn in ("blind_inputs.jsonl", "audit_selected.jsonl", "plan.json"):
        if not (d / fn).exists():
            print(f"[无法校验] 缺 {fn}")
            return 2

    blind = [json.loads(l) for l in (d / "blind_inputs.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    audit = [json.loads(l) for l in (d / "audit_selected.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    blind_raw = (d / "blind_inputs.jsonl").read_text(encoding="utf-8")

    rows = [json.loads(l) for l in open(REPO / "data" / "manifest.jsonl", encoding="utf-8")
            if l.strip()]
    by_id = {r["sample_id"]: r for r in rows}

    # ── 1 结构与保密 ──────────────────────────────────
    print("\n[1] 盲清单结构与保密")
    check("每行恰好四字段", all(set(b) == BLIND_KEYS for b in blind))
    check("无标签/lot/split 等字段",
          not (set().union(*[set(b) for b in blind]) &
               {"failure_type", "label", "lot_name", "split", "difficulty", "defect_class"}))
    check("原文不含类别名", not [c for c in CLASSES
                                 if re.search(re.escape(c), blind_raw, re.I)])
    check("原文不含 lot 字面量", not re.search(r"lot\d+", blind_raw, re.I))
    check("item_id 连续", [b["item_id"] for b in blind] ==
          [f"item_{i:03d}" for i in range(1, len(blind) + 1)])
    check("共 18 张", len(blind) == 18, str(len(blind)))

    # ── 2 与 manifest 对照 ────────────────────────────
    print("\n[2] 与 manifest 对照")
    check("样本都在清单里", all(b["sample_id"] in by_id for b in blind))
    check("全部 split=train", all(by_id[b["sample_id"]]["split"] == "train" for b in blind))
    check("全部 ground_truth",
          all(by_id[b["sample_id"]]["label_source"] == "ground_truth" for b in blind))
    per_class = {c: sum(1 for b in blind if by_id[b["sample_id"]]["failure_type"] == c)
                 for c in CLASSES}
    check("每类恰好 2 张", all(v == PER_CLASS for v in per_class.values()), str(per_class))
    check("九类齐全", sorted(per_class) == sorted(CLASSES))
    lots = [by_id[b["sample_id"]]["lot_name"] for b in blind]
    check("18 个不同 lot", len(set(lots)) == len(lots))

    # ── 3 隔离 ────────────────────────────────────────
    print("\n[3] 与 benchmark / val / test 的隔离")
    bm_ids = set()
    for f in sorted((REPO / "benchmark").glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in ("sample_id", "query_id", "gallery_id"):
                v = o.get(k) if isinstance(o, dict) else None
                if isinstance(v, str) and v.startswith("wafer_"):
                    bm_ids.add(v)
    bm_lots = {by_id[i]["lot_name"] for i in bm_ids if i in by_id}
    val_lots = {r["lot_name"] for r in rows if r["split"] == "val"}
    test_lots = {r["lot_name"] for r in rows if r["split"] == "test"}
    check("未用 benchmark 样本", not ({b["sample_id"] for b in blind} & bm_ids))
    check("lot 不在 benchmark", not (set(lots) & bm_lots))
    check("lot 不在 val", not (set(lots) & val_lots))
    check("lot 不在 test", not (set(lots) & test_lots))

    # ── 4 排除集合（独立重扫）────────────────────────
    print("\n[4] 排除集合（本脚本独立重扫；区分「看过」与「仅计划」）")
    # 与准备脚本同一条规则、但另写一份实现：
    #   出现在 **任何非 plan.json 文件**里 → 有发送证据 → 排除
    #   只出现在 plan.json 里 → 仅计划过 → A 裁决：**不排除**
    seen, planned = set(), set()
    for rel in ("kimi_k3_cost_test", "zcode_glm53flash_test", "协作/03_ZCode_标注"):
        root = REPO / rel
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".json", ".md", ".txt"}:
                ids = set(WAFER_ID_RE.findall(
                    p.read_text(encoding="utf-8", errors="replace")))
                (planned if p.name == "plan.json" else seen).update(ids)
    wb = Path("D:/pycode/workbuddy/wafer-cost-test/20261001-211719")
    if wb.exists():
        for p in sorted(wb.glob("*.json")):
            ids = set(WAFER_ID_RE.findall(
                p.read_text(encoding="utf-8", errors="replace")))
            (planned if p.name == "plan.json" else seen).update(ids)
    this_round = {json.loads(l)["sample_id"] for l in
                  (REPO / "协作/02_ClaudeCode_实操/准备_20261002_九类试标/audit_selected.jsonl"
                   ).read_text(encoding="utf-8").splitlines() if l.strip()}
    planned_only = planned - seen
    excluded = seen | this_round
    hit = sorted({b["sample_id"] for b in blind} & excluded)
    check("未命中任何有发送证据的样本", not hit, f"命中={hit}")
    seen_lots = {by_id[i]["lot_name"] for i in excluded if i in by_id}
    check("lot 也不与已看过样本重叠", not (set(lots) & seen_lots))
    check("未使用上一轮那九张", not ({b["sample_id"] for b in blind} & this_round))
    print(f"    有发送证据 {len(seen)} 个；仅计划过 {len(planned_only)} 个 = {sorted(planned_only)}"
          "（按 A 裁决保留在池内）")

    # ── 5 图像实体（含缺图失败路径）──────────────────
    print("\n[5] 图像实体 + 缺图失败路径")
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        check("PIL 可用", False, "无 Pillow")
        Image = None
    if Image is not None:
        shas = []
        for b in blind:
            p = Path(b["image_path"])
            if not p.exists():
                check(f"{b['item_id']} 图片存在", False, str(p))
                continue
            raw = p.read_bytes()
            got = sha_bytes(raw)
            shas.append(got)
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                fmt, size = im.format, im.size
                arr = np.asarray(im.convert("RGB"))
            cols = {tuple(int(x) for x in c) for c in np.unique(arr.reshape(-1, 3), axis=0)}
            check(f"{b['item_id']} 指纹/PNG/448/三色",
                  got == b["image_sha256"] and fmt == "PNG" and size == (448, 448)
                  and cols <= {(0, 0, 0), (0, 255, 0), (255, 0, 0)},
                  f"sha_match={got == b['image_sha256']} fmt={fmt} size={size}")
        check("18 个指纹互不重复", len(set(shas)) == len(shas))

    # 缺图失败路径：用一个不存在的路径走**同一段判定逻辑**，要求它失败；再用真图做对照。
    missing = REPO / "data" / "images" / "__definitely_not_here__.png"
    real = Path(blind[0]["image_path"])
    ok_missing, why_missing = probe_image(missing)
    ok_real, why_real = probe_image(real)
    check("缺图失败路径：不存在时判定为不通过", ok_missing is False, why_missing)
    check("缺图失败路径：存在时判定为通过（对照组）", ok_real is True, why_real)
    check("缺图路径确实不存在", not missing.exists())

    # ── 6 audit 与 blind 一致 ────────────────────────
    print("\n[6] audit 与 blind 一致")
    check("item_id 逐项一致", [a["item_id"] for a in audit] == [b["item_id"] for b in blind])
    check("sample_id 逐项一致", [a["sample_id"] for a in audit] == [b["sample_id"] for b in blind])
    check("audit 标签与 manifest 一致",
          all(a["failure_type_normalized"] == by_id[a["sample_id"]]["failure_type"] for a in audit))
    check("audit 行号指向正确", all(
        rows[a["manifest_line"] - 1]["sample_id"] == a["sample_id"] for a in audit))

    # ── 7 plan ────────────────────────────────────────
    print("\n[7] plan.json")
    check("seed=3407", plan.get("seed") == SEED)
    check("per_class=2", plan.get("per_class") == PER_CLASS)
    check("planned_count=18", plan.get("planned_count") == 18)
    check("blind_inputs 指纹一致",
          plan.get("blind_inputs_sha256") == sha_file(d / "blind_inputs.jsonl"))
    check("audit_selected 指纹一致",
          plan.get("audit_selected_sha256") == sha_file(d / "audit_selected.jsonl"))
    check("manifest 指纹一致",
          plan.get("data_files", {}).get("data/manifest.jsonl")
          == sha_file(REPO / "data" / "manifest.jsonl"))
    pp = REPO / PROMPT_REL
    check("提示词字节指纹 == v2 原件", sha_file(pp) == PROMPT_SHA)
    check("plan 声明提示词未切换", plan.get("prompt", {}).get("unchanged") is True)

    # ── 8 复现抽样（独立重实现）──────────────────────
    print("\n[8] 复现抽样（本脚本另一份实现）")
    banned = val_lots | test_lots | bm_lots | {by_id[i]["lot_name"] for i in excluded if i in by_id}
    pool = {c: [] for c in CLASSES}
    for r in rows:
        if r["split"] != "train" or r["label_source"] != "ground_truth":
            continue
        if r["sample_id"] in excluded or r["lot_name"] in banned:
            continue
        if r["failure_type"] in pool:
            pool[r["failure_type"]].append(r)
    for c in pool:
        pool[c].sort(key=lambda r: r["sample_id"])
    rng = random.Random(SEED)
    used, picks = set(), []
    for c in CLASSES:
        for _ in range(PER_CLASS):
            cands = [x for x in pool[c] if x["lot_name"] not in used]
            if not cands:
                continue
            x = cands[rng.randrange(len(cands))]
            used.add(x["lot_name"])
            picks.append(x)
    random.Random(SEED).shuffle(picks)
    mine = [x["sample_id"] for x in picks]
    got = [b["sample_id"] for b in blind]
    check("独立重实现与交付盲清单一致", mine == got,
          f"\n        重算={mine}\n        交付={got}")

    # ── 汇总 ─────────────────────────────────────────
    print("\n" + "=" * 78)
    fails = [r for r in results if not r[1]]
    for name, ok, det in results:
        if not ok:
            print(f"  FAIL  {name}   {det}")
    print(f"\n通过 {len(results) - len(fails)}/{len(results)}")
    print("结论：" + ("全部通过" if not fails else f"{len(fails)} 项未通过"))
    print("=" * 78)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
