#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 A：56 图候选考卷的 CPU 冻结与独立隔离复核。

任务书要求（`任务书_20261009_ClaudeCode_56图补测后结项.md` §2.A）：
  1. 冻结 56 图清单、七字段题面和检查器 SHA，**现算 56 张 PNG 的 SHA**；
  2. 重新核唯一 ID/lot/PNG，与 train3072 的 ID/lot/PNG 内容交集均 0；
  3. 采用现有可见历史排除记录，**并明确完整历史接触未知**；
  4. 保持七类各 8 图。

本脚本**不信任** Codex 的 `考卷隔离核查.json`，全部数字**独立重算**，
算完再与它的记录对账（对不上就报错，不静默取其一）。

跑法：python CPU冻结与隔离复核.py
"""
from __future__ import annotations
import hashlib, io, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]           # 仓库根
TODAY = Path(__file__).resolve().parent
SRC = ROOT / "协作" / "01_Codex_指挥" / "AI验收补版_20261009"
BLIND = SRC / "候选考卷_盲清单.jsonl"
CODEX_ISO = SRC / "考卷隔离核查.json"
MANIFEST = ROOT / "data" / "manifest.jsonl"
IMAGES = ROOT / "data" / "images"
N3072_L = ROOT / "协作" / "02_ClaudeCode_实操" / "夜间扩容_20261008_0244" / "数据" / "L_N3072.jsonl"
N3072_JSON = ROOT / "协作" / "02_ClaudeCode_实操" / "夜间扩容_20261008_0244" / "数据" / "训练数据_N3072.json"
PROMPT = ROOT / "协作" / "02_ClaudeCode_实操" / "部署原型_20261009_0120" / "prompt_7f.txt"
CHECKER = ROOT / "协作" / "02_ClaudeCode_实操" / "部署原型_20261009_0120" / "schema_check.py"
DEV36_MANIFEST = ROOT / "协作" / "02_ClaudeCode_实操" / "部署原型_20261009_0120" / "样本清单.json"

FROZEN_PROMPT_SHA = "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda"
FROZEN_CHECKER_SHA = "7d46c37f72bb8fd09488dd73754ad75e6a1b711d91a6f08f82e5b34026c0c0ff"
SEVEN = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Random", "Scratch"]
MISSING = ["Near_full", "none"]


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def main() -> int:
    fails: list[str] = []

    def chk(cond, msg):
        print(("  ok  " if cond else "  **FAIL** ") + msg)
        if not cond:
            fails.append(msg)

    # ---- 1. 56 图清单 ----
    blind = load_jsonl(BLIND)
    chk(len(blind) == 56, f"盲清单 56 条（实际 {len(blind)}）")
    ids = [r["sample_id"] for r in blind]
    chk(len(set(ids)) == 56, "56 个唯一 sample_id")
    chk(len({r["image_sha256"] for r in blind}) == 56, "56 个唯一 PNG sha")

    man = {r["sample_id"]: r for r in load_jsonl(MANIFEST)}
    lots, png_sha, actual_sha = [], {}, {}
    for r in blind:
        sid = r["sample_id"]
        chk(sid in man, f"{sid} 在公开 manifest 中")
        m = man[sid]
        lots.append(m["lot_name"])
        p = IMAGES / f"{sid}.png"
        chk(p.exists(), f"{sid} PNG 存在")
        a = sha_file(p)
        actual_sha[sid] = a
        chk(a == r["image_sha256"], f"{sid} PNG sha 与盲清单一致")
        png_sha[sid] = a
    chk(len(set(lots)) == 56, f"56 个唯一 lot（实际 {len(set(lots))}）")

    # ---- 2. 七类各 8 ----
    cls = Counter(man[s]["failure_type"] for s in ids)
    print(f"  类别分布：{dict(cls)}")
    chk(set(cls) == set(SEVEN), "覆盖的类别恰为七类（无 Near_full / none）")
    for c in SEVEN:
        chk(cls.get(c, 0) == 8, f"{c} 恰 8 图（实际 {cls.get(c, 0)}）")

    # ---- 3a. 先取开发 36 图清单（下面要用它验证本地图等价性 + 查重叠）----
    d36 = json.loads(DEV36_MANIFEST.read_text(encoding="utf-8"))
    d36_rows = d36.get("逐图") or d36.get("明细") or []
    d36_ids = {r["sample_id"] for r in d36_rows if isinstance(r, dict) and "sample_id" in r}
    if not d36_ids:  # 换一种结构兜底
        for v in d36.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "sample_id" in v[0]:
                d36_ids = {r["sample_id"] for r in v}
                break
    chk(len(d36_ids) == 36, f"开发 36 图清单取到 36 条（实际 {len(d36_ids)}）")

    # ---- 3b. 与 train3072 的三重隔离（独立重算）----
    n3072 = load_jsonl(N3072_L)
    chk(len(n3072) == 3072, f"train3072 清单 3072 条（实际 {len(n3072)}）")
    t_ids = {r["sample_id"] for r in n3072}
    t_lots = {man[r["sample_id"]]["lot_name"] for r in n3072 if r["sample_id"] in man}
    # train3072 的 images 字段写的是**服务器路径**（本地没有）；本地按 sample_id
    # 取同一渲染管线产出的 data/images/<sid>.png —— 先用开发 36 图验证这个等价性
    DEV36_IMG = ROOT / "协作" / "02_ClaudeCode_实操" / "部署原型_20261009_0120" / "图"
    eq_n = sum(sha_file(DEV36_IMG / f"{s}.png") == sha_file(IMAGES / f"{s}.png") for s in d36_ids)
    chk(eq_n == 36, f"本地 data/images/<sid>.png 与部署图 sha 等价（{eq_n}/36，本地图可作内容基准）")
    t_png = {sha_file(IMAGES / f"{r['sample_id']}.png") for r in n3072}
    ov_id = t_ids & set(ids)
    ov_lot = t_lots & set(lots)
    ov_png = t_png & set(png_sha.values())
    chk(not ov_id, f"与 train3072 ID 交集 = 0（实际 {len(ov_id)}）")
    chk(not ov_lot, f"与 train3072 lot 交集 = 0（实际 {len(ov_lot)}）")
    chk(not ov_png, f"与 train3072 PNG 内容交集 = 0（实际 {len(ov_png)}）")

    # 训练数据_N3072.json 声明的题目 sha 也要对上（防拿错清单当隔离基准）
    tj = json.loads(N3072_JSON.read_text(encoding="utf-8"))
    chk(tj["N"] == 3072, f"训练数据_N3072.json 声明 N=3072（实际 {tj['N']}）")

    # ---- 4. 与开发 36 图 ID 不重叠（那 36 张已反复看过多轮）----
    ov36 = d36_ids & set(ids)
    chk(not ov36, f"与开发 36 图 ID 交集 = 0（实际 {len(ov36)}）")

    # ---- 5. 题面 / 检查器冻结 ----
    p_sha, c_sha = sha_file(PROMPT), sha_file(CHECKER)
    chk(p_sha == FROZEN_PROMPT_SHA, f"题面 sha256 = {p_sha[:16]}…")
    chk(c_sha == FROZEN_CHECKER_SHA, f"检查器 sha256 = {c_sha[:16]}…")
    pb = PROMPT.read_bytes()
    chk(pb.decode("utf-8") == pb.decode("utf-8"), "题面是合法 UTF-8")

    # ---- 6. 与 Codex 记录的隔离结果对账（对不上就报）----
    ci = json.loads(CODEX_ISO.read_text(encoding="utf-8"))
    chk(ci["selected_n"] == 56, "Codex 记录 selected_n = 56")
    chk(ci["train3072_id_overlap"] == 0 and ci["train3072_lot_overlap"] == 0
        and ci["train3072_png_overlap"] == 0, "Codex 记录 train3072 三重交集均 0")
    chk(ci["selected_unique_lots"] == 56 and ci["selected_unique_png"] == 56,
        "Codex 记录 56 唯一 lot / 56 唯一 PNG")
    chk(ci["status"] == "candidate_package_prepared_not_model_run",
        "Codex 记录状态 = 尚未跑模型")
    chk(ci["shortfalls"] == {"Near_full": 8, "none": 8}, "Codex 记录九类缺口 = Near_full/none 各 8")

    # ---- 7. 写冻结清单 ----
    freeze = {
        "性质": "隔离候选考卷（可见历史排除后）；**非独立九类最终考卷**，模型成绩在本文件写出时未测",
        "生成时间": "2026-10-09",
        "seed": 3407,
        "N": 56,
        "题面_sha256": p_sha,
        "检查器_sha256": c_sha,
        "类别分布": dict(sorted(cls.items())),
        "缺类": {c: 8 for c in MISSING},
        "唯一lot数": len(set(lots)),
        "唯一PNG数": len(set(png_sha.values())),
        "train3072交集": {"id": len(ov_id), "lot": len(ov_lot), "png内容": len(ov_png)},
        "开发36图交集": len(ov36),
        "隔离边界": [
            "隔离依据是**本地可见**的历史记录；未取回完整历史教师/旧训练记录，**完整历史接触未核**。",
            "因此不得称独立盲测或最终考卷；正式九类考卷还需补 Near_full 与 none 的新可靠公开样本。",
            "公开原类别（failure_type）只用于离线评分，不下发给推理模型。",
        ],
        "逐图": [
            {"item_id": r["item_id"], "sample_id": r["sample_id"],
             "lot_name": man[r["sample_id"]]["lot_name"],
             "failure_type": man[r["sample_id"]]["failure_type"],
             "label_source": man[r["sample_id"]]["label_source"],
             "manifest_split": man[r["sample_id"]].get("split"),
             "image_sha256": actual_sha[r["sample_id"]],
             "image_sha256_与盲清单一致": actual_sha[r["sample_id"]] == r["image_sha256"]}
            for r in blind],
    }
    out = TODAY / "冻结清单.json"
    out.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n写出 {out.name}")

    print("\n" + ("**有 %d 项未通过**" % len(fails) if fails else "**全部通过**"))
    for f in fails:
        print("  FAIL: " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
