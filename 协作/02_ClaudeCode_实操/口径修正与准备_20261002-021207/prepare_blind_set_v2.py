#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下一轮盲清单准备 v2 —— 九类各两张，共 18 张 train 样本。

**沿用 `准备_20261002_九类试标/prepare_blind_set.py` 的选样方法**，只改两件事：
每类数量（1 → 2）与排除集合（并入本轮已看过的九张）。方法本身逐字保留：

  1. 候选 = `split=train` ∧ `label_source=ground_truth` ∧ 样本未在排除集合 ∧ lot 不在禁用集合
  2. 每类候选**按 `sample_id` 升序**
  3. 类别按正规化顺序逐个抽；同一 `random.Random(3407)` 串行使用；
     每类内抽 `PER_CLASS` 次，抽中的 lot 立即占用，后续不再碰
  4. 抽齐后另起 `random.Random(3407)` 做一次 shuffle 定盲序

**没有证据表明发出去过的旧计划样本（A 裁决）**：保留在池内，但在 plan 里如实披露。

只读；不联网；不调模型。
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

SEED = 3407
PER_CLASS = 2
SAMPLING_RULE_VERSION = "prepare-v2@2026-10-02 (per_class=2; v1 方法不变)"

CANONICAL_CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
                     "Near_full", "Random", "Scratch", "none"]
MANIFEST = REPO / "data" / "manifest.jsonl"
IMAGES_DIR = REPO / "data" / "images"
BENCHMARK_DIR = REPO / "benchmark"
PROMPT = REPO / "协作" / "01_Codex_指挥" / "prompt_image_only_v2_20261002.txt"
PROMPT_EXPECTED = "7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a"
WAFER_ID_RE = re.compile(r"wafer_\d{8}_\d{3}")

# Codex 的历史试标来源索引（含 future_exclusion_search_roots）
CODEX_INDEX = REPO / "协作" / "01_Codex_指挥" / "历史试标来源索引_20261002.json"
# 本轮（GLM 九张）已看过的样本
THIS_ROUND_AUDIT = REPO / "协作" / "02_ClaudeCode_实操" / "准备_20261002_九类试标" / "audit_selected.jsonl"

# 只进过旧计划、**没有发送证据**的样本。A 裁决：不排除，但披露。
NEVER_SENT_PLANNED = {
    "wafer_00043893_019": "Near_full",
    "wafer_00047160_025": "Scratch",
    "wafer_00017102_001": "Random（**本轮已被看过，已并入排除**）",
    "wafer_00044523_017": "none（**本轮已被看过，已并入排除**）",
}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(p.read_bytes())


def load_manifest() -> list[dict]:
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                r["__line__"] = i
                rows.append(r)
    return rows


def search_roots() -> list[Path]:
    """搜索根，去重并保持顺序。"""
    roots: list[Path] = [REPO / "kimi_k3_cost_test", REPO / "zcode_glm53flash_test",
                         REPO / "协作" / "03_ZCode_标注"]
    if CODEX_INDEX.exists():
        idx = json.loads(CODEX_INDEX.read_text(encoding="utf-8"))
        for p in idx.get("future_exclusion_search_roots", []):
            cand = Path(p)
            roots.append(cand if cand.is_absolute() else REPO / p)
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        k = str(r).lower()
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


# 只算「计划过」、**不算「看过」**的文件名（A 裁决：无发送证据的不排除）。
# 依据 Codex `历史试标来源索引_20261002.json` 的 `evidence_kind`：
#   response_record → 看过；started → 保守也算看过；planned → 只是计划。
PLANNED_ONLY_NAMES = {"plan.json"}


def scan_trial_ids() -> tuple[set[str], set[str], list[dict]]:
    """扫历史试标产物，正则抓 wafer 编号。**不读任何配置或凭据。**

    返回 (看过的 id, 仅计划过的 id, 逐文件证据)。
    "仅计划过"的文件单独归类 —— 否则上界扫描会把**从未发出**的旧计划样本也一并排除，
    比 A 裁决更严，属于没申报的偏离。
    """
    seen: dict[str, list[str]] = {}
    planned: dict[str, list[str]] = {}
    for root in search_roots():
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in {".json", ".md", ".txt"}:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            bucket = planned if p.name in PLANNED_ONLY_NAMES else seen
            rel = str(p.relative_to(REPO)) if str(p).startswith(str(REPO)) else str(p)
            for sid in set(WAFER_ID_RE.findall(text)):
                bucket.setdefault(sid, []).append(rel)
    # 只要有**任何一个非 plan 文件**提到它，就算"有发送证据"。
    # 注意不能写成 `set(seen) - set(planned)`：同一张图既可能被 plan.json 列过、
    # 又真的出现在结果文件里（Kimi 那批就是），那样会把它错误降级成"仅计划过"。
    seen_ids = set(seen)
    planned_only = set(planned) - seen_ids
    evidence = [{"sample_id": s, "kind": "seen", "files": sorted(seen[s])}
                for s in sorted(seen_ids)]
    evidence += [{"sample_id": s, "kind": "planned_only", "files": sorted(planned[s])}
                 for s in sorted(planned_only)]
    return seen_ids, planned_only, evidence


def benchmark_ids() -> set[str]:
    ids: set[str] = set()
    for p in sorted(BENCHMARK_DIR.glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(d, dict):
                    for k in ("sample_id", "query_id", "gallery_id"):
                        v = d.get(k)
                        if isinstance(v, str) and v.startswith("wafer_"):
                            ids.add(v)
    return ids


def this_round_ids() -> set[str]:
    if not THIS_ROUND_AUDIT.exists():
        return set()
    return {json.loads(l)["sample_id"]
            for l in THIS_ROUND_AUDIT.read_text(encoding="utf-8").splitlines() if l.strip()}


def main() -> int:
    write = "--write" in sys.argv
    out_dir = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else HERE

    print("=" * 78)
    print("下一轮盲清单准备 v2（九类各两张）")
    print("=" * 78)

    # ── 提示词锁定（不切换）────────────────────────────
    if not PROMPT.exists():
        print(f"[停止] 提示词不存在：{PROMPT}")
        return 2
    psha = sha_file(PROMPT)
    print(f"\n[1] 提示词 {PROMPT.relative_to(REPO)}")
    print(f"    字节 SHA256 : {psha}")
    print(f"    与上轮一致  : {'是' if psha == PROMPT_EXPECTED else '否 ← 本任务不切换提示词，停止'}")
    if psha != PROMPT_EXPECTED:
        return 2

    rows = load_manifest()
    by_id = {r["sample_id"]: r for r in rows}
    man_sha = sha_file(MANIFEST)
    print(f"\n[2] manifest {len(rows)} 行  SHA256 {man_sha}")

    # ── 排除集合 ───────────────────────────────────────
    seen_ids, planned_only_ids, evidence = scan_trial_ids()
    prev_round = this_round_ids()
    excluded = set(seen_ids) | set(prev_round)
    kept_never_sent = {k: v for k, v in NEVER_SENT_PLANNED.items()
                       if k in planned_only_ids and k not in excluded}
    print(f"\n[3] 排除集合")
    print(f"    扫描根 {len(search_roots())} 个（已去重）")
    print(f"    有发送证据的 {len(seen_ids)} 个 + 本轮已验证看过 {len(prev_round)} 张")
    print(f"    合计排除 {len(excluded)} 个样本")
    print(f"    仅出现在旧 plan.json、无发送证据的 {len(planned_only_ids)} 个：{sorted(planned_only_ids)}")
    print(f"    ↑ A 裁决：**不排除**，保留在池内并披露不确定性")
    miss = sorted(excluded - set(by_id))
    if miss:
        print(f"    ⚠ 排除表里不在 manifest 的：{miss}")
    print(f"    本次实际保留在池内的（A 裁决）：{sorted(kept_never_sent)}")

    bm_ids = benchmark_ids()
    bm_lots = {by_id[i]["lot_name"] for i in bm_ids if i in by_id}
    val_lots = {r["lot_name"] for r in rows if r["split"] == "val"}
    test_lots = {r["lot_name"] for r in rows if r["split"] == "test"}
    ex_lots = {by_id[i]["lot_name"] for i in excluded if i in by_id}
    banned_lots = val_lots | test_lots | bm_lots | ex_lots

    # ── 候选池 ─────────────────────────────────────────
    pool: dict[str, list[dict]] = {c: [] for c in CANONICAL_CLASSES}
    for r in rows:
        if r["split"] != "train" or r["label_source"] != "ground_truth":
            continue
        if r["sample_id"] in excluded or r["lot_name"] in banned_lots:
            continue
        if r["failure_type"] in pool:
            pool[r["failure_type"]].append(r)
    for c in pool:
        pool[c].sort(key=lambda r: r["sample_id"])
    print(f"\n[4] 候选池（禁用 {len(banned_lots)} 个 lot）")
    for c in CANONICAL_CLASSES:
        print(f"      {c:10s} {len(pool[c]):5d} 行 / {len({r['lot_name'] for r in pool[c]}):5d} lot")

    # ── 抽样 ───────────────────────────────────────────
    rng = random.Random(SEED)
    used_lots: set[str] = set()
    picks: list[dict] = []
    draw_log: list[dict] = []
    for cls in CANONICAL_CLASSES:
        for k in range(1, PER_CLASS + 1):
            cands = [r for r in pool[cls] if r["lot_name"] not in used_lots]
            entry = {"class": cls, "k": k, "eligible": len(cands)}
            if not cands:
                entry["picked"] = None
                entry["note"] = "该类别在未占用 lot 中已无候选 —— 按任务书如实报告，不回填。"
                draw_log.append(entry)
                continue
            idx = rng.randrange(len(cands))
            row = cands[idx]
            used_lots.add(row["lot_name"])
            picks.append(row)
            entry.update({"picked": row["sample_id"], "lot": row["lot_name"],
                          "index_into_eligible": idx})
            draw_log.append(entry)
    print(f"\n[5] 抽样：抽到 {len(picks)} 张（目标 {len(CANONICAL_CLASSES) * PER_CLASS}）")

    ordered = list(picks)
    random.Random(SEED).shuffle(ordered)
    print(f"\n[6] 盲序")
    for i, r in enumerate(ordered, 1):
        print(f"      item_{i:03d}  {r['sample_id']}")

    # ── 图像检查 ───────────────────────────────────────
    from PIL import Image
    import io as _io
    import numpy as np
    print(f"\n[7] 图像检查")
    checks, shas, problems = [], [], []
    for i, r in enumerate(ordered, 1):
        sid = r["sample_id"]
        p = IMAGES_DIR / f"{sid}.png"
        item = f"item_{i:03d}"
        if not p.exists():
            problems.append(f"{item} {sid}: 缺图 {p}")
            checks.append({"item_id": item, "sample_id": sid, "exists": False, "ok": False})
            continue
        raw = p.read_bytes()
        with Image.open(_io.BytesIO(raw)) as im:
            im.load()
            fmt, size = im.format, im.size
            arr = np.asarray(im.convert("RGB"))
        cols = {tuple(int(x) for x in c) for c in np.unique(arr.reshape(-1, 3), axis=0)}
        ok = fmt == "PNG" and size == (448, 448) and cols <= {(0, 0, 0), (0, 255, 0), (255, 0, 0)}
        rec = {"item_id": item, "sample_id": sid, "exists": True, "ok": ok,
               "sha256": sha_bytes(raw), "size": list(size), "format": fmt,
               "n_colors": len(cols), "bytes": len(raw)}
        checks.append(rec)
        shas.append(rec["sha256"])
        if not ok:
            problems.append(f"{item} {sid}: PNG/尺寸/颜色不合格")
        print(f"      {item}  {sid}  {size} {fmt} {len(cols)}色 {len(raw)}B ok={ok}")

    lots = [r["lot_name"] for r in ordered]
    dup_sha = len(set(shas)) != len(shas)
    dup_sid = len({r["sample_id"] for r in ordered}) != len(ordered)
    if len(set(lots)) != len(lots):
        problems.append(f"lot 重复：{lots}")
    if dup_sha:
        problems.append("PNG SHA256 有重复")
    if dup_sid:
        problems.append("sample_id 有重复")
    hit_ex = [r["sample_id"] for r in ordered if r["sample_id"] in excluded]
    if hit_ex:
        problems.append(f"命中排除集合：{hit_ex}")

    print(f"\n[8] 隔离检查")
    print(f"      不同 lot     : {'是' if len(set(lots)) == len(lots) else '否'}")
    print(f"      SHA256 不重复: {'是' if not dup_sha else '否'}")
    print(f"      sample_id 不重复: {'是' if not dup_sid else '否'}")
    print(f"      未命中排除集 : {'是' if not hit_ex else '否 ' + str(hit_ex)}")
    print(f"      每类张数     : " +
          str({c: sum(1 for r in ordered if r['failure_type'] == c) for c in CANONICAL_CLASSES}))
    if problems:
        print(f"\n[!] {len(problems)} 个问题：")
        for x in problems:
            print(f"      - {x}")

    # ── 落盘 ───────────────────────────────────────────
    blind_lines = [{"item_id": f"item_{i:03d}", "sample_id": r["sample_id"],
                    "image_path": str(IMAGES_DIR / f"{r['sample_id']}.png"),
                    "image_sha256": next(c["sha256"] for c in checks
                                         if c["sample_id"] == r["sample_id"])}
                   for i, r in enumerate(ordered, 1)]
    audit_lines = [{"item_id": f"item_{i:03d}", "sample_id": r["sample_id"],
                    "manifest_line": r["__line__"], "failure_type_normalized": r["failure_type"],
                    "label_source": r["label_source"], "split": r["split"],
                    "lot_name": r["lot_name"], "wafer_index": r["wafer_index"],
                    "source_index": r["source_index"], "matrix_shape": r["matrix_shape"]}
                   for i, r in enumerate(ordered, 1)]

    plan = {
        "task": "下一轮盲清单准备 v2（九类各两张）",
        "task_book": "协作/01_Codex_指挥/任务书_20261002_ClaudeCode_口径修正与新样本准备.md",
        "prepared_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "sampling_rule_version": SAMPLING_RULE_VERSION,
        "supersedes_rule": "准备_20261002_九类试标/prepare_blind_set.py 的 prepare-v1（方法不变，只改每类数量与排除集合）",
        "seed": SEED, "per_class": PER_CLASS,
        "planned_count": len(ordered),
        "class_order_used": CANONICAL_CLASSES,
        "blind_order": [b["item_id"] + ":" + b["sample_id"] for b in blind_lines],
        "data_files": {"data/manifest.jsonl": man_sha, "manifest_rows": len(rows)},
        "prompt": {"rel_path": "协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt",
                   "byte_sha256": psha, "expected": PROMPT_EXPECTED, "unchanged": psha == PROMPT_EXPECTED},
        "exclusion_sources": {
            "search_roots": [str(p) for p in search_roots()],
            "seen_ids_with_send_evidence": sorted(seen_ids),
            "planned_only_ids_no_send_evidence": sorted(planned_only_ids),
            "planned_only_source": "这些编号只出现在历史 plan.json 里（evidence_kind=planned），"
                                   "没有任何 call-*/started-* 记录。",
            "this_round_verified_seen": sorted(prev_round),
            "excluded_total": sorted(excluded),
            "never_sent_planned_kept_in_pool": kept_never_sent,
            "a_ruling_note": "A 裁决：没有发送证据的旧计划样本**不排除**，保留在池内并披露不确定性。"
                             "本脚本按文件名区分：plan.json → 仅计划；call-*/started-*/结果文件 → 看过。",
            "lot_policy": "排除 val ∪ test ∪ benchmark ∪ 已排除样本的 lot",
            "banned_lots_total": len(banned_lots),
            "evidence_table": evidence,
        },
        "checks": {"distinct_lots": len(set(lots)) == len(lots),
                   "sha256_unique": not dup_sha, "sample_id_unique": not dup_sid,
                   "no_excluded_hit": not hit_ex, "all_images_ok": all(c.get("ok") for c in checks),
                   "per_class": {c: sum(1 for r in ordered if r["failure_type"] == c)
                                 for c in CANONICAL_CLASSES},
                   "problems": problems, "per_image": checks},
        "draw_log": draw_log,
        "notes": ["blind_inputs.jsonl 只含 item_id/sample_id/image_path/image_sha256。",
                  "audit_selected.jsonl 与 plan.json 含答案，只给 Codex，严禁转 ZCode。",
                  "原图留在 data/images/，未搬动、未复制。",
                  "本批是开发测试，**不是**教师资格认证；十八张不足以保证任何质量结论。"],
    }

    if not write:
        print("\n" + "=" * 78)
        print("DRY-RUN：未写文件。加 --write 落盘。")
        print("=" * 78)
        return 0

    targets = ["blind_inputs.jsonl", "audit_selected.jsonl", "plan.json"]
    exist = [t for t in targets if (out_dir / t).exists()]
    if exist:
        print(f"\n[停止] 已存在，不覆盖：{exist}。用 --out 换目录。")
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "blind_inputs.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for b in blind_lines:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    plan["blind_inputs_sha256"] = sha_file(out_dir / "blind_inputs.jsonl")
    with open(out_dir / "audit_selected.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for a in audit_lines:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    plan["audit_selected_sha256"] = sha_file(out_dir / "audit_selected.jsonl")
    with open(out_dir / "plan.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\n[9] 已写出到 {out_dir}")
    for t in targets:
        print(f"      {t:26s} {sha_file(out_dir / t)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
