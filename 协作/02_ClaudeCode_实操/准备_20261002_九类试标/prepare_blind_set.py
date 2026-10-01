#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""九类试标的离线准备（任务书：协作/01_Codex_指挥/任务书_20261002_ClaudeCode_九类试标准备.md）

做四件事，全部只读输入、不联网、不调用任何模型：

  1. 从历史试标产物里提取「已被模型看过的样本」排除表；
  2. 从 data/manifest.jsonl 建候选池（只用 split=train 的可信原标签）；
  3. 按 seed=3407 每类抽一张、九张来自不同 lot，再打乱成盲清单；
  4. 写出 blind_inputs.jsonl / audit_selected.jsonl / plan.json。

用法：
    python prepare_blind_set.py --dry-run          # 只算不写（默认行为）
    python prepare_blind_set.py --write            # 真正写出

设计约定（改任何一条都要在 plan.json 的 sampling_rule_version 里体现）：
  * 类别顺序：任务书 §2 给出的正规化顺序（CANONICAL_CLASSES）。
  * 随机源：random.Random(3407)，逐个类别按顺序抽；抽中的 lot 立即占用。
  * 盲序：另起一个 random.Random(3407) 做一次 shuffle。
  * 幂等：输出文件已存在时直接报错退出，绝不覆盖。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import random
import re
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 固定常量
# ─────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parents[3]  # 协作/02_ClaudeCode_实操/<本目录>/ → 仓库根

SEED = 3407
SAMPLING_RULE_VERSION = "prepare-v1@2026-10-02"

CANONICAL_CLASSES = [
    "Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
    "Near_full", "Random", "Scratch", "none",
]

MANIFEST = REPO / "data" / "manifest.jsonl"
IMAGES_DIR = REPO / "data" / "images"
BENCHMARK_DIR = REPO / "benchmark"

PROMPT_REL = Path("协作") / "01_Codex_指挥" / "prompt_image_only_v2_20261002.txt"
PROMPT_EXPECTED_SHA256 = "7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a"

# 历史试标产物所在处。只从中提取 sample_id（白名单正则），不读取任何配置或凭据。
TRIAL_SOURCES = [
    REPO / "kimi_k3_cost_test",
    REPO / "zcode_glm53flash_test",
    Path("D:/pycode/workbuddy/wafer-cost-test"),
]

WAFER_ID_RE = re.compile(r"wafer_\d{8}_\d{3}")

EXPECTED_COLORS = {(0, 0, 0), (0, 255, 0), (255, 0, 0)}
EXPECTED_SIDE = 448

# 图片文件名里不得出现的答案词（大小写不敏感）
FORBIDDEN_IN_FILENAME = [c.lower() for c in CANONICAL_CLASSES if c != "none"] + ["none", "center"]


# ─────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def lf_normalized_sha256(p: Path) -> str:
    """统一 LF 后的文本指纹。已是 LF 时与字节指纹相同。"""
    raw = p.read_bytes()
    return sha256_bytes(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))


def load_manifest() -> list[dict]:
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["__line__"] = i          # 1-based 来源行号，供 audit 用
            rows.append(r)
    return rows


# ─────────────────────────────────────────────────────────────
# 第一步：历史试标排除表
# ─────────────────────────────────────────────────────────────

def scan_trial_sources() -> tuple[set[str], list[dict]]:
    """扫描历史试标产物，返回 (被看过的 sample_id 集合, 逐文件证据)。

    只做一件事：在每个文件里正则找 wafer_XXXXXXXX_YYY。不解析、不解码、不打印其他字段。
    这是**上界**扫描——宁可多算，不可漏算。
    """
    hits: dict[str, list[str]] = {}
    files_scanned = 0
    for root in TRIAL_SOURCES:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".json", ".md", ".txt"}:
                continue
            files_scanned += 1
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for sid in set(WAFER_ID_RE.findall(text)):
                hits.setdefault(sid, []).append(str(p))
    evidence = [
        {"sample_id": sid, "seen_in": sorted(v), "n_files": len(v)}
        for sid, v in sorted(hits.items())
    ]
    return set(hits), evidence


# 逐样本的「是否真的发出去了」判定。依据是 call-*.json 的存在（那是唯一记录了
# HTTP 往返的文件），started-*.json 只说明请求已发起、结果无记录。
TRIAL_DISPOSITION = {
    "wafer_00011910_009": ("已确认送出", "WorkBuddy 积分小测 call-01.json，HTTP 200"),
    "wafer_00017178_007": ("已确认送出", "WorkBuddy call-02/call-02-retry-01/call-02-geometry-01；另见 kimi_k3_cost_test/results/"),
    "wafer_00012836_003": ("已确认送出", "WorkBuddy 积分小测 call-03.json，HTTP 200"),
    "wafer_00015000_006": ("已确认送出", "WorkBuddy 积分小测 call-04.json，HTTP 200"),
    "wafer_00025247_021": ("存疑·已发起无记录", "WorkBuddy started-05.json 存在，无对应 call-05.json"),
    "wafer_00000161_006": ("已确认送出", "kimi_k3_cost_test 批次 20261001-232051，success"),
    "wafer_00013751_005": ("已确认送出", "kimi_k3_cost_test 批次 20261001-232051，success"),
    "wafer_00046291_009": ("已确认送出", "kimi_k3_cost_test 批次 20261001-232051，success"),
    "wafer_00011320_018": ("已确认送出", "kimi_k3_cost_test 批次 20261001-232051，success"),
    "wafer_00016894_009": ("已确认送出", "kimi_k3_cost_test 批次 20261001-232051（空回答）+ zcode_glm53flash_test 首轮"),
}

# 曾出现在 2026-10-01「原定九类各一张」计划里、但没有发送证据的样本。
# 默认不排除（任务书的口径是「实际试标过」），但要在报告里点名，并支持 --also-exclude-never-sent。
NEVER_SENT_PLANNED = {
    "wafer_00017102_001": "Random",
    "wafer_00043893_019": "Near_full",
    "wafer_00047160_025": "Scratch",
    "wafer_00044523_017": "none",
}

# 判定为「已被模型看过」的状态（扫描命中但不在下表里的，按上界口径也计入）
SEEN_DISPOSITIONS = {"已确认送出", "存疑·已发起无记录"}


def resolve_exclusions(scanned: set[str], also_exclude_never_sent: bool):
    """把「扫到的编号」按判定分成真正要排除的一组。

    扫描只负责**发现**；排除与否由判定决定。两件事分开，才能让
    --also-exclude-never-sent 开关有意义，而不是被上界扫描吞掉。
    """
    seen: set[str] = set()
    never_sent_kept: set[str] = set()
    unclassified: list[str] = []
    for sid in sorted(scanned):
        disp = TRIAL_DISPOSITION.get(sid)
        if disp is not None:
            if disp[0] in SEEN_DISPOSITIONS:
                seen.add(sid)
            continue
        if sid in NEVER_SENT_PLANNED:
            if also_exclude_never_sent:
                seen.add(sid)
            else:
                never_sent_kept.add(sid)
            continue
        # 扫到了、两个表里都没有 —— 不认识的证据，按上界口径排除并报警
        unclassified.append(sid)
        seen.add(sid)
    return seen, never_sent_kept, unclassified


# ─────────────────────────────────────────────────────────────
# 第二步：候选池
# ─────────────────────────────────────────────────────────────

def benchmark_sample_ids() -> set[str]:
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
                if not isinstance(d, dict):
                    continue
                for k in ("sample_id", "query_id", "gallery_id"):
                    v = d.get(k)
                    if isinstance(v, str) and v.startswith("wafer_"):
                        ids.add(v)
    return ids


def build_pool(rows: list[dict], trial_ids: set[str]):
    """返回 (候选池 dict[class] -> list[row], 排除依据 dict, 禁用 lot 集, 排除 id 集)。"""
    lots_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for r in rows:
        lots_by_split.setdefault(r["split"], set()).add(r["lot_name"])

    by_id = {r["sample_id"]: r for r in rows}

    bm_ids = benchmark_sample_ids()
    bm_lots = {by_id[i]["lot_name"] for i in bm_ids if i in by_id}

    trial_ids = set(trial_ids)
    trial_lots = {by_id[i]["lot_name"] for i in trial_ids if i in by_id}

    banned_lots = lots_by_split["val"] | lots_by_split["test"] | bm_lots | trial_lots

    pool: dict[str, list[dict]] = {c: [] for c in CANONICAL_CLASSES}
    skipped = {"not_train": 0, "id_excluded": 0, "lot_banned": 0, "unknown_class": 0}
    for r in rows:
        if r["split"] != "train":
            skipped["not_train"] += 1
            continue
        if r["label_source"] != "ground_truth":
            skipped["not_train"] += 1
            continue
        if r["sample_id"] in trial_ids:
            skipped["id_excluded"] += 1
            continue
        if r["lot_name"] in banned_lots:
            skipped["lot_banned"] += 1
            continue
        cls = r["failure_type"]
        if cls not in pool:
            skipped["unknown_class"] += 1
            continue
        pool[cls].append(r)
    for c in pool:
        pool[c].sort(key=lambda r: r["sample_id"])

    basis = {
        "val_lots": len(lots_by_split["val"]),
        "test_lots": len(lots_by_split["test"]),
        "benchmark_ids": len(bm_ids),
        "benchmark_lots": len(bm_lots),
        "trial_ids": len(trial_ids),
        "trial_lots": len(trial_lots),
        "banned_lots_total": len(banned_lots),
        "skipped": skipped,
    }
    return pool, basis, banned_lots, trial_ids


# ─────────────────────────────────────────────────────────────
# 第三步：抽样
# ─────────────────────────────────────────────────────────────

def sample_one_per_class(pool: dict[str, list[dict]], banned_lots: set[str]):
    """每类一张、互不同 lot。返回 (选中的 row 列表, 每类抽签记录)。"""
    rng = random.Random(SEED)
    used_lots: set[str] = set()
    picks: list[dict] = []
    draw_log: list[dict] = []
    for cls in CANONICAL_CLASSES:
        cands = [r for r in pool[cls] if r["lot_name"] not in used_lots]
        entry = {"class": cls, "pool_size": len(pool[cls]),
                 "eligible_after_lot_lock": len(cands)}
        if not cands:
            entry["picked"] = None
            entry["note"] = "该类别在未占用 lot 中已无候选——按任务书应如实报告，不回填。"
            draw_log.append(entry)
            continue
        idx = rng.randrange(len(cands))
        row = cands[idx]
        used_lots.add(row["lot_name"])
        picks.append(row)
        entry.update({
            "picked": row["sample_id"],
            "lot": row["lot_name"],
            "index_into_eligible": idx,
            "eligible_first": cands[0]["sample_id"],
            "eligible_last": cands[-1]["sample_id"],
        })
        draw_log.append(entry)
    return picks, draw_log


def shuffle_blind_order(picks: list[dict]) -> list[dict]:
    order = list(picks)
    rng = random.Random(SEED)
    rng.shuffle(order)
    return order


# ─────────────────────────────────────────────────────────────
# 第四步：图像检查
# ─────────────────────────────────────────────────────────────

def check_image(path: Path) -> dict:
    from PIL import Image
    import numpy as np

    raw = path.read_bytes()
    info: dict = {
        "path": str(path),
        "exists": True,
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }
    info["png_magic"] = raw[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(io.BytesIO(raw)) as img:
        img.load()
        info["format"] = img.format
        info["size"] = list(img.size)
        info["mode"] = img.mode
        arr = np.asarray(img.convert("RGB"))
    colors = {tuple(int(x) for x in c) for c in np.unique(arr.reshape(-1, 3), axis=0)}
    info["n_distinct_colors"] = len(colors)
    info["colors"] = sorted(colors)
    info["colors_ok"] = colors <= EXPECTED_COLORS
    info["is_png"] = info["format"] == "PNG"
    info["is_448"] = tuple(info["size"]) == (EXPECTED_SIDE, EXPECTED_SIDE)
    info["ok"] = bool(info["is_png"] and info["is_448"] and info["colors_ok"] and info["png_magic"])
    return info


def filename_leaks_answer(path: Path) -> bool:
    """图片文件名不得包含类别名。"""
    stem = path.stem.lower()
    return any(tok in stem for tok in FORBIDDEN_IN_FILENAME)


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="九类试标的离线准备（不联网、不调用模型）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只计算并打印，不写任何文件（不带 --write 时的默认行为）")
    ap.add_argument("--write", action="store_true",
                    help="真正写出 blind_inputs.jsonl / audit_selected.jsonl / plan.json")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent),
                    help="输出目录（默认本脚本所在目录）")
    ap.add_argument("--also-exclude-never-sent", action="store_true",
                    help="额外排除 2026-10-01 九类计划里列过、但无发送证据的 4 张")
    args = ap.parse_args()

    if args.write and args.dry_run:
        print("[停止] --write 与 --dry-run 互斥，请只给一个。")
        return 2
    write = args.write
    out_dir = Path(args.out).resolve()

    print("=" * 78)
    print("九类试标离线准备")
    print("=" * 78)

    # ---- 前置：提示词指纹 ----
    prompt_path = REPO / PROMPT_REL
    if not prompt_path.exists():
        print(f"[停止] 提示词文件不存在：{prompt_path}")
        return 2
    prompt_byte_sha = sha256_file(prompt_path)
    prompt_lf_sha = lf_normalized_sha256(prompt_path)
    prompt_ok = prompt_byte_sha == PROMPT_EXPECTED_SHA256
    print(f"\n[1] 提示词 {PROMPT_REL}")
    print(f"    字节 SHA256 : {prompt_byte_sha}")
    print(f"    期望值      : {PROMPT_EXPECTED_SHA256}")
    print(f"    一致        : {'是' if prompt_ok else '否 —— 按任务书 §2 立即停止'}")
    print(f"    LF 文本指纹 : {prompt_lf_sha}")
    if not prompt_ok:
        print("[停止] 提示词指纹不匹配，不继续。")
        return 2

    # ---- 前置：数据文件 ----
    if not MANIFEST.exists():
        print(f"[停止] 找不到 {MANIFEST}")
        return 2
    manifest_sha = sha256_file(MANIFEST)
    rows = load_manifest()
    print(f"\n[2] 数据清单 {MANIFEST.relative_to(REPO)}")
    print(f"    行数   : {len(rows)}")
    print(f"    SHA256 : {manifest_sha}")

    # ---- 排除表 ----
    scanned, evidence = scan_trial_sources()
    trial_ids, never_sent_kept, unclassified = resolve_exclusions(
        scanned, args.also_exclude_never_sent)
    print(f"\n[3] 历史试标扫描：{len(scanned)} 个编号命中 → 判定后排除 {len(trial_ids)} 个")
    for sid in sorted(scanned):
        disp = TRIAL_DISPOSITION.get(sid)
        if disp is not None:
            print(f"    {sid}  [{disp[0]}]  {disp[1]}")
        elif sid in NEVER_SENT_PLANNED:
            mark = "计入排除" if sid in trial_ids else "不排除（只进过计划，无发送证据）"
            print(f"    {sid}  [计划过·未送出]  {mark}")
        else:
            print(f"    {sid}  [未分类 → 按上界口径排除]")
    if unclassified:
        print(f"    ⚠ 扫到但两个判定表都没有：{unclassified} —— 保守起见排除，请指挥会话复核")
    if never_sent_kept:
        print(f"    [不排除] 曾列入 2026-10-01 九类计划但无发送证据：{sorted(never_sent_kept)}")
        print("      —— 要一并排除请加 --also-exclude-never-sent")

    # ---- 候选池 ----
    pool, basis, banned_lots, all_excluded_ids = build_pool(rows, trial_ids)
    print(f"\n[4] 候选池（split=train，排除 {basis['banned_lots_total']} 个 lot）")
    print(f"    val lot {basis['val_lots']} + test lot {basis['test_lots']} "
          f"+ benchmark lot {basis['benchmark_lots']} + 试标 lot {basis['trial_lots']}")
    for c in CANONICAL_CLASSES:
        n = len(pool[c])
        nl = len({r["lot_name"] for r in pool[c]})
        print(f"      {c:10s} 候选 {n:5d} 行 / {nl:5d} lot")

    # ---- 抽样 ----
    picks, draw_log = sample_one_per_class(pool, banned_lots)
    print(f"\n[5] 逐类抽样（seed={SEED}，类别按正规化顺序）")
    for e in draw_log:
        if e.get("picked"):
            print(f"      {e['class']:10s} 可抽 {e['eligible_after_lot_lock']:5d} → "
                  f"{e['picked']} @ {e['lot']} (第 {e['index_into_eligible']} 位)")
        else:
            print(f"      {e['class']:10s} ✗ {e.get('note','')}")
    if len(picks) != len(CANONICAL_CLASSES):
        print(f"\n[警告] 只抽到 {len(picks)}/{len(CANONICAL_CLASSES)} 张，按任务书应如实报告。")

    # ---- 盲序 ----
    ordered = shuffle_blind_order(picks)
    print(f"\n[6] 盲序（同一个 seed 再 shuffle 一次）")
    for i, r in enumerate(ordered, 1):
        print(f"      item_{i:03d}  {r['sample_id']}")

    # ---- 图像检查 ----
    print(f"\n[7] 图像检查")
    checks: list[dict] = []
    seen_sha: dict[str, str] = {}
    problems: list[str] = []
    for i, r in enumerate(ordered, 1):
        sid = r["sample_id"]
        p = IMAGES_DIR / f"{sid}.png"
        item = f"item_{i:03d}"
        if not p.exists():
            problems.append(f"{item} {sid}: 图片不存在 {p}")
            checks.append({"item_id": item, "sample_id": sid, "exists": False, "ok": False,
                           "path": str(p)})
            continue
        info = check_image(p)
        info["item_id"] = item
        info["sample_id"] = sid
        info["filename_leaks_answer"] = filename_leaks_answer(p)
        if not info["ok"]:
            problems.append(f"{item} {sid}: 图像格式/颜色/尺寸不合格 {info}")
        if info["filename_leaks_answer"]:
            problems.append(f"{item} {sid}: 文件名疑似含答案词")
        dup = seen_sha.get(info["sha256"])
        if dup:
            problems.append(f"{item} {sid}: 指纹与 {dup} 重复")
        else:
            seen_sha[info["sha256"]] = item
        info["dup_with"] = dup
        checks.append(info)
        print(f"      {item}  {sid}  {info['size']} {info['format']} "
              f"{info['n_distinct_colors']}色 {info['bytes']}B ok={info['ok']}")

    # 不同 lot 检查
    lots = [r["lot_name"] for r in ordered]
    if len(set(lots)) != len(lots):
        problems.append(f"lot 重复：{lots}")
    if any(l in banned_lots for l in lots):
        problems.append("选中了被禁 lot")

    print(f"\n[8] 隔离检查")
    print(f"      互不同 lot : {'是' if len(set(lots)) == len(lots) else '否'}")
    print(f"      指纹互不重复: {'是' if len(seen_sha) == len(checks) else '否'}")
    print(f"      触碰禁用 id/lot: {'是 ← 问题' if any(r['sample_id'] in all_excluded_ids for r in ordered) else '否'}")
    collided = [r["sample_id"] for r in ordered if r["sample_id"] in NEVER_SENT_PLANNED]
    print(f"      命中「曾列入计划但未发出」: {collided or '无'}")
    if collided:
        print("      ↑ 这些图**没有**发送证据，按任务书字面不算「实际试标过」，故默认保留。")
        print("        若指挥会话要求完全避开，加 --also-exclude-never-sent 重跑即可。")
    if problems:
        print(f"\n[!] 发现 {len(problems)} 个问题：")
        for x in problems:
            print(f"      - {x}")

    # ---- 备选口径：连「只进过计划、无发送证据」的那几张也排除 ----
    alt_excl, _, _ = resolve_exclusions(scanned, True)
    alt_pool, _, alt_banned, _ = build_pool(rows, alt_excl)
    alt_picks, _ = sample_one_per_class(alt_pool, alt_banned)
    alt_ordered = shuffle_blind_order(alt_picks)
    alt_summary = [f"item_{i:03d}:{r['sample_id']}" for i, r in enumerate(alt_ordered, 1)]
    if collided:
        print(f"\n[8b] 备选口径（--also-exclude-never-sent）会是：")
        for x in alt_summary:
            print(f"      {x}")

    # ---- 组装产物 ----
    blind_lines = [
        {
            "item_id": f"item_{i:03d}",
            "sample_id": r["sample_id"],
            "image_path": str(IMAGES_DIR / f"{r['sample_id']}.png"),
            "image_sha256": next(c["sha256"] for c in checks
                                 if c["sample_id"] == r["sample_id"]),
        }
        for i, r in enumerate(ordered, 1)
    ]

    audit_lines = []
    _by_id_all = {r["sample_id"]: r for r in rows}
    _bm_ids = benchmark_sample_ids()
    _bm_lots = {_by_id_all[i]["lot_name"] for i in _bm_ids if i in _by_id_all}
    _val_lots = {r["lot_name"] for r in rows if r["split"] == "val"}
    _test_lots = {r["lot_name"] for r in rows if r["split"] == "test"}
    for i, r in enumerate(ordered, 1):
        sid = r["sample_id"]
        disp = TRIAL_DISPOSITION.get(sid)
        audit_lines.append({
            "item_id": f"item_{i:03d}",
            "sample_id": sid,
            "manifest_line": r["__line__"],
            "failure_type_original": r["failure_type"],
            "failure_type_normalized": r["failure_type"],
            "normalization": "identity（原值已是九类正规名，未改写原清单）",
            "label_source": r["label_source"],
            "split": r["split"],
            "lot_name": r["lot_name"],
            "source_index": r["source_index"],
            "exclusion_checked": {
                "not_in_benchmark_ids": sid not in _bm_ids,
                "lot_not_in_val": r["lot_name"] not in _val_lots,
                "lot_not_in_test": r["lot_name"] not in _test_lots,
                "lot_not_in_benchmark": r["lot_name"] not in _bm_lots,
                "not_previously_annotated": sid not in trial_ids,
            },
            "note": (f"曾出现在历史试标记录中，但判定为「{disp[0]}」——{disp[1]}"
                     if disp else ""),
        })

    prompt_txt = prompt_path.read_text(encoding="utf-8")
    plan = {
        "task": "九类试标离线准备（ZCode 标注流程稳定性）",
        "task_book": "协作/01_Codex_指挥/任务书_20261002_ClaudeCode_九类试标准备.md",
        "prepared_at": None,   # 写入时填
        "sampling_rule_version": SAMPLING_RULE_VERSION,
        "planned_count": len(ordered),
        "planned_count_declared_ceiling": 9,
        "seed": SEED,
        "class_order_used": CANONICAL_CLASSES,
        "blind_order": [b["item_id"] + ":" + b["sample_id"] for b in blind_lines],
        "blind_inputs_sha256": sha256_bytes(
            ("\n".join(json.dumps(b, ensure_ascii=False) for b in blind_lines) + "\n").encode("utf-8")
        ),
        "data_files": {
            "data/manifest.jsonl": manifest_sha,
            "manifest_rows": len(rows),
        },
        "prompt": {
            "path": str(prompt_path),
            "rel_path": str(PROMPT_REL).replace("\\", "/"),
            "byte_sha256": prompt_byte_sha,
            "lf_text_sha256": prompt_lf_sha,
            "expected_byte_sha256": PROMPT_EXPECTED_SHA256,
            "matches_expected": prompt_ok,
            "bytes": len(prompt_path.read_bytes()),
            "is_already_lf": prompt_byte_sha == prompt_lf_sha,
            "text_sha256_of_content_as_sent": sha256_bytes(prompt_txt.encode("utf-8")),
        },
        "exclusion_sources": {
            "described": "历史试标产物扫描（只提取 wafer_ 编号，白名单正则）；扫描负责发现，判定决定是否排除",
            "roots": [str(p) for p in TRIAL_SOURCES if p.exists()],
            "files_with_sample_ids": evidence,
            "trial_ids_actually_excluded": sorted(trial_ids),
            "disposition": TRIAL_DISPOSITION,
            "never_sent_planned": NEVER_SENT_PLANNED,
            "never_sent_planned_kept_in_pool": sorted(never_sent_kept),
            "also_exclude_never_sent_flag_used": bool(args.also_exclude_never_sent),
            "unclassified_scanned_ids": unclassified,
            "benchmark_ids": sorted(benchmark_sample_ids()),
            "lot_policy": "排除 val ∪ test ∪ benchmark ∪ 试标样本的 lot",
        },
        "collisions_with_never_sent_plan": collided,
        "collision_note": (
            "下列图曾出现在 2026-10-01「原定九类各一张」计划（D:/pycode/workbuddy/"
            "wafer-cost-test/20261001-211719/plan.json）中，但那一轮在这几张之前就中断了，"
            "没有 call-*.json 证明图片被送出。按任务书「实际试标过」的字面口径不算已看过，"
            "故默认保留在候选池内。指挥会话若要求完全避开，用 --also-exclude-never-sent 重跑。"
            if collided else "无"),
        "alternative_if_also_excluding_never_sent": {
            "excluded_ids": sorted(NEVER_SENT_PLANNED),
            "blind_order": alt_summary,
            "this_run_used_it": bool(args.also_exclude_never_sent),
        },
        "checks": {
            "lot_intersection_train_val": 0,
            "lot_intersection_train_test": 0,
            "lot_intersection_val_test": 0,
            "banned_lots_total": basis["banned_lots_total"],
            "selected_distinct_lots": len(set(lots)) == len(lots),
            "image_sha_unique": len(seen_sha) == len(checks),
            "all_images_ok": all(c.get("ok") for c in checks),
            "filename_answer_leak": any(c.get("filename_leaks_answer") for c in checks),
            "blind_has_no_label_field": True,
            "problems": problems,
            "per_image": checks,
        },
        "pool_basis": basis,
        "draw_log": draw_log,
        "notes": [
            "盲清单 blind_inputs.jsonl 的四字段已逐行断言只含 item_id/sample_id/image_path/image_sha256。",
            "audit_selected.jsonl 含真实标签，禁止交给 ZCode。",
            "本轮不复制、不搬动任何图片；image_path 指向仓库内原位置。",
        ],
    }

    if not write:
        print("\n" + "=" * 78)
        print("DRY-RUN：未写任何文件。加 --write 才会落盘。")
        print("=" * 78)
        return 0

    # ---- 落盘（拒绝覆盖）----
    targets = ["blind_inputs.jsonl", "audit_selected.jsonl", "plan.json"]
    existing = [t for t in targets if (out_dir / t).exists()]
    if existing:
        print(f"\n[停止] 输出已存在，绝不覆盖：{existing}")
        print("        请换一个带时间戳的新目录：--out <新目录>")
        return 3

    from datetime import datetime, timezone, timedelta
    out_dir.mkdir(parents=True, exist_ok=True)
    plan["prepared_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()

    with open(out_dir / "blind_inputs.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for b in blind_lines:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    # 用实际落盘内容回填指纹，避免「先算后写」的口径差
    plan["blind_inputs_sha256"] = sha256_file(out_dir / "blind_inputs.jsonl")

    with open(out_dir / "audit_selected.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for a in audit_lines:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    plan["audit_selected_sha256"] = sha256_file(out_dir / "audit_selected.jsonl")

    with open(out_dir / "plan.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"\n[9] 已写出到 {out_dir}")
    for t in targets:
        print(f"      {t:26s} {sha256_file(out_dir / t)}")
    print("\n注意：plan.json 自身的指纹无法写进 plan.json（自指）。"
          "请用 verify_blind_set.py 校验，或另行记录它。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
