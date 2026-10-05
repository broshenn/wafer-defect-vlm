#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GLM 训练标注的数据准备 **v2**（清单与校验；看图作答由 ZCode 完成）。

**不覆盖 v1**（`GLM训练标注准备_20261002/prepare_glm_annotation.py`）。v1 保留作历史版本。

v2 改两处：
  1. `task_book` 指向**实际任务文件名** `任务书_20261002_ClaudeCode_GLM训练标注准备.md`
     （v1 写成 `…GLM标注准备.md`，少了「训练」二字，那个文件不存在）。
  2. `--resume` **复用已有的 `prepared_at`**，使 `plan.json` 在重复 resume 下**字节稳定**。
     v1 每次都刷新时间戳，导致 plan.json 的 hash 每次重写 —— 而 hash 是验收依据。

**本轮已冻结的输入不使用本版本**：ZCode 仍用已验收的旧索引，修复不改运行输入。

## 产出

  blind_inputs_all_180.jsonl   全量盲清单，180 行
  batch_01_blind.jsonl … batch_09_blind.jsonl   九个批次，各 20 行
  batch_01_worker_01.jsonl … batch_01_worker_04.jsonl   首批四份子清单，各 5 行
  audit_selected.jsonl         含标签/lot/split/manifest 行号；**只给 Codex，禁止给 ZCode**
  plan.json                    含审计信息，**不交标注会话**
  zcode_batch_01_inputs.json   **唯一给 ZCode 的输入索引**

## 分批规则（照任务书 §1）

  首批 batch_01 = `smoke_20.jsonl` 的**原顺序**（20 条）
  其余 160 = `train_180.jsonl` 的**原顺序**剔除首批 ID，按 20 切 8 批（batch_02…09）
  全局 item_id 按此顺序连续编号 item_001…item_180
  **不重新选样、不按教师表现调整名单、不按类别排序**

## 盲清单字段

每行**仅** item_id / sample_id / image_path / image_sha256。
不含标签、lot、split、难度、几何或任何评价。

用法：
    python prepare_glm_annotation.py --dry-run      # 只算不写（默认）
    python prepare_glm_annotation.py --write
    python prepare_glm_annotation.py --write --resume   # 允许复用字节一致的已有文件
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "协作" / "02_ClaudeCode_实操" / "学校训练准备_20261002-025848"

SOURCE_SHA = {
    "train_180.jsonl": "04febd99674f70bf4619c3b529fe4c1313a5b8d739e89c69c390fe686324db47",
    "smoke_20.jsonl": "f59ae6ba5ba917fae410a9a82f6747c886abde20df189878093467242dd98eec",
    "dev_18.jsonl": "08b267ea73def5b3119dbf6ca73d27630ef565bb4011f6e53b404d753f7972ac",
}
PROMPT = REPO / "协作" / "01_Codex_指挥" / "prompt_image_only_v2_20261002.txt"
PROMPT_SHA = "7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a"
CHECKER = (REPO / "协作" / "02_ClaudeCode_实操" / "口径修正与准备_20261002-021207"
           / "check_answer_v3.py")
CHECKER_SHA = "794a254e005eaa232362c539817abde98a1c2406ecb52574b9e9edb35303ec55"

MANIFEST = REPO / "data" / "manifest.jsonl"
BENCHMARK = REPO / "benchmark"

BATCH_SIZE = 20
N_BATCHES = 9
WORKERS = 4                      # 首批四个 worker
BLIND_KEYS = ["item_id", "sample_id", "image_path", "image_sha256"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def stop(msg: str) -> None:
    print(f"[停止] {msg}")
    sys.exit(3)


def load_and_verify() -> tuple[list[dict], list[dict], list[dict]]:
    """核对来源指纹；漂移即停，不静默修原数据。"""
    for name, want in SOURCE_SHA.items():
        p = SRC / name
        if not p.exists():
            stop(f"来源缺失：{p}")
        got = sha256_file(p)
        if got != want:
            stop(f"来源指纹漂移：{name}\n  实际 {got}\n  期望 {want}")
    if sha256_file(PROMPT) != PROMPT_SHA:
        stop(f"提示词指纹不符：{PROMPT}")
    if sha256_file(CHECKER) != CHECKER_SHA:
        stop(f"检查器指纹不符：{CHECKER}")
    return (read_jsonl(SRC / "train_180.jsonl"),
            read_jsonl(SRC / "smoke_20.jsonl"),
            read_jsonl(SRC / "dev_18.jsonl"))


def build_order(train: list[dict], smoke: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    """返回 (全局顺序 180 条, 九个批次)。"""
    smoke_ids = [r["sample_id"] for r in smoke]
    if len(set(smoke_ids)) != BATCH_SIZE:
        stop("smoke_20 有重复 sample_id")
    train_ids = {r["sample_id"] for r in train}
    missing = [s for s in smoke_ids if s not in train_ids]
    if missing:
        stop(f"smoke_20 里有 {len(missing)} 个不在 train_180：{missing[:5]}")

    b1 = list(smoke)                                          # 首批保持 smoke 原顺序
    rest = [r for r in train if r["sample_id"] not in set(smoke_ids)]  # 保持 train 原顺序
    if len(rest) != (N_BATCHES - 1) * BATCH_SIZE:
        stop(f"剩余 {len(rest)} 条，不等于 {(N_BATCHES-1)*BATCH_SIZE}")

    batches = [b1] + [rest[i:i + BATCH_SIZE] for i in range(0, len(rest), BATCH_SIZE)]
    flat = [r for b in batches for r in b]
    if len(flat) != N_BATCHES * BATCH_SIZE:
        stop(f"总条数 {len(flat)} != {N_BATCHES*BATCH_SIZE}")
    if len({r["sample_id"] for r in flat}) != len(flat):
        stop("全局 sample_id 有重复")
    return flat, batches


def check_isolation(rows: list[dict]) -> dict:
    """核对 manifest 对应、ground_truth 来源、以及 train/dev/test/benchmark 的 lot 隔离。"""
    man = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                man[r["sample_id"]] = (i, r)

    problems = []
    for r in rows:
        sid = r["sample_id"]
        if sid not in man:
            problems.append(f"{sid} 不在 manifest")
            continue
        line_no, m = man[sid]
        if m["label_source"] != "ground_truth":
            problems.append(f"{sid} label_source={m['label_source']}")
        if m["failure_type"] != r["label"]:
            problems.append(f"{sid} 标签不一致：清单 {r['label']} vs manifest {m['failure_type']}")
        if m["lot_name"] != r["lot_name"]:
            problems.append(f"{sid} lot 不一致")
        if r.get("manifest_line") != line_no:
            problems.append(f"{sid} manifest_line {r.get('manifest_line')} != 实际 {line_no}")

    lots = {s: {r["lot_name"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    bm_ids = set()
    for p in sorted(BENCHMARK.glob("*.jsonl")):
        for line in open(p, encoding="utf-8"):
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
    bm_lots = {man[i][1]["lot_name"] for i in bm_ids if i in man}

    inter = {
        "train∩val": len(lots["train"] & lots["val"]),
        "train∩test": len(lots["train"] & lots["test"]),
        "val∩test": len(lots["val"] & lots["test"]),
        "train∩benchmark": len(lots["train"] & bm_lots),
        "val∩benchmark": len(lots["val"] & bm_lots),
    }
    return {"problems": problems, "lot_intersections": inter,
            "manifest_rows": len(man), "benchmark_lots": len(bm_lots)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    if a.write and a.dry_run:
        print("[停止] --write 与 --dry-run 互斥")
        return 2
    out = Path(a.out).resolve()

    print("=" * 78)
    print("GLM 训练标注的数据准备")
    print("=" * 78)

    train, smoke, dev = load_and_verify()
    print(f"\n[1] 来源指纹全部吻合（train {len(train)} / smoke {len(smoke)} / dev {len(dev)}）")
    print(f"    提示词 {PROMPT_SHA[:16]}…  检查器 {CHECKER_SHA[:16]}…  均已核对")

    flat, batches = build_order(train, smoke)
    print(f"\n[2] 分批：{len(batches)} 批 × {BATCH_SIZE} = {len(flat)} 条")
    for i, b in enumerate(batches, 1):
        print(f"    batch_{i:02d}  {len(b):2d} 条  首个 {b[0]['sample_id']}")

    iso = check_isolation(flat)
    print(f"\n[3] 隔离与来源核对")
    for k, v in iso["lot_intersections"].items():
        print(f"    {k:18s} {v}")
    if iso["problems"]:
        print(f"    ⚠ {len(iso['problems'])} 个问题：")
        for x in iso["problems"][:10]:
            print(f"      - {x}")
    else:
        print("    manifest 对应 / ground_truth / 标签 / lot 全部一致，无问题")
    if any(iso["lot_intersections"].values()):
        stop("lot 泄漏，不放宽")

    # 图片指纹与存在性
    img_problems, img_sha = [], {}
    for r in flat:
        p = Path(r["image_path"])
        if not p.exists():
            img_problems.append(f"{r['sample_id']} 缺图")
            continue
        img_sha[r["sample_id"]] = sha256_file(p)
    print(f"    图片：{len(img_sha)}/{len(flat)} 张存在"
          + (f"，{len(img_problems)} 张缺图" if img_problems else "，无缺图"))
    if len(set(img_sha.values())) != len(img_sha):
        stop("图片指纹有重复")
    if img_problems:
        stop("有缺图，不静默跳过")

    # ── 组装各文件 ──────────────────────────────────────
    def blind_line(idx: int, r: dict) -> dict:
        return {"item_id": f"item_{idx:03d}", "sample_id": r["sample_id"],
                "image_path": r["image_path"], "image_sha256": img_sha[r["sample_id"]]}

    all_lines, batch_lines, cursor = [], {}, 0
    for bi, b in enumerate(batches, 1):
        lines = []
        for r in b:
            cursor += 1
            lines.append(blind_line(cursor, r))
        batch_lines[bi] = lines
        all_lines += lines

    worker_lines = {}
    for w in range(WORKERS):
        lo, hi = w * 5, w * 5 + 5
        worker_lines[w + 1] = batch_lines[1][lo:hi]

    # 子清单并集必须精确等于首批
    union = [x for w in worker_lines.values() for x in w]
    assert [x["item_id"] for x in union] == [x["item_id"] for x in batch_lines[1]], "子清单并集 != 首批"

    man_line = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                man_line[json.loads(line)["sample_id"]] = ln

    audit_lines = []
    for i, r in enumerate(flat, 1):
        audit_lines.append({"item_id": f"item_{i:03d}", "sample_id": r["sample_id"],
                            "manifest_line": man_line.get(r["sample_id"]),
                            "label": r["label"], "label_source": r["label_source"],
                            "lot_name": r["lot_name"], "split": r["split"],
                            "image_sha256": img_sha[r["sample_id"]]})

    # ── 落盘 ────────────────────────────────────────────
    def to_text(lines) -> str:
        return "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines)

    files: dict[str, str] = {"blind_inputs_all_180.jsonl": to_text(all_lines),
                             "audit_selected.jsonl": to_text(audit_lines)}
    for bi in range(1, N_BATCHES + 1):
        files[f"batch_{bi:02d}_blind.jsonl"] = to_text(batch_lines[bi])
    for w in range(1, WORKERS + 1):
        files[f"batch_01_worker_{w:02d}.jsonl"] = to_text(worker_lines[w])

    if not a.write:
        print(f"\n[4] dry-run：将写 {len(files)} 个清单文件 + plan.json + zcode_batch_01_inputs.json")
        for name in sorted(files):
            print(f"      {name:34s} {len(files[name].splitlines()):3d} 行  "
                  f"{sha256_text(files[name])[:16]}…")
        print("\n" + "=" * 78)
        print("DRY-RUN：未写任何文件。加 --write 落盘。")
        print("=" * 78)
        return 0

    # 拒绝覆盖
    targets = sorted(files) + ["plan.json", "zcode_batch_01_inputs.json"]
    existed = [t for t in targets if (out / t).exists()]
    if existed:
        if not a.resume:
            print(f"\n[停止] 已存在 {len(existed)} 个文件，拒绝覆盖：{existed[:4]}…")
            print("        加 --resume 仅当字节一致时才放行。")
            return 3
        drift = [t for t in existed if (out / t).read_text(encoding="utf-8") != files.get(t)]
        if drift and any(files.get(t) for t in drift):
            print(f"\n[停止] --resume 但内容不一致：{drift[:4]}…")
            return 3

    out.mkdir(parents=True, exist_ok=True)

    # resume 时**复用已有的 prepared_at**：时间戳若每次刷新，plan.json 的 hash 就每次都变，
    # 而 hash 是验收依据 —— 那就等于没有可核对的稳定产物。
    prepared_at = None
    existing_plan = out / "plan.json"
    if a.resume and existing_plan.exists():
        try:
            prepared_at = json.loads(existing_plan.read_text(encoding="utf-8")).get("prepared_at")
        except (OSError, json.JSONDecodeError):
            prepared_at = None
    if not prepared_at:
        prepared_at = datetime.now(timezone(timedelta(hours=8))).isoformat()

    written = {}
    for name, text in files.items():
        p = out / name
        p.write_text(text, encoding="utf-8", newline="\n")
        written[name] = sha256_file(p)

    plan = {
        "task": "GLM 训练标注的数据准备",
        "task_book": "协作/01_Codex_指挥/任务书_20261002_ClaudeCode_GLM训练标注准备.md",
        "prepared_at": prepared_at,
        "seed": 3407,
        "source_files_sha256": SOURCE_SHA,
        "prompt": {"path": str(PROMPT), "sha256": PROMPT_SHA},
        "checker": {"path": str(CHECKER), "sha256": CHECKER_SHA},
        "batching": {"rule": "首批=smoke_20 原顺序；其余=train_180 原顺序剔除首批后按 20 切",
                     "n_batches": N_BATCHES, "batch_size": BATCH_SIZE,
                     "global_item_ids": "item_001..item_180，按批次顺序连续编号",
                     "not_resampled": True, "not_adjusted_by_teacher": True},
        "isolation": iso,
        "image_problems": img_problems,
        "batch_first_item": {f"batch_{i:02d}": b[0]["sample_id"] for i, b in enumerate(batches, 1)},
        "worker_split": {f"batch_01_worker_{w:02d}": [x["item_id"] for x in worker_lines[w]]
                         for w in range(1, WORKERS + 1)},
        "file_sha256": written,
        "notes": [
            "盲清单只含 item_id/sample_id/image_path/image_sha256。",
            "audit_selected.jsonl 与 plan.json 含答案，只给 Codex/Claude，禁止给 ZCode。",
            "本轮未生成任何 GLM 答案，未启动标注子 agent。",
        ],
    }
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8", newline="\n")

    zcode = {
        "batch_id": "batch_01",
        "total_limit": BATCH_SIZE,
        "max_concurrency": WORKERS,
        "batch_manifest": {"path": str(out / "batch_01_blind.jsonl"),
                           "sha256": written["batch_01_blind.jsonl"]},
        "prompt": {"path": str(PROMPT), "sha256": PROMPT_SHA},
        "checker": {"path": str(CHECKER), "sha256": CHECKER_SHA},
        "workers": [
            {"worker_id": f"worker_{w:02d}",
             "input_path": str(out / f"batch_01_worker_{w:02d}.jsonl"),
             "input_sha256": written[f"batch_01_worker_{w:02d}.jsonl"],
             "count": 5} for w in range(1, WORKERS + 1)
        ],
    }
    (out / "zcode_batch_01_inputs.json").write_text(
        json.dumps(zcode, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\n[5] 已写出到 {out}")
    for name in sorted(written):
        print(f"      {name:34s} {written[name]}")
    print(f"      {'plan.json':34s} {sha256_file(out / 'plan.json')}")
    print(f"      {'zcode_batch_01_inputs.json':34s} {sha256_file(out / 'zcode_batch_01_inputs.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
