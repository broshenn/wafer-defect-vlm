#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 source_manifest.json（统一 TEST200 打分用）。

每条来源 = 一个模型在 TEST200 上的一次作答产物，带**哈希钉死的来源声明**：
  · rawfile + raw_sha256：原答文件本身；
  · identity：ids/reference/images/prompt 四个 SHA 必须与合同一致；
  · evidence：证明"同一份考卷"的哈希文件（服务器消费版考卷 + 该次运行的台账）；
  · inputfile：实际消费的考卷，由打分器逐 ID 复验题面/图片/参考。

**本脚本不修改任何原答**：只读 `sources/`、`evidence/`、`inputs/` 并生成清单。
未在本清单里的模型，等于"尚无 TEST200 原答"，需另行新推理或走百炼。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "sources"
EVIDENCE = HERE / "evidence"
INPUTS = HERE / "inputs"


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(HERE.resolve())).replace("\\", "/")


def main() -> int:
    contract = json.loads((HERE / "contract.json").read_text(encoding="utf-8"))
    IDS = contract["ids"]["canonical_sha256"]
    REF = contract["reference"]["sha256"]
    IMG = contract["images"]["sha256"]
    P_MAIN = contract["prompts"]["main"]["sha256"]
    P_L = contract["prompts"]["L"]["sha256"]
    SRV_EXAM = INPUTS / "server_test200_main.jsonl"
    SRV_L = INPUTS / "server_test_L.jsonl"
    L200 = INPUTS / "L200.jsonl"
    # 考卷目录：按台账声明的 dataset_sha256 匹配，而不是写死文件名
    EXAM_CATALOG = {}
    for p in (SRV_EXAM, SRV_L, L200):
        if p.exists():
            EXAM_CATALOG[sha_file(p)] = p

    # 来源类别：台账声明优先，其次显式声明文件，最后默认『复用原答』
    ov_path = HERE / "source_provenance_overrides.json"
    overrides = {}
    if ov_path.exists():
        ov = json.loads(ov_path.read_text(encoding="utf-8"))
        for cat in ("复用原答", "新推理", "百炼"):
            for nm in ov.get(cat, []):
                overrides[nm] = cat
    OV_SHA = sha_file(ov_path) if ov_path.exists() else None

    entries = []
    for raw in sorted(SOURCES.glob("*_raw.jsonl")):
        name = raw.name[:-len("_raw.jsonl")]
        ledger = EVIDENCE / f"{name}_ledger.jsonl"
        metrics = EVIDENCE / f"{name}_metrics.json"
        # 从台账读该次运行实际消费的考卷、N 与来源类别（不靠猜）
        ds_name = ds_sha = n = None
        src_decl = proto_decl = kind_decl = None
        api_prompt_sha = api_prompt_note = None
        if ledger.exists():
            for line in ledger.open(encoding="utf-8"):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("事件") == "冻结件":
                    ds_name, ds_sha, n = d.get("dataset"), d.get("dataset_sha256"), d.get("N")
                    src_decl, proto_decl, kind_decl = d.get("source"), d.get("protocol"), d.get("kind")
                    api_prompt_sha, api_prompt_note = d.get("api_prompt_sha256"), d.get("api_prompt_note")
                    break
        # 考卷：按台账声明的 SHA 在目录里匹配；匹配不到即失败（不靠文件名猜）
        if ds_sha:
            exam = EXAM_CATALOG.get(ds_sha)
            if exam is None:
                raise SystemExit(f"{name}: 台账声明的考卷 SHA {ds_sha[:12]}… 不在本地目录 "
                                 f"{[p.name for p in EXAM_CATALOG.values()]} 中")
        else:
            exam = SRV_L if kind_decl == "L" else SRV_EXAM
        exam_sha = sha_file(exam)
        is_L_exam = exam in (SRV_L, L200)
        if kind_decl in ("main", "L", "CNN"):
            kind = kind_decl
        else:
            kind = "L" if is_L_exam else "main"
        protocol = proto_decl or ("class_json" if kind == "L" else "v5_main")
        source = src_decl or overrides.get(name) or "复用原答"
        prompt_sha = P_L if kind == "L" else P_MAIN
        nominal = n or 200
        if nominal < 200:
            raise SystemExit(f"{name}: nominalN={nominal} 不足 200")
        ev = [{"path": rel(exam), "sha256": exam_sha}]
        if ledger.exists():
            ev.append({"path": rel(ledger), "sha256": sha_file(ledger)})
        if metrics.exists():
            ev.append({"path": rel(metrics), "sha256": sha_file(metrics)})
        runsum = EVIDENCE / f"{name}_run.json"
        if runsum.exists():
            ev.append({"path": rel(runsum), "sha256": sha_file(runsum)})
        entry = {
            "name": name, "kind": kind, "protocol": protocol, "source": source,
            "rawfile": rel(raw), "raw_sha256": sha_file(raw), "nominalN": nominal,
            "inputfile": rel(exam), "input_sha256": exam_sha, "seed": 3407,
            "identity": {"ids_sha256": IDS, "reference_sha256": REF, "images_sha256": IMG,
                         "prompt_sha256": prompt_sha, "evidence": ev},
            "note": f"台账声明 dataset={ds_name} sha256={(ds_sha or '')[:12]}…；本次直接复用该原答，未重跑",
        }
        if api_prompt_sha:
            entry["api_prompt_sha256"] = api_prompt_sha
            entry["api_prompt_note"] = api_prompt_note
            entry["note"] = (f"台账声明 dataset={ds_name} sha256={(ds_sha or '')[:12]}…；"
                             f"API 侧实发题面 sha256={api_prompt_sha[:12]}…（{api_prompt_note}）")
        entries.append(entry)
    # CNN：类别专用，无文本题面 → prompt_sha256 必须显式为 null
    cnn_raw = SOURCES / "CNN_per_sample200.jsonl"
    cnn_src = EVIDENCE / "CNN_source_per_sample.jsonl"
    cnn_metrics = EVIDENCE / "CNN_source_metrics.json"
    for f in (cnn_src, cnn_metrics):
        if not f.exists():
            raise SystemExit(f"缺 CNN 证据副本 {f}（应从本地 E_结果/pull2/eval/CNN/ 复制）")
    entries.append({
        "name": "CNN", "kind": "CNN", "protocol": "cnn_per_sample", "source": "复用原答",
        "rawfile": rel(cnn_raw), "raw_sha256": sha_file(cnn_raw), "nominalN": 589,
        "seed": 3407,
        "identity": {"ids_sha256": IDS, "reference_sha256": REF, "images_sha256": IMG,
                     "prompt_sha256": None,
                     "evidence": [{"path": rel(cnn_src), "sha256": sha_file(cnn_src)},
                                  {"path": rel(cnn_metrics), "sha256": sha_file(cnn_metrics)}]},
        "note": "ResNet18 类别分类器；几何分按合同写不适用，不记 0",
    })

    manifest = {
        "schema": "unified-test200-sources-v1",
        "note": "每条来源的实际消费考卷由台账声明并与本地拉取版逐 SHA 核对；"
                "identity 四 SHA 必须等于合同值，否则打分器拒绝。",
        "sources": entries,
    }
    # pairs 锁由 lock_pairs.py 事后写入；重建清单时保留，避免静默丢失锁定
    out = HERE / "source_manifest.json"
    if out.exists():
        try:
            prev_manifest = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev_manifest = {}
        if prev_manifest.get("pairs_sha256"):
            manifest["pairs_sha256"] = prev_manifest["pairs_sha256"]
            manifest["pairs_note"] = prev_manifest.get("pairs_note", "")
    # 稳定性：source 顺序固定，便于 --resume 字节一致
    entries.sort(key=lambda e: e["name"])
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if out.exists() and out.read_text(encoding="utf-8") != text:
        backup = HERE / "source_manifest.prev.json"
        backup.write_text(out.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        print(f"!! 清单变化，旧版另存 {backup.name}")
    out.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"来源数": len(entries),
                      "kind 分布": {k: sum(1 for e in entries if e["kind"] == k)
                                    for k in ("main", "L", "CNN")},
                      "清单 sha256": sha_file(out)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
