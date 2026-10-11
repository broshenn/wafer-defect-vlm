"""冻结外部原答机械验收和同图类别复算；不调用模型，不改输入。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
WB = REPO / "协作/04_WorkBuddy_复核/外部评测102_20261008_201229"
BLIND = REPO / "协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/客户端盲包/WorkBuddy_外部评测102"
SUPP = REPO / "协作/01_Codex_指挥/WorkBuddy外部102_执行补包_20261008"
NIGHT = REPO / "协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/结果"
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_full", "Random", "Scratch", "none"]
SEED = 3407
HASHES = {"index": "41d6e79c7ae6026fae34a609f39af4ced07aa080b163c18e813d1205c9dcd0c1",
          "prompt": "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda",
          "checker": "7d46c37f72bb8fd09488dd73754ad75e6a1b711d91a6f08f82e5b34026c0c0ff"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def nodup(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_key")
        value[key] = item
    return value


def loads(text):
    return json.loads(text, object_pairs_hook=nodup)


def read(path):
    return loads(path.read_text(encoding="utf-8"))


def readlines(path):
    return [loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_checker():
    path = SUPP / "schema_check.py"
    assert sha(path.read_bytes()) == HASHES["checker"]
    spec = importlib.util.spec_from_file_location("external_schema", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(gold, pred):
    per_class = {}
    for category in CLASSES:
        tp = sum(g == category and p == category for g, p in zip(gold, pred))
        fp = sum(g != category and p == category for g, p in zip(gold, pred))
        fn = sum(g == category and p != category for g, p in zip(gold, pred))
        per_class[category] = {"support": gold.count(category), "tp": tp, "fp": fp, "fn": fn,
                               "precision": tp / (tp + fp) if tp + fp else 0,
                               "recall": tp / (tp + fn) if tp + fn else 0,
                               "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}
    correct = sum(g == p for g, p in zip(gold, pred))
    confusion = {g: dict(Counter(p if p is not None else "INVALID" for gg, p in zip(gold, pred) if gg == g)) for g in CLASSES}
    return {"n": len(gold), "correct": correct, "accuracy": correct / len(gold),
            "macro_f1_fixed9": sum(x["f1"] for x in per_class.values()) / 9,
            "per_class": per_class, "confusion": confusion}


def paired(gold, a, b, bootstraps=5000):
    wins = sum(x == g and y != g for g, x, y in zip(gold, a, b))
    losses = sum(x != g and y == g for g, x, y in zip(gold, a, b))
    n = len(gold)
    ma, mb = metrics(gold, a), metrics(gold, b)
    rng = random.Random(SEED)
    differences_acc, differences_f1 = [], []
    for _ in range(bootstraps):
        inds = [rng.randrange(n) for _ in range(n)]
        gs, pa, pb = [[v[i] for i in inds] for v in (gold, a, b)]
        am, bm = metrics(gs, pa), metrics(gs, pb)
        differences_acc.append(am["accuracy"] - bm["accuracy"])
        differences_f1.append(am["macro_f1_fixed9"] - bm["macro_f1_fixed9"])
    interval = lambda xs: [sorted(xs)[int((bootstraps - 1) * q)] for q in (0.025, 0.975)]
    discordant = wins + losses
    exact_p = min(1, 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2 ** discordant) if discordant else 1
    return {"a_minus_b_accuracy": ma["accuracy"] - mb["accuracy"],
            "accuracy_paired_bootstrap95": interval(differences_acc),
            "a_minus_b_macro_f1": ma["macro_f1_fixed9"] - mb["macro_f1_fixed9"],
            "macro_f1_paired_bootstrap95": interval(differences_f1),
            "a_win_b_loss": wins, "a_loss_b_win": losses,
            "mcnemar_exact_two_sided_p": exact_p, "bootstrap_n": bootstraps,
            "seed": SEED, "unit": "同一晶圆配对；本102图恰为102个不同lot；开发集条件结果"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    checker = load_checker()
    index_path = BLIND / "WB_eval102_索引.jsonl"
    assert sha(index_path.read_bytes()) == HASHES["index"]
    assert sha((SUPP / "prompt_7f.txt").read_bytes()) == HASHES["prompt"]
    index = readlines(index_path)
    assert len(index) == 102 and len({r["item_id"] for r in index}) == len({r["sample_id"] for r in index}) == 102
    union = [r for p in sorted(BLIND.glob("WB_eval102_分片*.jsonl")) for r in readlines(p)]
    assert union == index
    source_paths = [index_path, SUPP / "prompt_7f.txt", SUPP / "schema_check.py", REPO / "data/manifest.jsonl"]
    manifest = {r["sample_id"]: r for r in readlines(REPO / "data/manifest.jsonl")}
    reference = readlines(NIGHT / "L-N3072-3407_7f_raw.jsonl")
    assert len(reference) == 102 and len({r["sample_id"] for r in reference}) == 102
    ref = {r["sample_id"]: r for r in reference}
    assert set(ref) == {r["sample_id"] for r in index}
    aliases = {"Edge-Loc": "Edge_Loc", "Edge-Ring": "Edge_Ring", "Near-full": "Near_full"}
    for sid, r in ref.items():
        assert manifest[sid]["label_source"] == "ground_truth"
        assert aliases.get(manifest[sid]["failure_type"], manifest[sid]["failure_type"]) == r["label"]
        assert manifest[sid]["split"] in ("val", "validation")
    assert len({manifest[sid]["lot_name"] for sid in ref}) == 102
    raw_paths = list((WB / "raw").glob("*.txt"))
    assert {p.stem for p in raw_paths} == {r["item_id"] for r in index}
    integrity, answers = [], {}
    keys = ["strict_json", "parse_status", "schema_ok", "schema_problems", "contract_warnings", "raw_sha256"]
    for row in index:
        item, sid = row["item_id"], row["sample_id"]
        raw_path = WB / "raw" / f"{item}.txt"
        raw_bytes = raw_path.read_bytes()
        raw = raw_bytes.decode("utf-8")
        result = checker.check_answer(raw)
        assert sha(Path(row["image_path"]).read_bytes()) == row["image_sha256"]
        meta_path = WB / "meta" / f"{item}_meta.json"
        req_path = WB / "request_records" / f"{item}.json"
        meta, req = read(meta_path), read(req_path)
        assert meta["item_id"] == item and meta["sample_id"] == sid
        assert meta["raw_answer_sha256"] == sha(raw_bytes) == result["raw_sha256"]
        assert sha(req["submitted_prompt_text"].encode("utf-8")) == HASHES["prompt"]
        assert req["image_sha256_runtime"] == meta["image_sha256_runtime"] == row["image_sha256"]
        assert (WB / "done" / f"{item}.done").exists()
        source_paths += [raw_path, meta_path, req_path]
        for dirname, suffix in (("checks", "check"), ("checks_rerun", "rerun")):
            path = WB / dirname / f"{item}_{suffix}.json"
            stored = read(path)
            assert all(stored.get(k) == result[k] for k in keys), (item, dirname)
            assert stored["validator_sha256"] == HASHES["checker"]
            source_paths.append(path)
        obj, status = checker.parse_strict(raw)
        answers[sid] = {"raw": raw, "obj": obj, "pred": obj.get("defect_class") if status == "ok" and obj.get("defect_class") in CLASSES else None,
                        "result": result, "item_id": item}
        integrity.append({"item_id": item, "sample_id": sid, "image_sha256": row["image_sha256"], "raw_sha256": sha(raw_bytes),
                          "strict_json": result["strict_json"], "schema_ok": result["schema_ok"], "warnings": result["contract_warnings"]})
    predictions = {"GLM-WorkBuddy": {sid: r["pred"] for sid, r in answers.items()}}
    schema_full = {"GLM-WorkBuddy": {"strict_json": sum(r["result"]["strict_json"] for r in answers.values()),
                                   "schema_ok": sum(r["result"]["schema_ok"] for r in answers.values()),
                                   "clock_string_null": 0, "warnings": sum(len(r["result"]["contract_warnings"]) for r in answers.values())}}
    fence = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n?\s*```\s*$", re.S)
    for tag in ("Base", "L-N3072-3407", "D-N3072-3407", "D-N3072-3408", "D-N3072-3409"):
        path = NIGHT / f"{tag}_7f_raw.jsonl"
        rows = readlines(path)
        source_paths.append(path)
        assert len(rows) == 102 and {r["sample_id"] for r in rows} == set(ref)
        predictions[tag] = {}
        counts = Counter()
        for r in rows:
            assert (r["label"], r["part"]) == (ref[r["sample_id"]]["label"], ref[r["sample_id"]]["part"])
            result = checker.check_answer(r["raw"])
            obj, status = checker.parse_strict(r["raw"])
            predictions[tag][r["sample_id"]] = obj.get("defect_class") if status == "ok" and obj.get("defect_class") in CLASSES else None
            counts["strict_json"] += result["strict_json"]
            counts["schema_ok"] += result["schema_ok"]
            counts["clock_string_null"] += obj is not None and obj.get("clock_direction") == "null"
            if tag == "Base":
                stripped = checker.EMPTY_THINK.sub("", r["raw"].strip())
                match = fence.fullmatch(stripped)
                diag, ds = checker.parse_strict(match.group(1) if match else stripped)
                predictions.setdefault("Base-fence-diagnostic", {})[r["sample_id"]] = diag.get("defect_class") if ds == "ok" and diag.get("defect_class") in CLASSES else None
        schema_full[tag] = dict(counts)
    by_subset, comparisons = {}, {}
    for part in ("dev18", "val84", "full102"):
        ids = [r["sample_id"] for r in index if part == "full102" or ref[r["sample_id"]]["part"] == ("val90" if part == "val84" else part)]
        gold = [ref[sid]["label"] for sid in ids]
        by_subset[part] = {tag: metrics(gold, [ps[sid] for sid in ids]) for tag, ps in predictions.items()}
        if part == "val84":
            assert by_subset[part]["L-N3072-3407"]["correct"] == 77
            assert by_subset[part]["D-N3072-3407"]["correct"] == 70
            assert by_subset[part]["Base"]["correct"] == 4
            assert by_subset[part]["Base-fence-diagnostic"]["correct"] == 15
            comparisons[part] = {tag + " minus GLM": paired(gold, [predictions[tag][sid] for sid in ids], [predictions["GLM-WorkBuddy"][sid] for sid in ids])
                                 for tag in ("L-N3072-3407", "D-N3072-3407")}
    report = {"date": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
              "topic": "相同七字段题面中的类别成绩；不是描述正确率", "integrity": {"rows": len(integrity),
              "strict_json": schema_full["GLM-WorkBuddy"]["strict_json"], "schema_ok": schema_full["GLM-WorkBuddy"]["schema_ok"],
              "warnings": schema_full["GLM-WorkBuddy"]["warnings"], "chain_verified": True},
              "metrics": by_subset, "format_full102": schema_full, "paired": comparisons,
              "backend_disclosure": "GLM为WorkBuddy显示型号；后台注入/真实参数/精确后端标识未知；客户端记录不证明实际后端请求",
              "gold_source": "公开manifest.failure_type；只用于类别评分。形态/位置/方向/Caption仍无独立gold。",
              "raw_records": integrity,
              "sources_sha256": {str(p.relative_to(REPO)).replace(chr(92), "/"): sha(p.read_bytes()) for p in source_paths}}
    if args.write:
        target = OUT / "独立复算.json"
        if target.exists():
            raise SystemExit("结果已存在，拒绝覆盖；只读重算可不加--write")
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        detail = []
        for row in index:
            sid = row["sample_id"]
            detail.append({"item_id": row["item_id"], "sample_id": sid, "part": ref[sid]["part"], "gold_class": ref[sid]["label"],
                           "predictions": {t: p[sid] for t, p in predictions.items()}})
        (OUT / "逐图类别对照.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in detail), encoding="utf-8")
    print(json.dumps({"integrity": report["integrity"], "metrics": {p: {t: {k: v[k] for k in ("correct", "n", "accuracy", "macro_f1_fixed9")} for t, v in ts.items()} for p, ts in by_subset.items()},
                      "paired": comparisons, "format_full102": schema_full}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
