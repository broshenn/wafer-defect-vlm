"""补核内容重复、API失败分母和嵌套图片指纹；只读，不改变冻结成绩。"""
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "协作/02_ClaudeCode_实操/百炼多模型业务对照_20261009_1203"
PACK = ROOT / "协作/02_ClaudeCode_实操/自动接续交付_20261009_1247"
OUT = Path(__file__).parent
readlines = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
spec = importlib.util.spec_from_file_location("locked_analysis", RUN / "离线类目分析.py")
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
blind = readlines(ROOT / "协作/01_Codex_指挥/百炼多模型对照_20261009/blind36.jsonl")
ids = [r["sample_id"] for r in blind]
by_id = {r["sample_id"]: r for r in blind}
gold = {r["sample_id"]: r["公开类别"] for r in json.loads((ROOT / "协作/02_ClaudeCode_实操/部署原型_20261009_0120/样本清单.json").read_text(encoding="utf-8"))["样本"]}
data = {r["sample_id"]: r for r in readlines(PACK / "清单_5904_本地相对路径.jsonl")}
train = readlines(ROOT / "协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/数据/L_N3072.jsonl")
train_hashes = defaultdict(list)
for r in train: train_hashes[data[r["sample_id"]]["sha256"]].append(r["sample_id"])
overlap = {sid: train_hashes[by_id[sid]["image_sha256"]] for sid in ids if by_id[sid]["image_sha256"] in train_hashes}
for sid in overlap:
    h = by_id[sid]["image_sha256"]
    assert hashlib.sha256((ROOT / by_id[sid]["image_path"]).read_bytes()).hexdigest() == h
    for tid in overlap[sid]:
        assert hashlib.sha256((ROOT / data[tid]["local_relative_path"]).read_bytes()).hexdigest() == h

models, status = {}, {}
for path in (RUN / "原答_API").glob("*_raw.jsonl"):
    rs = readlines(path); name = path.name.removesuffix("_raw.jsonl")
    assert len(rs) == len({r["sample_id"] for r in rs}) == 36
    models[name] = {r["sample_id"]: r.get("content") or "" for r in rs}
    status[name] = {r["sample_id"]: r["status"] == "ok" for r in rs}
for name in ("Base", "L-N3072-3407"):
    rs = readlines(RUN / "原答_GPU" / (name + "_raw.jsonl"))
    models[name] = {r["sample_id"]: r["raw"] for r in rs}
dep = readlines(ROOT / "协作/02_ClaudeCode_实操/部署原型_20261009_0120/性能与自测/http_replay.jsonl")
models["D-N3072-3407"] = {r["sample_id"]: r["response"]["raw_answer"] for r in dep}
d_hash = {r["sample_id"]: r["response"]["image"]["image_sha256"] for r in dep}
ext = ROOT / "协作/04_WorkBuddy_复核/外部评测102_20261008_201229"
ext_rows = [r for r in readlines(ext / "原答清单.jsonl") if r["sample_id"] in by_id]
models["GLM-WorkBuddy"] = {r["sample_id"]: (ext / "raw" / (r["item_id"] + ".txt")).read_text(encoding="utf-8") for r in ext_rows}
glm_hash = {r["sample_id"]: r["image_sha256"] for r in ext_rows}
common = [sid for sid in ids if all(status[m][sid] for m in status)]
sets = {"frozen36": ids, "exclude_train_same_image": [s for s in ids if s not in overlap],
        "api_all_returned_common": common,
        "api_all_returned_and_no_train_same_image": [s for s in common if s not in overlap]}
tables = {}
for set_name, selected in sets.items():
    table = {}
    for name, raws in models.items():
        rows = {}
        for method, parser in (("strict", a.parse_json), ("full_fence_diagnostic", a.fence_parse)):
            pairs = [(gold[s], a.pred_of(parser(raws[s])[0])) for s in selected]
            rows[method] = {"n": len(selected), "correct": sum(g == p for g, p in pairs),
                            "accuracy": sum(g == p for g, p in pairs) / len(selected),
                            "macro_f1_fixed9": a.prf(pairs)[0]}
        table[name] = rows
    tables[set_name] = table
api_failures = {m: [s for s in ids if not st[s]] for m, st in status.items()}
result = {"scope": "补充诊断；原36主表、原答、标签和请求不改；不同晶圆ID可有相同PNG，不据此推为同一物理晶圆",
          "eval_vs_train_content_overlap": overlap, "train3072_distinct_png_sha256": len(train_hashes),
          "d_nested_image_hash_verified": sum(d_hash[s] == by_id[s]["image_sha256"] for s in ids),
          "glm_subset_image_hash_verified": sum(glm_hash[s] == by_id[s]["image_sha256"] for s in ids),
          "api_failures": api_failures, "subset_ids": sets, "metrics": tables,
          "money_note": "成功响应usage按价目推算约4.9034元；不是平台实际实付。失败后台用量/免费抵扣/实际账单未核。"}
(OUT / "13点补核.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"image_overlap": overlap, "d_hash_ok": result["d_nested_image_hash_verified"],
                  "glm_hash_ok": result["glm_subset_image_hash_verified"],
                  "subset_n": {k: len(v) for k, v in sets.items()},
                  "selected_results": {k: {m: v["full_fence_diagnostic"] for m, v in t.items()} for k, t in tables.items()}}, ensure_ascii=False, indent=2))
