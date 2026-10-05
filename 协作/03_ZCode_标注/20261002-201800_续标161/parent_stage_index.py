"""父会话阶段索引生成器：对指定组汇总 worker_log.jsonl 并核验产物，写 stage_index.json。
只读汇总；不修改任何原答。用法：python parent_stage_index.py <group_id>"""
import json, os, sys, hashlib, glob, time

ROOT = r"D:\pycode\晶圆图研究\协作\03_ZCode_标注\20261002-201800_续标161"
IDX = r"D:\pycode\晶圆图研究\协作\01_Codex_指挥\GLM续标_20261002_无答案输入"
PARENT_KNOWN_ANSWERS = {"wafer_00047178_011"}  # 父会话在九张试标中作答过的图（20261002-012138），仅作披露标记

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

gid = sys.argv[1]
idx = json.load(open(os.path.join(IDX, "inputs.json"), encoding="utf-8"))
g = next(x for x in idx["groups"] if x["group_id"] == gid)
expected = [json.loads(l) for l in open(g["input_path"], encoding="utf-8") if l.strip()]
exp_by_id = {r["sample_id"]: r for r in expected}

gdir = os.path.join(ROOT, gid)
workers = sorted(w for w in os.listdir(gdir) if w.startswith("worker_"))
items, anomalies = [], []
for w in workers:
    wd = os.path.join(gdir, w)
    logp = os.path.join(wd, "worker_log.jsonl")
    rows = [json.loads(l) for l in open(logp, encoding="utf-8") if l.strip()] if os.path.exists(logp) else []
    for r in rows:
        sid = r["sample_id"]
        rec = {"sample_id": sid, "worker": w, "status": r.get("status"),
               "checker1_exit": r.get("checker1_exit"), "normalizer_exit": r.get("normalizer_exit"),
               "checker2_exit": r.get("checker2_exit"), "pending_review": r.get("pending_review"),
               "alias_changes": r.get("alias_changes"), "duration_ms": r.get("duration_ms")}
        rawp = os.path.join(wd, sid + "_raw.json")
        if os.path.exists(rawp) and sid in exp_by_id:
            rec["raw_sha256"] = sha(rawp)
            rec["raw_sha_match"] = (rec["raw_sha256"] == r.get("raw_sha256"))
            rec["image_sha_match"] = (exp_by_id[sid]["image_sha256"] == r.get("image_sha256"))
            obj = json.load(open(rawp, encoding="utf-8"))
            rec["defect_class_raw"] = obj.get("defect_class")
            if obj.get("defect_class") == "unknown":
                rec["is_unknown"] = True
        else:
            anomalies.append({"worker": w, "sample_id": sid, "issue": "raw_missing_or_id_not_in_group"})
        if sid in PARENT_KNOWN_ANSWERS:
            rec["parent_disclosure"] = "父会话在九张试标(20261002-012138)对本图作答过；本标注由全新worker完成，父会话知识未进入worker输入"
        items.append(rec)

done_ids = {i["sample_id"] for i in items}
not_run = [r["sample_id"] for r in expected if r["sample_id"] not in done_ids]
counts = {
    "planned": len(expected),
    "attempted": len(items),
    "raw_pass": sum(1 for i in items if i["status"] == "raw_pass"),
    "raw_fail_canonical_pass": sum(1 for i in items if i["status"] == "raw_fail_canonical_pass"),
    "raw_fail": sum(1 for i in items if i["status"] == "raw_fail"),
    "isolated_bad_json": sum(1 for i in items if i["status"] == "isolated_bad_json"),
    "not_run": len(not_run),
    "unknown": sum(1 for i in items if i.get("is_unknown")),
    "pending_review": sum(1 for i in items if i.get("pending_review")),
    "raw_sha_mismatch": sum(1 for i in items if i.get("raw_sha_match") is False),
    "image_sha_mismatch": sum(1 for i in items if i.get("image_sha_match") is False),
}
stopped = {w: [os.path.basename(x) for x in glob.glob(os.path.join(gdir, w, "STOPPED*"))] for w in workers}
stage = {
    "group_id": gid, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "group_input_path": g["input_path"], "group_input_sha256_expected": g["input_sha256"],
    "counts": counts, "not_run_ids": not_run, "stopped_markers": stopped,
    "anomalies": anomalies, "items": items,
    "note": "父会话只读汇总；raw_pass/raw_fail等以worker_log.jsonl自报status为准，父会话另核验raw文件存在与哈希一致",
}
out = os.path.join(gdir, "stage_index.json")
json.dump(stage, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(gid, json.dumps(counts, ensure_ascii=False), "anomalies:", len(anomalies),
      "stopped:", {k: v for k, v in stopped.items() if v}, "not_run:", not_run[:5])
