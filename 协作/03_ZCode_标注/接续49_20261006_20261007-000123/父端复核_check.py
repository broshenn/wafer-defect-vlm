# -*- coding: utf-8 -*-
"""父端复核：复跑49原答v4检查器、核验五件套与meta哈希、拒重复键写batch_log、生成指纹日志。

只读本目录与输入盲清单/检查器；不读仓库其他标注目录。
"""
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
IN = ROOT / "协作/01_Codex_指挥/ZCode接续49_20261006"
OUT = Path(__file__).resolve().parent

# 以与 v4 相同方式加载锁定检查器（内含 v3 SHA256 断言）
V4 = ROOT / "协作/01_Codex_指挥/验收_20261003_GLM180/check_answer_v4.py"
spec = importlib.util.spec_from_file_location("checker_v4_parent", V4)
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)

blind = [json.loads(l) for l in (IN / "blind_remaining49.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
assert len(blind) == 49, len(blind)

CODE_NAME = {0: "通过", 2: "非JSON", 3: "顶层非对象", 4: "字段集合不符",
             5: "类型错误", 6: "枚举未知", 7: "extent_r非null", 8: "描述含数值"}

records = []
seen_ids = set()
problems = []
for b in blind:
    item = b["item_id"]
    if item in seen_ids:
        problems.append(f"重复键: {item}")
        continue
    seen_ids.add(item)
    worker = f"worker_{(int(item.split('_')[1]) - 150) // 7 + 1:02d}"
    wdir = OUT / worker
    rec = {"item_id": item, "sample_id": b["sample_id"], "worker": worker}
    paths = {
        "started": wdir / "started" / f"{item}_started.json",
        "raw": wdir / f"{item}_raw.json",
        "check": wdir / f"{item}_check.txt",
        "meta": wdir / f"{item}_meta.json",
        "done": wdir / "done" / f"{item}.done",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        rec["missing"] = missing
        problems.append(f"{item}: 缺 {missing}")
        records.append(rec)
        continue
    raw_bytes = paths["raw"].read_bytes()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    # 父端独立复跑检查器
    rep = v4.check_text(raw_bytes.decode("utf-8-sig"))
    indep_exit = rep["exit_code"]
    # worker 留档的 process_exit_code
    check_txt = paths["check"].read_text(encoding="utf-8", errors="replace")
    m = re.search(r"process_exit_code=(\d+)", check_txt)
    saved_exit = int(m.group(1)) if m else None
    # meta 核验
    meta = json.loads(paths["meta"].read_text(encoding="utf-8-sig"))
    meta_raw_sha_ok = meta.get("raw_sha256") == raw_sha
    meta_img_sha_ok = meta.get("image_sha256") == b["image_sha256"]
    rec.update({
        "independent_checker_exit": indep_exit,
        "independent_checker_verdict": CODE_NAME.get(indep_exit, f"码{indep_exit}"),
        "saved_process_exit_code": saved_exit,
        "exit_match": saved_exit == indep_exit,
        "meta_raw_sha256_match": meta_raw_sha_ok,
        "meta_image_sha256_match": meta_img_sha_ok,
        "glm_visual_attempt_count": meta.get("glm_visual_attempt_count"),
        "defect_class": rep and json.loads(raw_bytes.decode("utf-8-sig")).get("defect_class"),
        "extent_r_is_null": json.loads(raw_bytes.decode("utf-8-sig")).get("extent_r") is None,
        "raw_sha256": raw_sha,
    })
    if not rec["exit_match"]:
        problems.append(f"{item}: worker留档退出码{saved_exit} != 父端复跑{indep_exit}")
    if not meta_raw_sha_ok:
        problems.append(f"{item}: meta.raw_sha256 与实际 raw 不符")
    if not meta_img_sha_ok:
        problems.append(f"{item}: meta.image_sha256 与盲清单不符")
    if not rec["extent_r_is_null"]:
        problems.append(f"{item}: extent_r 非 null")
    records.append(rec)

# 未闭合 started 检查（全目录扫）
for wdir in sorted(OUT.glob("worker_*")):
    for s in sorted((wdir / "started").glob("*_started.json")):
        item = s.name.replace("_started.json", "")
        if not (wdir / "done" / f"{item}.done").exists():
            problems.append(f"未闭合 started: {wdir.name}/{item}")
    # 多余 raw 检查
    for raw in sorted(wdir.glob("*_raw.json")):
        item = raw.name.replace("_raw.json", "")
        if item not in seen_ids:
            problems.append(f"清单外 raw: {wdir.name}/{item}")

n_attempt = len([r for r in records if "missing" not in r])
n_pass = len([r for r in records if r.get("independent_checker_exit") == 0])
n_fail = len([r for r in records if r.get("independent_checker_exit") not in (0, None)])
n_unknown = len([r for r in records if r.get("defect_class") == "unknown"])
n_missing = len([r for r in records if "missing" in r])

batch_log = {
    "run": "接续49_20261006_20261007-000123",
    "author": "ZCode父会话",
    "date": "2026-10-07",
    "scope": "zc_150..zc_198 共49图；GLM_original 107 + GPT_supplement 42 原答保留未动",
    "model_declaration": "GLM-5.3-Flash via ZCode客户端子agent；实际后端未提供",
    "extra_payment": 0,
    "attempt_note": "尝试=标注尝试次数（每图1次视觉读取+1次作答），不等同于API调用次数；费用/token/真实后端请求数未提供",
    "counts": {
        "planned": 49, "attempted": n_attempt, "format_pass": n_pass,
        "format_fail": n_fail, "unknown_class": n_unknown,
        "isolated_preserved": [r["item_id"] for r in records if r.get("independent_checker_exit") not in (0, None)],
        "not_executed": n_missing,
    },
    "prior_note_zc150_156": "GPT中断分片：先前是否已看图未确认；旧GPT尝试不计入GLM次数，也不记0",
    "prior_note_others": "Codex移交时无已保存raw；更早未保存的尝试状态未逐一确认，未确认者不计入GLM次数也不记0",
    "problems": problems,
    "items": records,
}

# 指纹日志：输出树全部文件
fingerprints = {}
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.name not in ("batch_log.json", "指纹日志.json", "父端复核_check.py"):
        fingerprints[str(p.relative_to(OUT)).replace("\\", "/")] = hashlib.sha256(p.read_bytes()).hexdigest()

batch_log["n_files_fingerprinted"] = len(fingerprints) + 2  # 含本日志与指纹日志自身

(OUT / "batch_log.json").write_text(json.dumps(batch_log, ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "指纹日志.json").write_text(json.dumps(fingerprints, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")

print(json.dumps(batch_log["counts"], ensure_ascii=False))
print("problems:", problems if problems else "无")
print("fingerprinted files:", len(fingerprints))
same = {}
for r in records:
    same.setdefault(r.get("raw_sha256"), []).append(r["item_id"])
dup_raw = {k: v for k, v in same.items() if len(v) > 1}
print("identical raw content groups:", dup_raw if dup_raw else "无")
sys.exit(0 if not problems else 1)
