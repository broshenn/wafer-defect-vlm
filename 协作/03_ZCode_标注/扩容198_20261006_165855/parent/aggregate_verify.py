# -*- coding: utf-8 -*-
"""父会话复核器：对指定 worker 目录做结构/一致性核验并独立复跑检查器。
不打印、不保存任何作答文本；batch_log 只含元数据。用法：
  python aggregate_verify.py worker_01 worker_02 worker_03
"""
import hashlib, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(r"D:\pycode\晶圆图研究")
RUN = ROOT / "协作" / "03_ZCode_标注" / "扩容198_20261006_165855"
IN = ROOT / "协作" / "02_ClaudeCode_实操" / "ZCode标注准备_20261006"
CHECKER = ROOT / "协作" / "01_Codex_指挥" / "验收_20261003_GLM180" / "check_answer_v4.py"
V4_EXIT_CODES = {0, 2, 3, 4, 5, 6, 7, 8}

WORKER_OF_ITEM = {}
for i in range(1, 8):    WORKER_OF_ITEM[f"zc_{i:03d}"] = "worker_01"
for i in range(8, 15):   WORKER_OF_ITEM[f"zc_{i:03d}"] = "worker_02"
for i in range(15, 21):  WORKER_OF_ITEM[f"zc_{i:03d}"] = "worker_03"
for b in range(2, 28):   # batch_02..27 -> worker_04..29
    pass
# batch_02..26 各7条、batch_27 3条；索引顺序 zc_021..zc_198
idx = json.loads((IN / "zcode_input_index.json").read_text(encoding="utf-8"))
seq = [it["item_id"] for it in idx["items"] if it["item_id"] not in WORKER_OF_ITEM]  # zc_021..zc_198
pos = 0
for wnum in range(4, 30):
    cnt = 3 if wnum == 29 else 7
    for it in seq[pos:pos + cnt]:
        WORKER_OF_ITEM[it] = f"worker_{wnum:02d}"
    pos += cnt

ITEMS = {it["item_id"]: it for it in idx["items"]}
EXPECTED_FIELDS = {"defect_class", "morphology", "radial_zone", "clock_direction", "extent_r", "caption_zh", "uncertainty"}

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def no_dup_pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate key {k}")
        d[k] = v
    return d

def checker_exit_from_log(p):
    m = re.search(r"exit_code=(\d+)", p.read_text(encoding="utf-8", errors="replace"))
    return int(m.group(1)) if m else None

def rerun_exit(raw_path):
    r = subprocess.run([sys.executable, str(CHECKER), str(raw_path)], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={"PYTHONIOENCODING": "utf-8", "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")})
    return r.returncode

def main(worker_dirs):
    problems = []
    rows = []
    stats = {"format_pass": 0, "unknown": 0, "format_fail": 0, "missing": 0}
    seen_ids = set()
    for wd in worker_dirs:
        wdir = RUN / wd
        raws = sorted(wdir.glob("*_raw.json"))
        for raw in raws:
            item_id = raw.name[:-len("_raw.json")]
            if item_id in seen_ids:
                problems.append(f"{item_id}: raw 重复出现（{wd}）")
                continue
            seen_ids.add(item_id)
            it = ITEMS.get(item_id)
            if it is None:
                problems.append(f"{item_id}: 不在锁定索引")
                continue
            expect_worker = WORKER_OF_ITEM.get(item_id)
            if expect_worker != wd:
                problems.append(f"{item_id}: 在 {wd}，映射应为 {expect_worker}")
            row = {"item_id": item_id, "sample_id": it["sample_id"], "worker": wd,
                   "image_sha256": it["image_sha256"], "image_path": it["image_path"]}
            # raw 结构（不输出内容）
            try:
                obj = json.loads(raw.read_text(encoding="utf-8-sig"), object_pairs_hook=no_dup_pairs)
            except Exception as e:
                problems.append(f"{item_id}: raw 非严格JSON/重复键: {e}")
                row["strict_json"] = False
                row["status"] = "format_fail"
                stats["format_fail"] += 1
                rows.append(row)
                continue
            row["strict_json"] = True
            if not isinstance(obj, dict) or set(obj.keys()) != EXPECTED_FIELDS:
                problems.append(f"{item_id}: raw 字段集合不符")
            if isinstance(obj, dict) and obj.get("extent_r") is not None:
                problems.append(f"{item_id}: extent_r 非 null")
            cls = obj.get("defect_class") if isinstance(obj, dict) else None
            # entry 一致性
            entry_path = wdir / f"{item_id}_entry.json"
            entry = None
            if entry_path.exists():
                try:
                    entry = json.loads(entry_path.read_text(encoding="utf-8"), object_pairs_hook=no_dup_pairs)
                except Exception as e:
                    problems.append(f"{item_id}: entry 非严格JSON: {e}")
            else:
                problems.append(f"{item_id}: 缺 entry")
            if entry:
                if entry.get("item_id") != item_id:
                    problems.append(f"{item_id}: entry item_id 不符")
                if entry.get("raw_sha256") != sha256_file(raw):
                    problems.append(f"{item_id}: entry raw_sha256 与实际不符")
                if entry.get("visual_attempts") != 1:
                    problems.append(f"{item_id}: visual_attempts != 1")
                if entry.get("image_sha256") != it["image_sha256"]:
                    problems.append(f"{item_id}: entry image_sha256 不符")
            row["raw_sha256"] = sha256_file(raw)
            row["visual_attempts"] = entry.get("visual_attempts") if entry else None
            row["start_time"] = entry.get("start_time") if entry else None
            row["end_time"] = entry.get("end_time") if entry else None
            row["model_claim"] = entry.get("model_claim") if entry else None
            row["backend_identity"] = entry.get("backend_identity") if entry else None
            # worker 检查器日志 vs 独立复跑
            log_path = wdir / f"{item_id}_checker.log"
            if not log_path.exists():
                problems.append(f"{item_id}: 缺 checker.log")
                row["checker_exit_code"] = None
            else:
                ce = checker_exit_from_log(log_path)
                row["checker_exit_code"] = ce
                if entry and entry.get("checker_exit_code") != ce:
                    problems.append(f"{item_id}: entry checker_exit_code({entry.get('checker_exit_code')}) 与 log({ce}) 不符")
            re_ec = rerun_exit(raw)
            row["rerun_exit_code"] = re_ec
            if row["checker_exit_code"] is not None and row["checker_exit_code"] != re_ec:
                problems.append(f"{item_id}: 复跑退出码({re_ec}) 与 worker 日志({row['checker_exit_code']}) 不符")
            # 状态分类（不打印类别值）
            eff = re_ec if row["checker_exit_code"] is None else row["checker_exit_code"]
            if eff == 0:
                st = "unknown" if cls == "unknown" else "format_pass"
            else:
                st = "format_fail"
            row["status"] = st
            stats[st] += 1
            rows.append(row)
        # 缺失检查
        expected_items = sorted(k for k, v in WORKER_OF_ITEM.items() if v == wd)
        got = {r["item_id"] for r in rows if r["worker"] == wd}
        miss = set(expected_items) - got
        if miss:
            problems.append(f"{wd}: 缺 raw 的 item: {sorted(miss)}")
            stats["missing"] += len(miss)

    out = RUN / "parent" / ("batch_log_" + "_".join(w.replace("worker_", "") for w in worker_dirs) + ".jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"items={len(rows)} unique={len(seen_ids)} stats={stats}")
    print(f"batch_log={out}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("PROBLEMS=none")

if __name__ == "__main__":
    main(sys.argv[1:])
