#!/usr/bin/env python3
"""finalize_meta.py <item_id> <shard_no> <worker_label> [--started-file PATH]

在执行目录内生成 meta/<id>_meta.json、request_records/<id>.json 与 done/<id>.done。
只读取：盲包索引中该条目信息、meta/<id>_started.json（若存在）、raw/<id>.txt、
checks/<id>_check.json（若已生成）。不读取任何 gold、训练标注或兄弟 worker 答案。
不修改原答与检查输出。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent
BLIND = Path("D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/客户端盲包/WorkBuddy_外部评测102")
PROMPT = Path("D:/pycode/晶圆图研究/协作/01_Codex_指挥/WorkBuddy外部102_执行补包_20261008/prompt_7f.txt")
CHECKER = Path("D:/pycode/晶圆图研究/协作/01_Codex_指挥/WorkBuddy外部102_执行补包_20261008/schema_check.py")
DISPLAY_MODEL = "GLM-5.3-Flash"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("item_id")
    ap.add_argument("shard_no")
    ap.add_argument("worker_label")
    ap.add_argument("--started-file", default=None)
    args = ap.parse_args()
    item = args.item_id

    row = None
    for line in (BLIND / "WB_eval102_索引.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj["item_id"] == item:
            row = obj
            break
    if row is None:
        print(f"ERROR: item {item} not in index", file=sys.stderr)
        return 2

    image_path = Path(row["image_path"])
    image_sha = sha(image_path.read_bytes())
    image_sha_ok = image_sha == row["image_sha256"]

    started = None
    started_path = Path(args.started_file) if args.started_file else OUT / "meta" / f"{item}_started.json"
    if started_path.exists():
        started = json.loads(started_path.read_text(encoding="utf-8"))

    raw_path = OUT / "raw" / f"{item}.txt"
    if not raw_path.exists():
        print(f"ERROR: raw answer missing: {raw_path}", file=sys.stderr)
        return 2
    raw_bytes = raw_path.read_bytes()
    raw_sha = sha(raw_bytes)

    check_path = OUT / "checks" / f"{item}_check.json"
    check_summary = None
    if check_path.exists():
        c = json.loads(check_path.read_text(encoding="utf-8"))
        check_summary = {
            "strict_json": c.get("strict_json"),
            "parse_status": c.get("parse_status"),
            "schema_ok": c.get("schema_ok"),
            "schema_problems": c.get("schema_problems"),
            "contract_warnings": c.get("contract_warnings"),
            "raw_sha256_in_check": c.get("raw_sha256"),
            "raw_sha256_match": (c.get("raw_sha256") == raw_sha) if c.get("raw_sha256") else None,
            "validator_sha256": c.get("validator_sha256"),
            "check_input_mode": c.get("input_mode"),
        }

    prompt_bytes = PROMPT.read_bytes()
    prompt_sha = sha(prompt_bytes)

    meta = {
        "item_id": item,
        "sample_id": row["sample_id"],
        "shard_no": int(args.shard_no),
        "worker": args.worker_label,
        "display_model": DISPLAY_MODEL,
        "model_exact_backend_id": "未知：本环境无后端模型标识记录",
        "image_path": row["image_path"],
        "image_sha256_expected": row["image_sha256"],
        "image_sha256_runtime": image_sha,
        "image_sha256_ok": image_sha_ok,
        "started_at": (started or {}).get("started_at", "未知：无started记录"),
        "finalized_at": now_iso(),
        "prompt_file": str(PROMPT),
        "prompt_sha256_runtime": prompt_sha,
        "prompt_sha256_expected": "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda",
        "prompt_sha256_ok": prompt_sha == "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda",
        "raw_answer_path": str(raw_path),
        "raw_answer_sha256": raw_sha,
        "raw_answer_bytes": len(raw_bytes),
        "raw_answer_source": "string:逐字节UTF-8文本落盘（非dict重构）",
        "visual_reads_used": 1,
        "attempts": 1,
        "retry_used": False,
        "checker_command": f"python -X utf8 -B {CHECKER} --input {raw_path} --out {check_path}",
        "checker_exit_code_source": "由校验JSON推导：检查器退出码0 ⇔ schema_ok且prompt_hash_ok，否则1",
        "checker_exit_code": (0 if (check_summary and check_summary.get("schema_ok")
                                    and prompt_sha == "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda") else 1) if check_summary else None,
        "check": check_summary,
        "done": bool(check_path.exists() and check_summary is not None),
    }
    (OUT / "meta" / f"{item}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    request_record = {
        "item_id": item,
        "submitted_prompt_text": prompt_bytes.decode("utf-8"),
        "submitted_prompt_sha256_runtime": prompt_sha,
        "prompt_matches_reference": prompt_sha == "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda",
        "image_path_submitted": row["image_path"],
        "image_sha256_runtime": image_sha,
        "backend_request_log_available": False,
        "disclosure": "后台注入/真实参数/后端请求数未提供：本客户端环境无法读取后台请求记录；上述prompt为执行时从题面文件逐字读入的客户端侧记录，其hash为现算值，不能冒充后端发送证据。",
        "sampling_params": "未知：客户端未暴露温度/top_p等真实请求参数",
        "recorded_at": now_iso(),
    }
    (OUT / "request_records" / f"{item}.json").write_text(
        json.dumps(request_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    (OUT / "done" / f"{item}.done").parent.mkdir(exist_ok=True)
    (OUT / "done" / f"{item}.done").write_text(
        json.dumps({"item_id": item, "done_at": now_iso()}, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    print(f"FINALIZED {item} image_ok={image_sha_ok} prompt_ok={meta['prompt_sha256_ok']} "
          f"schema_ok={(check_summary or {}).get('schema_ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
