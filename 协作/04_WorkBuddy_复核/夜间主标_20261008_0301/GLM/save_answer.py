# -*- coding: utf-8 -*-
"""原答落盘辅助（worker 版）：stdin 传 JSON {"item_id","sample_id","image_sha256","raw_answer"}
用法: python save_answer.py <输出jsonl路径>
自动补 hash/模型/时间并追加写入。仅本任务使用。"""
import json, hashlib, datetime, sys, os

PROMPT_SHA = "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda"

def main():
    out = sys.argv[1]
    d = json.loads(sys.stdin.read())
    raw = d["raw_answer"]
    obj = json.loads(raw)  # 自检可解析
    fields = ["defect_class","morphology","radial_zone","clock_direction","extent_r","caption_zh","uncertainty"]
    missing = [f for f in fields if f not in obj]
    rec = {
        "item_id": d["item_id"], "sample_id": d["sample_id"],
        "image_sha256": d["image_sha256"], "image_hash_checked": d.get("image_hash_checked", True),
        "prompt_sha256": PROMPT_SHA,
        "raw_answer": raw,
        "answer_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "model": "GLM-5.3-Flash (WorkBuddy 主会话worker)",
        "answer_time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "check": "json_ok_7fields" if not missing else "missing:" + ",".join(missing),
    }
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("saved", rec["item_id"], rec["answer_sha256"][:16], rec["check"])

if __name__ == "__main__":
    main()
