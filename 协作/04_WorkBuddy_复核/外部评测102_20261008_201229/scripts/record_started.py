#!/usr/bin/env python3
"""record_started.py <item_id> — 在视觉读取前记录 started 与图片hash（运行时现算）。"""
import hashlib, json, sys
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent
BLIND = Path("D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/客户端盲包/WorkBuddy_外部评测102")

def sha(b): return hashlib.sha256(b).hexdigest()

def main() -> int:
    item = sys.argv[1]
    note = sys.argv[2] if len(sys.argv) > 2 else None
    row = None
    for line in (BLIND / "WB_eval102_索引.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if obj["item_id"] == item:
                row = obj
                break
    if row is None:
        print(f"ERROR: {item} not in index", file=sys.stderr)
        return 2
    p = Path(row["image_path"])
    rec = {
        "item_id": item,
        "sample_id": row["sample_id"],
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "image_sha256_runtime": sha(p.read_bytes()),
        "image_sha256_ok": sha(p.read_bytes()) == row["image_sha256"],
    }
    if note:
        rec["note"] = note
    dest = OUT / "meta" / f"{item}_started.json"
    if dest.exists():
        print(f"SKIP started record exists: {dest}")
        return 0
    dest.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"STARTED {item} image_ok={rec['image_sha256_ok']} at={rec['started_at']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
