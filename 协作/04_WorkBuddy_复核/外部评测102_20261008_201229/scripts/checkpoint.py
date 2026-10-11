#!/usr/bin/env python3
"""checkpoint.py <标签> — 汇总当前已完成原答的检查结果，写检查点JSON，不读取真值。"""
import json, sys
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent
BLIND = Path("D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/客户端盲包/WorkBuddy_外部评测102")

def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%H%M%S")
    idx = [json.loads(l) for l in (BLIND / "WB_eval102_索引.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    done_ids = sorted(p.stem for p in (OUT / "done").glob("*.done"))
    raws = sorted(p.stem for p in (OUT / "raw").glob("*.txt"))
    metas = sorted(p.stem[:-5] for p in (OUT / "meta").glob("*_meta.json"))
    reqs = sorted(p.stem for p in (OUT / "request_records").glob("*.json"))
    counts = {"strict_json": 0, "schema_ok": 0, "contract_warnings": 0,
              "schema_failures": 0, "missing_check": 0}
    for i in done_ids:
        cp = OUT / "checks" / f"{i}_check.json"
        if not cp.exists():
            counts["missing_check"] += 1
            continue
        c = json.loads(cp.read_text(encoding="utf-8"))
        counts["strict_json"] += bool(c.get("strict_json"))
        counts["schema_ok"] += bool(c.get("schema_ok"))
        counts["contract_warnings"] += len(c.get("contract_warnings") or [])
        if not c.get("schema_ok"):
            counts["schema_failures"] += 1
    summary = {
        "checkpoint": label,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "planned_total": len(idx),
        "done": len(done_ids),
        "raw_files": len(raws),
        "meta_final_files": len(metas),
        "request_record_files": len(reqs),
        "unexecuted": len(idx) - len(done_ids),
        "counts": counts,
        "done_ids": done_ids,
    }
    dest = OUT / "checkpoints" / f"checkpoint_{label}.json"
    dest.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "done_ids"}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
