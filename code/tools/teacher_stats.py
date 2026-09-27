"""Summarise teacher-annotation token usage and provenance.

Rows are counted whether or not they succeeded, because a failed call still
consumes tokens and a cost report that hides them is misleading. Provenance is
reported per model so the fallback from the exhausted free-tier model stays
visible in the record.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def summarise(path: Path) -> dict:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    usage = [row.get("usage") or {} for row in rows]
    models = collections.Counter(str(row.get("model")) for row in rows)
    versions = collections.Counter(str(row.get("prompt_version")) for row in rows)
    ok = sum(1 for row in rows if row.get("status") == "ok")
    return {
        "file": path.name,
        "rows": len(rows),
        "ok": ok,
        "errors": len(rows) - ok,
        "prompt_tokens": sum(int(u.get("prompt_tokens", 0) or 0) for u in usage),
        "completion_tokens": sum(int(u.get("completion_tokens", 0) or 0) for u in usage),
        "total_tokens": sum(int(u.get("total_tokens", 0) or 0) for u in usage),
        "models": dict(models),
        "prompt_versions": dict(versions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    # Derived files (e.g. plan.jsonl) are not API output and carry no status
    # field; counting them as errors would misreport the teacher failure rate.
    candidates = []
    for path in sorted(Path(args.dir).glob("*.jsonl")):
        first = next((line for line in path.open(encoding="utf-8") if line.strip()), "")
        if first and "status" in json.loads(first):
            candidates.append(path)
    reports = [summarise(p) for p in candidates]
    total = {
        "files": len(reports),
        "rows": sum(r["rows"] for r in reports),
        "total_tokens": sum(r["total_tokens"] for r in reports),
        "prompt_tokens": sum(r["prompt_tokens"] for r in reports),
        "completion_tokens": sum(r["completion_tokens"] for r in reports),
    }
    report = {"per_file": reports, "total": total}
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
