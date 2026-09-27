"""Write a manifest JSONL restricted to the given splits."""
import json, sys
from pathlib import Path

src, dst, splits = Path(sys.argv[1]), Path(sys.argv[2]), set(sys.argv[3].split(","))
rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
keep = [r for r in rows if r["split"] in splits]
with dst.open("w", encoding="utf-8", newline="\n") as fh:
    for r in keep:
        fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
print(json.dumps({"source_rows": len(rows), "kept": len(keep), "splits": sorted(splits)}))
