"""Re-attach sample_id/task to ms-swift inference results.

`swift infer` returns only response/images/messages, so the benchmark keys are
lost. Order is preserved and every row carries its source image path, which is
checked here so a silently misaligned file cannot reach the scorer.

ms-swift writes ``images`` entries as ``{'bytes': None, 'path': ...}`` dicts
while the request file carries bare path strings, so both sides are normalised
to path strings before comparison; comparing them raw flags every row.
"""
import argparse, json, sys
from pathlib import Path


def load(path):
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def image_paths(value):
    """Normalise ms-swift image entries to a list of path strings."""
    paths = []
    for item in value or []:
        if isinstance(item, dict):
            path = item.get("path")
            if path:
                paths.append(str(path))
        elif item is not None:
            paths.append(str(item))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    requests, results = load(args.requests), load(args.results)
    if len(results) != len(requests):
        print("WARN length mismatch requests=%d results=%d" % (len(requests), len(results)),
              file=sys.stderr)
    n = min(len(requests), len(results))
    mismatches = []
    rows = []
    for i in range(n):
        req, res = requests[i], results[i]
        req_img = image_paths(req.get("images"))
        res_img = image_paths(res.get("images"))
        if req_img and res_img and req_img != res_img:
            mismatches.append((i, req.get("sample_id"), req_img[:1], res_img[:1]))
        rows.append({
            "sample_id": req.get("sample_id"),
            "task": req.get("task"),
            "image_path": req_img[0] if req_img else None,
            "response": res.get("response") or "",
        })
    if mismatches:
        print("ERROR: %d image-path mismatches, first 3:" % len(mismatches))
        for m in mismatches[:3]:
            print("  ", m)
        return 1
    with Path(args.output).open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"aligned": len(rows), "requests": len(requests), "results": len(results),
                      "empty_responses": sum(1 for r in rows if not r["response"].strip())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
