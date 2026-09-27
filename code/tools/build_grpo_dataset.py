"""Build the GRPO prompt set from the curated, teacher-approved samples.

The gold carried into ``ground_truth`` is the trusted class label plus the
deterministic geometry the image itself produced — the same two sources the
benchmark scores against. The teacher's prose is deliberately not used as gold
here: rewarding the policy for agreeing with a teacher would just distil the
teacher again, with none of the label's authority behind it.

The prompt is byte-identical to the structured prompt used for SFT, so the RL
policy is optimised on the same task it was fine-tuned on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects" / "wafer-defect-vlm" / "src"))

from wafer_vlm.curate import SYSTEM  # noqa: E402

STRUCTURED_PROMPT = "请以 JSON 输出缺陷类别、形态、径向区域、钟点方向、密度、连续性、尺寸及中英文描述。"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gold_for(record: dict) -> dict:
    features = record.get("features") or {}
    return {
        "defect_type": record["failure_type"],
        "radial_zone": features.get("radial_zone"),
        "clock_direction": features.get("clock_sector"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0, help="0 keeps every eligible sample")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    records = [r for r in read_jsonl(Path(args.accepted)) if r.get("split") == args.split]
    random.Random(args.seed).shuffle(records)
    if args.limit:
        records = records[:args.limit]

    rows, skipped = [], Counter()
    for record in records:
        image = record.get("image_path")
        if not image or not Path(image).is_file():
            skipped["missing_image"] += 1
            continue
        gold = gold_for(record)
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"<image>\n{STRUCTURED_PROMPT}"},
            ],
            "images": [image],
            "ground_truth": json.dumps(gold, ensure_ascii=False, separators=(",", ":")),
            "sample_id": record["sample_id"],
            "lot_name": record["lot_name"],
            "failure_type": record["failure_type"],
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {
        "output": str(output),
        "sha256": digest,
        "split": args.split,
        "rows": len(rows),
        "skipped": dict(skipped),
        "classes": dict(Counter(r["failure_type"] for r in rows)),
        "lots": len({r["lot_name"] for r in rows}),
        "gold_source": "trusted label + deterministic geometry (radial_zone, clock_sector)",
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
