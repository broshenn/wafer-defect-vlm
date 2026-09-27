"""Measure how much of the SFT score is lost to the train/serve prompt mismatch.

The benchmark deliberately froze its own phrasing, with no system message, and
the zero-shot baseline was scored on it — so Base vs SFT is a fair comparison on
identical prompts. But the SFT data was built with a system message and a terser
question. A fine-tuned model can therefore score below the baseline for a reason
that has nothing to do with what it learned.

This tool asks the same 252 core wafers the question in the *training* format,
so the two numbers can be read side by side:

    benchmark format : what the frozen leaderboard says
    training format  : what the model was actually taught to answer

The prompts are imported from the training code rather than retyped, so the
diagnostic cannot drift away from the thing it is diagnosing.

    build:  trainfmt_check.py build --benchmark <dir> --output <requests.jsonl>
    score:  trainfmt_check.py score --benchmark <dir> --predictions <results.jsonl>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects" / "wafer-defect-vlm" / "src"))

from wafer_vlm.curate import SYSTEM  # noqa: E402
from wafer_vlm.evaluate import align_predictions, classification_metrics, read_jsonl  # noqa: E402

TRAIN_PROMPT = "请判断这张晶圆图的缺陷类别，只回答类别名称。"


def build(args: argparse.Namespace) -> None:
    core = read_jsonl(Path(args.benchmark) / "core.jsonl")
    rows = [{
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<image>\n{TRAIN_PROMPT}"},
        ],
        "images": [row["image_path"]],
        "sample_id": row["sample_id"],
        "task": "classification",
    } for row in core]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "requests": len(rows),
                      "prompt": TRAIN_PROMPT, "system": SYSTEM}, ensure_ascii=False, indent=2))


def score(args: argparse.Namespace) -> None:
    benchmark = Path(args.benchmark)
    requests = read_jsonl(args.requests) if args.requests else [{
        "sample_id": row["sample_id"], "task": "classification",
    } for row in read_jsonl(benchmark / "core.jsonl")]
    aligned = align_predictions(requests, read_jsonl(args.predictions))
    gold = {r["sample_id"]: r["gold_label"] for r in read_jsonl(benchmark / "classification.jsonl")}
    metrics = classification_metrics(aligned, gold)
    report = {
        "requests": len(requests),
        "scored": metrics["samples"],
        "classification_in_training_format": metrics,
        "note": "diagnostic only; the frozen benchmark number is the one to quote",
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scored": metrics["samples"], "accuracy": metrics["accuracy"],
                      "macro_f1": metrics["macro_f1"],
                      "macro_f1_95ci": metrics["macro_f1_95ci"]}, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--benchmark", required=True)
    builder.add_argument("--output", required=True)
    builder.set_defaults(func=build)
    scorer = sub.add_parser("score")
    scorer.add_argument("--benchmark", required=True)
    scorer.add_argument("--predictions", required=True)
    scorer.add_argument("--requests")
    scorer.add_argument("--output")
    scorer.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
