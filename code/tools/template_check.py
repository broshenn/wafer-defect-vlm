"""Decode curated training examples back through ms-swift's own chat template.

A training run can look healthy while the template silently drops the image or
mangles the system turn. This renders one example per task with the same
template ms-swift uses for training and inference, then re-decodes the token ids
so what the model actually consumes is visible as text instead of assumed.

Note: ``processor.apply_chat_template`` is deliberately NOT used here. It leaves
``<image>`` literal and never emits vision tokens, so checking it would report a
failure that the real training path does not have.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

DEFAULT_SWIFT_SRC = "/root/autodl-tmp/wafer-vlm/src/ms-swift"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--template", default="qwen3_5")
    parser.add_argument("--swift-src", default=DEFAULT_SWIFT_SRC)
    parser.add_argument("--output")
    args = parser.parse_args()

    sys.path.insert(0, args.swift_src)
    from swift.model import get_model_processor
    from swift.template import get_template

    rows = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: dict[str, dict] = {}
    for row in rows:
        seen.setdefault(row.get("task", "unknown"), row)

    _, processor = get_model_processor(args.model, load_model=False)
    template = get_template(processor, template_type=args.template)
    pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")

    report: dict[str, object] = {"template": args.template, "tasks_checked": [], "problems": []}
    for task, row in sorted(seen.items()):
        messages = row["messages"]
        with Image.open(row["images"][0]) as handle:
            image = handle.convert("RGB")
        encoded = template.encode({"messages": messages, "images": [image]})
        ids = encoded["input_ids"]
        decoded = processor.tokenizer.decode(ids, skip_special_tokens=False)
        entry = {
            "task": task,
            "image": row["images"][0],
            "tokens": len(ids),
            "image_pad_tokens": ids.count(pad_id),
            "vision_start_tokens": ids.count(start_id),
            "pixel_values_shape": list(encoded["pixel_values"].shape) if "pixel_values" in encoded else None,
            "has_system_turn": messages[0]["role"] == "system",
            "assistant_chars": len(messages[-1]["content"]),
            "decoded_head": decoded[:200],
        }
        report["tasks_checked"].append(entry)
        if "<image>" in decoded:
            report["problems"].append(f"{task}: <image> was left unexpanded")
        if entry["image_pad_tokens"] == 0:
            report["problems"].append(f"{task}: no image tokens were emitted")
        if not entry["pixel_values_shape"]:
            report["problems"].append(f"{task}: no pixel_values were produced")
        if not entry["has_system_turn"]:
            report["problems"].append(f"{task}: system turn missing")

    report["ok"] = not report["problems"]
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
