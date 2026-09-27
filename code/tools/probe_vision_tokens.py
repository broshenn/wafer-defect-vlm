"""Check that a processor invocation actually injects vision tokens.

A processor will happily return ``pixel_values`` for a prompt that contains no
image placeholders. The vision tower then runs and its output has nowhere to
go, so the embeddings are text-only and any retrieval metric computed from them
is meaningless. The failure is silent, which is why it is worth an explicit
check rather than trusting the return type.

Usage:
    python probe_vision_tokens.py --model <path> --image <png> [--image <png> ...]
"""

from __future__ import annotations

import argparse

from PIL import Image
from transformers import AutoProcessor

PROBE = "Describe the defect pattern in this wafer map."


def counts(processor, ids: list[int]) -> dict:
    tokenizer = processor.tokenizer
    out = {}
    for name in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
        token_id = tokenizer.convert_tokens_to_ids(name)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            out[name] = ids.count(token_id)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", action="append", required=True)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    images = [Image.open(p).convert("RGB") for p in args.image]

    # The naive call: a bare string with no placeholders.
    naive = processor(text=[PROBE] * len(images), images=images,
                      return_tensors="pt", padding=True)
    naive_ids = naive["input_ids"][0].tolist()
    print("naive text= string")
    print("  seq_len:", len(naive_ids))
    print("  vision tokens:", counts(processor, naive_ids))

    # The canonical call: the chat template supplies the placeholders.
    messages = [
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROBE},
        ]}]
        for image in images
    ]
    templated = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
    chat = processor(text=templated, images=images, return_tensors="pt", padding=True)
    chat_ids = chat["input_ids"][0].tolist()
    print("apply_chat_template")
    print("  seq_len:", len(chat_ids))
    print("  vision tokens:", counts(processor, chat_ids))
    print("  pixel_values:", tuple(chat["pixel_values"].shape))

    ok = sum(counts(processor, chat_ids).values()) > 0
    print("VERDICT:", "chat template injects vision tokens" if ok
          else "STILL BROKEN: no vision tokens even with the chat template")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
