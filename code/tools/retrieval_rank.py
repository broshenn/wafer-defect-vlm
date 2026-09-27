"""Rank retrieval gallery items for each benchmark query with VLM embeddings.

Mean-pools the last hidden state of a fixed probe prompt (image + text) and
ranks gallery entries by cosine similarity. Same-lot pairs are excluded because
they are also excluded from the qrels, so leaving them in would cap recall.
"""
import argparse, json, sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROBE = "Describe the defect pattern in this wafer map."


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def embed(model, processor, paths, device, batch_size):
    vectors = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in chunk]
        # The chat template is what inserts the vision placeholders. Handing the
        # processor a bare string still returns pixel_values, but input_ids then
        # holds no image tokens, so the vision tower's output has nowhere to go:
        # the pooled vector is text-only and every retrieval metric computed from
        # it is meaningless. Nothing about that failure is visible in the return
        # types, so the check below makes it loud instead.
        messages = [
            [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROBE},
            ]}]
            for image in images
        ]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=text, images=images,
                           return_tensors="pt", padding=True).to(device)
        if start == 0:
            pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if pad_id is None or pad_id not in inputs["input_ids"][0].tolist():
                raise SystemExit(
                    "processor produced no <|image_pad|> tokens, so the embeddings "
                    "would be text-only; refusing to write meaningless rankings"
                )
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1].float()
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        vectors.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    return torch.cat(vectors)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    model.eval()

    bench = Path(args.benchmark)
    queries = load_jsonl(bench / "retrieval_queries.jsonl")
    gallery = load_jsonl(bench / "retrieval_gallery.jsonl")
    print("queries=%d gallery=%d" % (len(queries), len(gallery)))

    qvec = embed(model, processor, [q["image_path"] for q in queries], device, args.batch_size)
    gvec = embed(model, processor, [g["image_path"] for g in gallery], device, args.batch_size)
    sims = qvec @ gvec.T

    rows = []
    for i, query in enumerate(queries):
        ranked = []
        for j in torch.argsort(sims[i], descending=True).tolist():
            cand = gallery[j]
            if cand["gallery_id"] == query["query_id"] or cand["lot_name"] == query["lot_name"]:
                continue
            ranked.append(cand["gallery_id"])
            if len(ranked) >= args.top_k:
                break
        rows.append({"query_id": query["query_id"], "gallery_ids": ranked})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"ranked_queries": len(rows), "top_k": args.top_k}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
