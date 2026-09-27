"""The walkthrough's evidence page: what each of the five tasks asks for, and what the
model actually answered. Everything is read from the records; nothing is rewritten.
"""
import json
import math
import pathlib

R = pathlib.Path("/root/autodl-fs/wafer-vlm")
B = R / "benchmarks/wafer_bench_v1"


def load(p):
    return [json.loads(l) for l in open(p)]


def first(p, n=1, limit=1400):
    recs = load(p)[:n]
    for r in recs:
        s = json.dumps(r, ensure_ascii=False, indent=2)
        print(s if len(s) <= limit else s[:limit] + f"\n... [+{len(s)-limit} chars]")


print("=" * 78)
print("A. WHAT THE BENCHMARK ASKS FOR -- one record from each task file")
print("=" * 78)
for name in ("classification.jsonl", "structured.jsonl", "caption.jsonl"):
    print(f"\n--- {name} ---")
    first(B / name, 1, 1200)

print("\n--- retrieval_queries.jsonl (first) ---")
first(B / "retrieval_queries.jsonl", 1, 700)
print("\n--- retrieval_gallery.jsonl (first) ---")
first(B / "retrieval_gallery.jsonl", 1, 700)

print()
print("=" * 78)
print("B. WHAT THE MODEL ANSWERED -- classification, base vs SFT")
print("=" * 78)


def loadpreds(p):
    out = {}
    for r in load(p):
        for im in r["images"]:
            out[pathlib.Path(im["path"]).stem] = r
    return out


base = loadpreds(R / "outputs/baselines/base_cls_logprobs.jsonl")
sft = loadpreds(R / "outputs/baselines/sft_cls_logprobs.jsonl")


def score(rec):
    toks = rec["logprobs"]["content"]
    vals = [t["logprob"] for t in toks if isinstance(t.get("logprob"), float)]
    return math.exp(sum(vals) / len(vals)) if vals else None, toks


CASES = [
    ("wafer_00012768_013", "SFT 很有把握地答对"),
    ("wafer_00044283_021", "SFT 最自信的一个错误（0.995）"),
    ("wafer_00045584_023", "SFT 把 Donut 看成 Loc（0.958）"),
]

for sid, why in CASES:
    print(f"\n########## {sid} -- {why} ##########")
    for tag, table in (("BASE  ", base), ("SFT   ", sft)):
        rec = table.get(sid)
        if rec is None:
            print(f"  {tag}: absent")
            continue
        c, toks = score(rec)
        prompt = [m for m in rec["messages"] if m["role"] == "user"]
        print(f"  {tag} raw: {rec['response']!r}")
        print(f"  {tag} conf = {c:.4f}   n_answer_tokens = {len(toks)}")
    print("  the model's own top-5 at the first answer token (SFT):")
    toks = sft[sid]["logprobs"]["content"]
    if toks:
        for alt in (toks[0].get("top_logprobs") or [])[:5]:
            print(f"    {alt['token']!r:<14} logprob {alt['logprob']:.4f}")
    print("  prompt (identical for both):")
    print("    " + prompt[0]["content"].replace("\n", "\n    "))
