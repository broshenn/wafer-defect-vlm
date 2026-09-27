"""Verbatim model outputs + the gold labels, with the record shapes printed rather
than assumed (the previous attempt assumed a list and got a dict).
"""
import json
import math
import pathlib

R = pathlib.Path("/root/autodl-fs/wafer-vlm")


def load(p):
    return [json.loads(l) for l in open(p)]


base = {pathlib.Path(im["path"]).stem: r
        for r in load(R / "outputs/baselines/base_cls_logprobs.jsonl")
        for im in r["images"]}
sft = {pathlib.Path(im["path"]).stem: r
       for r in load(R / "outputs/baselines/sft_cls_logprobs.jsonl")
       for im in r["images"]}

r0 = list(sft.values())[0]
print("logprobs: top-level type =", type(r0["logprobs"]).__name__)
lp = r0["logprobs"]
if isinstance(lp, dict):
    print("  dict keys:", list(lp.keys())[:10])
    for k in list(lp.keys())[:2]:
        v = lp[k]
        print(f"  [{k}] -> {type(v).__name__}",
              (list(v.keys())[:8] if isinstance(v, dict) else str(v)[:120]))
else:
    print("  list len", len(lp), "first:", json.dumps(lp[0], ensure_ascii=False)[:300])


def conf(rec):
    lp = rec.get("logprobs")
    vals = []
    if isinstance(lp, dict):
        for v in lp.values():
            if isinstance(v, dict) and isinstance(v.get("logprob"), float):
                vals.append(v["logprob"])
            elif isinstance(v, list):
                vals += [t["logprob"] for t in v if isinstance(t.get("logprob"), float)]
    elif isinstance(lp, list):
        for v in lp:
            if isinstance(v, dict) and isinstance(v.get("logprob"), float):
                vals.append(v["logprob"])
            elif isinstance(v, dict) and isinstance(v.get("content"), list):
                vals += [t["logprob"] for t in v["content"]
                         if isinstance(t.get("logprob"), float)]
    return math.exp(sum(vals) / len(vals)) if vals else None


print()
print("## gold labels: looking for the file that maps sample_id -> defect class")
for pat in ("benchmarks/wafer_bench_v1",):
    for p in sorted((R / pat).rglob("*")):
        if p.is_file():
            print(f"   {p.relative_to(R)}  {p.stat().st_size} B")
