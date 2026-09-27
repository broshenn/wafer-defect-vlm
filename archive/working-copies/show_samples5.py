"""What one trained model actually emits on each task: classification, the structured
JSON, a caption, and a robustness variant.

Reads outputs/baselines/qwen35_9b_gspo_g4_lr1e5.jsonl (the GSPO G=4 lr 1e-5 run) and
prints the model's raw text; nothing is paraphrased.
"""
import json
import pathlib
import re
from collections import Counter

R = pathlib.Path("/root/autodl-fs/wafer-vlm")
FILE = R / "outputs/baselines/qwen35_9b_gspo_g4_lr1e5.jsonl"

recs = [json.loads(l) for l in open(FILE)]
print(f"{FILE.name}: {len(recs)} records")
print("keys:", list(recs[0].keys()))

# What distinguishes the tasks? The user prompt text (and the dataset field).
def prompt_of(r):
    for m in r.get("messages", []):
        if m["role"] == "user":
            return m["content"]
    return ""


def tag_of(r):
    p = prompt_of(r)
    if "只回答类别名称" in p:
        return "分类"
    if "JSON" in p or "json" in p or "字段" in p:
        return "结构化"
    if "描述" in p or "caption" in p.lower() or "说明" in p:
        return "Caption"
    return "其他: " + p[:40].replace("\n", " ")


c = Counter(tag_of(r) for r in recs)
print("\ntask mix:", dict(c))

seen = set()
for r in recs:
    t = tag_of(r)
    if t in seen or not t.startswith(("分类", "结构化", "Caption")):
        continue
    seen.add(t)
    print("\n" + "=" * 78)
    print(f"TASK: {t}")
    print("=" * 78)
    print("PROMPT:")
    print("  " + prompt_of(r).replace("\n", "\n  "))
    print("\nMODEL RESPONSE (verbatim):")
    print("  " + r["response"].replace("\n", "\n  "))
    print(f"\nimage: {[i['path'] for i in r.get('images', [])]}")
    if len(seen) == 3:
        break

print("\n" + "=" * 78)
print("ROBUSTNESS -- the same wafer under a transform")
print("=" * 78)
for r in recs:
    img = (r.get("images") or [{}])[0].get("path", "")
    if "__" in pathlib.Path(img).stem:
        print(f"image: {pathlib.Path(img).name}")
        print("PROMPT:   " + prompt_of(r).replace("\n", " ")[:160])
        print("RESPONSE: " + repr(r["response"]))
        break
