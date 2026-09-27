"""Dump verbatim model outputs for the walkthrough -- no summarising, no rewriting.

The user asked to see what the model actually emits. Everything printed here is read
straight out of the records; nothing is reconstructed.
"""
import json
import pathlib

R = pathlib.Path("/root/autodl-fs/wafer-vlm")
reqs = [json.loads(l) for l in open(R / "outputs/reports/aux_classification_requests.jsonl")]
preds = [json.loads(l) for l in open(R / "outputs/baselines/sft_cls_logprobs.jsonl")]

print(f"requests: {len(reqs)}   predictions: {len(preds)}")
print(f"request keys:    {list(reqs[0].keys())}")
print(f"prediction keys: {list(preds[0].keys())}")
print()

print("=" * 78)
print("REQUEST 0 -- verbatim, truncated to 2600 chars if longer")
print("=" * 78)
s = json.dumps(reqs[0], ensure_ascii=False, indent=2)
print(s if len(s) <= 2600 else s[:2600] + "\n... [%d more chars]" % (len(s) - 2600))
print()

print("=" * 78)
print("PREDICTION 0 -- verbatim")
print("=" * 78)
print(json.dumps(preds[0], ensure_ascii=False, indent=2))
print()

# The three the user most wants to see: a confident success, the most confident
# failure, and a failure where the model was wrong in an interesting way.
WANT = ["wafer_00012768_013", "wafer_00044283_021", "wafer_00045584_023"]
by_id = {p.get("sample_id"): p for p in preds}
by_id_req = {q.get("sample_id"): q for q in reqs}

for sid in WANT:
    p = by_id.get(sid)
    print("=" * 78)
    print(f"{sid}")
    print("=" * 78)
    if p is None:
        print("  not found by sample_id; keys seen:", list(preds[0].keys()))
        break
    print(json.dumps(p, ensure_ascii=False, indent=2))
    print()
