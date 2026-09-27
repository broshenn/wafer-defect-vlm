"""Verbatim model outputs, stripped of the per-token logprob firehose.

Reads the same records the reports are derived from; prints the prompt, the raw
response, and the score, and nothing that was not in the record.
"""
import json
import math
import pathlib

R = pathlib.Path("/root/autodl-fs/wafer-vlm")


def conf(rec):
    """exp(mean(token logprob)) over the tokens after the thinking block."""
    lp = rec.get("logprobs") or []
    vals = []
    if lp and isinstance(lp[0], dict):
        inner = lp[0].get("content") or []
        vals = [t["logprob"] for t in inner if isinstance(t.get("logprob"), float)]
    if not vals:
        return None
    return math.exp(sum(vals) / len(vals))


def sid(rec):
    for im in rec.get("images") or []:
        p = im.get("path") or ""
        if p:
            return pathlib.Path(p).stem
    return None


for tag, predfile in (("BASE (zero-shot)", "base_cls_logprobs.jsonl"),
                      ("SFT (QLoRA adapter)", "sft_cls_logprobs.jsonl")):
    preds = [json.loads(l) for l in open(R / "outputs/baselines" / predfile)]
    by = {sid(p): p for p in preds}
    cases = json.load(open(R / "outputs/reports/error_cases_sft.json"))
    print("#" * 78)
    print(f"# {tag}   {len(preds)} samples")
    print("#" * 78)
    for want, why in (("wafer_00012768_013", "对，且很有把握"),
                      ("wafer_00044283_021", "最自信的一个错误"),
                      ("wafer_00045584_023", "另一个高自信错误")):
        rec = by.get(want)
        if rec is None:
            print(f"  {want}: absent from this file")
            continue
        msgs = rec["messages"]
        gold = None
        for f in cases["all_failures"]:
            if f.get("sample") == want:
                gold = f.get("gold")
        print(f"\n----- {want}  ({why}) -----")
        print("PROMPT:")
        for m in msgs:
            if m["role"] == "user":
                print("  " + m["content"].replace("\n", "\n  "))
        print("RAW RESPONSE (repr, so whitespace is visible):")
        print("  " + repr(rec["response"]))
        print(f"  gold: {gold}   confidence: {conf(rec):.4f}")
    print()
