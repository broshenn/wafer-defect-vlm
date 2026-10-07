#!/bin/bash
# 服务端：A/B/C 的 token 长度与「只监督回答」的模板探针（CPU，0 GPU）
set -u
R=/WS
export CUDA_VISIBLE_DEVICES=""
export IMAGE_MAX_TOKEN_NUM=256
exec $R/envs/wafer/bin/python - <<'PY'
import json, io, os, statistics, sys
from datasets import Dataset
from swift.model import get_model_processor
from swift.template import get_template
import swift

R = "/WS"
MODEL = R + "/models/Qwen3.5-9B"
D = R + "/datasets/abc"
proc = get_model_processor(MODEL, model_type="qwen3_5", torch_dtype=None)[1]
tpl = get_template(proc, template_type="qwen3_5")
pre = swift.EncodePreprocessor(tpl)
tok = proc.tokenizer

print("=" * 76)
print("A/B/C token 长度与监督掩码（CPU）")
print("=" * 76)
out = {}
for name in ("A", "B", "C"):
    p = f"{D}/abc_{name}.server.jsonl"
    rows = [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]
    ds = Dataset.from_list([{"messages": r["messages"], "images": r["images"],
                             "label": r["label"]} for r in rows])
    enc = pre(ds)
    lens, sup, img = [], [], []
    bad = 0
    for i in range(len(enc)):
        ids = enc[i]["input_ids"]
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        lens.append(len(ids))
        img.append(sum(1 for t in ids if t == tpl.image_token_id))
        if i < 40:   # 抽 40 条做掩码检查
            b = tpl.data_collator([dict(enc[i], pixel_values=[
                __import__("torch").as_tensor(x) if not __import__("torch").is_tensor(x) else x
                for x in enc[i]["pixel_values"]])])
            lab = b["labels"][0].tolist()
            v = [j for j, x in enumerate(lab) if x != -100]
            if not v or rows[i]["messages"][1]["content"] not in tok.decode([lab[j] for j in v]):
                bad += 1
            sup.append(len(v))
    out[name] = {"n": len(rows), "seq_min": min(lens), "seq_med": int(statistics.median(lens)),
                 "seq_max": max(lens), "seq_mean": round(statistics.mean(lens), 1),
                 "image_tokens": img[0],
                 "sup_med": int(statistics.median(sup)) if sup else None,
                 "sup_min": min(sup) if sup else None, "sup_max": max(sup) if sup else None,
                 "mask_probe_bad": bad, "mask_probe_n": len(sup)}
    print(f"\n--- {name} ---")
    print(f"  序列 token: min {out[name]['seq_min']} 中位 {out[name]['seq_med']} "
          f"max {out[name]['seq_max']} 均值 {out[name]['seq_mean']}")
    print(f"  其中 image token {img[0]}")
    print(f"  被监督 token: min {out[name]['sup_min']} 中位 {out[name]['sup_med']} "
          f"max {out[name]['sup_max']}")
    print(f"  掩码探针 {out[name]['mask_probe_n']} 条，异常 {bad}")

print("\n" + "=" * 76)
print("总监督 token（1 epoch 全量）")
tot = {}
for name in ("A", "B", "C"):
    rows = [json.loads(l) for l in io.open(f"{D}/abc_{name}.server.jsonl", encoding="utf-8") if l.strip()]
    s = out[name]
    # 用中位监督 token 估总量
    est = s["sup_med"] * s["n"]
    tot[name] = {"n": s["n"], "sup_med": s["sup_med"], "sup_total_est": est,
                 "seq_med": s["seq_med"], "seq_total_est": s["seq_med"] * s["n"]}
    print(f"  {name}: {s['n']} 条 × 中位监督 {s['sup_med']} = **{est:,}** 监督 token；"
          f"序列总量约 {s['seq_med']*s['n']:,}")
json.dump(out, io.open(R + "/logs/abc_token_len.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
json.dump(tot, io.open(R + "/logs/abc_token_total.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n写出 /WS/logs/abc_token_len.json 与 abc_token_total.json")
PY
