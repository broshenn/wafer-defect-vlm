#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在同一台租机上，用**冻结七字段题面**对固定 36 图跑 Base 或 L。

与部署服务的区别：这是**批量离线跑**，不启 HTTP。
推理接口、模板、解码参数与 `wafer_service.py` **完全一致**
（`TransformersEngine` + `InferRequest` + `RequestConfig(max_tokens=256, temperature=0, seed=3407)`），
**不传 `enable_thinking`**（该参数在 ms-swift 4.5.3 不存在）。

用法：
  CUDA_VISIBLE_DEVICES=<uuid> python 批量36.py \
    --model /WS/models/Qwen3.5-9B --tag Base \
    --prompt /WS/deploy/prompt_7f.txt \
    --images /WS/deploy/图 --out /WS/baidian36/原答

  ... --adapter /WS/adapters/L-N3072-3407 --tag L-N3072-3407
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, sys, time
from pathlib import Path


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--images", required=True, help="36 图目录")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}（须由父 shell 设 UUID）")
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "父 shell 未锁定 GPU"
    torch.cuda.init()
    n = torch.cuda.device_count()
    print(f"可见 GPU 数 = {n}  device0 = {torch.cuda.get_device_name(0)}")
    assert n == 1, "必须真单卡"

    pb = Path(a.prompt).read_bytes()
    prompt = pb.decode("utf-8")
    psha = sha(pb)
    print(f"题面 sha256 {psha}")

    imgs = sorted(Path(a.images).glob("*.png"))
    assert len(imgs) == 36, f"应有 36 图，实际 {len(imgs)}"
    print(f"图 {len(imgs)} 张")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"{a.tag}_raw.jsonl"
    if outfile.exists():
        print(f"**{outfile} 已存在，拒绝覆盖**"); return 3

    from swift import InferRequest, RequestConfig
    from swift.infer_engine import TransformersEngine

    adapters = [a.adapter] if a.adapter else None
    ad_sha = sha(Path(a.adapter, "adapter_model.safetensors").read_bytes()) if a.adapter else None
    t0 = time.perf_counter()
    eng = TransformersEngine(a.model, model_type="qwen3_5", template_type="qwen3_5",
                             adapters=adapters)
    load_s = time.perf_counter() - t0
    print(f"加载 {load_s:.1f}s  adapter_sha256={ad_sha}")

    cfg = RequestConfig(max_tokens=256, temperature=0.0, seed=3407)
    fh = io.open(outfile, "w", encoding="utf-8", newline="\n")
    try:
        for i, p in enumerate(imgs, 1):
            b = p.read_bytes()
            t1 = time.perf_counter()
            resp = eng.infer([InferRequest(
                messages=[{"role": "system", "content": "你是半导体晶圆缺陷分析专家。"},
                          {"role": "user", "content": "<image>" + prompt}],
                images=[str(p)])], cfg)[0]
            ch = resp.choices[0]
            rec = {"tag": a.tag, "sample_id": p.stem,
                   "image_sha256": sha(b), "prompt_sha256": psha,
                   "adapter_sha256": ad_sha,
                   "raw": ch.message.content,
                   "finish_reason": getattr(ch, "finish_reason", None),
                   "seconds": round(time.perf_counter() - t1, 2)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            print(f"  [{i}/36] {p.stem} {rec['seconds']}s {rec['finish_reason']}", flush=True)
    finally:
        fh.close()
    print(f"写出 {outfile}")
    print(f"加载 {load_s:.1f}s + 推理 {time.perf_counter()-t0-load_s:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
