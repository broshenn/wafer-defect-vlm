#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多字段 RL v2：确认池同题评测（M0 / M1 / M2 + 可选 Base）。

与既有 36/56 图运行同一口径：`TransformersEngine` + `InferRequest` +
`RequestConfig(max_tokens=256, temperature=0.0, seed=3407)`，不传 enable_thinking。

纪律：先写 started 台账再写原答；逐张核图 sha；失败原样保留不重试；
系统错误停调度；断点续跑（已有原答的跳过）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", required=True, help="confirm120.jsonl（不含答案）")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=int, default=9 * 60)
    a = ap.parse_args()

    t_start = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / f"台账_{a.tag}.jsonl"
    rawfile = out / f"{a.tag}_raw.jsonl"

    def led(kind: str, **kw):
        rec = {"ts": round(time.time() - t_start, 2), "tag": a.tag, "事件": kind, **kw}
        with ledger.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "父 shell 未锁 GPU"
    import torch
    torch.cuda.init()
    n = torch.cuda.device_count()
    assert n == 1, f"必须真单卡，实际 {n} 张"
    host = os.uname().nodename
    led("环境", host=host, cuda_visible=os.environ["CUDA_VISIBLE_DEVICES"],
        device_name=torch.cuda.get_device_name(0), device_count=n, torch=torch.__version__)
    print(f"单卡 {torch.cuda.get_device_name(0)}  host={host}")

    rows = [json.loads(l) for l in open(a.dataset, encoding="utf-8") if l.strip()]
    print(f"题面样本 {len(rows)} 条")
    ad_sha = sha_file(Path(a.adapter, "adapter_model.safetensors")) if a.adapter else None
    led("冻结件", adapter=ad_sha, N=len(rows), dataset_sha256=sha_file(Path(a.dataset)))
    print(f"adapter {ad_sha and ad_sha[:16]}…")

    imgs = {}
    for r in rows:
        sid = r["sample_id"]
        p = Path(a.images) / f"{sid}.png"
        if not p.exists():
            led("系统错误", 类型="缺图", sample_id=sid); print(f"**缺图 {sid}，停调度**"); return 4
        imgs[sid] = p

    done = set()
    if rawfile.exists():
        for l in rawfile.open(encoding="utf-8"):
            if l.strip():
                try:
                    done.add(json.loads(l)["sample_id"])
                except Exception:
                    pass
        print(f"已有原答 {len(done)} 条，跳过")
    todo = [r for r in rows if r["sample_id"] not in done]

    from swift import InferRequest, RequestConfig
    from swift.infer_engine import TransformersEngine
    adapters = [a.adapter] if a.adapter else None
    t0 = time.perf_counter()
    eng = TransformersEngine(a.model, model_type="qwen3_5", template_type="qwen3_5",
                             adapters=adapters)
    load_s = time.perf_counter() - t0
    led("模型加载", 秒=round(load_s, 2))
    print(f"加载 {load_s:.1f}s")

    cfg = RequestConfig(max_tokens=256, temperature=0.0, seed=3407)
    ok = err = 0
    fh = rawfile.open("a", encoding="utf-8", newline="\n")
    try:
        for i, r in enumerate(todo, 1):
            if time.time() - t_start > a.budget_s:
                led("硬截止", 已完成=ok, 剩余=len(todo) - i + 1)
                print(f"**到达硬截止，停（已完成 {ok}，剩 {len(todo)-i+1}）**")
                break
            sid = r["sample_id"]
            led("started", sample_id=sid, 序号=f"{i}/{len(todo)}")
            t1 = time.perf_counter()
            try:
                resp = eng.infer([InferRequest(messages=r["messages"],
                                               images=[str(imgs[sid])])], cfg)[0]
                ch = resp.choices[0]
                rec = {"tag": a.tag, "sample_id": sid, "adapter_sha256": ad_sha,
                       "base_model": a.model, "host": host, "raw": ch.message.content,
                       "finish_reason": getattr(ch, "finish_reason", None),
                       "usage": (getattr(resp, "usage", None) and
                                 json.loads(json.dumps(getattr(resp, "usage"), default=str))),
                       "seconds": round(time.perf_counter() - t1, 2)}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                led("done", sample_id=sid, 秒=rec["seconds"])
                ok += 1
                print(f"  [{i}/{len(todo)}] {sid} {rec['seconds']}s", flush=True)
            except Exception as e:
                err += 1
                led("单车失败", sample_id=sid, 类型=type(e).__name__, 信息=str(e)[:300])
                print(f"  [{i}/{len(todo)}] {sid} **失败 {type(e).__name__}**", flush=True)
    finally:
        fh.close()
    led("结束", 成功=ok, 失败=err, 加载秒=round(load_s, 2),
        总秒=round(time.time() - t_start, 2))
    print(f"结束：成功 {ok} 失败 {err}；总 {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
