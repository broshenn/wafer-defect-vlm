#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5 推理评测（B 卡 6；模板：一天半优化_20261010/server_v3_eval.py）。
用法：python eval_v5.py --model <基座> [--adapter <dir>|--adapter-from-line <线名>] --tag <模型名>
      --dataset <jsonl> --out <dir> [--max-tokens 384] [--batch-size 8] [--budget-s 毫秒不设上限?]
改动 v3 版之处：图片路径直接用数据行 images[0]；批推理（批内某条失败退回逐条重试一次）；
--adapter-from-line 找最大步数 checkpoint；记 s/张；L 指针文件由 --adapter 直接给。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

EMPTY_THINK = re.compile(r"^\s*<think\b[^>]*>\s*</think\s*>\s*", re.S)
FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.S)


def strip_envelope(text: str) -> str:
    s = EMPTY_THINK.sub("", text.strip())
    m = FENCE.match(s)
    return m.group(1).strip() if m else s


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def find_max_checkpoint(line_dir: Path):
    cks = []
    for p in line_dir.rglob("checkpoint-*"):
        try:
            cks.append((int(p.name.split("checkpoint-")[1]), p))
        except (IndexError, ValueError):
            continue
    if not cks:
        return None, None
    step, path = max(cks)
    return step, path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--adapter-from-line", default=None)
    ap.add_argument("--runs-root", default="/WS/v5/runs")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--budget-s", type=int, default=0)
    a = ap.parse_args()

    t_start = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / f"台账_{a.tag}.jsonl"
    rawfile = out / f"{a.tag}_raw.jsonl"

    def led(kind: str, **kw):
        with ledger.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"ts": round(time.time() - t_start, 2), "tag": a.tag,
                                "事件": kind, **kw}, ensure_ascii=False) + "\n")

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "父 shell 未锁 GPU"
    import torch
    torch.cuda.init()
    n = torch.cuda.device_count()
    assert n == 1, f"必须真单卡，实际 {n} 张"
    host = os.uname().nodename
    led("环境", host=host, device=torch.cuda.get_device_name(0), cards=n, torch=torch.__version__)
    print(f"单卡 {torch.cuda.get_device_name(0)}")

    adapter = a.adapter
    used_step = None
    if a.adapter_from_line:
        step, p = find_max_checkpoint(Path(a.runs_root) / a.adapter_from_line)
        if p is None:
            led("系统错误", 类型="找不到checkpoint", 线=a.adapter_from_line)
            print("**找不到 checkpoint，停"); return 4
        adapter, used_step = str(p), step
        print(f"adapter-from-line {a.adapter_from_line} → {adapter} (step {step})")
    if a.adapter and os.path.isfile(a.adapter):
        adapter = Path(a.adapter).read_text(encoding="utf-8").strip()  # 指针文件
        print(f"指针文件 → {adapter}")

    rows = [json.loads(l) for l in open(a.dataset, encoding="utf-8") if l.strip()]
    ad_sha = sha_file(Path(adapter, "adapter_model.safetensors")) if adapter else None
    led("冻结件", adapter=ad_sha, adapter_path=adapter, step=used_step, N=len(rows),
        dataset=Path(a.dataset).name, dataset_sha256=sha_file(Path(a.dataset)))
    print(f"题面 {Path(a.dataset).name} {len(rows)} 条 adapter {ad_sha and ad_sha[:16]}…")

    for r in rows:
        if not Path(r["images"][0]).exists():
            led("系统错误", 类型="缺图", sample_id=r["sample_id"], path=r["images"][0])
            print(f"**缺图 {r['sample_id']}，停"); return 4

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
    adapters = [adapter] if adapter else None
    t0 = time.perf_counter()
    eng = TransformersEngine(a.model, model_type="qwen3_5", template_type="qwen3_5",
                             adapters=adapters)
    load_s = time.perf_counter() - t0
    led("模型加载", 秒=round(load_s, 2))
    print(f"加载 {load_s:.1f}s")

    cfg = RequestConfig(max_tokens=a.max_tokens, temperature=0.0, seed=3407)
    ok = err = 0
    secs = []
    fh = rawfile.open("a", encoding="utf-8", newline="\n")
    try:
        for i0 in range(0, len(todo), max(1, a.batch_size)):
            if a.budget_s and time.time() - t_start > a.budget_s:
                led("硬截止", 已完成=ok, 剩余=len(todo) - i0)
                print(f"**到达硬截止，已完成 {ok}"); break
            batch = todo[i0:i0 + max(1, a.batch_size)]
            t1 = time.perf_counter()
            reqs = [InferRequest(messages=r["messages"], images=[r["images"][0]]) for r in batch]
            resps = None
            try:
                resps = eng.infer(reqs, cfg)
                if len(resps) != len(batch):
                    raise RuntimeError(f"批返回 {len(resps)} != {len(batch)}")
            except Exception as e:
                led("批量失败", 类型=type(e).__name__, 信息=str(e)[:200], n=len(batch))
                resps = []
                for r in batch:  # 退回逐条重试一次
                    try:
                        resps.append(eng.infer([InferRequest(messages=r["messages"],
                                                             images=[r["images"][0]])], cfg)[0])
                    except Exception:
                        resps.append(None)
            dt_batch = time.perf_counter() - t1
            for j, (r, resp) in enumerate(zip(batch, resps)):
                sid = r["sample_id"]
                if resp is None:
                    err += 1
                    led("单车失败", sample_id=sid, 类型="infer_none")
                    print(f"  [{i0 + j + 1}/{len(todo)}] {sid} **失败**", flush=True)
                    continue
                ch = resp.choices[0]
                raw = ch.message.content
                rec = {"tag": a.tag, "sample_id": sid, "adapter_sha256": ad_sha,
                       "adapter_path": adapter, "step": used_step, "base_model": a.model,
                       "host": host, "raw": raw,
                       "raw_is_bare_measure": raw.lstrip().startswith("<measure>"),
                       "stripped": strip_envelope(raw),
                       "finish_reason": getattr(ch, "finish_reason", None),
                       "seconds": round(dt_batch / len(batch), 2)}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                led("done", sample_id=sid)
                ok += 1
                secs.append(rec["seconds"])
            if (i0 // max(1, a.batch_size)) % 5 == 0:
                avg = sum(secs) / len(secs) if secs else 0
                print(f"  [{min(i0 + len(batch), len(todo))}/{len(todo)}] 均 {avg:.1f}s/张", flush=True)
    finally:
        fh.close()
    avg = round(sum(secs) / len(secs), 2) if secs else None
    led("结束", 成功=ok, 失败=err, 加载秒=round(load_s, 2), 均秒每张=avg,
        总秒=round(time.time() - t_start, 2))
    print(f"结束：成功 {ok} 失败 {err} 均 {avg}s/张")
    return 0


if __name__ == "__main__":
    sys.exit(main())
