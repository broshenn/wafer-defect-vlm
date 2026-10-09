#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在获准的学校 5090 上，对 56 图隔离候选跑 Base / L / D **同题**推理。

与部署服务和已有 36 图运行保持同一口径：
  `TransformersEngine` + `InferRequest` + `RequestConfig(max_tokens=256,
   temperature=0.0, seed=3407)`，**不传 `enable_thinking`**
  （ms-swift 4.5.3 的 RequestConfig 没有这个字段，传了会 TypeError）。

纪律（任务书 §2.B）：
  · 每题**先写 started 台账**，跑完写原答，再写 done 台账；
  · 图片/题面/adapter 哈希**在跑前逐张核**，不符 = 系统错误 → **停调度**；
  · 模型格式或内容失败**原样保留**，不重试同一已完成 model×图；
  · **系统错误停调度**（输入/哈希/环境）；超时停本人任务并保存部分结果；
  · 断点续跑：已有原答的 (tag, sample_id) 跳过，不重跑。

用法（父 shell 必须先 `export CUDA_VISIBLE_DEVICES=<GPU-UUID>`）：
  python 跑56图.py --model <基座> --tag Base \
    --prompt <题面> --checker <检查器> --images <图目录> \
    --frozen <冻结清单（仅 id+sha，不含 gold）> --out <输出目录>
  … 加 --adapter <adapter目录> 跑 L / D
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

HARD_STOP_S = 28 * 60          # 28 分钟硬截止（30 设备分钟上限留 2 分钟余量）


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--checker", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--frozen", required=True, help="冻结清单：只含 item_id/sample_id/image_sha256")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=int, default=HARD_STOP_S)
    a = ap.parse_args()

    t_start = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ledger = out / f"台账_{a.tag}.jsonl"
    rawfile = out / f"{a.tag}_raw.jsonl"

    def led(kind: str, **kw):
        rec = {"ts": round(time.time() - t_start, 2), "tag": a.tag, "事件": kind, **kw}
        with ledger.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- 环境锁 ----
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "父 shell 未锁 GPU（须在 import torch 前设 UUID）"
    import torch
    torch.cuda.init()
    n = torch.cuda.device_count()
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"可见 GPU 数={n}  device0={torch.cuda.get_device_name(0)}")
    assert n == 1, f"必须真单卡，实际 {n} 张"
    host = os.uname().nodename
    led("环境", host=host, cuda_visible=os.environ["CUDA_VISIBLE_DEVICES"],
        device_name=torch.cuda.get_device_name(0), device_count=n,
        torch=torch.__version__)

    # ---- 冻结件核验（不符 = 系统错误，直接停）----
    pb = Path(a.prompt).read_bytes()
    prompt = pb.decode("utf-8")
    p_sha = sha_bytes(pb)
    c_sha = sha_file(Path(a.checker))
    frozen = json.loads(Path(a.frozen).read_text(encoding="utf-8"))
    expect = {r["sample_id"]: r["image_sha256"] for r in frozen["逐图"]}
    assert frozen.get("题面_sha256") == p_sha, f"题面 sha 不符 {p_sha}"
    assert frozen.get("检查器_sha256") == c_sha, f"检查器 sha 不符 {c_sha}"
    ad_sha = sha_file(Path(a.adapter, "adapter_model.safetensors")) if a.adapter else None
    led("冻结件", 题面=p_sha, 检查器=c_sha, adapter=ad_sha, N=len(expect))
    print(f"题面 {p_sha[:16]}…  检查器 {c_sha[:16]}…  adapter {ad_sha and ad_sha[:16]}…")

    # 逐张核图（本地 56 张上传副本）
    imgs = {}
    for sid, esha in expect.items():
        p = Path(a.images) / f"{sid}.png"
        if not p.exists():
            led("系统错误", 类型="缺图", sample_id=sid); print(f"**缺图 {sid}，停调度**"); return 4
        got = sha_file(p)
        if got != esha:
            led("系统错误", 类型="图sha不符", sample_id=sid, 期望=esha, 实际=got)
            print(f"**{sid} 图 sha 不符，停调度**"); return 4
        imgs[sid] = p
    print(f"56 图 sha 全部与冻结清单一致")

    # ---- 断点续跑 ----
    done = set()
    if rawfile.exists():
        for l in rawfile.open(encoding="utf-8"):
            if l.strip():
                try:
                    done.add(json.loads(l)["sample_id"])
                except Exception:
                    pass
        print(f"已有原答 {len(done)} 条，跳过")

    order = [r["sample_id"] for r in frozen["逐图"]]
    todo = [s for s in order if s not in done]

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
        for i, sid in enumerate(todo, 1):
            if time.time() - t_start > a.budget_s:
                led("硬截止", 已完成=ok, 剩余=len(todo) - i + 1)
                print(f"**到达硬截止，停（已完成 {ok}，剩 {len(todo)-i+1}）**")
                break
            p = imgs[sid]
            led("started", sample_id=sid, 序号=f"{i}/{len(todo)}")
            t1 = time.perf_counter()
            try:
                resp = eng.infer([InferRequest(
                    messages=[{"role": "system", "content": "你是半导体晶圆缺陷分析专家。"},
                              {"role": "user", "content": "<image>" + prompt}],
                    images=[str(p)])], cfg)[0]
                ch = resp.choices[0]
                rec = {"tag": a.tag, "sample_id": sid,
                       "image_sha256": expect[sid], "prompt_sha256": p_sha,
                       "checker_sha256": c_sha, "adapter_sha256": ad_sha,
                       "base_model": a.model, "host": host,
                       "raw": ch.message.content,
                       "finish_reason": getattr(ch, "finish_reason", None),
                       "usage": (getattr(resp, "usage", None) and
                                 json.loads(json.dumps(getattr(resp, "usage"), default=str))),
                       "seconds": round(time.perf_counter() - t1, 2)}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                led("done", sample_id=sid, 秒=rec["seconds"], finish=rec["finish_reason"])
                ok += 1
                print(f"  [{i}/{len(todo)}] {sid} {rec['seconds']}s {rec['finish_reason']}", flush=True)
            except Exception as e:                       # 单车故障：保留、继续
                err += 1
                led("单车失败", sample_id=sid, 类型=type(e).__name__, 信息=str(e)[:300])
                print(f"  [{i}/{len(todo)}] {sid} **失败 {type(e).__name__}: {str(e)[:120]}**", flush=True)
    finally:
        fh.close()
    infer_s = time.perf_counter() - t0 - load_s
    led("结束", 成功=ok, 失败=err, 加载秒=round(load_s, 2), 推理秒=round(infer_s, 2),
        总秒=round(time.time() - t_start, 2))
    print(f"结束：成功 {ok} 失败 {err}；加载 {load_s:.1f}s 推理 {infer_s:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
