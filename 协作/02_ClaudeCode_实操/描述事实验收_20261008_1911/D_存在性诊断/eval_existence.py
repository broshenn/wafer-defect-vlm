#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D 段：一次**图像存在性诊断**（5090，真单卡）。

回答「去红后仍猜缺陷类」的问题：模型是**不能辨别红色存在**，
还是**分类选择/输出习惯**的问题？本诊断是辅助解释，**不改主指标**。

固定：原 Base、L3072-3407、D3072-3407；同 5090 环境；seed3407；真单卡。
每图 4 个条件 × dev18 18 张 = **72 次/模型**，三模型共 216。

  条件       图像输入            gold
  original   原 PNG + <image>    图中是否≥1个纯红像素（dev18 全为 true）
  dered      去红 PNG + <image>  false（红像素恒为 0）
  allred     全红 PNG + <image>  true（红像素 == 有效像素数）
  noimage    **不给图、删 <image> 占位**  输入不足 → 只统计未知/无依据断言

冻结题面（见 题目.json）：temperature 0、seed 3407、max_tokens 64、关闭 thinking。
主口径严格解析；统一诊断副口径（剥围栏 + 字符串布尔归一）另列，**不改主分**。

跑法（父进程必须先设 CUDA_VISIBLE_DEVICES）：
  CUDA_VISIBLE_DEVICES=3 python eval_existence.py --model base=<dir> --outdir ... --assets ...
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from 解析回归 import parse_strict, parse_diag, _REJ  # noqa: E402

SYSTEM = "你是半导体晶圆缺陷分析专家。"
QUESTION = ("图片里是否存在红色失效die？只输出一个JSON对象，"
            "has_red_defect填true或false；图片没有提供或无法可靠判断时填null")
CONDS = ["original", "dered", "allred", "noimage"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with io.open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="tag=<base>::<adapter>")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--assets", required=True,
                    help="派生图目录（含 派生清单.json）")
    a = ap.parse_args()

    import torch
    print("=" * 78)
    print(f"D 段存在性诊断  {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')!r}（父进程设置）")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("!! 父进程未设置 —— 拒绝运行"); return 3
    torch.cuda.init()
    n = torch.cuda.device_count()
    print(f"可见 GPU 数 = {n}  device0 = {torch.cuda.get_device_name(0)}")
    if n != 1:
        print("!! 不是 1 张卡 —— 拒绝运行"); return 3
    import torch.distributed as dist
    if dist.is_initialized() and dist.get_world_size() != 1:
        print("!! world_size != 1 —— 拒绝运行"); return 3

    tag, _, path = a.model.partition("=")
    base, _, adapter = path.partition("::")
    assets = Path(a.assets)
    man = json.loads(io.open(assets / "派生清单.json", encoding="utf-8").read())
    samples = man["样本"]
    print(f"模型 {tag}  基座 {base}  adapter {adapter or '（无）'}")
    print(f"诊断样本 {len(samples)} 张 × {len(CONDS)} 条件 = {len(samples)*len(CONDS)} 次")
    print(f"题面 sha256 {hashlib.sha256(QUESTION.encode()).hexdigest()}")

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    io.open(outdir / "题目.json", "w", encoding="utf-8", newline="\n").write(
        json.dumps({"system": SYSTEM, "question": QUESTION,
                    "question_sha256": hashlib.sha256(QUESTION.encode()).hexdigest(),
                    "条件": CONDS,
                    "解码参数": {"max_tokens": 64, "temperature": 0.0, "seed": 3407,
                              "thinking": "关闭"},
                    "gold": {"original": "图中是否≥1个纯红像素",
                             "dered": False, "allred": True,
                             "noimage": "输入不足，不参与图像事实正确率"}},
                   ensure_ascii=False, indent=2) + "\n")

    from swift import InferRequest, RequestConfig
    from swift.infer_engine import TransformersEngine

    t0 = time.time()
    eng = TransformersEngine(base, model_type="qwen3_5", template_type="qwen3_5",
                             adapters=[adapter] if adapter else None)
    print(f"加载 {time.time()-t0:.1f} s", flush=True)

    cfg = RequestConfig(max_tokens=64, temperature=0.0, seed=3407, enable_thinking=False)
    out_path = outdir / f"{tag}_existence_raw.jsonl"
    fh = io.open(out_path, "w", encoding="utf-8", newline="\n")
    k = 0
    try:
        for s in samples:
            for cond in CONDS:
                k += 1
                img = None
                gold = None
                if cond == "original":
                    img, gold = s["原图"], s["gold_原图"]
                elif cond == "dered":
                    img, gold = s["去红"], s["gold_去红"]
                elif cond == "allred":
                    img, gold = s["全红"], s["gold_全红"]
                # noimage：不给图、不写 <image> 占位

                content = ("<image>" + QUESTION) if img else QUESTION
                req = InferRequest(messages=[{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": content}],
                                   images=[img] if img else None)
                t1 = time.time()
                try:
                    resp = eng.infer([req], cfg)[0]
                    ch = resp.choices[0]
                    raw = ch.message.content
                    fr = getattr(ch, "finish_reason", None)
                except Exception as e:                      # 单条失败隔离，不重试扩额
                    raw, fr = f"<INFER_ERROR {type(e).__name__}: {e}>", "error"
                # 前两条记录真实读到的图与视觉 token，证明图像数据真的进了编码器
                ntok = None
                if img and k <= 2:
                    try:
                        mm = eng.template.processor(images=[__import__("PIL.Image", fromlist=["Image"])
                                                            .open(img).convert("RGB")],
                                                    return_tensors="pt")
                        ntok = int(mm["image_grid_thw"].prod().item()) if "image_grid_thw" in mm else None
                    except Exception:
                        ntok = None
                val, how = parse_strict(raw)
                dval, dhow = parse_diag(raw)
                rec = {"model_tag": tag, "sample_id": s["sample_id"], "parent": s["parent"],
                       "cond": cond, "gold": gold,
                       "image_path": img, "image_sha256": sha256_file(Path(img)) if img else None,
                       "image_tokens": ntok,
                       "raw": raw, "finish_reason": fr,
                       "val": (None if val is _REJ else val), "how": how,
                       "strict_ok": how in ("ok", "unknown"),
                       "diag_val": (None if dval is _REJ else dval), "diag_how": dhow,
                       "diag_ok": dhow in ("ok", "ok_str_bool", "unknown"),
                       "gen_seconds": round(time.time() - t1, 2)}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()                                  # 每条即写即 flush
                print(f"[{k:>3}/{len(samples)*len(CONDS)}] {s['sample_id']:<22}{cond:<9}"
                      f"gold={str(gold):<5} how={how:<16} val={val!r}", flush=True)
    finally:
        fh.close()
    print(f"写出 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
