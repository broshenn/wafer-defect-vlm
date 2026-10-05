#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M：把旧 Adapter 合并进基座，再在原 dev18 上同题比较（合并**不覆盖**基座与旧 Adapter）。

隔离同上：CUDA_VISIBLE_DEVICES 必须由父进程设置，断言只见 1 张卡。

用法：
    CUDA_VISIBLE_DEVICES=<uuid> python merge_card.py --adapter <ckpt60> --outdir <新目录>
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import inspect
import io
import json
import os
import sys

os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/WS/models/Qwen3.5-9B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--data", default="/WS/datasets/dev_18.server.jsonl")
    ap.add_argument("--out", default="/WS/checkpoints/merged_dev18.jsonl")
    a = ap.parse_args()

    import torch
    print("=" * 78)
    print(f"M 合并等价检查  {datetime.datetime.now().isoformat()}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')!r}（父进程设置）")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("!! 父进程未设置 CUDA_VISIBLE_DEVICES —— 拒绝运行")
        return 3
    torch.cuda.init()
    n = torch.cuda.device_count()
    print(f"可见 GPU 数 = {n}   device 0 = {torch.cuda.get_device_name(0)}")
    if n != 1:
        print(f"!! 可见 {n} 张卡 —— 拒绝运行")
        return 3

    print(f"adapter : {a.adapter}")
    print(f"adapter_config sha256: {sha(os.path.join(a.adapter, 'adapter_config.json'))}")
    print(f"输出目录: {a.outdir}（**新目录，不覆盖基座/旧 Adapter**）")
    # 注意：ExportArguments 会**拒绝写入已存在的目录**（防覆盖）。
    # 所以这里只建父目录，目标目录必须不存在，由 ms-swift 自己创建。
    if os.path.exists(a.outdir):
        print(f"!! 输出目录已存在，拒绝运行（避免覆盖）：{a.outdir}")
        return 3
    os.makedirs(os.path.dirname(a.outdir.rstrip("/")), exist_ok=True)

    from swift import ExportArguments
    from swift.pipelines import export_main

    try:
        fields = {f.name for f in dataclasses.fields(ExportArguments)}
    except Exception:
        fields = set(inspect.signature(ExportArguments.__init__).parameters)
    print(f"ExportArguments 字段数 {len(fields)}；含 merge_lora: {'merge_lora' in fields}")

    t0 = datetime.datetime.now()
    args = ExportArguments(model=a.model, adapters=[a.adapter],
                           merge_lora=True, output_dir=a.outdir)
    export_main(args)
    dt = (datetime.datetime.now() - t0).total_seconds()
    print(f"\n合并完成 {dt:.1f}s -> {a.outdir}", flush=True)

    import glob
    files = sorted(os.path.basename(x) for x in glob.glob(os.path.join(a.outdir, "*")))
    print(f"输出文件 {len(files)} 个: {files[:12]}")
    has_cfg = os.path.exists(os.path.join(a.outdir, "config.json"))

    # ── 合并后同题推理 ──────────────────────────────────
    from swift import InferRequest, RequestConfig
    from swift.infer_engine import TransformersEngine
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    rows = [json.loads(l) for l in io.open(a.data, encoding="utf-8") if l.strip()]
    print(f"\n合并后模型在 {len(rows)} 条上推理")
    engine = TransformersEngine(a.outdir)
    cfg = RequestConfig(max_tokens=64, temperature=0.0, seed=3407)

    import re
    CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
               "Near_full", "Random", "Scratch", "none"]

    def parse_class(text):
        if not text:
            return None, "empty"
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                o = json.loads(m.group(0))
                v = o.get("defect_class")
                if isinstance(v, str):
                    return (v if v in CLASSES + ["unknown"] else None), \
                           ("ok" if v in CLASSES + ["unknown"] else "bad_enum")
            except json.JSONDecodeError:
                pass
        for c in CLASSES + ["unknown"]:
            if c in text:
                return c, "loose_match"
        return None, "unparsed"

    QUESTION = ("这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。"
                "请从 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_full、Random、Scratch、none "
                "中选一个最符合主要图案的类别。只输出一个 JSON 对象，形如 "
                '{"defect_class":"类别"}；无法可靠判断时填 {"defect_class":"unknown"}。')

    out = io.open(a.out, "w", encoding="utf-8", newline="\n")
    for i, r in enumerate(rows, 1):
        resp = engine.infer([InferRequest(
            messages=[{"role": "user", "content": "<image>" + QUESTION}],
            images=[r["image_path"]])], cfg)[0]
        raw = resp.choices[0].message.content
        pred, why = parse_class(raw)
        out.write(json.dumps({"sample_id": r["sample_id"], "label": r["label"], "raw": raw,
                              "parsed": pred, "parse_how": why,
                              "correct": bool(pred == r["label"])}, ensure_ascii=False) + "\n")
        out.flush()
        print(f"  [{i}/{len(rows)}] {r['sample_id']} 真值={r['label']:10s} 预测={pred}", flush=True)
    out.close()

    res = [json.loads(l) for l in io.open(a.out, encoding="utf-8") if l.strip()]
    n_ok = sum(1 for x in res if x["correct"])
    peak = torch.cuda.max_memory_allocated(0) / 2**30
    summary = {"model": a.model, "adapter": a.adapter, "outdir": a.outdir,
               "adapter_config_sha256": sha(os.path.join(a.adapter, "adapter_config.json")),
               "merged_output_files": files, "has_config_json": has_cfg,
               "merge_seconds": round(dt, 1),
               "dev18": {"n": len(res), "correct": n_ok,
                         "accuracy": round(n_ok / len(res), 4) if res else 0},
               "peak_allocated_GiB_own_process": round(peak, 3),
               "answers": res}
    with io.open(a.out + ".merge.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"\n合并后 dev18: {n_ok}/{len(res)} = {n_ok/len(res):.4f}")
    print("注意：这只是**类别一致**的比较，差异保留不编为等价。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
