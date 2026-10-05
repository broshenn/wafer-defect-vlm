#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SFT-A 标签监督训练驱动（20 步 smoke / 60 步主训练）。

用 ms-swift 自己的管道 `sft_main(SftArguments)`，不是 CLI —— 原因是实测：

    `swift/utils/transformers_utils.py::get_target_modules` 第一行：

        if isinstance(args.target_modules, str):
            return args.target_modules          # ← 直接返回，freeze_* 根本不读

    于是 target_modules 传**字符串** "all-linear" 时冻结被静默跳过：

        字符串   → 可训练 51.265M，视觉侧 220/553 个张量可训练   ❌
        列表     → 可训练 43.278M，视觉侧 0/333 可训练           ✅

    传列表才会走 get_multimodal_target_regex(freeze_vit=..., freeze_aligner=...)，
    实际生成的正则只匹配 `model.language_model.*`。

所以这里**显式传列表**，训练后再从 checkpoint 的 `adapter_config.json`
把 target_modules 读回来核对 —— 那是持久证据，不是日志里的一句话。

用法：
    python train_sft_a.py --stage smoke
    python train_sft_a.py --stage main
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")

import torch                                       # noqa: E402
from swift import SftArguments                     # noqa: E402
from swift.pipelines import sft_main               # noqa: E402

MODEL = "/WS/models/Qwen3.5-9B"
DATA = "/WS/datasets/sft_a"

STAGES = {
    "smoke": dict(train="sft_a_smoke_20.jsonl",
                  out="/WS/checkpoints/sft_a_smoke20_20step",
                  max_steps=20, ga=1),
    "main": dict(train="sft_a_train_180.jsonl",
                 out="/WS/checkpoints/sft_a_train180_60step",
                 max_steps=60, ga=4),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    st = STAGES[a.stage]
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

    print("=" * 78)
    print(f"阶段 {a.stage}   GPU={a.gpu}   开始 {datetime.datetime.now().isoformat()}")
    print(f"torch {torch.__version__} cuda {torch.version.cuda} "
          f"可见 GPU {torch.cuda.device_count()}")
    print("=" * 78, flush=True)

    train_f = f"{DATA}/{st['train']}"
    os.makedirs(st["out"], exist_ok=True)

    args = SftArguments(
        model=MODEL,
        dataset=train_f,
        val_dataset=f"{DATA}/sft_a_smoke_20.jsonl",   # 全是训练集样本，不碰 dev18 信息
        tuner_type="lora",
        target_modules=["all-linear"],                 # ← 必须是列表，见文件头
        freeze_vit=True,
        freeze_aligner=True,
        torch_dtype="bfloat16",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        max_length=2048,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=st["ga"],
        max_steps=st["max_steps"],
        gradient_checkpointing=True,
        seed=3407,
        data_seed=3407,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, st["max_steps"] // 2),
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=max(1, st["max_steps"] // 2),
        output_dir=st["out"],
        report_to=[],
    )

    print("\n=== 冻结配置 ===")
    for k in ("tuner_type", "target_modules", "freeze_vit", "freeze_aligner", "freeze_llm",
              "lora_rank", "lora_alpha", "lora_dropout", "learning_rate", "max_length",
              "per_device_train_batch_size", "gradient_accumulation_steps", "max_steps",
              "gradient_checkpointing", "seed", "torch_dtype"):
        print(f"  {k} = {getattr(args, k, '<缺失>')}")
    print(f"  有效 batch = {args.per_device_train_batch_size * args.gradient_accumulation_steps}"
          f"   数据文件 = {train_f}")
    print(f"  数据 sha256 = "
          f"{hashlib.sha256(open(train_f, 'rb').read()).hexdigest()}", flush=True)

    cfg = {
        "stage": a.stage, "gpu": a.gpu, "train_file": train_f,
        "train_sha256": hashlib.sha256(open(train_f, "rb").read()).hexdigest(),
        "max_steps": args.max_steps, "ga": st["ga"],
        "effective_batch": args.per_device_train_batch_size * st["ga"],
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, "lr": args.learning_rate,
        "max_length": args.max_length, "seed": args.seed,
        "target_modules_arg": list(args.target_modules),
        "freeze_vit": args.freeze_vit, "freeze_aligner": args.freeze_aligner,
        "started_at": datetime.datetime.now().isoformat(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    with open(f"{st['out']}/train_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"\n=== 开始训练：{args.max_steps} 步 ===", flush=True)
    t0 = datetime.datetime.now()
    sft_main(args)                                  # 训练实际入口
    dt = (datetime.datetime.now() - t0).total_seconds()
    print(f"\n=== 训练结束：{dt/60:.2f} 分钟 ===", flush=True)

    # ── 事后核验：从持久化的 adapter_config 读回 LoRA 目标正则 ──
    import glob
    cands = sorted(glob.glob(f"{st['out']}/**/adapter_config.json", recursive=True))
    print(f"找到 adapter_config.json: {len(cands)} 个")
    vis_hit = []
    for p in cands:
        ac = json.load(open(p, encoding="utf-8"))
        tm = ac.get("target_modules")
        print(f"  {p}\n    target_modules = {tm!r}")
        s = str(tm)
        if "visual" in s or "vision" in s:
            vis_hit.append(p)
    if cands and not vis_hit:
        print("  ✅ 全部 adapter 的 target_modules 都不含 visual/vision")
    elif vis_hit:
        print("  ❌ 有 adapter 把 visual 作为目标:", vis_hit)

    cfg.update({"seconds": dt, "finished_at": datetime.datetime.now().isoformat(),
                "adapter_configs": cands,
                "target_modules_effective": [json.load(open(p, encoding="utf-8")).get("target_modules")
                                             for p in cands],
                "visual_in_targets": bool(vis_hit)})
    with open(f"{st['out']}/train_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
