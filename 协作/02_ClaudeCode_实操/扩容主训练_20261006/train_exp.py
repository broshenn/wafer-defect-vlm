#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩容主训练（B 部分）：从**未经本项目训练的原 Qwen3.5-9B 基座**起，新 LoRA。

配置按任务书 B 冻结：
  LR 1e-4、warmup_ratio 0.05、cosine、LoRA 16/32/0.05、target_modules **显式列表**、
  freeze_vit/aligner、BF16、image token 256、max_length 2048、
  gradient_checkpointing、per_device_train_batch_size=1 / GA=4、data_seed 3407。

1 epoch 实际覆盖：步数 = 规模 / 4。步数不足则记 partial，不把未呈现的图说已学习。

隔离：CUDA_VISIBLE_DEVICES 由父进程设置，断言 device_count=1 / world_size=1。
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import io
import json
import os
import sys

os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")

import torch                                              # noqa: E402
from swift import SftArguments                            # noqa: E402
from swift.pipelines import sft_main                      # noqa: E402

BASE = "/WS/models/Qwen3.5-9B"
D = "/WS/datasets"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True, choices=[180, 360, 720, 1440])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=0, help="0 = 自动 = level/4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-type", default="qwen3_5")
    ap.add_argument("--template-type", default="qwen3_5")
    a = ap.parse_args()
    steps = a.steps or a.level // 4

    print("=" * 78)
    print(f"扩容训练 L{a.level} seed{a.seed} steps={steps}  "
          f"{datetime.datetime.now().isoformat()}")
    print("=" * 78)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"CUDA_VISIBLE_DEVICES = {cvd!r}（父进程设置）")
    if cvd is None:
        print("!! 父进程未设置 —— 拒绝运行"); return 3
    torch.cuda.init()
    n = torch.cuda.device_count()
    print(f"可见 GPU 数 = {n}   device 0 = {torch.cuda.get_device_name(0)}")
    if n != 1:
        print("!! 不是 1 张卡 —— 拒绝运行"); return 3
    try:
        import torch.distributed as dist
        ws = dist.get_world_size() if dist.is_initialized() else 1
    except Exception:
        ws = 1
    print(f"world_size = {ws}")
    if ws != 1:
        print("!! world_size != 1 —— 拒绝运行"); return 3

    data = f"{D}/expand_v2_sft_scale_{a.level}_v2.server.jsonl"
    rows = [json.loads(l) for l in io.open(data, encoding="utf-8") if l.strip()]
    n_full = len(rows)               # 处理后的实际有效行数
    covered = steps * 4              # 实际呈现的样本次数（有效 batch 4）
    print(f"\n数据 {data}")
    print(f"  {n_full} 行  sha256 {sha(data)}")
    print(f"  名义步数 {steps}  →  呈现 {covered} 次  "
          f"覆盖 {covered/n_full:.3f} epoch  "
          f"{'（完整 1 epoch）' if covered >= n_full else '**partial，未覆盖全部**'}")

    os.makedirs(a.out, exist_ok=True)
    args = SftArguments(
        model=BASE, model_type=a.model_type, template=a.template_type,
        dataset=data, val_dataset=data,
        tuner_type="lora", target_modules=["all-linear"],   # 列表，冻结才生效
        freeze_vit=True, freeze_aligner=True, torch_dtype="bfloat16",
        lora_rank=16, lora_alpha=32, lora_dropout=0.05,
        learning_rate=1e-4, warmup_ratio=0.05, lr_scheduler_type="cosine",
        max_length=2048,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=4, max_steps=steps,
        gradient_checkpointing=True, seed=a.seed, data_seed=3407,
        dataset_shuffle=True, train_dataloader_shuffle=True,   # 1 epoch 覆盖，打乱是通行做法
        logging_steps=max(1, steps // 20), save_strategy="steps",
        save_steps=max(1, steps // 2), save_total_limit=3,
        eval_strategy="no", output_dir=a.out, report_to=[],
    )
    print("\n=== 冻结配置 ===")
    for k in ("learning_rate", "warmup_ratio", "lr_scheduler_type", "max_steps",
              "gradient_accumulation_steps", "per_device_train_batch_size",
              "tuner_type", "target_modules", "freeze_vit", "freeze_aligner",
              "lora_rank", "lora_alpha", "lora_dropout", "max_length", "seed",
              "data_seed", "gradient_checkpointing", "torch_dtype"):
        print(f"  {k:30s} = {getattr(args, k, '<缺失>')}")

    cfg = {
        "level": a.level, "seed": a.seed, "steps": steps, "epochs_covered": round(covered / n_full, 4),
        "model": BASE, "model_note": "未经本项目训练的原基座（不是 F0/F2）",
        "data": data, "data_sha256": sha(data), "n_rows": n_full,
        "cuda_visible_devices_parent": cvd, "visible_gpu_count": n, "world_size": ws,
        "device": torch.cuda.get_device_name(0),
        "config": {k: str(getattr(args, k, None)) for k in
                   ("learning_rate", "warmup_ratio", "lr_scheduler_type", "max_steps",
                    "gradient_accumulation_steps", "per_device_train_batch_size",
                    "tuner_type", "target_modules", "freeze_vit", "freeze_aligner",
                    "lora_rank", "lora_alpha", "lora_dropout", "max_length", "seed",
                    "data_seed", "gradient_checkpointing")},
        "started_at": datetime.datetime.now().isoformat(),
    }
    with io.open(f"{a.out}/exp_config.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")

    print(f"\n=== 开始训练 {steps} 步 ===", flush=True)
    t0 = datetime.datetime.now()
    sft_main(args)
    dt = (datetime.datetime.now() - t0).total_seconds()
    peak = torch.cuda.max_memory_allocated(0) / 2**30
    print(f"\n结束 {dt/60:.2f} 分钟  峰值 {peak:.2f} GiB", flush=True)

    cks = sorted(glob.glob(f"{a.out}/**/checkpoint-*", recursive=True))
    # 末步预选（不按 val 选最佳）
    final = [c for c in cks if c.endswith(f"checkpoint-{steps}")]
    cfg.update({"seconds": dt, "peak_allocated_GiB": round(peak, 3),
                "checkpoints": cks, "final_checkpoint": final[0] if final else None,
                "selection": "末步预选（不按 val 挑最佳）",
                "finished_at": datetime.datetime.now().isoformat()})
    with io.open(f"{a.out}/exp_config.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    print(f"checkpoints: {cks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
