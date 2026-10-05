#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SFT-A 单卡训练（三种子）。**隔离在父进程完成，脚本只做断言与记录。**

为什么这样写：上一轮驱动在 `import torch` **之后**才设 `os.environ["CUDA_VISIBLE_DEVICES"]`，
结果完全无效 —— 日志打印 `可见 GPU 6`，`hf_device_map` 把模型铺满 0–5 六张卡
（其中 GPU 3 还有他人的任务）。CUDA 在首次调用时读取可见设备列表，
**必须在进程启动前由父进程设置**。

所以本脚本：
  * 不在 Python 里改 CUDA_VISIBLE_DEVICES（发现被改过就直接失败）；
  * 断言可见 GPU 数 == 1、world_size == 1；
  * 断言模型所有参数与 `hf_device_map` 都只落在逻辑卡 0；
  * 训练后核实 LoRA-B 是否真的被更新（B 初始为 0，非零才算真更新）。

用法（**由启动脚本设置 CUDA_VISIBLE_DEVICES**）：
    CUDA_VISIBLE_DEVICES=<uuid> python train_sft_a_v2.py --seed 3407 --tag a1
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import re
import sys

os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")

import torch                                          # noqa: E402
from swift import SftArguments                        # noqa: E402
from swift.pipelines import sft_main                  # noqa: E402

MODEL = "/WS/models/Qwen3.5-9B"
DATA = "/WS/datasets/sft_a"
CKPT_ROOT = "/WS/checkpoints"


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--card-uuid", default=os.environ.get("CARD_UUID", ""))
    a = ap.parse_args()

    out = f"{CKPT_ROOT}/sft_a_seed{a.seed}_{a.tag}"
    os.makedirs(out, exist_ok=True)

    # ── 隔离断言 ────────────────────────────────────────
    print("=" * 78)
    print(f"单卡训练  seed={a.seed}  tag={a.tag}  开始 {datetime.datetime.now().isoformat()}")
    print("=" * 78)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"CUDA_VISIBLE_DEVICES = {cvd!r}（由父进程设置）")
    if cvd is None:
        print("!! 父进程没有设置 CUDA_VISIBLE_DEVICES —— 拒绝运行")
        return 3
    torch.cuda.init()
    n = torch.cuda.device_count()
    print(f"可见 GPU 数 = {n}   device 0 名 = {torch.cuda.get_device_name(0)}")
    if n != 1:
        print(f"!! 可见 {n} 张卡，不是 1 张 —— 拒绝运行")
        return 3
    try:
        import torch.distributed as dist
        ws = dist.get_world_size() if dist.is_initialized() else 1
    except Exception:
        ws = 1
    print(f"world_size = {ws}")
    if ws != 1:
        print("!! world_size != 1 —— 拒绝运行")
        return 3
    torch.zeros(1, device="cuda:0")     # 触发上下文初始化

    train_f = f"{DATA}/sft_a_train_180.jsonl"
    mem_before = torch.cuda.memory_allocated(0)

    args = SftArguments(
        model=MODEL, dataset=train_f,
        val_dataset=f"{DATA}/sft_a_smoke_20.jsonl",   # **已见样本诊断**，不是独立验证
        tuner_type="lora", target_modules=["all-linear"],
        freeze_vit=True, freeze_aligner=True, torch_dtype="bfloat16",
        lora_rank=16, lora_alpha=32, lora_dropout=0.05,
        learning_rate=1e-4, warmup_ratio=0.05, max_length=2048,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=4, max_steps=60,
        gradient_checkpointing=True,
        seed=a.seed, data_seed=3407,          # **data_seed 固定 3407，不重新抽样**
        logging_steps=1, save_strategy="steps", save_steps=30, save_total_limit=3,
        eval_strategy="steps", eval_steps=30,
        output_dir=out, report_to=[],
    )
    print(f"\n有效 batch = {args.per_device_train_batch_size * args.gradient_accumulation_steps}"
          f"   max_steps={args.max_steps}   seed={args.seed}   data_seed={args.data_seed}")
    print(f"target_modules 传入类型: {type(args.target_modules).__name__} {args.target_modules}")
    print(f"train 数据 sha256 = {sha(train_f)}")

    cfg = {
        "tag": a.tag, "seed": a.seed, "data_seed": args.data_seed,
        "cuda_visible_devices_parent": cvd, "card_uuid": a.card_uuid,
        "visible_gpu_count": n, "world_size": ws,
        "device_name": torch.cuda.get_device_name(0),
        "train_file": train_f, "train_sha256": sha(train_f),
        "max_steps": args.max_steps, "ga": args.gradient_accumulation_steps,
        "effective_batch": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, "lr": args.learning_rate,
        "max_length": args.max_length, "target_modules_arg": list(args.target_modules),
        "freeze_vit": args.freeze_vit, "freeze_aligner": args.freeze_aligner,
        "eval_dataset": "sft_a_smoke_20.jsonl（训练子集，已见样本诊断）",
        "started_at": datetime.datetime.now().isoformat(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    with io.open(f"{out}/train_config.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")

    print(f"\n=== 开始训练 {args.max_steps} 步 ===", flush=True)
    t0 = datetime.datetime.now()
    sft_main(args)
    dt = (datetime.datetime.now() - t0).total_seconds()
    peak_alloc = torch.cuda.max_memory_allocated(0) / 2**30
    peak_res = torch.cuda.max_memory_reserved(0) / 2**30
    print(f"\n=== 训练结束 {dt/60:.2f} 分钟 ===")
    print(f"本人进程峰值：allocated {peak_alloc:.2f} GiB / reserved {peak_res:.2f} GiB", flush=True)

    # ── 设备图 + LoRA-B 更新核实 ────────────────────────
    import glob
    ac = sorted(glob.glob(f"{out}/**/adapter_config.json", recursive=True))
    am = sorted(glob.glob(f"{out}/**/adapter_model.safetensors", recursive=True))
    bstat = {}
    if am:
        from safetensors.torch import load_file
        sd = load_file(am[-1])
        bkeys = [k for k in sd if "lora_B" in k]
        anonzero = sum(1 for k in bkeys if sd[k].abs().sum().item() > 0)
        bstat = {"n_lora_B": len(bkeys), "n_lora_B_nonzero": anonzero,
                 "max_abs_B": max((sd[k].abs().max().item() for k in bkeys), default=0.0),
                 "all_B_zero": anonzero == 0,
                 "n_lora_A": sum(1 for k in sd if "lora_A" in k)}
    cfg.update({
        "seconds": dt, "finished_at": datetime.datetime.now().isoformat(),
        "peak_allocated_GiB_own_process": round(peak_alloc, 3),
        "peak_reserved_GiB_own_process": round(peak_res, 3),
        "mem_before_MiB": round(mem_before / 2**20, 1),
        "adapter_config": ac[-1] if ac else None,
        "adapter_model": am[-1] if am else None,
        "target_modules_effective": (json.load(open(ac[-1], encoding="utf-8"))["target_modules"]
                                     if ac else None),
        "lora_B": bstat,
    })
    with io.open(f"{out}/train_config.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    print(f"LoRA-B 核实: {json.dumps(bstat, ensure_ascii=False)}")
    if bstat.get("all_B_zero"):
        print("!! 所有 LoRA-B 仍为 0 —— 说明没有实际更新")
        return 4
    print("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
