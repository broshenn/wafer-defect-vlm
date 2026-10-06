#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""收尾：R2(GRPO) 与 R1(继续SFT) 从同一 R0 起，90 步全池覆盖。

R2  GRPO：每优化步 1 张唯一图 × G=4，90 步 = 全池 90 图各一次
R1  SFT ：每图正确答案连续 4 行，batch1/GA4 → 每步 4 行同图，90 步 = 全池各一次
两者**都关闭** dataset / train_dataloader 打乱，顺序来自同一 order90。

隔离：CUDA_VISIBLE_DEVICES 由父进程设置；断言 device_count=1 / world_size=1。
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch                                                          # noqa: E402
from reward_v1 import WaferClassReward                                # noqa: E402

R0 = "/WS/checkpoints/Qwen3.5-9B-R0merged"
D = "/WS/datasets/rl90"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rl", "sft"], required=True)
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-type", default="qwen3_5")
    ap.add_argument("--template-type", default="qwen3_5")
    a = ap.parse_args()

    print("=" * 78)
    print(f"R{'2' if a.mode=='rl' else '1'} ({a.mode})  {datetime.datetime.now().isoformat()}")
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

    data = f"{D}/rl90_grpo.server.jsonl" if a.mode == "rl" else f"{D}/rl90_sft_repeat4.server.jsonl"
    rows = [json.loads(l) for l in io.open(data, encoding="utf-8") if l.strip()]
    print(f"\n数据 {data}\n  {len(rows)} 行  sha256 {sha(data)}")
    if a.mode == "sft":
        ok = all(len({rows[i+j]['sample_id'] for j in range(4)}) == 1
                 for i in range(0, len(rows), 4))
        print(f"  每 4 行同图: {ok}")
    os.makedirs(a.out, exist_ok=True)
    t0 = datetime.datetime.now()

    if a.mode == "rl":
        from swift import RLHFArguments
        from swift.pipelines import rlhf_main
        reward = WaferClassReward()
        args = RLHFArguments(
            model=R0, model_type=a.model_type, template=a.template_type,
            dataset=data, rlhf_type="grpo", reward_funcs=[reward],
            tuner_type="lora", target_modules=["all-linear"],
            freeze_vit=True, freeze_aligner=True, torch_dtype="bfloat16",
            lora_rank=16, lora_alpha=32, lora_dropout=0.05,
            use_vllm=False, num_generations=4, temperature=1.0, top_p=1.0,
            max_completion_length=64, learning_rate=1e-5, beta=0.04, epsilon=0.2,
            importance_sampling_level="token", num_iterations=1,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            max_steps=a.steps, max_length=2048, gradient_checkpointing=True,
            seed=3407, data_seed=3407, dataset_shuffle=False,
            train_dataloader_shuffle=False, logging_steps=5,
            save_strategy="steps", save_steps=a.steps, save_total_limit=2,
            output_dir=a.out, report_to=[])
        print(f"\n  G={args.num_generations} temperature={args.temperature} "
              f"max_completion_length={args.max_completion_length} beta={args.beta} "
              f"epsilon={args.epsilon} use_vllm={args.use_vllm}")
        print(f"  dataset_shuffle={args.dataset_shuffle} "
              f"train_dataloader_shuffle={args.train_dataloader_shuffle}")
        print(f"\n=== 开始 GRPO {a.steps} 步 ===", flush=True)
        rlhf_main(args)
        with io.open(f"{a.out}/reward_audit.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in reward.audit:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"reward 审计 {len(reward.audit)} 条")
    else:
        from swift import SftArguments
        from swift.pipelines import sft_main
        args = SftArguments(
            model=R0, model_type=a.model_type, template=a.template_type,
            dataset=data, val_dataset=data,
            tuner_type="lora", target_modules=["all-linear"],
            freeze_vit=True, freeze_aligner=True, torch_dtype="bfloat16",
            lora_rank=16, lora_alpha=32, lora_dropout=0.05,
            learning_rate=1e-5, warmup_ratio=0.0, lr_scheduler_type="cosine",
            max_length=2048, per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=4, max_steps=a.steps,
            gradient_checkpointing=True, seed=3407, data_seed=3407,
            dataset_shuffle=False, train_dataloader_shuffle=False,
            dataloader_drop_last=False, logging_steps=5,
            save_strategy="steps", save_steps=a.steps, save_total_limit=2,
            eval_strategy="no", output_dir=a.out, report_to=[])
        print(f"\n  LR={args.learning_rate} cosine warmup={args.warmup_ratio} "
              f"dataset_shuffle={args.dataset_shuffle} "
              f"train_dataloader_shuffle={args.train_dataloader_shuffle}")
        print(f"\n=== 开始 SFT {a.steps} 步 ===", flush=True)
        sft_main(args)

    dt = (datetime.datetime.now() - t0).total_seconds()
    peak = torch.cuda.max_memory_allocated(0) / 2**30
    cks = sorted(glob.glob(f"{a.out}/**/checkpoint-*", recursive=True))
    final = [c for c in cks if c.endswith(f"checkpoint-{a.steps}")]
    cfg = {"mode": a.mode, "steps": a.steps, "model_R0": R0, "data": data,
           "data_sha256": sha(data), "n_rows": len(rows),
           "seconds": dt, "peak_allocated_GiB": round(peak, 3),
           "checkpoints": cks, "final_checkpoint": final[0] if final else None,
           "selection": "末步预选", "cuda_visible_devices": cvd,
           "device": torch.cuda.get_device_name(0), "world_size": ws,
           "finished_at": datetime.datetime.now().isoformat()}
    with io.open(f"{a.out}/rl90_config.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    print(f"\n结束 {dt/60:.2f} 分钟  峰值 {peak:.2f} GiB\ncheckpoints: {cks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
