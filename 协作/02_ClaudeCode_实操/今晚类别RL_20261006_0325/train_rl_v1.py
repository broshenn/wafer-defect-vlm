#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F2：类别奖励 GRPO 驱动（两步小测试 / 20 步正式试跑）。

配置按任务书 §3 冻结，**不扫参**：
  rlhf_type=grpo，use_vllm=False，token 级重要性采样，
  num_generations=4，temperature=1.0，top_p=1.0，max_completion_length=64，
  LR=1e-5，beta=0.04，epsilon(=clip)=0.2，num_iterations=1（每组只更新一轮），
  per_device_train_batch_size=1，gradient_accumulation_steps=4，
  LoRA 16/32/0.05，freeze_vit/aligner，BF16，max_length=2048，image token 256，
  gradient checkpointing，seed/data_seed=3407。

已核清的 batch 语义（`swift/rlhf_trainers/args_mixin.py::_init_generation_batch_params`）：
  global_batch_size     = pdb × world = 1
  steps_per_generation  = GA = 4
  generation_batch_size = 1 × 4 = 4      ← 必须被 num_generations 整除（4 % 4 == 0 ✓）
  采样器 batch_size = 4 // 4 = 1 张 prompt，mini_repeat_count = 4
  → **每个优化步 1 张图 × G=4**，GA 把 4 个 completion 累积成 1 次更新，即"每组只更新一轮"。

隔离：CUDA_VISIBLE_DEVICES 由父进程设置，断言只见 1 张卡、world_size=1。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import io
import json
import os
import sys

os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward_v1 import WaferClassReward                       # noqa: E402


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke2", "run20"], required=True)
    ap.add_argument("--model", default="/WS/checkpoints/Qwen3.5-9B-f0a1merged")
    ap.add_argument("--data", default="/WS/datasets/rl_pool90/rl_pool90_grpo.server.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="f2")
    ap.add_argument("--model-type", default="qwen3_5")
    ap.add_argument("--template-type", default="qwen3_5")
    a = ap.parse_args()

    steps = 2 if a.stage == "smoke2" else 20

    print("=" * 78)
    print(f"F2 GRPO {a.stage}  {datetime.datetime.now().isoformat()}")
    print("=" * 78)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"CUDA_VISIBLE_DEVICES = {cvd!r}（父进程设置）")
    if cvd is None:
        print("!! 父进程未设置 —— 拒绝运行"); return 3
    import torch
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

    from swift import RLHFArguments
    from swift.pipelines import rlhf_main

    os.makedirs(a.out, exist_ok=True)
    reward = WaferClassReward()

    args = RLHFArguments(
        model=a.model,
        model_type=a.model_type,
        template=a.template_type,
        dataset=a.data,
        rlhf_type="grpo",
        reward_funcs=[reward],          # **可调用对象**，不改 site-packages 的 orms 注册表
        tuner_type="lora", target_modules=["all-linear"],
        freeze_vit=True, freeze_aligner=True, torch_dtype="bfloat16",
        lora_rank=16, lora_alpha=32, lora_dropout=0.05,
        use_vllm=False,
        num_generations=4,
        temperature=1.0, top_p=1.0,
        max_completion_length=64,
        learning_rate=1e-5,
        beta=0.04,
        epsilon=0.2,
        importance_sampling_level="token",
        num_iterations=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=steps,
        max_length=2048,
        gradient_checkpointing=True,
        seed=3407, data_seed=3407,
        logging_steps=1,
        save_strategy="steps",
        save_steps=10 if steps >= 20 else 2,
        save_total_limit=4,
        output_dir=a.out,
        report_to=[],
    )

    print("\n=== 冻结配置实值 ===")
    for k in ("rlhf_type", "num_generations", "temperature", "top_p", "max_completion_length",
              "learning_rate", "beta", "epsilon", "importance_sampling_level", "num_iterations",
              "per_device_train_batch_size", "gradient_accumulation_steps", "max_steps",
              "use_vllm", "tuner_type", "target_modules", "freeze_vit", "freeze_aligner",
              "lora_rank", "lora_alpha", "lora_dropout", "max_length", "seed", "data_seed",
              "gradient_checkpointing", "scale_rewards", "loss_type"):
        print(f"  {k:32s} = {getattr(args, k, '<缺失>')}")
    print(f"  generation_batch_size            = {getattr(args, 'generation_batch_size', '?')}")
    print(f"  steps_per_generation             = {getattr(args, 'steps_per_generation', '?')}")
    print(f"  reward_funcs 类型                 = {type(args.reward_funcs[0]).__name__}")

    cfg = {
        "stage": a.stage, "tag": a.tag, "model": a.model, "data": a.data,
        "data_sha256": sha(a.data), "steps": steps,
        "cuda_visible_devices_parent": cvd, "visible_gpu_count": n, "world_size": ws,
        "device_name": torch.cuda.get_device_name(0),
        "config": {k: str(getattr(args, k, None)) for k in
                   ("rlhf_type", "num_generations", "temperature", "top_p",
                    "max_completion_length", "learning_rate", "beta", "epsilon",
                    "importance_sampling_level", "num_iterations",
                    "per_device_train_batch_size", "gradient_accumulation_steps",
                    "max_steps", "use_vllm", "tuner_type", "target_modules",
                    "freeze_vit", "freeze_aligner", "lora_rank", "lora_alpha",
                    "lora_dropout", "max_length", "seed", "data_seed",
                    "gradient_checkpointing", "scale_rewards", "loss_type")},
        "started_at": datetime.datetime.now().isoformat(),
    }
    with io.open(f"{a.out}/rl_config.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")

    print(f"\n=== 开始 GRPO：{steps} 步 ===", flush=True)
    t0 = datetime.datetime.now()
    rlhf_main(args)
    dt = (datetime.datetime.now() - t0).total_seconds()
    peak = torch.cuda.max_memory_allocated(0) / 2**30
    print(f"\n=== 结束 {dt/60:.2f} 分钟  峰值 {peak:.2f} GiB ===", flush=True)

    # reward 审计留痕：证明 gold 与 response 的对应关系
    audit_path = f"{a.out}/reward_audit.jsonl"
    with io.open(audit_path, "w", encoding="utf-8", newline="\n") as f:
        for r in reward.audit:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"reward 审计 {len(reward.audit)} 条 -> {audit_path}")

    import glob
    cks = sorted(glob.glob(f"{a.out}/**/checkpoint-*", recursive=True))
    cfg.update({"seconds": dt, "peak_allocated_GiB_own_process": round(peak, 3),
                "checkpoints": cks, "n_reward_audit": len(reward.audit),
                "finished_at": datetime.datetime.now().isoformat()})
    with io.open(f"{a.out}/rl_config.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
