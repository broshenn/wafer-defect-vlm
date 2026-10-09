#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多字段 RL v2 首轮：M0 schema 适配 / M1 匹配继续 SFT / M2 固定规则 GRPO。

与既有 RL90 运行同一接线方式（swift 4.5.3 的 `SftArguments` / `RLHFArguments`），
区别只在：数据是新 schema、奖励换成 reward_v2、起点是合并了旧 D adapter 的基座。

纪律：
  · 父 shell 必须先 export CUDA_VISIBLE_DEVICES=<GPU-UUID>（在 import torch 之前）；
  · 断言 device_count==1 且 world_size==1，否则拒绝运行；
  · **首轮 2 步门槛**：跑完 2 步先做接线检查（图像前向、LoRA 梯度、奖励各分量、
    G 个候选同 ID），门槛不过就停，不硬跑完整程；
  · 保存全部原答与奖励审计。

用法：
  # 门槛（2 步）
  python v2_train.py --mode adapt --steps 2 --out /WS/v2run/out/adapt_probe
  # 正式
  python v2_train.py --mode adapt --steps 64 --out /WS/v2run/out/M0
  python v2_train.py --mode sft   --steps 60 --out /WS/v2run/out/M1
  python v2_train.py --mode rl    --steps 60 --out /WS/v2run/out/M2
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

BASE = os.environ["WAFER_V2_WORKSPACE"]
MERGED = os.environ["WAFER_D_MERGED"]
DATA = {
    "adapt": f"{BASE}/数据/adapt_sft.jsonl",
    "sft": f"{BASE}/数据/group60_sft4.jsonl",
    "rl": f"{BASE}/数据/group60_grpo.jsonl",
}


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["adapt", "sft", "rl"], required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MERGED)
    ap.add_argument("--start-adapter", default=None,
                    help="从已有 adapter 继续训练（M1/M2 从 M0 的 checkpoint 起）；"
                         "与 resume_only_model 一起用，避免意外恢复旧优化器状态")
    ap.add_argument("--model-type", default="qwen3_5")
    ap.add_argument("--template-type", default="qwen3_5")
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None)
    a = ap.parse_args()
    if a.mode in ("sft", "rl") and not a.start_adapter:
        print("!! sft/rl必须显式提供共同M0 start-adapter，拒绝从默认基座重新起跑")
        return 3

    print("=" * 78)
    print(f"v2 {a.mode}  {a.steps} 步  {datetime.datetime.now().isoformat()}")
    print("=" * 78)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"CUDA_VISIBLE_DEVICES = {cvd!r}（父进程设置）")
    if not cvd:
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
    if ws != 1:
        print("!! world_size != 1 —— 拒绝运行"); return 3

    data = DATA[a.mode]
    rows = [json.loads(l) for l in io.open(data, encoding="utf-8") if l.strip()]
    print(f"\n数据 {data}\n  {len(rows)} 行  sha256 {sha(data)}")
    print("  首行 images:", rows[0].get("images"))
    if a.mode in ("sft", "rl"):
        if a.mode == "sft":
            ok = all(len({rows[i + j]['sample_id'] for j in range(4)}) == 1
                     for i in range(0, len(rows), 4))
            print(f"  每 4 行同图: {ok}")
        else:
            print(f"  唯一图数: {len({r['sample_id'] for r in rows})}")
            print(f"  参考列: {sorted(k for k in rows[0] if k.startswith('gold_'))}")

    if a.start_adapter:
        import os as _os
        if not _os.path.isdir(a.start_adapter):
            print(f"!! --start-adapter 不存在：{a.start_adapter}"); return 3
        print(f"从已有 adapter 继续：{a.start_adapter}")
        print("  其中 adapter_model.safetensors sha256:",
              sha(_os.path.join(a.start_adapter, "adapter_model.safetensors")))

    os.makedirs(a.out, exist_ok=True)
    t0 = datetime.datetime.now()
    extra = ({"adapters": [a.start_adapter], "resume_only_model": True}
             if a.start_adapter else {})

    if a.mode == "rl":
        from swift import RLHFArguments
        from swift.pipelines import rlhf_main
        from reward_v2 import WaferMultiFieldReward
        reward = WaferMultiFieldReward()
        args = RLHFArguments(
            model=a.model, model_type=a.model_type, template=a.template_type,
            dataset=data, rlhf_type="grpo", reward_funcs=[reward],
            tuner_type="lora", target_modules=["all-linear"],
            freeze_vit=True, freeze_aligner=True, torch_dtype="bfloat16",
            lora_rank=16, lora_alpha=32, lora_dropout=0.05,
            use_vllm=False, num_generations=4, temperature=1.0, top_p=1.0,
            max_completion_length=a.max_completion_length,
            learning_rate=(a.lr if a.lr is not None else 1e-5),
            beta=0.04, epsilon=0.2,
            importance_sampling_level="token", num_iterations=1,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            max_steps=a.steps, max_length=2048, gradient_checkpointing=True,
            seed=3407, data_seed=3407, dataset_shuffle=False,
            train_dataloader_shuffle=False, logging_steps=1,
            save_strategy="steps", save_steps=max(1, a.steps // 2), save_total_limit=3,
            output_dir=a.out, report_to=[], **extra)
        print(f"\n  G={args.num_generations} temperature={args.temperature} "
              f"max_completion_length={args.max_completion_length} beta={args.beta} "
              f"lr={args.learning_rate} use_vllm={args.use_vllm}")
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
            model=a.model, model_type=a.model_type, template=a.template_type,
            dataset=data, val_dataset=data,
            tuner_type="lora", target_modules=["all-linear"],
            freeze_vit=True, freeze_aligner=True, torch_dtype="bfloat16",
            lora_rank=16, lora_alpha=32, lora_dropout=0.05,
            learning_rate=(a.lr if a.lr is not None else 1e-5),
            warmup_ratio=0.0, lr_scheduler_type="cosine",
            max_length=2048, per_device_train_batch_size=1,
            per_device_eval_batch_size=1, gradient_accumulation_steps=4,
            max_steps=a.steps, gradient_checkpointing=True,
            seed=3407, data_seed=3407, dataset_shuffle=False,
            train_dataloader_shuffle=False, dataloader_drop_last=False,
            logging_steps=1, save_strategy="steps",
            save_steps=max(1, a.steps // 2), save_total_limit=3,
            eval_strategy="no", output_dir=a.out, report_to=[], **extra)
        print(f"\n  LR={args.learning_rate} max_steps={a.steps} "
              f"eff_batch={args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print(f"\n=== 开始 SFT({a.mode}) {a.steps} 步 ===", flush=True)
        sft_main(args)

    dt = (datetime.datetime.now() - t0).total_seconds()
    peak = torch.cuda.max_memory_allocated(0) / 2 ** 30
    cks = sorted(glob.glob(f"{a.out}/**/checkpoint-*", recursive=True))
    final = [c for c in cks if c.endswith(f"checkpoint-{a.steps}")]
    cfg = {"mode": a.mode, "steps": a.steps, "model_start": a.model,
           "start_adapter": a.start_adapter, "data": data,
           "data_sha256": sha(data), "n_rows": len(rows),
           "seconds": dt, "device_minutes": round(dt / 60, 3),
           "peak_allocated_GiB": round(peak, 3),
           "checkpoints": cks, "final_checkpoint": final[0] if final else None,
           "selection": "末步预选", "cuda_visible_devices": cvd,
           "device": torch.cuda.get_device_name(0), "world_size": ws,
           "finished_at": datetime.datetime.now().isoformat()}
    with io.open(f"{a.out}/v2_config.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    print(f"\n结束 {dt/60:.2f} 分钟  峰值 {peak:.2f} GiB\ncheckpoints: {cks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
