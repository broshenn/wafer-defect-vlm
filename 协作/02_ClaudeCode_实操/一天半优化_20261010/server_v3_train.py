#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多字段 RL v3：M0_v3 六字段适配 / M1_v3 匹配 SFT / M2_v3 固定规则 GRPO。

接线沿用 Codex 已实跑验证的 `v2_train_fixed.py`（同一框架路径、同一运行时门槛）：
  · sft / rl **必须**显式给 `--start-adapter`，否则拒绝启动；
  · rl 同时设 `ref_adapters=[同一 M0_v3]`，由 runtime_guard_v2fix 断言
    "初始 policy 等于 M0_v3、参考名为 ref_adapter 且等于 M0_v3、参考前向确实发生、两步后参考未变"。
  · 奖励换成 v3 的复核修复版 `reward_v3.py`（剥完整空 think 包装 + 扇区分不给白分）。

用法（父 shell 必须先锁卡）：
  python v3_train.py --mode adapt --steps 128 --out .../out/M0_v3
  python v3_train.py --mode sft --steps 256 --start-adapter <M0_v3 ckpt> --out .../out/M1_v3
  python v3_train.py --mode rl  --steps 256 --start-adapter <M0_v3 ckpt> --out .../out/M2_v3
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

BASE = "/root/autodl-tmp/ws/v3run"
MERGED = "/root/autodl-tmp/ws/models/Qwen3.5-9B-Dmerged"
DATA = {
    "adapt": f"{BASE}/数据/adapt512_sft.jsonl",
    "sft": f"{BASE}/数据/group256_sft4.jsonl",
    "rl": f"{BASE}/数据/group256_grpo.jsonl",
}
EXPECT_STEPS = {"adapt": 128, "sft": 256, "rl": 256}


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
    ap.add_argument("--model-type", default="qwen3_5")
    ap.add_argument("--template-type", default="qwen3_5")
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--start-adapter", default=None)
    ap.add_argument("--probe", action="store_true",
                    help="门槛模式：允许 2 步；其余合同闸门（起点必须显式给出）照旧")
    a = ap.parse_args()

    print("=" * 78)
    print(f"v3 {a.mode}  {a.steps} 步  {datetime.datetime.now().isoformat()}")
    print("=" * 78)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        print("!! 父进程未锁卡 —— 拒绝运行"); return 3
    import torch
    torch.cuda.init()
    n = torch.cuda.device_count()
    print(f"CUDA_VISIBLE_DEVICES={cvd!r}  可见卡数={n}  device0={torch.cuda.get_device_name(0)}")
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
    if a.mode == "rl":
        print(f"  唯一图数: {len({r['sample_id'] for r in rows})}")
        print(f"  reference_json 在列: {'reference_json' in rows[0]}")

    # —— 合同硬闸门：缺起点、步数不符，一律拒绝，不靠打印 ——
    if a.mode in ("sft", "rl") and not a.start_adapter:
        raise RuntimeError("v3 合同：sft/rl 必须显式给 --start-adapter（M0_v3）")
    if not a.probe and a.steps != EXPECT_STEPS[a.mode]:
        raise RuntimeError(f"v3 合同固定 {a.mode} = {EXPECT_STEPS[a.mode]} 步，收到 {a.steps}")
    if a.probe:
        print(f"** 门槛模式：{a.steps} 步（合同步数 {EXPECT_STEPS[a.mode]} 只在正式跑强制）")
    if a.start_adapter:
        if not os.path.isdir(a.start_adapter):
            print(f"!! --start-adapter 不存在：{a.start_adapter}"); return 3
        print(f"起点 adapter：{a.start_adapter}")
        print("  adapter_model.safetensors sha256:",
              sha(os.path.join(a.start_adapter, "adapter_model.safetensors")))

    os.makedirs(a.out, exist_ok=True)
    t0 = datetime.datetime.now()
    extra = ({"adapters": [a.start_adapter], "resume_only_model": True}
             if a.start_adapter else {})
    if a.mode == "rl":
        extra["ref_adapters"] = [a.start_adapter]
    if a.start_adapter:
        os.environ["WAFER_START_ADAPTER"] = a.start_adapter
        os.environ["WAFER_FIX_MODE"] = a.mode
        os.environ["WAFER_GUARD_OUT"] = f"{a.out}/runtime_gate.json"
        from runtime_guard_v2fix import install
        install()
        print("运行时门槛已装载（初始权重/参考策略断言）")

    if a.mode == "rl":
        from swift import RLHFArguments
        from swift.pipelines import rlhf_main
        from reward_v3 import WaferV3Reward
        reward = WaferV3Reward()
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
        print(f"\n  G={args.num_generations} T={args.temperature} beta={args.beta} "
              f"lr={args.learning_rate} max_completion_length={args.max_completion_length}")
        print(f"  adapters={args.adapters} ref_adapters={getattr(args,'ref_adapters',None)}")
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
              f"adapters={args.adapters} resume_only_model={args.resume_only_model}")
        print(f"\n=== 开始 SFT({a.mode}) {a.steps} 步 ===", flush=True)
        sft_main(args)

    dt = (datetime.datetime.now() - t0).total_seconds()
    peak = torch.cuda.max_memory_allocated(0) / 2 ** 30
    cks = sorted(glob.glob(f"{a.out}/**/checkpoint-*", recursive=True))
    final = [c for c in cks if c.endswith(f"checkpoint-{a.steps}")]
    cfg = {"mode": a.mode, "steps": a.steps, "model_start": a.model,
           "start_adapter": a.start_adapter, "data": data, "data_sha256": sha(data),
           "n_rows": len(rows), "seconds": dt, "device_minutes": round(dt / 60, 3),
           "peak_allocated_GiB": round(peak, 3), "checkpoints": cks,
           "final_checkpoint": final[0] if final else None,
           "selection": "末步预选（不挑最好 checkpoint）",
           "cuda_visible_devices": cvd, "device": torch.cuda.get_device_name(0),
           "world_size": ws, "finished_at": datetime.datetime.now().isoformat()}
    # 把训练器**实际解析后**的实参也存下来（不是意图）
    try:
        dump = {k: (v if isinstance(v, (int, float, str, bool, type(None), list))
                    else str(v)) for k, v in vars(args).items()
                if k in ("adapters", "ref_adapters", "resume_only_model", "learning_rate",
                         "max_steps", "num_generations", "temperature", "beta",
                         "max_completion_length", "use_vllm", "dataset_shuffle",
                         "train_dataloader_shuffle", "lora_rank", "lora_alpha",
                         "freeze_vit", "freeze_aligner", "seed", "data_seed")}
        cfg["trainer_actual_args"] = dump
    except Exception as exc:  # 不让它掩盖主流程
        cfg["trainer_actual_args_error"] = f"{type(exc).__name__}: {exc}"
    with io.open(f"{a.out}/v3_config.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n")
    print(f"\n结束 {dt/60:.2f} 分钟  峰值 {peak:.2f} GiB\ncheckpoints: {cks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
