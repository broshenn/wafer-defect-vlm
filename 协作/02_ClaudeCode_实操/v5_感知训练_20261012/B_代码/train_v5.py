#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5 训练脚本（B 卡 3）。模板：`day-half/server_v3_train.py`（本地复制修改）。
用法：python train_v5.py --config configs/<线名>.json --out <输出目录> [--probe-steps N] [--dry-run]
- 在 import torch/swift 之前读 config 并设 IMAGE_MAX_TOKEN_NUM / WAFER_REWARD_MODE。
- 模式：sft_base（不加载 adapter）/ sft_cont（R-SFT：adapters=[起点] + guard）/ rl（+ ref_adapters=[起点]）。
- start_adapter 可为字面路径或 "FROM_WINNER"（读 /WS/v5/night/winner.json 的 checkpoint 字段，
  并从赢家 config 继承 freeze_vit/freeze_aligner/image_max_token_num）。
- --dry-run 不 import torch/swift，只打印解析后的参数字典并退出 0。
- 结束写 v5_config.json（训练器实际参数 + 全部指纹）。
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def resolve_config(a):
    """在 import torch 之前完成：读 config、解析起点与继承项、返回 (params, meta)。
    dry-run 时 FROM_WINNER 不读 winner.json（本机没有），用占位符。"""
    cfg = json.loads(io.open(a.config, encoding="utf-8").read())
    mode = cfg["mode"]
    start = cfg.get("start_adapter")
    winner = None
    if start == "FROM_WINNER":
        if a.dry_run:
            start = "<FROM_WINNER:winner.json#checkpoint>"
        else:
            winner_path = cfg.get("winner_json", "/WS/v5/night/winner.json")
            winner = json.loads(io.open(winner_path, encoding="utf-8").read())
            start = winner["checkpoint"]
            wcfg = json.loads(io.open(os.path.join(HERE, "configs", f"{winner['winner']}.json"),
                                      encoding="utf-8").read())
            cfg["freeze_vit"] = wcfg["freeze_vit"]
            cfg["freeze_aligner"] = wcfg["freeze_aligner"]
            cfg["image_max_token_num"] = wcfg["image_max_token_num"]
    # 环境变量必须在 torch 之前设（dry-run 不启动训练，跳过）
    if not a.dry_run:
        if cfg["image_max_token_num"] is None:
            raise SystemExit("image_max_token_num 为 null：应来自 FROM_WINNER 继承或配置里显式给出")
        os.environ["IMAGE_MAX_TOKEN_NUM"] = str(cfg["image_max_token_num"])
        if mode == "rl":
            os.environ["WAFER_REWARD_MODE"] = cfg["reward_mode"]
    steps = a.probe_steps if a.probe_steps else cfg["max_steps"]
    if a.probe_steps and "probe" not in a.out:
        raise SystemExit("--probe-steps 的输出目录必须含 'probe'（与正式目录不同）")
    save_steps = a.probe_steps if a.probe_steps else cfg["save_steps"]
    params = dict(
        model=cfg["model"], model_type=cfg["model_type"], template=cfg["template"],
        dataset=cfg["data"], tuner_type="lora", target_modules=cfg["target_modules"],
        freeze_vit=cfg["freeze_vit"], freeze_aligner=cfg["freeze_aligner"],
        torch_dtype="bfloat16", lora_rank=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"], learning_rate=cfg["lr"],
        max_steps=steps, max_length=cfg["max_length"], gradient_checkpointing=True,
        seed=cfg["seed"], data_seed=cfg["data_seed"], dataset_shuffle=False,
        train_dataloader_shuffle=False, logging_steps=1,
        save_strategy="steps", save_steps=max(1, save_steps),
        save_total_limit=cfg.get("save_total_limit", 4),
        output_dir=a.out, report_to=[])
    extra = {}
    if start:
        if not a.dry_run and not os.path.isdir(start):
            raise SystemExit(f"起点 adapter 不存在：{start}")
        extra = {"adapters": [start], "resume_only_model": True}
    if mode == "rl":
        extra["ref_adapters"] = [start]
        params.update(rlhf_type="grpo", use_vllm=False,
                      num_generations=cfg["num_generations"], temperature=cfg["temperature"],
                      top_p=cfg["top_p"], beta=cfg["beta"], epsilon=cfg["epsilon"],
                      importance_sampling_level="token", num_iterations=1,
                      max_completion_length=cfg["max_completion_length"],
                      per_device_train_batch_size=cfg["per_device_train_batch_size"],
                      gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
                      warmup_ratio=0, lr_scheduler_type="cosine")
        dyn = "checked-at-runtime" if a.dry_run else "pending"
    else:
        params.update(val_dataset=cfg["data"], per_device_eval_batch_size=1,
                      per_device_train_batch_size=cfg["per_device_train_batch_size"],
                      gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
                      warmup_ratio=cfg["warmup_ratio"], lr_scheduler_type="cosine",
                      dataloader_drop_last=False, eval_strategy="no")
        dyn = "n/a(sft)"
    params.update(extra)
    meta = {"line": cfg["line"], "mode": mode, "start_adapter": start,
            "reward_mode": cfg.get("reward_mode"), "dynamic_sample": dyn,
            "config_sha256": sha(a.config), "config": cfg, "winner": winner}
    return params, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-steps", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    print("=" * 78)
    print(f"v5 train {datetime.datetime.now().isoformat()}")
    print("=" * 78)
    params, meta = resolve_config(a)
    if a.dry_run:
        printable = {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))
                     for k, v in params.items()}
        print(json.dumps({"params": printable, "meta": {k: v for k, v in meta.items() if k != "config"}},
                         ensure_ascii=False, indent=1))
        return 0

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        print("!! 父进程未锁卡 —— 拒绝运行")
        return 3
    import torch
    torch.cuda.init()
    n = torch.cuda.device_count()
    print(f"CUDA_VISIBLE_DEVICES={cvd!r} 可见卡数={n} device0={torch.cuda.get_device_name(0)}")
    if n != 1:
        print("!! 不是 1 张卡 —— 拒绝运行")
        return 3
    try:
        import torch.distributed as dist
        ws = dist.get_world_size() if dist.is_initialized() else 1
    except Exception:
        ws = 1
    if ws != 1:
        print("!! world_size != 1 —— 拒绝运行")
        return 3

    data = params["dataset"]
    rows = [json.loads(l) for l in io.open(data, encoding="utf-8") if l.strip()]
    print(f"\n数据 {data}\n  {len(rows)} 行 sha256 {sha(data)}")
    print("  首行 images:", rows[0].get("images"))
    if meta["mode"] == "rl":
        print(f"  唯一图数 {len({r['sample_id'] for r in rows})} reference_json 在列 "
              f"{'reference_json' in rows[0]} sample_id 在列 {'sample_id' in rows[0]}")

    os.makedirs(a.out, exist_ok=True)
    t0 = datetime.datetime.now()

    if meta["start_adapter"]:
        # 起点张量门槛（WAFER_EXPECT_M0_SHA 自动取起点文件 SHA）
        start = meta["start_adapter"]
        os.environ["WAFER_START_ADAPTER"] = start
        os.environ["WAFER_FIX_MODE"] = "rl" if meta["mode"] == "rl" else "sft"
        os.environ["WAFER_GUARD_OUT"] = f"{a.out}/runtime_gate.json"
        os.environ["WAFER_EXPECT_M0_SHA"] = sha(os.path.join(start, "adapter_model.safetensors"))
        if not params["freeze_vit"]:
            os.environ["WAFER_ALLOW_VISUAL_TRAINABLE"] = "1"
        from runtime_guard_v3 import install
        install()
        print("运行时门槛已装载；起点 adapter_model.safetensors sha256 =",
              os.environ["WAFER_EXPECT_M0_SHA"])

    if meta["mode"] == "rl":
        # 动态采样：不猜参数是否存在，运行时检查 RLHFArguments 字段（合同 6.3）
        import dataclasses
        from swift import RLHFArguments
        if "dynamic_sample" in {f.name for f in dataclasses.fields(RLHFArguments)}:
            params["dynamic_sample"] = True
            params["max_resample_times"] = 3
            meta["dynamic_sample"] = "enabled"
        else:
            meta["dynamic_sample"] = "unsupported"
        reward = None
        from swift.pipelines import rlhf_main
        from reward_v5 import WaferV5Reward
        reward = WaferV5Reward()
        args = RLHFArguments(**params, reward_funcs=[reward])
        print(f"\n  G={args.num_generations} T={args.temperature} beta={args.beta} lr={args.learning_rate} "
              f"max_completion_length={args.max_completion_length}")
        print(f"  adapters={args.adapters} ref_adapters={getattr(args, 'ref_adapters', None)}")
        print(f"\n=== 开始 GRPO {params['max_steps']} 步 ===", flush=True)
        rlhf_main(args)
        with io.open(f"{a.out}/reward_audit.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in reward.audit:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"reward 审计 {len(reward.audit)} 条")
    else:
        from swift import SftArguments
        from swift.pipelines import sft_main
        args = SftArguments(**params)
        print(f"\n  LR={args.learning_rate} max_steps={args.max_steps} adapters={args.adapters} "
              f"resume_only_model={getattr(args, 'resume_only_model', None)}")
        print(f"\n=== 开始 SFT({meta['mode']}) {params['max_steps']} 步 ===", flush=True)
        sft_main(args)

    dt = (datetime.datetime.now() - t0).total_seconds()
    peak = torch.cuda.max_memory_allocated(0) / 2 ** 30
    cks = sorted(glob.glob(f"{a.out}/**/checkpoint-*", recursive=True))
    final = [c for c in cks if c.endswith(f"checkpoint-{params['max_steps']}")]
    cfg_out = {"line": meta["line"], "mode": meta["mode"], "steps": params["max_steps"],
               "model_start": params["model"], "start_adapter": meta["start_adapter"],
               "start_adapter_sha256": (sha(os.path.join(meta["start_adapter"], "adapter_model.safetensors"))
                                        if meta["start_adapter"] else None),
               "data": data, "data_sha256": sha(data), "n_rows": len(rows),
               "reward_mode": meta["reward_mode"], "dynamic_sample": meta["dynamic_sample"],
               "config_sha256": meta["config_sha256"], "seconds": dt,
               "device_minutes": round(dt / 60, 3), "peak_allocated_GiB": round(peak, 3),
               "checkpoints": cks, "final_checkpoint": final[0] if final else None,
               "selection": "末步预选（不挑最好 checkpoint）",
               "cuda_visible_devices": cvd, "device": torch.cuda.get_device_name(0),
               "world_size": ws, "finished_at": datetime.datetime.now().isoformat()}
    try:
        dump = {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))
                for k, v in vars(args).items()
                if k in ("adapters", "ref_adapters", "resume_only_model", "learning_rate",
                         "max_steps", "num_generations", "temperature", "beta",
                         "max_completion_length", "use_vllm", "dataset_shuffle",
                         "train_dataloader_shuffle", "lora_rank", "lora_alpha",
                         "freeze_vit", "freeze_aligner", "seed", "data_seed",
                         "dynamic_sample", "max_resample_times", "per_device_train_batch_size",
                         "gradient_accumulation_steps", "importance_sampling_level")}
        cfg_out["trainer_actual_args"] = dump
    except Exception as exc:
        cfg_out["trainer_actual_args_error"] = f"{type(exc).__name__}: {exc}"
    with io.open(f"{a.out}/v5_config.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cfg_out, ensure_ascii=False, indent=1) + "\n")
    print(f"\n结束 {dt / 60:.2f} 分钟 峰值 {peak:.2f} GiB\ncheckpoints: {cks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
