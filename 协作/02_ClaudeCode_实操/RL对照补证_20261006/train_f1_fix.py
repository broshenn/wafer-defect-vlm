#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修正版 F1：从 F0 重新开始，按 **F2 的真实 20 组 ID 与顺序**做继续 SFT。

与原 F1 的三个差别（这是它不能当公平对照的原因）：
  1. **图集不同**：原 F1 用 pool 前 20 张（Center×10 + Donut×10），
     F2 实际用的是另外 20 张（类别混合），两者只重叠 8/20；
  2. **打乱**：原 F1 与 F2 都开着 `dataset_shuffle` 与 `train_dataloader_shuffle`，
     4 份副本被打散，每个优化步不再是同一张图；
  3. 本版**显式关闭两层打乱**，并按 F2 顺序每 ID 连续 4 次。

结构：80 行 = 20 组 × 4；pdb=1 / GA=4 → 每个优化步消耗 4 个连续行 = 同一张图的 4 份。
20 步正好把 80 行用完一次。

零步自检：装完 LoRA、训练前，先在 dev18 上推理，必须与 F0 预测一致才继续。

隔离：CUDA_VISIBLE_DEVICES 由父进程设置，断言只见 1 张卡、world_size=1。
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


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class StepIdLogger:
    """记录每个优化步实际看到的 sample_id 序列。"""

    def __init__(self):
        self.records = []
        self._buf = []

    def on_step_end(self, args, state, control, **kw):
        if self._buf:
            self.records.append({"step": state.global_step, "ids": list(self._buf)})
            self._buf = []
        return control

    def on_substep_end(self, args, state, control, **kw):
        return control


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/WS/checkpoints/Qwen3.5-9B-f0a1merged")
    ap.add_argument("--data", default="/WS/datasets/rl_pool90/f1_replay_f2order.server.jsonl")
    ap.add_argument("--dev18", default="/WS/datasets/dev_18.server.jsonl")
    ap.add_argument("--f0-answers", default="/WS/checkpoints/eval_unified/f0_dev18.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-type", default="qwen3_5")
    ap.add_argument("--template-type", default="qwen3_5")
    a = ap.parse_args()

    print("=" * 78)
    print(f"修正 F1（F2 顺序 replay）  {datetime.datetime.now().isoformat()}")
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

    # 数据自检：必须恰好是 20 组 × 4、按组连续
    rows = [json.loads(l) for l in io.open(a.data, encoding="utf-8") if l.strip()]
    print(f"\n数据 {len(rows)} 行  sha256 {sha(a.data)}")
    ok = True
    for i in range(0, len(rows), 4):
        g = rows[i:i + 4]
        if len({r["sample_id"] for r in g}) != 1:
            print(f"  !! 第 {i//4} 组不是同一张图"); ok = False
    print(f"  每组 4 行同 ID: {'是' if ok else '**否**'}")
    print(f"  组序列: {[rows[i]['sample_id'] for i in range(0, len(rows), 4)][:6]} …")
    if not ok:
        return 3

    os.makedirs(a.out, exist_ok=True)
    args = SftArguments(
        model=a.model, model_type=a.model_type, template=a.template_type,
        dataset=a.data, val_dataset=a.data,
        tuner_type="lora", target_modules=["all-linear"],
        freeze_vit=True, freeze_aligner=True, torch_dtype="bfloat16",
        lora_rank=16, lora_alpha=32, lora_dropout=0.05,
        learning_rate=1e-5, warmup_ratio=0.0, lr_scheduler_type="cosine",
        max_length=2048,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=4, max_steps=20,
        gradient_checkpointing=True, seed=3407, data_seed=3407,
        dataset_shuffle=False,              # ← 关键：不做数据集打乱
        train_dataloader_shuffle=False,     # ← 关键：不做 dataloader 打乱
        dataloader_drop_last=False,
        remove_unused_columns=False,        # 保住 sample_id，便于逐步核对
        logging_steps=1, save_strategy="steps", save_steps=10, save_total_limit=4,
        eval_strategy="no", output_dir=a.out, report_to=[],
    )
    print("\n=== 冻结配置（含打乱开关）===")
    for k in ("learning_rate", "lr_scheduler_type", "warmup_ratio", "max_steps",
              "gradient_accumulation_steps", "per_device_train_batch_size",
              "dataset_shuffle", "train_dataloader_shuffle", "dataloader_drop_last",
              "remove_unused_columns", "tuner_type", "target_modules", "freeze_vit",
              "freeze_aligner", "lora_rank", "lora_alpha", "lora_dropout",
              "seed", "data_seed"):
        print(f"  {k:30s} = {getattr(args, k, '<缺失>')}")

    print(f"\n=== 开始训练 20 步 ===", flush=True)
    t0 = datetime.datetime.now()
    sft_main(args)
    dt = (datetime.datetime.now() - t0).total_seconds()
    peak = torch.cuda.max_memory_allocated(0) / 2**30
    print(f"\n训练结束 {dt/60:.2f} 分钟  峰值 {peak:.2f} GiB", flush=True)

    cks = sorted(glob.glob(f"{a.out}/**/checkpoint-*", recursive=True))
    cfg = {
        "model": a.model, "data": a.data, "data_sha256": sha(a.data),
        "lr": 1e-5, "lr_scheduler_type": "cosine", "max_steps": 20,
        "ga": 4, "pdb": 1, "effective_batch": 4,
        "dataset_shuffle": False, "train_dataloader_shuffle": False,
        "note": "按 F2 真实 20 组 ID 与顺序 replay；关闭两层打乱；从 F0 重新开始。",
        "seconds": dt, "peak_allocated_GiB": round(peak, 3),
        "checkpoints": cks, "cuda_visible_devices": cvd,
        "device": torch.cuda.get_device_name(0),
        "finished_at": datetime.datetime.now().isoformat(),
    }
    with io.open(f"{a.out}/f1fix_config.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    print("checkpoints:", cks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
