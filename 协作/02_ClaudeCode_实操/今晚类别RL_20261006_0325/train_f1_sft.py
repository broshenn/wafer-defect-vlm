#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F1：从 F0 继续类别 SFT（对照 F2）。LR=1e-5，20 步，pdb=1/GA=4。"""
import argparse, datetime, hashlib, io, json, os, sys
os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")
import torch
from swift import SftArguments
from swift.pipelines import sft_main

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/WS/checkpoints/Qwen3.5-9B-f0a1merged")
ap.add_argument("--data", default="/WS/datasets/rl_pool90/rl_pool90_sft_f1_repeat4.server.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--model-type", default="qwen3_5")
ap.add_argument("--template-type", default="qwen3_5")
a = ap.parse_args()

print("="*78); print(f"F1 继续 SFT  {datetime.datetime.now().isoformat()}"); print("="*78)
cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
print(f"CUDA_VISIBLE_DEVICES = {cvd!r}（父进程设置）")
if cvd is None: print("!! 父进程未设置"); sys.exit(3)
torch.cuda.init(); n = torch.cuda.device_count()
print(f"可见 GPU 数 = {n}   device 0 = {torch.cuda.get_device_name(0)}")
if n != 1: print("!! 不是 1 张卡"); sys.exit(3)
try:
    import torch.distributed as dist
    ws = dist.get_world_size() if dist.is_initialized() else 1
except Exception: ws = 1
print(f"world_size = {ws}")
if ws != 1: sys.exit(3)

os.makedirs(a.out, exist_ok=True)
args = SftArguments(
    model=a.model, model_type=a.model_type, template=a.template_type,
    dataset=a.data, val_dataset=a.data,
    tuner_type="lora", target_modules=["all-linear"],
    freeze_vit=True, freeze_aligner=True, torch_dtype="bfloat16",
    lora_rank=16, lora_alpha=32, lora_dropout=0.05,
    learning_rate=1e-5, warmup_ratio=0.0, max_length=2048,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=4, max_steps=20,
    gradient_checkpointing=True, seed=3407, data_seed=3407,
    logging_steps=1, save_strategy="steps", save_steps=10, save_total_limit=4,
    eval_strategy="no", output_dir=a.out, report_to=[],
)
for k in ("learning_rate","max_steps","gradient_accumulation_steps",
          "per_device_train_batch_size","tuner_type","target_modules","freeze_vit",
          "freeze_aligner","lora_rank","lora_alpha","seed","data_seed"):
    print(f"  {k:30s} = {getattr(args,k)}")
t0 = datetime.datetime.now()
sft_main(args)
dt = (datetime.datetime.now()-t0).total_seconds()
peak = torch.cuda.max_memory_allocated(0)/2**30
print(f"\nF1 结束 {dt/60:.2f} 分钟  峰值 {peak:.2f} GiB")
import glob
ck = sorted(glob.glob(f"{a.out}/**/checkpoint-*", recursive=True))
io.open(f"{a.out}/f1_config.json","w",encoding="utf-8").write(json.dumps({
    "model":a.model,"data":a.data,
    "data_sha256":hashlib.sha256(open(a.data,"rb").read()).hexdigest(),
    "lr":1e-5,"max_steps":20,"ga":4,"pdb":1,"effective_batch":4,
    "seconds":dt,"peak_allocated_GiB":round(peak,3),"checkpoints":ck,
    "cuda_visible_devices":cvd,"device":torch.cuda.get_device_name(0),
    "finished_at":datetime.datetime.now().isoformat()},ensure_ascii=False,indent=2)+"\n")
print("checkpoints:", ck)
