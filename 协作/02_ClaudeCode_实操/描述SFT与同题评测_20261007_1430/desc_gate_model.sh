#!/bin/bash
# 说明：`/WS` 是真实私有工作区的**别名**，真实值只存在于仓库外的接入资料与服务器上的
# 可执行副本中。本文件入库前已按约定脱敏。
# 描述 SFT CPU 门槛 D：正式 rank16 配置的真实可训练参数 / 视觉侧计数 / adapter_config
# 0 GPU（use_cpu=True）；本脚本不写任何训练产物，只落 adapter 配置与计数证据。
set -u
R=/WS
export CUDA_VISIBLE_DEVICES=""          # 必须在 Python 启动前设
export IMAGE_MAX_TOKEN_NUM=256
export PYTHONIOENCODING=utf-8
exec $R/envs/wafer/bin/python - <<'PY'
import io, json, os, re, sys
from collections import Counter

R = "/WS"
MODEL = R + "/models/Qwen3.5-9B"
DATA = R + "/datasets/abc/desc_v2.server.jsonl"
OUTD = R + "/tmp/desc_gate_adapter"
LOGS = R + "/logs"
VIS = re.compile(r"visual|vision|vit|aligner|merger|patch_embed", re.I)

res = []
def ck(n, ok, d=""):
    res.append({"项": n, "通过": bool(ok), "说明": str(d)})
    print(f"  {'OK  ' if ok else 'FAIL'} {n}  {d}")

import torch, transformers, peft
import swift
print("=" * 76)
print("D. 正式 rank16 配置（CPU 建模，0 GPU）")
print("=" * 76)
print(f"  swift {swift.__version__} / torch {torch.__version__} / "
      f"transformers {transformers.__version__} / peft {peft.__version__}")
print(f"  cuda.is_available = {torch.cuda.is_available()}（必须为 False）")
ck("CPU 建模，未看见 GPU", torch.cuda.is_available() is False)

from swift import SftArguments
from swift.pipelines import SwiftSft

args = SftArguments(model=MODEL, model_type="qwen3_5", template="qwen3_5",
                    dataset=DATA, val_dataset=DATA,
                    tuner_type="lora", target_modules=["all-linear"],
                    freeze_vit=True, freeze_aligner=True, torch_dtype="bfloat16",
                    lora_rank=16, lora_alpha=32, lora_dropout=0.05,
                    max_length=2048, seed=3407, data_seed=3407, use_cpu=True,
                    output_dir=R + "/tmp/desc_gate_probe")
print(f"\n  target_modules 实际类型 = {type(args.target_modules).__name__} "
      f"{args.target_modules}   ← 字符串会使冻结静默失效")
ck("target_modules 是列表（冻结才生效）", isinstance(args.target_modules, list),
   str(args.target_modules))

pipe = SwiftSft(args)
m = pipe.prepare_model(args, pipe.model, template=pipe.template)

tr = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
tot = sum(p.numel() for p in m.parameters())
ntrn = sum(p.numel() for _, p in tr)
vis_tr = [n for n, _ in tr if VIS.search(n)]
lora_tr = [n for n, _ in tr if "lora_" in n.lower()]
vis_all = [n for n, _ in m.named_parameters() if VIS.search(n)]

print(f"\n  总参数        {tot/1e6:.1f}M")
print(f"  可训练参数    {ntrn/1e6:.3f}M   （{ntrn} 个）")
print(f"  可训练张量    {len(tr)}")
print(f"  其中 lora_    {len(lora_tr)}")
print(f"  视觉侧张量    {len(vis_all)}  其中可训练 {len(vis_tr)}")
ck("视觉侧完全冻结", not vis_tr, f"视觉侧可训练 {len(vis_tr)} {vis_tr[:3]}")
ck("全部可训练张量都是 LoRA", len(lora_tr) == len(tr), f"{len(lora_tr)}/{len(tr)}")

# rank 由**实际张量形状**核对，不靠配置标题
A = [(n, tuple(p.shape)) for n, p in tr if n.endswith("lora_A.weight")
     or ".lora_A." in n]
B = [(n, tuple(p.shape)) for n, p in tr if n.endswith("lora_B.weight")
     or ".lora_B." in n]
rA = Counter(s[0] for _, s in A)
rB = Counter(s[1] for _, s in B)
print(f"\n  lora_A 形状首维分布 {dict(rA)}   共 {len(A)} 个")
print(f"  lora_B 形状次维分布 {dict(rB)}   共 {len(B)} 个")
ck("实际张量确认 rank=16（lora_A 首维）", set(rA) == {16}, dict(rA))
ck("实际张量确认 rank=16（lora_B 次维）", set(rB) == {16}, dict(rB))
ck("lora_A 与 lora_B 成对", len(A) == len(B) and len(A) > 0, f"{len(A)}/{len(B)}")

# 用实际形状算期望参数量，与实测比对
exp = sum(a * b for _, (a, b) in A) + sum(a * b for _, (a, b) in B)
ck("实测可训练参数量 == 由实际形状推算", exp == ntrn, f"推算 {exp} vs 实测 {ntrn}")

# adapter_config 落盘
os.makedirs(OUTD, exist_ok=True)
m.save_pretrained(OUTD)
ap = os.path.join(OUTD, "adapter_config.json")
cfg = json.load(io.open(ap, encoding="utf-8"))
print(f"\n  adapter_config: r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
      f"dropout={cfg.get('lora_dropout')} target={cfg.get('target_modules')}")
ck("adapter_config r == 16", cfg.get("r") == 16, str(cfg.get("r")))
ck("adapter_config alpha == 32", cfg.get("lora_alpha") == 32, str(cfg.get("lora_alpha")))
ck("adapter_config dropout == 0.05", abs(float(cfg.get("lora_dropout", -1)) - 0.05) < 1e-9,
   str(cfg.get("lora_dropout")))

report = {"环境": {"swift": swift.__version__, "torch": torch.__version__,
                   "transformers": transformers.__version__, "peft": peft.__version__},
          "target_modules": str(args.target_modules),
          "总参数": tot, "可训练参数": ntrn, "可训练张量": len(tr),
          "lora张量": len(lora_tr), "视觉侧张量": len(vis_all),
          "视觉侧可训练": len(vis_tr), "lora_A个数": len(A), "lora_B个数": len(B),
          "adapter_config": cfg,
          "note": ("CPU 建模（use_cpu=True）。证明的是**参数选择与冻结**，"
                   "不证明 GPU 上的显存占用、多步收敛或前向数值。"),
          "检查": res}
os.makedirs(LOGS, exist_ok=True)
json.dump(report, io.open(LOGS + "/desc_gate_model.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

fails = [x for x in res if not x["通过"]]
print()
print("=" * 76)
print(f"通过 {len(res)-len(fails)}/{len(res)}")
for f in fails:
    print(f"  未通过：{f['项']}  {f['说明']}")
print(f"写出 {LOGS}/desc_gate_model.json   adapter_config {ap}")
sys.exit(0 if not fails else 1)
PY
