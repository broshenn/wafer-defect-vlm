#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D1 描述 SFT —— 修正版 v2（Codex 2026-10-07 三点修正之 §1）。

相对 v1 的改动：
  1. **门槛失败真的会停**：`_ck` 记问题，阶段末统一保存诊断并抛 `GateFailure`，
     不再只打印 FAIL 就继续。训练结束后另有一道断言，确认门槛**确实执行过且通过**。
  2. **两步检查改到第 2 步 loss 日志取得之后**：由 `on_log` 触发（此时该步日志已入
     `state.log_history`，优化器也已 step 过两次），不再在 `on_step_end` 预设日志存在，
     也不会在空列表上取 `ls[0]`。
  3. **不再把「第 2 步 loss 更高」当硬停条件**：两个不同小批次的 loss 本就无需单调。
     硬门槛是 loss 有限、视觉前向有效、非零 LoRA 梯度、参数确有更新。
  4. **快照含 lora_B**：LoRA 的 B 是零初始化，第 1 步 `dL/dA = (dL/dout)·B·x = 0`，
     只看 lora_A 会误报「未更新」。硬判定用 lora_B，lora_A 另行如实记录。
  5. 全程记录 loss 趋势，**不用两步判断收敛**。

设备隔离：CUDA_VISIBLE_DEVICES 由父进程在 Python 启动前按 GPU UUID 设置。
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import io
import json
import math
import os
import re
import sys

os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")

import torch                                              # noqa: E402
import transformers                                       # noqa: E402
from transformers import TrainerCallback                  # noqa: E402
from swift import SftArguments                            # noqa: E402
from swift.pipelines import sft_main                      # noqa: E402

BASE = "/WS/models/Qwen3.5-9B"
DATA = "/WS/datasets/abc/desc_v2.server.jsonl"
VIS_RE = re.compile(r"visual|vision|vit|aligner|merger|patch_embed", re.I)

_CAP: dict = {}
_ORIG = transformers.Trainer.__init__


def _patched(self, *a, **kw):
    _ORIG(self, *a, **kw)
    _CAP["trainer"] = self


transformers.Trainer.__init__ = _patched


class GateFailure(RuntimeError):
    pass


class DescGate(TrainerCallback):
    """前 2 个优化步的运行内门槛。任一硬项不过 → 保存诊断并抛 GateFailure。"""

    def __init__(self, out: str) -> None:
        self.out = out
        self.rec = {"losses": [], "checks": [], "grad_steps": [], "vision_calls": 0,
                    "params": {}, "snapshots": {}}
        self.failed: list = []
        self.done = False
        self.ran = False

    # ── 记录与判据 ────────────────────────────────────────
    def _ck(self, name, ok, d=""):
        self.rec["checks"].append({"项": name, "通过": bool(ok), "说明": str(d)})
        print(f"    [gate] {'OK  ' if ok else 'FAIL'} {name}  {d}", flush=True)
        if not ok:
            self.failed.append(name)
        return bool(ok)

    def _dump(self):
        os.makedirs(self.out, exist_ok=True)
        p = os.path.join(self.out, "gate_report.json")
        with io.open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(self.rec, ensure_ascii=False, indent=2) + "\n")
        print(f"    [gate] 诊断已写入 {p}", flush=True)

    def _fail_now(self, stage):
        if self.failed:
            self._dump()
            raise GateFailure(f"{stage} 门槛未通过：{self.failed}")

    # ── 训练开始：参数统计、快照、插桩 ────────────────────
    def on_train_begin(self, args, state, control, **kw):
        tr = _CAP.get("trainer")
        m = tr.model
        trn = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        vis = [n for n, _ in trn if VIS_RE.search(n)]
        lora = [n for n, _ in trn if "lora_" in n.lower()]
        tot = sum(p.numel() for p in m.parameters())
        ntrn = sum(p.numel() for _, p in trn)
        self.rec["params"] = {"总参数": tot, "可训练参数": ntrn, "可训练张量": len(trn),
                              "lora张量": len(lora), "视觉侧可训练张量": len(vis),
                              "视觉侧总张量": sum(1 for n, _ in m.named_parameters()
                                                  if VIS_RE.search(n))}
        print(f"  [gate] 可训练 {ntrn/1e6:.3f}M / 总 {tot/1e6:.1f}M   张量 {len(trn)}"
              f"   lora {len(lora)}   视觉侧可训练 {len(vis)}", flush=True)
        self._ck("正式配置下视觉侧完全冻结", not vis, f"视觉侧可训练 {len(vis)}")
        self._ck("全部可训练张量都是 LoRA", len(lora) == len(trn), f"{len(lora)}/{len(trn)}")
        self._ck("单卡断言 device_count==1", torch.cuda.device_count() == 1,
                 f"device_count={torch.cuda.device_count()} "
                 f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")

        # 快照 lora_B（零初始化，第 1 步真正被更新的那个）与 lora_A
        for n, p in trn:
            if "lora_B" in n and "loraB" not in self.rec["snapshots"]:
                self.rec["snapshots"]["loraB"] = n
                self.snapB_name, self.snapB = n, p.detach().float().clone()
            elif "lora_A" in n and "loraA" not in self.rec["snapshots"]:
                self.rec["snapshots"]["loraA"] = n
                self.snapA_name, self.snapA = n, p.detach().float().clone()
        self._ck("取到 lora_B 快照（更新判定的硬依据）",
                 getattr(self, "snapB", None) is not None, str(self.rec["snapshots"].get("loraB")))
        self._ck("取到 lora_A 快照", getattr(self, "snapA", None) is not None,
                 str(self.rec["snapshots"].get("loraA")))

        # 视觉模块被调用计数：证明图像真的过了编码器
        def _vh(mod, inp, out):
            self.rec["vision_calls"] += 1
        vis_mods = [(n, mm) for n, mm in m.named_modules() if VIS_RE.search(n)]
        if vis_mods:
            vis_mods[0][1].register_forward_hook(_vh)
        self.rec["vision_probe_module"] = vis_mods[0][0] if vis_mods else None

        # 优化器 step 前记录梯度范数（on_log 时梯度已被清空）
        real_step = tr.optimizer.step

        def wrapped_step(*a, **k):
            row = []
            for n, p in m.named_parameters():
                if p.grad is not None:
                    row.append([n, float(p.grad.detach().float().norm())])
            self.rec["grad_steps"].append(row)
            return real_step(*a, **k)

        tr.optimizer.step = wrapped_step
        self._fail_now("训练开始")

    # ── 日志：loss 趋势；第 2 步之后一次性执行门槛 ──────────
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.rec["losses"].append({"step": int(state.global_step),
                                       "loss": float(logs["loss"]),
                                       "lr": float(logs.get("learning_rate", float("nan"))),
                                       "grad_norm": float(logs.get("grad_norm", float("nan")))})
        if (not self.done) and state.global_step >= 2 and len(self.rec["losses"]) >= 2:
            self.done = True
            self._run_gate(state)

    def _run_gate(self, state):
        self.ran = True
        print("\n  ===== 前 2 个优化步门槛 =====", flush=True)
        ls = [x["loss"] for x in self.rec["losses"]]
        self._ck("已取得第 2 步的 loss 日志", len(ls) >= 2, f"{len(ls)} 条 {ls}")
        self._ck("loss 全部有限", bool(ls) and all(math.isfinite(v) for v in ls), f"{ls}")

        m = _CAP["trainer"].model
        params = dict(m.named_parameters())
        for key in ("loraB", "loraA"):
            nm = self.rec["snapshots"].get(key)
            if not nm or nm not in params:
                self._ck(f"{key} 能取回当前权重", False, str(nm)); continue
            before = self.snapB if key == "loraB" else self.snapA
            d = float((params[nm].detach().float() - before).abs().max())
            if key == "loraB":
                self._ck("lora_B 权重确有更新（硬门槛）", d > 0, f"{nm} max|Δ| = {d:.3e}")
            else:
                print(f"    [gate] 参考 lora_A max|Δ| = {d:.3e}"
                      f"（第 1 步 dL/dA 经零初始化 B 回传为 0，故不作硬判定）", flush=True)
            self.rec.setdefault("权重变化", {})[key] = d

        gs = self.rec["grad_steps"]
        self._ck("记录了 2 次优化器 step 的梯度", len(gs) >= 2, f"{len(gs)} 次")
        last = gs[-1] if gs else []
        names = [n for n, _ in last]
        self._ck("存在非零 LoRA 梯度", any(v > 0 for _, v in last),
                 f"{len(last)} 个，最大 {max([v for _, v in last] or [0]):.3e}")
        self._ck("梯度只落在 LoRA 上", bool(names) and all("lora_" in n.lower() for n in names),
                 f"非 lora 梯度 {[n for n in names if 'lora_' not in n.lower()][:3]}")

        ncall = self.rec["vision_calls"]
        self._ck("视觉模块确被调用（图像前向有效）", ncall >= 8,
                 f"{self.rec.get('vision_probe_module')} 调用 {ncall} 次")
        devs = {str(p.device) for p in m.parameters()}
        self._ck("模型参数只在单卡", len(devs) == 1, str(devs))

        print("  ===== 门槛结束 =====\n", flush=True)
        self._fail_now("前 2 步")

    # ── 结束：落盘 loss 趋势 ──────────────────────────────
    def on_train_end(self, args, state, control, **kw):
        ls = [x["loss"] for x in self.rec["losses"]]
        self.rec["loss趋势"] = {
            "步数": len(ls), "首": ls[0] if ls else None, "末": ls[-1] if ls else None,
            "最小": min(ls) if ls else None, "最大": max(ls) if ls else None,
            "均值": round(sum(ls) / len(ls), 4) if ls else None,
            "全序列": ls,
            "说明": "仅记录趋势，**不用两步判断收敛**；收敛需另行分析。",
        }
        self._dump()


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=89)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--model-type", default="qwen3_5")
    ap.add_argument("--template-type", default="qwen3_5")
    a = ap.parse_args()

    print("=" * 78)
    print(f"D1 描述 SFT（修正版 v2）steps={a.steps}  {datetime.datetime.now().isoformat()}")
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
    import torch.distributed as dist
    ws = dist.get_world_size() if dist.is_initialized() else 1
    print(f"world_size = {ws}")
    if ws != 1:
        print("!! world_size != 1 —— 拒绝运行"); return 3

    rows = [json.loads(l) for l in io.open(a.data, encoding="utf-8") if l.strip()]
    covered = a.steps * 4
    ep = covered / len(rows)
    print(f"\n数据 {a.data}\n  {len(rows)} 行  sha256 {sha(a.data)}")
    print(f"  步数 {a.steps} → 呈现 {covered} 次  覆盖 {ep:.4f} epoch "
          f"{'（完整 1 epoch）' if covered >= len(rows) else '**partial**'}")

    os.makedirs(a.out, exist_ok=True)
    gate = DescGate(a.out)
    args = SftArguments(
        model=BASE, model_type=a.model_type, template=a.template_type,
        dataset=a.data, val_dataset=a.data,
        tuner_type="lora", target_modules=["all-linear"],   # 列表，冻结才生效
        freeze_vit=True, freeze_aligner=True, torch_dtype="bfloat16",
        lora_rank=16, lora_alpha=32, lora_dropout=0.05,
        learning_rate=1e-4, warmup_ratio=0.05, lr_scheduler_type="cosine",
        max_length=2048,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=4, max_steps=a.steps,
        gradient_checkpointing=True, seed=a.seed, data_seed=a.seed,
        dataset_shuffle=True, train_dataloader_shuffle=True,
        strict=True,          # 编码失败即报错，禁止 LazyLLMDataset 换样本顶替
        logging_steps=1, save_strategy="steps", save_steps=a.steps, save_total_limit=2,
        eval_strategy="no", output_dir=a.out, report_to=[],
        callbacks=[gate],
    )
    print("\n=== 冻结配置 ===")
    for k in ("learning_rate", "warmup_ratio", "lr_scheduler_type", "max_steps",
              "gradient_accumulation_steps", "per_device_train_batch_size", "tuner_type",
              "target_modules", "freeze_vit", "freeze_aligner", "lora_rank", "lora_alpha",
              "lora_dropout", "max_length", "seed", "data_seed", "gradient_checkpointing",
              "dataset_shuffle", "train_dataloader_shuffle", "strict", "torch_dtype"):
        print(f"  {k:30s} = {getattr(args, k, '<缺失>')}")

    cfg = {"run": "D1 描述 SFT（v2）", "model": BASE,
           "model_note": "未经本项目训练的原基座（不是 R0/F2）",
           "data": a.data, "data_sha256": sha(a.data), "n_rows": len(rows),
           "steps": a.steps, "presentations": covered, "epochs_covered": round(ep, 4),
           "cuda_visible_devices_parent": cvd, "world_size": ws,
           "device": torch.cuda.get_device_name(0),
           "config": {k: str(getattr(args, k, None)) for k in
                      ("learning_rate", "warmup_ratio", "lr_scheduler_type", "max_steps",
                       "gradient_accumulation_steps", "per_device_train_batch_size",
                       "tuner_type", "target_modules", "freeze_vit", "freeze_aligner",
                       "lora_rank", "lora_alpha", "lora_dropout", "max_length", "seed",
                       "data_seed", "gradient_checkpointing", "dataset_shuffle",
                       "train_dataloader_shuffle", "strict")},
           "selection": "末步预选（不按 val 挑最佳）",
           "started_at": datetime.datetime.now().isoformat()}
    with io.open(f"{a.out}/exp_config.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")

    print(f"\n=== 开始训练 {a.steps} 步 ===", flush=True)
    t0 = datetime.datetime.now()
    rc = 0
    try:
        sft_main(args)
    except Exception as e:
        rc = 1
        print(f"\n!! 训练异常：{type(e).__name__}: {e}", flush=True)
        raise
    finally:
        dt = (datetime.datetime.now() - t0).total_seconds()
        gate.rec["seconds"] = dt
        gate.rec["返回码"] = rc
        gate.rec["门槛已执行"] = gate.ran
        gate.rec["门槛未通过项"] = gate.failed
        gate._dump()

    # 结束断言：门槛必须**确实执行过且通过**，否则本轮不算成功
    if not gate.ran:
        print("!! 门槛从未执行（未取得第 2 步 loss 日志）—— 本轮不成立")
        return 5
    if gate.failed:
        print(f"!! 门槛未通过：{gate.failed} —— 本轮不成立")
        return 5

    peak = torch.cuda.max_memory_allocated(0) / 2 ** 30
    print(f"\n结束 {dt/60:.2f} 分钟  峰值 {peak:.2f} GiB", flush=True)
    print(f"loss 趋势：{gate.rec['loss趋势']['步数']} 步  "
          f"首 {gate.rec['loss趋势']['首']:.4f} → 末 {gate.rec['loss趋势']['末']:.4f}  "
          f"最小 {gate.rec['loss趋势']['最小']:.4f}", flush=True)

    cks = sorted(glob.glob(f"{a.out}/**/checkpoint-*", recursive=True))
    final = [c for c in cks if c.endswith(f"checkpoint-{a.steps}")]
    cfg.update({"seconds": dt, "minutes": round(dt / 60, 3),
                "peak_allocated_GiB": round(peak, 3), "checkpoints": cks,
                "final_checkpoint": final[0] if final else None,
                "gate_passed": True, "gate_ran": True,
                "finished_at": datetime.datetime.now().isoformat()})
    with io.open(f"{a.out}/exp_config.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    print(f"checkpoints: {cks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
