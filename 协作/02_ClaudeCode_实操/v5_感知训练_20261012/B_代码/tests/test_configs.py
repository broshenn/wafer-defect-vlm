#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""configs 测试（合同 B 卡 4：逐项断言，尤其 S2 freeze=False、S3 token=1024、S1p seed=3408、RL G=8、375/160 步）。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = HERE.parent / "configs"


def main():
    results = []

    def add(name, cond, extra=""):
        results.append((name, bool(cond), extra))

    files = sorted(p.stem for p in CFG.glob("*.json"))
    add("11 个配置文件", files == ["RSFT", "Rbinary", "Rfull", "Rfullp", "Rnoproc",
                                   "S0", "S1", "S1p", "S2", "S3", "S4"], files)
    cfg = {n: json.loads((CFG / f"{n}.json").read_text(encoding="utf-8")) for n in files}

    sft_lines = ["S0", "S1", "S2", "S3", "S4", "S1p"]
    for n in sft_lines:
        c = cfg[n]
        ok = (c["mode"] == "sft_base" and c["model"] == "/WS/models/Qwen3.5-9B"
              and c["model_type"] == "qwen3_5" and c["template"] == "qwen3_5"
              and c["lora_rank"] == 16 and c["lora_alpha"] == 32 and c["lora_dropout"] == 0.05
              and c["target_modules"] == ["all-linear"] and c["lr"] == 1e-4
              and c["warmup_ratio"] == 0.03 and c["per_device_train_batch_size"] == 8
              and c["gradient_accumulation_steps"] == 4 and c["max_steps"] == 375
              and c["save_steps"] == 125 and c["save_total_limit"] == 4 and c["max_length"] == 2048
              and c["data"] == f"/WS/v5/data/sft_{n}.jsonl" and c["start_adapter"] is None
              and c["time_box_min"] == 300)
        add(f"{n} SFT 共同项", ok)
    add("S2 freeze=True 反例（必须 False）", cfg["S2"]["freeze_vit"] is False
        and cfg["S2"]["freeze_aligner"] is False)
    add("其余五线 freeze=True",
        all(cfg[n]["freeze_vit"] is True and cfg[n]["freeze_aligner"] is True
            for n in ["S0", "S1", "S3", "S4", "S1p"]))
    add("S3 token=1024", cfg["S3"]["image_max_token_num"] == 1024)
    add("其余五线 token=256",
        all(cfg[n]["image_max_token_num"] == 256 for n in ["S0", "S1", "S2", "S4", "S1p"]))
    add("S1p seed=3408（seed+data_seed）", cfg["S1p"]["seed"] == 3408 and cfg["S1p"]["data_seed"] == 3408)
    add("其余五线 seed=3407",
        all(cfg[n]["seed"] == 3407 and cfg[n]["data_seed"] == 3407
            for n in ["S0", "S1", "S2", "S3", "S4"]))

    for n, rm in (("Rfull", "full"), ("Rbinary", "binary"), ("Rnoproc", "noproc"), ("Rfullp", "full")):
        c = cfg[n]
        ok = (c["mode"] == "rl" and c["num_generations"] == 8 and c["temperature"] == 1.0
              and c["top_p"] == 1.0 and c["beta"] == 0.04 and c["epsilon"] == 0.2
              and c["max_completion_length"] == 384 and c["per_device_train_batch_size"] == 1
              and c["gradient_accumulation_steps"] == 8 and c["max_steps"] == 160
              and c["lr"] == 1e-5 and c["start_adapter"] == "FROM_WINNER"
              and c["freeze_vit"] is None and c["freeze_aligner"] is None
              and c["image_max_token_num"] is None and c["reward_mode"] == rm
              and c["data"] == "/WS/v5/data/rl_pool320.jsonl" and c["time_box_min"] == 210)
        add(f"{n} RL 共同项+reward={rm}", ok)
    add("Rfullp seed=3408", cfg["Rfullp"]["seed"] == 3408 and cfg["Rfullp"]["data_seed"] == 3408)
    add("Rfull/Rbinary/Rnoproc seed=3407",
        all(cfg[n]["seed"] == 3407 for n in ("Rfull", "Rbinary", "Rnoproc")))
    c = cfg["RSFT"]
    add("RSFT: sft_cont、batch1 GA4、160 步、rsft_pool320",
        c["mode"] == "sft_cont" and c["per_device_train_batch_size"] == 1
        and c["gradient_accumulation_steps"] == 4 and c["max_steps"] == 160
        and c["data"] == "/WS/v5/data/rsft_pool320.jsonl"
        and c["start_adapter"] == "FROM_WINNER" and c["lr"] == 1e-5)

    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, extra in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name} {extra if not ok else ''}")
    print(f"{n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
