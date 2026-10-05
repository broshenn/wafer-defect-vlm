#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E 阶段：采样检查 —— 有没有**类别奖励差异信号**。

照任务书 §4：
  * 在**新池**里按固定顺序取九类各前 2 张（不足取全部），每图 G=4；
  * 最多 18 组 / 72 个回答，预算最多 4 设备分钟；
  * 保存全部回答、奖励向量、解析失败、截断、组内 std、同分组比例和耗时；
  * **只读训练池**，不用 val84 找奖励漏洞或挑训练图。

判据：**至少一个组出现类别奖励差异**才进 RL 小测试。
若全部同分 → 回交"当前奖励/采样没有类别优势信号"，
不以 KL 引起参数变化冒称有效 RL，**不自动提高温度、G 或输出预算**。

隔离：CUDA_VISIBLE_DEVICES 必须由父进程设置，断言只见 1 张卡。
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import statistics
import sys
import time
from collections import Counter

os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_v1 import WaferClassReward, parse_answer, CLASSES        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/WS/datasets/rl_pool90/rl_pool90_grpo.server.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--g", type=int, default=4)
    ap.add_argument("--per-class", type=int, default=2)
    # 合并目录的 config.json 被 transformers 5.16.1 重写后，
    # vision_config.model_type 变成 qwen3_5_vision，ms-swift 无法自动判别，
    # 必须显式指定（这是 ms-swift 报错信息里给出的官方办法）。
    ap.add_argument("--model-type", default="qwen3_5")
    # 合并目录不在 ms-swift 的模型组命名里，model_type 与 template_type
    # 都无法自动判别，两个都要显式给。
    ap.add_argument("--template-type", default="qwen3_5")
    a = ap.parse_args()

    import torch
    print("=" * 78)
    print(f"E 阶段采样检查  {datetime.datetime.now().isoformat()}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')!r}（父进程设置）")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("!! 父进程未设置 —— 拒绝运行"); return 3
    torch.cuda.init()
    n = torch.cuda.device_count()
    print(f"可见 GPU 数 = {n}   device 0 = {torch.cuda.get_device_name(0)}")
    if n != 1:
        print("!! 不是 1 张卡 —— 拒绝运行"); return 3

    from swift import InferRequest, RequestConfig
    from swift.infer_engine import TransformersEngine

    rows = [json.loads(l) for l in io.open(a.data, encoding="utf-8") if l.strip()]
    # 固定顺序取每类前 N 张
    picked, cnt = [], Counter()
    for r in rows:
        if cnt[r["ground_truth"]] < a.per_class:
            picked.append(r); cnt[r["ground_truth"]] += 1
    print(f"池 {len(rows)} 条 -> 选中 {len(picked)} 图 × G={a.g} = {len(picked)*a.g} 个回答")
    print(f"  类别覆盖 {dict(cnt)}")

    t0 = time.time()
    engine = TransformersEngine(a.model, model_type=a.model_type, template_type=a.template_type)
    load_s = time.time() - t0
    # 采样配置：temperature=1.0, top_p=1.0（GRPO 要多样性）
    cfg = RequestConfig(max_tokens=64, temperature=1.0, top_p=1.0, seed=3407)
    rw = WaferClassReward()

    groups, t_gen = [], 0.0
    for i, r in enumerate(picked, 1):
        reqs = [InferRequest(messages=[{"role": "user", "content": r["messages"][0]["content"]}],
                             images=r["images"]) for _ in range(a.g)]
        t1 = time.time()
        resps = engine.infer(reqs, cfg)
        t_gen += time.time() - t1
        comps = [x.choices[0].message.content for x in resps]
        rewards = rw(completions=comps, ground_truth=[r["ground_truth"]] * a.g,
                     sample_id=[r["sample_id"]] * a.g)
        std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        parsed = [parse_answer(c) for c in comps]
        lens = [len(x.choices[0].message.content or "") for x in resps]
        groups.append({
            "sample_id": r["sample_id"], "gold": r["ground_truth"],
            "rewards": rewards, "mean": sum(rewards) / len(rewards), "std": round(std, 4),
            "same_group": std == 0.0,
            "preds": [p for p, _ in parsed],
            "parse_tags": [t for _, t in parsed],
            "n_parse_fail": sum(1 for p, _ in parsed if p is None),
            "completions": comps,
            "char_lens": lens,
        })
        print(f"  [{i}/{len(picked)}] {r['sample_id']} gold={r['ground_truth']:10s} "
              f"rewards={rewards} std={std:.3f} preds={[p for p, _ in parsed]}", flush=True)

    n_same = sum(1 for g in groups if g["same_group"])
    n_diff = len(groups) - n_same
    all_r = [x for g in groups for x in g["rewards"]]
    is_trunc = sum(1 for g in groups for c, t in zip(g["completions"], g["parse_tags"])
                   if t == "truncated" or (c and not c.rstrip().endswith("}") and "truncated" in t))

    summary = {
        "model": a.model, "data": a.data, "g": a.g, "n_groups": len(groups),
        "n_responses": len(all_r),
        "temperature": 1.0, "top_p": 1.0, "max_tokens": 64,
        "reward_mean": round(sum(all_r) / len(all_r), 4) if all_r else None,
        "n_groups_with_variance": n_diff,
        "n_groups_same": n_same,
        "same_group_ratio": round(n_same / len(groups), 4) if groups else None,
        "n_parse_fail": sum(g["n_parse_fail"] for g in groups),
        "parse_tag_dist": dict(Counter(t for g in groups for t in g["parse_tags"])),
        "load_seconds": round(load_s, 1),
        "total_generation_seconds": round(t_gen, 1),
        "device_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_allocated_GiB_own_process": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
        "groups": groups,
        "verdict": ("有类别奖励差异信号，可进 RL 小测试" if n_diff > 0
                    else "**全部组同分 —— 当前奖励/采样没有类别优势信号**"),
        "finished_at": datetime.datetime.now().isoformat(),
    }
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print()
    print("=" * 78)
    print(f"组数 {len(groups)}  回答 {len(all_r)}")
    print(f"奖励均值 {summary['reward_mean']}  同分组 {n_same}/{len(groups)} "
          f"({summary['same_group_ratio']})")
    print(f"解析失败 {summary['n_parse_fail']}   解析标签 {summary['parse_tag_dist']}")
    print(f"加载 {load_s:.1f}s  生成 {t_gen:.1f}s  峰值 "
          f"{summary['peak_allocated_GiB_own_process']} GiB")
    print(f"结论：{summary['verdict']}")
    print("=" * 78)
    return 0 if n_diff > 0 else 5


if __name__ == "__main__":
    sys.exit(main())
