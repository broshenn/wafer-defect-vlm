#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成夜跑计划 night_plan.json + 每个任务的执行脚本（B 卡 15）。
输入：smoke_timing.json、gpus.json（C 阶段产物，同一目录）、环境变量 WS_ROOT（真实根，不在仓库）。
输出：night_plan.json、scripts/*.sh、打印预计时间表。
卡分配按总纲 0.3：g1=S1、g2=S0、g3=S2、g4=S1p、g5–g7=评测/CNN/P2；
RL 阶段 g1=RSFT、g2=Rfull、g3=Rbinary、g4=Rnoproc、g5=Rfullp（P2 行，有空卡才启动）。
硬截止 08:30：调度器对 training=true 的任务发 SIGTERM；之后只跑评测/打分/备份。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-timing", default=str(HERE / "smoke_timing.json"))
    ap.add_argument("--gpus", default=str(HERE / "gpus.json"))
    ap.add_argument("--out", default=str(HERE / "night_plan.json"))
    ap.add_argument("--scripts-dir", default=str(HERE / "scripts"))
    ap.add_argument("--hard-stop", default="08:30")
    a = ap.parse_args()
    WS = os.environ.get("WS_ROOT", "/WS")
    PY = f"{WS}/envs/wafer/bin/python"
    timing = json.loads(Path(a.smoke_timing).read_text(encoding="utf-8"))
    gpus = json.loads(Path(a.gpus).read_text(encoding="utf-8"))
    # gpus.json: {"g1": uuid, ..., "gN": uuid}；最多 7 张
    gpu_names = [f"g{i}" for i in range(1, len(gpus) + 1)]
    g1, g2, g3 = gpu_names[0], gpu_names[1], gpu_names[2]
    g4 = gpu_names[3] if len(gpu_names) > 3 else g1
    g5 = gpu_names[4] if len(gpu_names) > 4 else g2
    g6 = gpu_names[5] if len(gpu_names) > 5 else g3
    g7 = gpu_names[6] if len(gpu_names) > 6 else g4
    LASTRES = [g5, g6, g7]
    ALLRES = gpu_names

    sft_s_per_step = timing.get("sft_s_per_step", 20.0)
    rl_s_per_step = timing.get("rl_s_per_step", sft_s_per_step * 3)
    eval_s_per_img = timing.get("eval_s_per_img", 6.0)
    load_s = timing.get("load_s", 120.0)
    bs = timing.get("eval_batch_size", 8)
    sft_steps, rl_steps = 375, 160

    def est_sft_min():
        return (sft_s_per_step * sft_steps + load_s) / 60.0

    def est_rl_min():
        return (rl_s_per_step * rl_steps + load_s) / 60.0

    def est_eval_min(n):
        return max(20.0, (n * eval_s_per_img + load_s) / 60.0 * 1.5)

    scripts = Path(a.scripts_dir)
    scripts.mkdir(parents=True, exist_ok=True)
    jobs = []

    def sh(name, body):
        p = scripts / name
        p.write_text("#!/bin/bash\nset -u\ncd {}/v5/code || exit 9\n{}\n".format(WS, body),
                     encoding="utf-8", newline="\n")
        os.chmod(p, 0o755)
        return ["bash", str(p)]

    def add(jid, cmd, gpu, after=None, timeout_min=60, timeout_ok=False, retries=0,
            training=False, cwd=None):
        jobs.append({"id": jid, "gpus": gpu if isinstance(gpu, list) else [gpu],
                     "cmd": cmd, "after": after or [], "after_mode": "all_ok",
                     "timeout_min": int(timeout_min), "timeout_ok": timeout_ok,
                     "retries": retries, "training": training,
                     "log": f"{WS}/v5/night/logs/{jid}.log", "cwd": cwd})

    # ── 评测 + 打分（score 为 cpu）──
    def add_eval(jid, dataset, tag, adapter_line=None, adapter_ptr=None, max_tokens=384,
                 gpu=None, after=None, n=200, out_set=None):
        out_set = out_set or jid.replace("eval_", "")
        args = [f"--model {WS}/models/Qwen3.5-9B"]
        if adapter_line:
            args.append(f"--adapter-from-line {adapter_line} --runs-root {WS}/v5/runs")
        if adapter_ptr:
            args.append(f"--adapter {adapter_ptr}")
        args += [f"--tag {tag}", f"--dataset {WS}/v5/data/{dataset}",
                 f"--out {WS}/v5/eval/{tag}/{out_set}", f"--max-tokens {max_tokens}",
                 f"--batch-size {bs}"]
        add(jid, sh(f"{jid}.sh", f"{PY} eval_v5.py {' '.join(args)}"),
            gpu, after=after, timeout_min=est_eval_min(n), timeout_ok=True, retries=1)
        base = dataset.replace("_x2", "")
        kind = "L" if dataset.endswith("_L.jsonl") else "main"
        add(f"score_{jid}", sh(f"score_{jid}.sh",
            f"{PY} score_v5.py --raw {WS}/v5/eval/{tag}/{out_set}/{tag}_raw.jsonl "
            f"--ref {WS}/v5/data/{base} --kind {kind} "
            f"--out {WS}/v5/eval/{tag}/{out_set}/metrics.json "
            f"--per-sample {WS}/v5/eval/{tag}/{out_set}/per_sample.jsonl"),
            "cpu", after=[jid], timeout_min=15)
        return jid

    # ── N0：四个 gate（g1–g4）+ 基座/L 评测（g5–g7）──
    for line, gcard in (("S1", g1), ("S0", g2), ("S2", g3), ("S1p", g4)):
        add(f"gate_{line}",
            sh(f"gate_{line}.sh",
               f"{PY} train_v5.py --config configs/{line}.json --out {WS}/v5/runs/{line}_probe --probe-steps 2 && "
               f"CKPT=$(find {WS}/v5/runs/{line}_probe -name checkpoint-2 -type d | head -1) && "
               f"test -n \"$CKPT\" && {PY} gate_sft.py --config configs/{line}.json --ckpt \"$CKPT\" "
               f"--out {WS}/v5/runs/{line}_probe/gate_sft.json"),
            gcard, timeout_min=20, retries=0, training=True)
    add_eval("eval_base_select", "select200_main.jsonl", "BASE", gpu=g5, n=200, out_set="select200")
    add_eval("eval_base_rot100", "rot100_main.jsonl", "BASE", gpu=g5,
             after=["eval_base_select"], n=100, out_set="rot100")
    add_eval("eval_base_test200", "test200_main.jsonl", "BASE", gpu=g7, n=200, out_set="test200")
    add_eval("eval_base_shuf50", "shuf50_main.jsonl", "BASE", gpu=g7,
             after=["eval_base_test200"], n=50, out_set="shuf50")
    add_eval("eval_L_select", "select200_L.jsonl", "L", adapter_ptr=f"{WS}/night/L-N3072-3407.ckpt",
             max_tokens=64, gpu=g6, n=200, out_set="select200")
    add_eval("eval_L_test200", "test_L.jsonl", "L", adapter_ptr=f"{WS}/night/L-N3072-3407.ckpt",
             max_tokens=64, gpu=g6, after=["eval_L_select"], n=200, out_set="test200")

    # ── N1：四个 SFT（各自原卡）+ P2 gate_S3/S4（g6/g7，排在 L 评测之后）──
    for line, gcard in (("S1", g1), ("S0", g2), ("S2", g3), ("S1p", g4)):
        add(f"sft_{line}", sh(f"sft_{line}.sh",
            f"{PY} train_v5.py --config configs/{line}.json --out {WS}/v5/runs/{line}"),
            gcard, after=[f"gate_{line}"], timeout_min=300, retries=0, training=True)
    for line, cand in (("S3", [g7, g6]), ("S4", [g6, g7])):
        add(f"gate_{line}", sh(f"gate_{line}.sh",
            f"{PY} train_v5.py --config configs/{line}.json --out {WS}/v5/runs/{line}_probe --probe-steps 2 && "
            f"CKPT=$(find {WS}/v5/runs/{line}_probe -name checkpoint-2 -type d | head -1) && "
            f"test -n \"$CKPT\" && {PY} gate_sft.py --config configs/{line}.json --ckpt \"$CKPT\" "
            f"--out {WS}/v5/runs/{line}_probe/gate_sft.json"),
            cand, after=["eval_L_test200"], timeout_min=20, retries=0, training=True)
        add(f"sft_{line}", sh(f"sft_{line}.sh",
            f"{PY} train_v5.py --config configs/{line}.json --out {WS}/v5/runs/{line}"),
            cand, after=[f"gate_{line}"], timeout_min=300, retries=0, training=True)

    # ── N1 评测（g5–g7 继续）：CNN ──
    add("cnn", sh("cnn.sh",
        f"{PY} cnn_v5.py --train-jsonl {WS}/v5/data/sft_meta_train.jsonl "
        f"--test-jsonl {WS}/v5/data/test_main.jsonl --out {WS}/v5/eval/CNN --seed 3407"),
        LASTRES, after=["eval_base_rot100"], timeout_min=90, timeout_ok=True, retries=0)

    # ── N2：各 SFT 线的评测（select + test200/rot100/shuf50；S3 用 _x2）──
    for line, gcard in (("S1", g1), ("S0", g2), ("S2", g3), ("S1p", g4), ("S3", g7), ("S4", g6)):
        x2 = "_x2" if line == "S3" else ""
        cand = [gcard] + LASTRES
        if line != "S1p":
            add_eval(f"eval_{line}_select", f"select200_main{x2}.jsonl", line, adapter_line=line,
                     gpu=cand, after=[f"sft_{line}"], n=200, out_set="select200")
        add_eval(f"eval_{line}_test200", f"test200_main{x2}.jsonl", line, adapter_line=line,
                 gpu=cand, after=[f"sft_{line}"], n=200, out_set="test200")
        add_eval(f"eval_{line}_rot100", f"rot100_main{x2}.jsonl", line, adapter_line=line,
                 gpu=cand, after=[f"eval_{line}_test200"], n=100, out_set="rot100")
        add_eval(f"eval_{line}_shuf50", f"shuf50_main{x2}.jsonl", line, adapter_line=line,
                 gpu=cand, after=[f"eval_{line}_rot100"], n=50, out_set="shuf50")
    add("select_winner", sh("select_winner.sh",
        f"{PY} select_winner.py --runs-root {WS}/v5/runs --eval-root {WS}/v5/eval "
        f"--l-metrics {WS}/v5/eval/L/select200/metrics.json "
        f"--out {WS}/v5/night/winner.json"),
        "cpu", after=[f"eval_{l}_select" for l in ("S0", "S1", "S2", "S3", "S4")]
        + ["eval_L_select"], timeout_min=10)
    # select_winner 需要 all_finished 语义（S3/S4 可能 partial）→ 由 make 函数后处理
    jobs[-1]["after_mode"] = "all_finished"

    # ── N2：RL 图池筛选 6 分片 + 合并 ──
    for i in range(6):
        add(f"screen_{i}", sh(f"screen_{i}.sh",
            f"WAFER_REWARD_MODE=full {PY} screen_rl_pool.py --shard {i} --nshards 6 "
            f"--winner {WS}/v5/night/winner.json --model {WS}/models/Qwen3.5-9B "
            f"--candidates {WS}/v5/data/rl_candidates800.jsonl "
            f"--out-dir {WS}/v5/night/screen --batch-size {bs}"),
            [gpu_names[i] if i < len(gpu_names) else g1], after=["select_winner"],
            timeout_min=60, retries=1)
    add("screen_merge", sh("screen_merge.sh",
        f"{PY} screen_rl_pool.py --merge --out-dir {WS}/v5/night/screen "
        f"--candidates {WS}/v5/data/rl_candidates800.jsonl "
        f"--template {WS}/v5/data/rsft_rows_template.jsonl --out-pool {WS}/v5/data/rl_pool320.jsonl "
        f"--out-rsft {WS}/v5/data/rsft_pool320.jsonl --passk {WS}/v5/night/passk.json"),
        "cpu", after=[f"screen_{i}" for i in range(6)], timeout_min=15)

    # ── N3：RL 五线 ──
    for line, gcard in (("RSFT", g1), ("Rfull", g2), ("Rbinary", g3), ("Rnoproc", g4), ("Rfullp", g5)):
        add(f"rl_{line}", sh(f"rl_{line}.sh",
            f"{PY} train_v5.py --config configs/{line}.json --out {WS}/v5/runs/{line}"),
            gcard, after=["screen_merge"], timeout_min=210, retries=0, training=True)
    # RL 评测
    for line in ("RSFT", "Rfull", "Rbinary", "Rnoproc", "Rfullp"):
        cand = ALLRES
        add_eval(f"eval_{line}_test200", "test200_main.jsonl", line, adapter_line=line,
                 gpu=cand, after=[f"rl_{line}"], n=200, out_set="test200")
        add_eval(f"eval_{line}_rot100", "rot100_main.jsonl", line, adapter_line=line,
                 gpu=cand, after=[f"eval_{line}_test200"], n=100, out_set="rot100")
        add_eval(f"eval_{line}_shuf50", "shuf50_main.jsonl", line, adapter_line=line,
                 gpu=cand, after=[f"eval_{line}_rot100"], n=50, out_set="shuf50")

    # ── P1：TEST 全量评测（有卡才跑；排在最后）──
    for m, al in (("BASE", None), ("S0", "S0"), ("S1", "S1"), ("RSFT", "RSFT"), ("Rfull", "Rfull")):
        dep = "eval_base_test200" if m == "BASE" else f"eval_{m}_test200"
        add_eval(f"eval_{m}_testfull", "test_main.jsonl", m, adapter_line=al,
                 gpu=LASTRES + [g4, g5], after=[dep], n=589, out_set="testfull")
    # ── P2：线性探针（有卡才跑）──
    add("probe", sh("probe.sh",
        f"{PY} probe_v5.py --model {WS}/models/Qwen3.5-9B "
        f"--train-jsonl {WS}/v5/data/sft_meta_train.jsonl --test-jsonl {WS}/v5/data/test_main.jsonl "
        f"--out {WS}/v5/eval/PROBE --seed 3407"),
        [g7, g6, g5], after=["eval_L_test200"], timeout_min=120, timeout_ok=True, retries=0)

    # ── N5：备份 ──
    add("backup", ["bash", f"{WS}/v5/code/backup.sh"], "cpu", after=[j["id"] for j in jobs],
        timeout_min=30)

    # 依赖全部存在性检查（防手误）
    ids = {j["id"] for j in jobs}
    for j in jobs:
        for d in j["after"]:
            assert d in ids, f"{j['id']} 依赖不存在的 {d}"

    plan = {"gpus": {k: v for k, v in gpus.items()}, "jobs": jobs, "hard_stop": a.hard_stop,
            "generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "deviations": [], "timing": timing}
    Path(a.out).write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8", newline="\n")
    # 预计时间表
    print(f"计划写出 {a.out}：{len(jobs)} 个任务")
    print(f"预计：SFT 单线 {est_sft_min():.0f} 分钟；RL 单线 {est_rl_min():.0f} 分钟；"
          f"评测 {eval_s_per_img}s/张")
    per_card = {k: 0.0 for k in gpu_names}
    for j in jobs:
        if j["gpus"] != ["cpu"]:
            for g in j["gpus"]:
                per_card[g] = per_card.get(g, 0) + j["timeout_min"] / max(1, len(j["gpus"]))
    for k, v in per_card.items():
        print(f"  {k}: 预算上限合计 {v:.0f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
