#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C 阶段冒烟驱动（服务器端；父 shell 必须先 CUDA_VISIBLE_DEVICES=<一张空卡UUID>）。
依次执行并把结果写 night/smoke_report.json + night/smoke_timing.json：
  1) 纯 Python 测试（tests/*.py，服务器环境再跑一遍）
  2) S1 2 步 probe + gate；S2 2 步 probe + gate（必须 visual 键>0）
  3) Rfull 2 步 probe（起点=S1 probe checkpoint-2，生成 Rfull_probe.json）：
     runtime_gate.json 的 policy/reference 张量计数 >0；reward_audit 非全 -1
  4) eval 10 张 + score_v5 出 metrics
  5) scheduler 假任务（sleep 5）×3 + 1 失败连锁 + 1 all_finished：DONE/FAILED/SKIPPED 正确
  6) 从日志解析 s/step 与 s/张，写 smoke_timing.json
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WS = os.environ.get("WS_ROOT")
PY = f"{WS}/envs/wafer/bin/python"
V5 = Path(WS) / "v5"
CODE = V5 / "code"
NIGHT = V5 / "night"
NIGHT.mkdir(parents=True, exist_ok=True)
LOGS = NIGHT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
report = {"stages": {}, "timing": {}}


def run(cmd, log_name, cwd=CODE, env_extra=None, timeout=1800):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    with open(LOGS / log_name, "w", encoding="utf-8") as lf:
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                           timeout=timeout)
    return p.returncode, time.time() - t0, (LOGS / log_name)


def parse_rate(log_path):
    """从日志里找最后一个 '\d.\ds/it' 或 '\d.\dit/s'。"""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    sit = re.findall(r"([\d.]+)s/it", text)
    its = re.findall(r"([\d.]+)it/s", text)
    if sit:
        return float(sit[-1]), "s/it"
    if its:
        return 1.0 / float(its[-1]), "it/s"
    return None, None


def stage(name, ok, info=""):
    report["stages"][name] = {"ok": bool(ok), "info": str(info)[:500]}
    print(f"[smoke] {name}: {'PASS' if ok else 'FAIL'} {str(info)[:200]}", flush=True)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "父 shell 未锁卡"
    import torch
    torch.cuda.init()
    assert torch.cuda.device_count() == 1, f"必须真单卡，实际 {torch.cuda.device_count()}"
    print("冒烟卡:", torch.cuda.get_device_name(0))

    # 1) 纯测试
    tests = ["test_common", "test_reward", "test_configs", "test_score", "test_scheduler_logic",
             "test_rewrite", "test_screen_merge", "test_bootstrap", "test_gate"]
    n_ok = 0
    fails = []
    for t in tests:
        rc, dt, log = run([PY, f"tests/{t}.py"], f"smoke_{t}.log", timeout=600)
        if rc == 0:
            n_ok += 1
        else:
            fails.append(t)
    stage("pure_tests", n_ok == len(tests), f"{n_ok}/{len(tests)}；fail={fails}")
    rc, dt, log = run([PY, "tests/test_winner_and_plan.py"], "smoke_test_winner.log", timeout=600)
    stage("test_winner_and_plan", rc == 0, "see log")

    # 2) S1/S2 probe + gate
    timing = {}
    for line in ("S1", "S2"):
        outdir = V5 / "runs" / f"{line}_probe"
        rc, dt, log = run([PY, "train_v5.py", "--config", f"configs/{line}.json",
                           "--out", str(outdir), "--probe-steps", "2"], f"smoke_{line}_probe.log",
                          timeout=1800)
        s_raw = None
        if rc == 0:
            v5cfg = outdir / "v5_config.json"
            if v5cfg.exists():
                cfg = json.loads(v5cfg.read_text(encoding="utf-8"))
                timing[f"{line}_probe"] = cfg
        rate, unit = parse_rate(LOGS / f"smoke_{line}_probe.log")
        stage(f"{line}_probe2", rc == 0, f"rc={rc} {dt:.0f}s rate={rate} {unit}")
        gate_out = outdir / "gate_sft.json"
        try:
            ck = find_ckpt(outdir)
        except SystemExit as exc:
            stage(f"{line}_gate", False, f"找不到 checkpoint：{exc}")
            continue
        rc2, dt2, log2 = run([PY, "gate_sft.py", "--config", f"configs/{line}.json",
                              "--ckpt", str(ck), "--out", str(gate_out)],
                             f"smoke_{line}_gate.log", timeout=300)
        g = json.loads(gate_out.read_text(encoding="utf-8")) if gate_out.exists() else {}
        if line == "S2":
            stage("S2_gate_visual", rc2 == 0 and g.get("visual_key_count", 0) > 0, g)
        else:
            stage(f"{line}_gate", rc2 == 0 and g.get("visual_key_count") == 0, g)
        if rate:
            timing[f"{line}_s_per_step"] = rate

    # 3) Rfull probe（起点= S1 probe checkpoint-2）
    try:
        s1_ckpt = find_ckpt(V5 / "runs" / "S1_probe")
    except SystemExit as exc:
        stage("Rfull_probe2", False, f"无 S1 checkpoint 作起点：{exc}")
        s1_ckpt = None
    if s1_ckpt is None:
        # 直接跳到 scheduler 假任务与计时汇总
        pass
    cfg = json.loads((CODE / "configs" / "Rfull.json").read_text(encoding="utf-8"))
    cfg["start_adapter"] = str(s1_ckpt)
    cfg["line"] = "Rfull_probe"
    cfg["data"] = str(V5 / "data" / "rl_candidates800.jsonl")
cfg["image_max_token_num"] = 256  # 冒烟时 rl_pool320 尚未生成
    (CODE / "configs" / "Rfull_probe.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    rc, dt, log = run([PY, "train_v5.py", "--config", "configs/Rfull_probe.json",
                       "--out", str(V5 / "runs" / "Rfull_probe"), "--probe-steps", "2"],
                      "smoke_Rfull_probe.log", timeout=1800)
    rate, unit = parse_rate(LOGS / "smoke_Rfull_probe.log")
    if rate:
        timing["rl_s_per_step"] = rate
    gate = V5 / "runs" / "Rfull_probe" / "runtime_gate.json"
    audit = V5 / "runs" / "Rfull_probe" / "reward_audit.jsonl"
    gate_j = json.loads(gate.read_text(encoding="utf-8")) if gate.exists() else {}
    n_audit = n_not_minus1 = 0
    if audit.exists():
        for l in audit.read_text(encoding="utf-8").splitlines():
            if l.strip():
                n_audit += 1
                if json.loads(l).get("reward", -1) != -1:
                    n_not_minus1 += 1
    stage("Rfull_probe2", rc == 0, f"rc={rc} {dt:.0f}s rate={rate}")
    stage("runtime_gate", gate_j.get("policy_equal_M0_tensor_count", 0) > 0
          and gate_j.get("reference_equal_M0_tensor_count", 0) > 0
          and gate_j.get("two_step_gate") == "passed", gate_j)
    stage("reward_audit", n_audit > 0 and n_not_minus1 > 0,
          f"审计 {n_audit} 条，非 -1 的 {n_not_minus1} 条")

    # 4) eval 10 张 + score
    smoke_data = V5 / "data" / "test200_smoke10.jsonl"
    lines = (V5 / "data" / "test200_main.jsonl").read_text(encoding="utf-8").splitlines()[:10]
    smoke_data.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    rc, dt, log = run([PY, "eval_v5.py", "--model", f"{WS}/models/Qwen3.5-9B",
                       "--adapter-from-line", "S1_probe", "--runs-root", str(V5 / "runs"),
                       "--tag", "SMOKE", "--dataset", str(smoke_data),
                       "--out", str(V5 / "eval" / "SMOKE" / "test200_smoke10"),
                       "--max-tokens", "384", "--batch-size", "4"], "smoke_eval10.log", timeout=1800)
    ledger = V5 / "eval" / "SMOKE" / "test200_smoke10" / "台账_SMOKE.jsonl"
    avg_s = load_s = None
    if ledger.exists():
        for l in ledger.read_text(encoding="utf-8").splitlines():
            d = json.loads(l)
            if d.get("事件") == "结束":
                avg_s = d.get("均秒每张")
            if d.get("事件") == "模型加载":
                load_s = d.get("秒")
    n_raw = 0
    rawf = V5 / "eval" / "SMOKE" / "test200_smoke10" / "SMOKE_raw.jsonl"
    if rawf.exists():
        n_raw = sum(1 for l in rawf.read_text(encoding="utf-8").splitlines() if l.strip())
    stage("eval10", rc == 0 and n_raw >= 8, f"rc={rc} raw={n_raw} avg={avg_s}s/张 load={load_s}s")
    if avg_s:
        timing["eval_s_per_img"] = avg_s
    if load_s:
        timing["load_s"] = load_s
    rc, dt, log = run([PY, "score_v5.py", "--raw", str(rawf), "--ref", str(smoke_data),
                       "--kind", "main", "--out", str(V5 / "eval" / "SMOKE" / "test200_smoke10" / "metrics.json"),
                       "--per-sample", str(V5 / "eval" / "SMOKE" / "test200_smoke10" / "per_sample.jsonl")],
                      "smoke_score10.log", timeout=300)
    stage("score10", rc == 0, f"rc={rc}")

    # 5) scheduler 假任务
    fake_dir = NIGHT / "sched_fake"
    fake_dir.mkdir(exist_ok=True)
    fake_plan = {
        "gpus": {}, "hard_stop": "23:59",
        "jobs": [
            {"id": "t_ok", "gpus": ["cpu"], "cmd": ["bash", "-c", "sleep 5; exit 0"], "after": [],
             "after_mode": "all_ok", "timeout_min": 2, "timeout_ok": True, "retries": 0,
             "log": str(fake_dir / "t_ok.log")},
            {"id": "t_fail", "gpus": ["cpu"], "cmd": ["bash", "-c", "sleep 5; exit 3"], "after": [],
             "after_mode": "all_ok", "timeout_min": 2, "timeout_ok": False, "retries": 0,
             "log": str(fake_dir / "t_fail.log")},
            {"id": "t_skip", "gpus": ["cpu"], "cmd": ["bash", "-c", "exit 0"],
             "after": ["t_fail"], "after_mode": "all_ok", "timeout_min": 2, "timeout_ok": False,
             "retries": 0, "log": str(fake_dir / "t_skip.log")},
            {"id": "t_allfin", "gpus": ["cpu"], "cmd": ["bash", "-c", "exit 0"],
             "after": ["t_fail"], "after_mode": "all_finished", "timeout_min": 2,
             "timeout_ok": False, "retries": 0, "log": str(fake_dir / "t_allfin.log")},
        ]}
    (fake_dir / "plan.json").write_text(json.dumps(fake_plan, ensure_ascii=False, indent=1),
                                        encoding="utf-8", newline="\n")
    rc, dt, log = run([PY, "scheduler.py", "--plan", str(fake_dir / "plan.json"),
                       "--state", str(fake_dir / "state.json"), "--poll", "3"],
                      "smoke_sched.log", timeout=300)
    st = json.loads((fake_dir / "state.json").read_text(encoding="utf-8"))["jobs"]
    ok_states = (st["t_ok"]["status"] == "DONE" and st["t_fail"]["status"] == "FAILED"
                 and st["t_skip"]["status"] == "SKIPPED" and st["t_allfin"]["status"] == "DONE")
    stage("scheduler_fake", rc == 0 and ok_states, {k: v["status"] for k, v in st.items()})

    timing["eval_batch_size"] = 4
    (NIGHT / "smoke_timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    report["timing"] = timing
    (NIGHT / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    n_fail = sum(1 for s in report["stages"].values() if not s["ok"])
    print(f"\n冒烟完成：{len(report['stages'])} 项，失败 {n_fail}；timing={timing}")
    return 0 if n_fail == 0 else 1


def find_ckpt(d):
    d = Path(d)
    best = None
    for p in d.rglob("checkpoint-*"):
        try:
            s = int(p.name.split("checkpoint-")[1])
        except ValueError:
            continue
        if best is None or s > best[0]:
            best = (s, p)
    if best is None:
        raise SystemExit(f"找不到 checkpoint: {d}")
    return best[1]


if __name__ == "__main__":
    sys.exit(main())
