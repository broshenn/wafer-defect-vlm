#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rfull 冒烟段重跑（首次因 rl_pool320 尚未生成而崩；改用 rl_candidates800）。
更新 smoke_report.json 三个阶段 + smoke_timing.json 的 rl_s_per_step，并写入 note。"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WS = os.environ["WS_ROOT"]
PY = f"{WS}/envs/wafer/bin/python"
V5 = Path(WS) / "v5"
CODE = V5 / "code"
LOGS = V5 / "night" / "logs"


def find_ckpt(d):
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


s1_ckpt = find_ckpt(V5 / "runs" / "S1_probe")
cfg = json.loads((CODE / "configs" / "Rfull.json").read_text(encoding="utf-8"))
cfg["start_adapter"] = str(s1_ckpt)
cfg["line"] = "Rfull_probe"
cfg["data"] = str(V5 / "data" / "rl_candidates800.jsonl")
cfg["image_max_token_num"] = 256
(CODE / "configs" / "Rfull_probe.json").write_text(
    json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")

t0 = time.time()
with open(LOGS / "smoke_Rfull_probe_rerun.log", "w", encoding="utf-8") as lf:
    rc = subprocess.run([PY, "train_v5.py", "--config", "configs/Rfull_probe.json",
                         "--out", str(V5 / "runs" / "Rfull_probe"), "--probe-steps", "2"],
                        cwd=CODE, stdout=lf, stderr=subprocess.STDOUT, timeout=1800).returncode
dt = time.time() - t0
text = (LOGS / "smoke_Rfull_probe_rerun.log").read_text(encoding="utf-8", errors="ignore")
sit = re.findall(r"([\d.]+)s/it", text)
rate = float(sit[-1]) if sit else None
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
ok2 = (rc == 0)
ok_gate = (gate_j.get("policy_equal_M0_tensor_count", 0) > 0
           and gate_j.get("reference_equal_M0_tensor_count", 0) > 0
           and gate_j.get("two_step_gate") == "passed")
ok_audit = n_audit > 0 and n_not_minus1 > 0

rp = V5 / "night" / "smoke_report.json"
rep = json.loads(rp.read_text(encoding="utf-8"))
note = ("首轮 Rfull 探针 rc=1 因 rl_pool320.jsonl 尚未生成（夜跑时才生成）；"
        "按 C 卡意图改用 rl_candidates800.jsonl 作探针数据重跑，结果覆盖此三项。")
rep["stages"]["Rfull_probe2"] = {"ok": ok2, "info": f"rc={rc} {dt:.0f}s rate={rate}（重跑）"}
rep["stages"]["runtime_gate"] = {"ok": ok_gate, "info": gate_j}
rep["stages"]["reward_audit"] = {"ok": ok_audit, "info": f"审计 {n_audit} 条，非 -1 的 {n_not_minus1} 条"}
rep.setdefault("notes", []).append(note)
rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")

tp = V5 / "night" / "smoke_timing.json"
tim = json.loads(tp.read_text(encoding="utf-8"))
if rate:
    tim["rl_s_per_step"] = rate
tp.write_text(json.dumps(tim, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print(f"Rfull 重跑: rc={rc} {dt:.0f}s rate={rate} gate_ok={ok_gate} audit={n_not_minus1}/{n_audit}")
print("timing:", tim)
sys.exit(0 if (ok2 and ok_gate and ok_audit) else 1)
