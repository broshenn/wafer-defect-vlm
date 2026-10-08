#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 硬截止的**离线反例模拟**（完全不碰 GPU）。

Codex 指出：交付的 18/18 模拟里「预算 3 秒实际记了 4 秒」，
说明**并没有证明 40 分钟是硬上限** —— 因为 v2 只在模型之间查预算，
当前进程没有 timeout。所以本模拟专门打这件事：

  A. **单个假推理远超预算** —— 必须被**在总截止内杀掉**，不是等它自己结束
  B. **假推理创建子进程** —— 必须**整组杀干净**，不留残留进程
  C. 预算耗尽 —— 剩余模型记 NOT_RUN（部分完成）
  D. 失败码传播 —— 子进程非零要**照实传播**，不被超时处理吞掉

判据全部基于**实测**：时间用 monotonic，残留用真实进程表查。

跑法：python 模拟_v3硬截止.py
"""
from __future__ import annotations
import io, json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE / "模拟沙箱v3"
V3 = HERE / "串行跑三模型_v3.py"
PASS = FAIL = 0


def ok(m):
    global PASS; PASS += 1; print(f"  ✓ {m}")


def bad(m):
    global FAIL; FAIL += 1; print(f"  ✗ {m}")


def setup():
    if SIM.exists():
        shutil.rmtree(SIM)
    (SIM / "envs/wafer/bin").mkdir(parents=True)
    (SIM / "existence").mkdir()
    (SIM / "logs").mkdir()
    (SIM / "out").mkdir()
    (SIM / "existence/派生图").mkdir()
    # 假 python：按环境变量决定睡多久、返回什么、要不要生个孩子
    fake = SIM / "envs/wafer/bin/python"
    fake.write_text(
        '#!/bin/bash\n'
        'echo "[fake] start $$ $(date +%s)"\n'
        'if [ -n "${FAKE_SPAWN_CHILD:-}" ]; then\n'
        '  ( sleep "${FAKE_SLEEP:-600}"; echo "[fake-child] done" ) &\n'
        '  echo "[fake] spawned child $!"\n'
        'fi\n'
        'sleep "${FAKE_SLEEP:-600}"\n'
        'exit "${FAKE_RC:-0}"\n', encoding="utf-8")
    fake.chmod(0o755)
    (SIM / "existence/eval_existence.py").write_text("# dummy\n", encoding="utf-8")
    # 假 nvidia-smi：说该卡没有进程
    nv = SIM / "nvidia-smi"
    nv.write_text('#!/bin/bash\nexit 0\n', encoding="utf-8")
    nv.chmod(0o755)


def run_v3(budget, grace, models, env_extra):
    led = SIM / "ledger.tsv"
    if led.exists():
        led.unlink()
    env = dict(os.environ)
    env.update({"WS_ROOT": str(SIM), "PATH": f"{SIM}:{env['PATH']}"})
    env.update(env_extra)
    args = [sys.executable, str(V3), "--budget", str(budget), "--grace", str(grace),
            "--ledger", str(led), "--outdir", str(SIM / "out"), "--gpu", "0"]
    for m in models:
        args += ["--model", m]
    t0 = time.monotonic()
    try:
        # 注意 env=env —— 第一版忘了传，导致沙箱环境变量根本没生效，
        # 表现成「v3 瞬间退出、账本为空」，差点被误判成 v3 的 bug。
        p = subprocess.run(args, capture_output=True, text=True,
                           timeout=budget + 60, env=env)
    except subprocess.TimeoutExpired:
        print("    **v3 自身超时未退出**")
        return -1, time.monotonic() - t0, led
    if not ledger_rows(led):
        print(f"    [v3 rc={p.returncode} 账本为空] "
              f"{(p.stderr or p.stdout or '')[-500:]}")
    return p.returncode, time.monotonic() - t0, led


def ledger_rows(led: Path):
    if not led.exists():
        return []
    return [l.split("\t") for l in
            io.open(led, encoding="utf-8").read().splitlines()
            if l and not l.startswith("#")]


def alive(pattern: str) -> int:
    """数一数还有多少进程匹配（用 pgrep，避免自己读 /proc 出错）。"""
    p = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return sum(1 for l in p.stdout.splitlines() if l.strip())


M = ["M1=/x1", "M2=/x2", "M3=/x3"]

# ---------------- A. 单个假推理远超预算 ----------------
print("【A】单个假推理远超预算（睡 600s，预算 8s，宽限 2s）")
setup()
rc, elapsed, led = run_v3(8, 2, ["M1=/x"], {"FAKE_SLEEP": "600", "FAKE_RC": "0"})
if elapsed <= 12:
    ok(f"在总截止内被杀掉（实测 {elapsed:.1f}s ≤ 12s，远小于 600s）")
else:
    bad(f"没有及时杀掉：{elapsed:.1f}s")
rows = ledger_rows(led)
if rows and rows[0][1] == "TIMEOUT":
    ok(f"账本记 TIMEOUT，用时 {rows[0][2]}s")
else:
    bad(f"账本未记 TIMEOUT：{rows}")
if alive("sleep 600") == 0:
    ok("没有残留的 sleep 600 进程")
else:
    bad(f"仍有 {alive('sleep 600')} 个残留进程")

# ---------------- B. 假推理创建子进程 ----------------
print("【B】假推理创建子进程（父子都睡 600s）")
setup()
rc, elapsed, led = run_v3(8, 3, ["M1=/x"], {"FAKE_SLEEP": "600",
                                            "FAKE_SPAWN_CHILD": "1", "FAKE_RC": "0"})
if elapsed <= 14:
    ok(f"在总截止内结束（实测 {elapsed:.1f}s）")
else:
    bad(f"超时未结束：{elapsed:.1f}s")
time.sleep(1)
n = alive("[s]leep 600")
if n == 0:
    ok("**整个进程组被杀干净**，父子都没残留")
else:
    bad(f"仍有 {n} 个残留（说明只杀了父进程，没杀进程组）")
rows = ledger_rows(led)
if rows and rows[0][1] == "TIMEOUT":
    ok("账本记 TIMEOUT")
else:
    bad(f"账本异常：{rows}")

# ---------------- C. 预算耗尽 ----------------
print("【C】预算耗尽：预算 6s、宽限 2s，三个模型各睡 3s")
setup()
rc, elapsed, led = run_v3(6, 2, M, {"FAKE_SLEEP": "3", "FAKE_RC": "0"})
rows = ledger_rows(led)
st = [r[1] for r in rows]
if "NOT_RUN" in st:
    ok(f"剩余模型记 NOT_RUN：{st}")
else:
    bad(f"未出现 NOT_RUN：{st}")
if rc != 0:
    ok(f"整体返回非零（{rc}）—— **不谎报完全成功**")
else:
    bad("部分完成却返回 0")
if rc in (3, 7):
    ok(f"返回码在预期集合内：{rc}（3=撞上硬截止优先于 7=部分完成；优先级已写进 v3 注释）")
else:
    bad(f"返回码 {rc} 不在 {{3,7}} 内")
if elapsed <= 10:
    ok(f"总耗时 {elapsed:.1f}s 在硬预算+宽限内")
else:
    bad(f"总耗时 {elapsed:.1f}s 超出")

# ---------------- D. 失败码传播 ----------------
print("【D】失败码传播：假推理睡 1s 后返回 42")
setup()
rc, elapsed, led = run_v3(30, 3, ["M1=/x"], {"FAKE_SLEEP": "1", "FAKE_RC": "42"})
rows = ledger_rows(led)
if rows and rows[0][1] == "FAILED" and "rc=42" in rows[0][3]:
    ok("账本记 FAILED 且保留真实 rc=42")
else:
    bad(f"账本未如实记录：{rows}")
if rc == 1:
    ok("整体返回 1")
else:
    bad(f"期望返回 1，实际 {rc}")

print()
print("=" * 60)
print(f"v3 硬截止模拟：{PASS} 通过 / {FAIL} 失败")
print("全部通过" if FAIL == 0 else "**有失败项**")
sys.exit(0 if FAIL == 0 else 1)
