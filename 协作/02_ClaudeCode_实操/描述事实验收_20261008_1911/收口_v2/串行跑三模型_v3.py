#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D 段 v3：**全局截止时间 + 进程组硬超时**的串行跑法。

Codex 复核发现 v2 的缺口：
  · `串行跑三模型_v2.sh` 只在**下一个模型开始前**查 `USED`，
    **当前这个 Python 进程没有任何 timeout** —— 一旦它卡住，可无限跑下去；
  · 交付的模拟里「预算 3 秒实际记了 4 秒」，
    说明 **18/18 模拟并没有证明 40 分钟是硬上限**。

v3 的做法（每条都对上一个缺口）：
  1. **单调时钟 + 全局截止**：`time.monotonic()` 记 D 段的绝对截止时刻；
  2. **进程组硬超时**：子进程用 `start_new_session=True` 起，
     给它**剩余时间**；到点先 `SIGTERM` 整组、再 `SIGKILL` 整组 ——
     **不只是停止派下一个模型**；
  3. **宽限也计入 40**：终止与 CUDA 释放留 `GRACE` 秒，**这段也在预算内**；
  4. **超时后留证**：原答是评测脚本**逐条 flush** 的，已写盘的不丢；
     另外记录已用秒数、未执行计数；
  5. **退出后核 GPU 释放**：查一次该卡是否还有本进程组的残留；
  6. **失败返回码照实传播**（不因为超时处理而吞掉真实 rc）。

用法:
  CUDA_VISIBLE_DEVICES=<uuid> python 串行跑三模型_v3.py \
      --budget 2400 --grace 30 --ledger <账本> --outdir <原答目录> \
      --model Base=<base> --model L=...=<base>::<adapter> ...
"""
from __future__ import annotations
import argparse, io, json, os, signal, subprocess, sys, time
from pathlib import Path

R = os.environ.get("WS_ROOT", "/WS")
# 解释器路径可覆盖：学校机器在 /WS/envs/wafer/bin/python，
# 租用实例用自带的 conda base。默认值保持与原环境一致。
PY = os.environ.get("D_PY", f"{R}/envs/wafer/bin/python")


def gpu_free(dev: str):
    """查指定卡上还有没有进程。查询失败返回 None（**不与「0 个进程」混为一谈**）。"""
    try:
        p = subprocess.run(["timeout", "15", "nvidia-smi", "-i", dev,
                            "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=25)
    except Exception:
        return None
    if p.returncode != 0:
        return None
    return sum(1 for l in p.stdout.splitlines() if l.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True, help="D 段总预算（秒）")
    ap.add_argument("--grace", type=int, default=30, help="终止+CUDA释放宽限（秒，计入预算）")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", action="append", required=True,
                    help="tag=base 或 tag=base::adapter，可重复")
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    a = ap.parse_args()

    START = time.monotonic()
    DEADLINE = START + a.budget
    # 目录必须先建：日志目录缺失会让 v3 在开日志时就崩，
    # 而那时账本还没写任何行 —— 表现成「账本为空、瞬间退出」，很难查。
    Path(a.ledger).parent.mkdir(parents=True, exist_ok=True)
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    Path(f"{R}/logs").mkdir(parents=True, exist_ok=True)
    ledger = io.open(a.ledger, "a", encoding="utf-8", newline="\n")

    def rec(tag, status, used, note=""):
        ledger.write(f"{tag}\t{status}\t{used:.1f}\t{note}\n"); ledger.flush()

    print(f"=== D-v3 开始 预算 {a.budget}s 宽限 {a.grace}s "
          f"GPU={a.gpu!r} {time.strftime('%F %T')}", flush=True)
    ledger.write(f"# v3 开始 预算={a.budget}s 宽限={a.grace}s "
                 f"{time.strftime('%F %T')}\n"); ledger.flush()

    n_run = n_fail = n_notrun = 0
    exit_rc = 0
    for spec in a.model:
        tag, _, path = spec.partition("=")
        remain = DEADLINE - time.monotonic()
        if remain <= a.grace:
            rec(tag, "NOT_RUN", 0.0, f"预算耗尽（剩 {remain:.0f}s ≤ 宽限 {a.grace}s）")
            print(f"**预算耗尽，未启动 {tag}**", flush=True)
            n_notrun += 1
            continue

        limit = remain - a.grace          # 留给终止与显存释放
        log = io.open(f"{R}/logs/existence_v3_{tag}.log", "a",
                      encoding="utf-8", newline="\n")
        cmd = [PY, f"{R}/existence/eval_existence.py",
               "--model", f"{tag}={path}", "--outdir", a.outdir,
               "--assets", f"{R}/existence/派生图"]
        t0 = time.monotonic()
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)   # ← 自成进程组，便于整组杀
        killed = None
        try:
            rc = p.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            killed = "剩余时间用尽"
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=max(5, a.grace // 2))
                killed += " → SIGTERM 后退出"
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                try:
                    p.wait(timeout=10)
                    killed += " → SIGKILL 后退出"
                except subprocess.TimeoutExpired:
                    killed += " → **SIGKILL 后仍未退出**"
            rc = -9
        used = time.monotonic() - t0
        log.close()

        if killed:
            rec(tag, "TIMEOUT", used, f"{killed}；原答已 flush 的部分保留")
            print(f"**{tag} 超时（{used:.0f}s）：{killed}**", flush=True)
            n_fail += 1; exit_rc = 3
        elif rc == 0:
            rec(tag, "OK", used)
            n_run += 1                      # ← 首版漏了这一行，导致汇总行写「跑 0」
            print(f"{tag} 完成 {used:.0f}s", flush=True)
        else:
            rec(tag, "FAILED", used, f"rc={rc}（已计入预算）")
            print(f"**{tag} 失败 rc={rc}，耗时 {used:.0f}s 已计入**", flush=True)
            n_fail += 1; exit_rc = 1

    total = time.monotonic() - START
    ledger.write(f"# v3 结束 总用 {total:.0f}s/{a.budget}s，跑 {n_run}，失败 {n_fail}，"
                 f"未启动 {n_notrun} {time.strftime('%F %T')}\n")
    ledger.close()

    # ---- 退出后核 GPU 是否真的释放 ----
    dev = a.gpu.strip()
    left = gpu_free(dev) if dev else None
    print(f"=== GPU 残留检查：{'查询失败(未知)' if left is None else str(left)+' 个进程'}")
    if left is None:
        exit_rc = exit_rc or 5
    elif left > 0:
        print("**该卡上仍有进程 —— 需要人工确认是否为本任务残留**")
        exit_rc = exit_rc or 5

    print(f"=== 总用 {total:.0f}s / {a.budget}s（硬上限）")
    if total > a.budget:
        print("**超出了硬预算 —— 记录为超支，不掩饰**")
        exit_rc = exit_rc or 3
    if n_notrun:
        exit_rc = exit_rc or 7
    if n_run == 0 and n_fail == 0:
        exit_rc = exit_rc or 4
    # 返回码**优先级**（同时命中多个时取最靠前的那个；全部情形都已写进账本，
    # 所以不存在「信息丢失」，只是单一返回码装不下多个状态）：
    #   4 一个都没跑  >  3 超时（撞上硬截止，最严重）  >  1 有模型失败  >  7 部分完成  >  0 全成功
    return exit_rc


if __name__ == "__main__":
    sys.exit(main())
