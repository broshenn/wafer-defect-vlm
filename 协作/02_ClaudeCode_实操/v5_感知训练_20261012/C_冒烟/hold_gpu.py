#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""占位锁卡进程（被 launch_holds.sh 以 CUDA_VISIBLE_DEVICES=<uuid> 拉起）。
用途：在空闲卡上分配一块显存并保持，防止别人占用；到点/收到信号/发现停止文件即释放退出。
输入：--outdir（holds 目录，写 <uuid>.json 记录本进程 pid）、--stop-file、--mem-gb、--duration-s。
退出：正常释放退出码 0；分配失败退出码 2。只释放自己，不影响任何他人进程。
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--mem-gb", type=float, default=24.0)
    ap.add_argument("--duration-s", type=int, default=12600)  # 默认 3.5h 自动过期兜底
    args = ap.parse_args()

    if torch.cuda.device_count() != 1:
        print(f"device_count={torch.cuda.device_count()}，必须单卡（CUDA_VISIBLE_DEVICES 未生效）", flush=True)
        return 2
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "NA")
    nbytes = int(args.mem_gb * (1024 ** 3))
    try:
        buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        buf[::4096] = 1  # 触页，确保显存真实驻留
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"显存分配失败: {type(exc).__name__}: {exc}", flush=True)
        return 2

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    info = {"pid": os.getpid(), "uuid": uuid, "mem_gb": args.mem_gb,
            "started_epoch": int(time.time()), "started": time.strftime("%F %T"),
            "duration_s": args.duration_s, "stop_file": args.stop_file}
    Path(args.outdir, f"{uuid}.json").write_text(json.dumps(info), encoding="utf-8")
    print(f"HOLD pid={info['pid']} uuid={uuid} mem={args.mem_gb}GB", flush=True)

    running = True

    def on_term(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    deadline = time.time() + args.duration_s
    while running:
        if os.path.exists(args.stop_file):
            print("发现停止文件，释放", flush=True)
            break
        if time.time() > deadline:
            print("到期自动释放", flush=True)
            break
        time.sleep(15)
    del buf
    torch.cuda.empty_cache()
    print("HOLD released", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
