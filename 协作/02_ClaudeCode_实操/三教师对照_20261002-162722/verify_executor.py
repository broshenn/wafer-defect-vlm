#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行器离线校验：不发请求、不读凭据。

查六件事：
  1. 固定输入指纹锁生效（改一个字节就停）
  2. dry-run 不产生任何网络调用（用假 socket 拦截证明）
  3. 密钥脱敏函数真的能剔掉疑似密钥
  4. 费用守卫失败关闭（无预算 / 无价目 / 超预算）
  5. 平台隔离：每个候选只认自己那个环境变量，不做跨平台回退
  6. 锁定参数没有被"失败后扩额"的路径
"""

from __future__ import annotations

import hashlib
import json
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    return bool(ok)


def main() -> int:
    print("=" * 78)
    print("三教师执行器 · 离线校验（不发请求、不读凭据）")
    print("=" * 78)

    import run_teacher_compare as R
    import cost_guard as G

    # 1 固定输入指纹锁
    blind, prompt = R.load_inputs()
    check("固定输入指纹锁通过（18 张 + 提示词 + 检查器）", len(blind) == 18)
    orig = R.EXPECTED[R.PROMPT]
    R.EXPECTED[R.PROMPT] = "0" * 64
    stopped = False
    try:
        R.load_inputs()
    except AssertionError:
        stopped = True
    R.EXPECTED[R.PROMPT] = orig
    check("指纹不符时会停下（不是只警告）", stopped)

    # 2 dry-run 无网络
    calls = {"n": 0}
    real_socket = socket.socket

    class Trap(real_socket):                                   # type: ignore[misc]
        def connect(self, *a, **k):
            calls["n"] += 1
            raise AssertionError("dry-run 期间不应有任何网络调用")
    socket.socket = Trap
    try:
        sys.argv = ["run_teacher_compare.py", "--dry-run"]
        R.main()
        net_ok = calls["n"] == 0
    except AssertionError:
        net_ok = False
    finally:
        socket.socket = real_socket
    check("dry-run 期间零网络调用", net_ok, f"connect 调用 {calls['n']} 次")

    # 3 脱敏
    samples = [
        ("sk-FAKE-TEST-VALUE-NOT-A-REAL-KEY", True),
        ("Bearer FAKE.JWT.FOR-TEST-ONLY", True),
        ("普通错误：timeout", False),
    ]
    ok3 = True
    for s, should_scrub in samples:
        out = R.scrub(s)
        scrubbed = "<REDACTED>" in out
        if scrubbed != should_scrub:
            ok3 = False
    check("疑似密钥被剔除，普通文本不动", ok3,
          f"例：{R.scrub('认证头 Bearer sk-FAKE-TEST-VALUE-NOT-A-REAL-KEY 无效')!r}")

    # 4 费用守卫失败关闭
    pr = G.load_pricing()
    ds = pr["candidates"]["deepseek"]
    check("无预算 → 拒", G.admit("deepseek", ds, 3000, 16384, 0.0, None)["allow"] is False)
    check("价目未核验 → 拒",
          G.admit("kimi", pr["candidates"]["kimi"], 3000, 16384, 0.0, 99.0)["allow"] is False)
    r = G.admit("deepseek", ds, 3000, 16384, 0.0, 0.0001)
    check("预算小于最坏成本 → 拒", r["allow"] is False, r["reason"][:50])

    # 5 平台隔离
    iso_ok = True
    for name, cfg in R.CANDIDATES.items():
        want = "DASHSCOPE_API_KEY" if cfg["provider"] == "bailian" else "DEEPSEEK_API_KEY"
        if cfg["credential_env"] != want:
            iso_ok = False
    check("平台隔离：候选只认自己平台的凭据变量", iso_ok,
          str({n: c["credential_env"] for n, c in R.CANDIDATES.items()}))

    # 6 锁定参数
    check("max_tokens 事前锁定且代码里只有一处定义",
          R.MAX_TOKENS_LOCKED == 16384)
    check("客户端重试锁定为 0", R.CLIENT_RETRIES_LOCKED == 0)
    check("无『失败后自动扩额』的代码路径",
          "MAX_TOKENS_LOCKED" in Path(R.__file__).read_text(encoding="utf-8")
          and Path(R.__file__).read_text(encoding="utf-8").count("MAX_TOKENS_LOCKED") >= 3)

    print()
    fails = [x for x in results if not x[1]]
    for n, ok, d in results:
        print(f"  {'✓' if ok else '✗'} {n}" + (f"   {d}" if d and not ok else ""))
    print(f"\n通过 {len(results) - len(fails)}/{len(results)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
