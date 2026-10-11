#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scheduler.py 纯函数测试（就绪判断、连锁跳过、选卡）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scheduler import dep_failed_hard, dep_satisfied, job_readiness, pick_gpu  # noqa: E402


def main():
    results = []
    add = lambda n, c, x="": results.append((n, bool(c), x))  # noqa: E731

    # 就绪
    jobs = {"a": {"id": "a", "after": [], "after_mode": "all_ok"},
            "b": {"id": "b", "after": ["a"], "after_mode": "all_ok", "dep_timeout_ok": {"a": False}},
            "c": {"id": "c", "after": ["a"], "after_mode": "all_finished", "dep_timeout_ok": {"a": False}},
            "d": {"id": "d", "after": ["a"], "after_mode": "all_ok", "dep_timeout_ok": {"a": True}}}
    add("无依赖 PENDING → ready", job_readiness(jobs["a"], {"a": "PENDING"}, {}) == (True, "ready"))
    add("依赖 RUNNING → wait", not job_readiness(jobs["b"], {"b": "PENDING", "a": "RUNNING"}, {})[0])
    add("依赖 DONE → ready", job_readiness(jobs["b"], {"b": "PENDING", "a": "DONE"}, {}) == (True, "ready"))
    add("依赖 FAILED all_ok → skip", job_readiness(jobs["b"], {"b": "PENDING", "a": "FAILED"}, {})[1].startswith("skip:"))
    add("依赖 TIMEOUT(timeout_ok=True) → ready",
        job_readiness(jobs["d"], {"d": "PENDING", "a": "TIMEOUT"}, {}) == (True, "ready"))
    add("依赖 TIMEOUT(timeout_ok=False) → skip",
        job_readiness(jobs["b"], {"b": "PENDING", "a": "TIMEOUT"}, {})[1].startswith("skip:"))
    add("all_finished 依赖 FAILED → ready",
        job_readiness(jobs["c"], {"c": "PENDING", "a": "FAILED"}, {}) == (True, "ready"))
    add("非 PENDING 不重复就绪", not job_readiness(jobs["a"], {"a": "RUNNING"}, {})[0])

    # 连锁
    add("DONE 不算失败", dep_failed_hard("DONE", False, "all_ok") is False)
    add("SKIPPED 是失败侧", dep_failed_hard("SKIPPED", False, "all_ok") is True)
    add("all_finished 不看失败", dep_failed_hard("FAILED", False, "all_finished") is False)
    add("dep_satisfied: TIMEOUT+ok", dep_satisfied("TIMEOUT", True, "all_ok") is True)

    # 选卡
    gpus = {"g1": "uuid-1", "g2": "uuid-2", "g3": "uuid-3"}
    add("cpu 直通", pick_gpu(["cpu"], set(), gpus, set()) == ("cpu", None))
    add("第一张空闲卡被选", pick_gpu(["g1", "g2"], set(), gpus, set()) == ("g1", "uuid-1"))
    add("本调度器占用则跳过", pick_gpu(["g1", "g2"], {"g1"}, gpus, set()) == ("g2", "uuid-2"))
    add("他人进程占用则跳过", pick_gpu(["g1", "g2"], set(), gpus, {"uuid-1"}) == ("g2", "uuid-2"))
    add("全占用 → None", pick_gpu(["g1", "g2"], {"g1"}, gpus, {"uuid-2"}) == (None, None))
    add("未知卡名跳过", pick_gpu(["gX", "g1"], set(), gpus, set()) == ("g1", "uuid-1"))

    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, extra in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name} {extra if not ok else ''}")
    print(f"{n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
