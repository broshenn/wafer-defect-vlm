#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""isolation_check 的反例测试（合同 4.4）。4 项必须全 PASS：
①正常输入通过；②train 副本加 1 个 TEST 的 ID → 必须报错；③空 train → 必须报错；
④构造与 TEST 某图 SHA 相同的假 train 条目 → 必须报错。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isolation_check import IsolationError, check  # noqa: E402

META = {
    "a1": {"lot_name": "lotA", "png_sha256": "sha_a1"},
    "a2": {"lot_name": "lotB", "png_sha256": "sha_a2"},
    "b1": {"lot_name": "lotC", "png_sha256": "sha_b1"},
    "b2": {"lot_name": "lotD", "png_sha256": "sha_b2"},
}


def main():
    results = []

    # ① 正常输入通过
    try:
        r = check(["a1", "a2"], ["b1", "b2"], META)
        results.append(("①正常输入通过", r["id_overlap"] == 0 and r["lot_overlap"] == 0))
    except IsolationError as exc:
        results.append(("①正常输入通过", False)); print("  意外报错:", exc)

    # ② train 副本里加 1 个 TEST 的 ID → 必须报错
    try:
        check(["a1", "a2", "b1"], ["b1", "b2"], META)
        results.append(("②ID 交集必须报错", False))
    except IsolationError:
        results.append(("②ID 交集必须报错", True))

    # ③ 空 train → 必须报错
    try:
        check([], ["b1", "b2"], META)
        results.append(("③空集合必须报错", False))
    except IsolationError:
        results.append(("③空集合必须报错", True))

    # ④ 假 train 条目与 TEST 某图 SHA 相同 → 必须报错
    meta2 = dict(META)
    meta2["a3"] = {"lot_name": "lotE", "png_sha256": "sha_b1"}  # 与 b1 同 SHA
    try:
        check(["a1", "a2", "a3"], ["b1", "b2"], meta2)
        results.append(("④SHA 交集必须报错", False))
    except IsolationError:
        results.append(("④SHA 交集必须报错", True))

    # 附加：lot 交集必须报错
    meta3 = dict(META)
    meta3["a4"] = {"lot_name": "lotC", "png_sha256": "sha_x"}
    try:
        check(["a1", "a4"], ["b1", "b2"], meta3)
        results.append(("⑤lot 交集必须报错", False))
    except IsolationError:
        results.append(("⑤lot 交集必须报错", True))

    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    print(f"{n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
