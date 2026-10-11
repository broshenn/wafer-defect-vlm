#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rewrite_paths 测试：替换次数与 SHA 记录（fixtures）。"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = HERE.parent
FIX = HERE / "fixtures"


def main():
    FIX.mkdir(exist_ok=True)
    src = FIX / "rw_in.jsonl"
    src.write_text(
        '{"messages":[{"role":"user","content":"<image>\\nX"}],"images":["__IMG__/a.png"],"sample_id":"a"}\n'
        '{"messages":[{"role":"user","content":"<image>\\nX"}],"images":["__IMG__/b_r090.png"],"sample_id":"b_r090"}\n',
        encoding="utf-8", newline="\n")
    src_x2 = FIX / "rw_in_x2.jsonl"
    src_x2.write_text(
        '{"messages":[{"role":"user","content":"<image>\\nX"}],"images":["__IMG__/a_x2.png"],"sample_id":"a"}\n',
        encoding="utf-8", newline="\n")
    out_dir = FIX / "rw_out"
    log = FIX / "rw_log.json"
    r = subprocess.run([sys.executable, "-I", str(B / "rewrite_paths.py"),
                        "--files", str(src), str(src_x2),
                        "--img448", "/REAL/img448", "--img896", "/REAL/img896",
                        "--root-from", "/WS/v5", "--root-to", "/REAL/v5",
                        "--out-dir", str(out_dir), "--log", str(log)],
                       capture_output=True, text=True, cwd=str(B))
    results = []
    add = lambda n, c, x="": results.append((n, bool(c), x))  # noqa: E731
    add("退出码 0", r.returncode == 0, r.stderr[-200:])
    l = json.loads(log.read_text(encoding="utf-8"))
    by = {x["file"]: x for x in l}
    add("普通文件 __IMG__ ×1 且指向 img448", by["rw_in.jsonl"]["img_replacements"] == 2
        and by["rw_in.jsonl"]["img_prefix"] == "/REAL/img448")
    add("_x2 文件指向 img896", by["rw_in_x2.jsonl"]["img_prefix"] == "/REAL/img896")
    out1 = (out_dir / "rw_in.jsonl").read_text(encoding="utf-8")
    add("输出内容替换正确", '"/REAL/img448/a.png"' in out1 and '"/REAL/img448/b_r090.png"' in out1)
    add("before/after SHA 均记录且不同", by["rw_in.jsonl"]["before_sha256"] != by["rw_in.jsonl"]["after_sha256"]
        and len(by["rw_in.jsonl"]["before_sha256"]) == 64)
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, extra in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name} {extra if not ok else ''}")
    print(f"{n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
