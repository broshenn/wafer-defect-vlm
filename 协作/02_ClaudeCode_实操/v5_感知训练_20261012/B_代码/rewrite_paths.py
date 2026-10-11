#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路径改写（B 卡 13）。
把数据 jsonl 里的 `__IMG__/` 换成给定前缀（img448 或 img896 真实目录），
把 config 里的 /WS/v5 换成真实根。输出新文件（不覆盖原件），写 rewrite_log.json
（每文件改写前后 SHA、替换次数）。用法：
  python rewrite_paths.py --files <f1> [<f2> ...] --img448 <dir448> --img896 <dir896> \
      --root-from /WS/v5 --root-to <真实根> --out-dir <目录> --log rewrite_log.json
规则：文件名含 `_x2` 的 jsonl 的 __IMG__ 指到 img896，其余指到 img448。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rewrite_text(text, img_prefix, root_from, root_to):
    n1 = text.count("__IMG__/")
    text = text.replace("__IMG__/", img_prefix + "/")
    n2 = text.count(root_from)
    text = text.replace(root_from, root_to)
    return text, n1, n2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--img448", required=True)
    ap.add_argument("--img896", required=True)
    ap.add_argument("--root-from", default="/WS/v5")
    ap.add_argument("--root-to", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--log", default="rewrite_log.json")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for f in a.files:
        src = Path(f)
        img_prefix = a.img896 if "_x2" in src.name else a.img448
        before_sha = sha(src)
        text = src.read_text(encoding="utf-8")
        new_text, n_img, n_root = rewrite_text(text, img_prefix, a.root_from, a.root_to)
        dst = out_dir / src.name
        dst.write_text(new_text, encoding="utf-8", newline="\n")
        after_sha = sha(dst)
        log.append({"file": src.name, "before_sha256": before_sha, "after_sha256": after_sha,
                    "img_replacements": n_img, "root_replacements": n_root,
                    "img_prefix": img_prefix, "out": str(dst)})
        print(f"{src.name}: __IMG__×{n_img} root×{n_root} {before_sha[:12]}→{after_sha[:12]}")
    Path(a.log).write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
