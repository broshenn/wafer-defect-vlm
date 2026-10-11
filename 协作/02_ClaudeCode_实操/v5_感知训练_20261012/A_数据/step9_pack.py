#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 9 步（前半）：打包 img448/ 与 img896/ 为 tar.gz，记录文件数与 SHA。
输出：img448.tgz、img896.tgz；打印各自文件数、字节数、SHA256。
"""
import hashlib
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pack(dirname, outname):
    src = HERE / dirname
    out = HERE / outname
    files = sorted(p for p in src.rglob("*") if p.is_file())
    with tarfile.open(out, "w:gz", compresslevel=6) as tf:
        for p in files:
            tf.add(p, arcname=str(p.relative_to(src)))
    sha = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            sha.update(chunk)
    print(f"{outname}: {len(files)} 个文件, {out.stat().st_size} 字节, SHA256={sha.hexdigest()}")
    return len(files), out.stat().st_size, sha.hexdigest()


def main():
    pack("img448", "img448.tgz")
    pack("img896", "img896.tgz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
