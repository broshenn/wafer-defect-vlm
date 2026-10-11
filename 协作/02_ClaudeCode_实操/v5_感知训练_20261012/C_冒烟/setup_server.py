#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""服务器端：解包 + SHA 全量校验 + 路径改写（C 阶段第 3 步）。
在服务器上用 /WS/envs/wafer/bin/python 运行：
  WS_ROOT=/data/... python3 setup_server.py
步骤：①校验 4 个 bundle 的 SHA256 与 上传清单.json 一致；
     ②解包：数据 bundle→v5/data/_orig/、代码 bundle→v5/code/、
       img448.tgz→v5/data/img448/、img896.tgz→v5/data/img896/；
     ③全量校验 _orig 下每个 jsonl 与代码文件的 SHA；
     ④rewrite_paths：数据 jsonl 的 __IMG__→真实 img448/img896，写入 v5/data/（原件留在 _orig/）；
       configs 的 /WS/v5→真实根（原地覆盖，先存 code/_orig_configs/）；
     ⑤抽 3 行检查图片路径真实存在；打印 img448/img896 文件数。
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

WS = os.environ.get("WS_ROOT")
assert WS, "必须提供 WS_ROOT"
V5 = Path(WS) / "v5"
PY = f"{WS}/envs/wafer/bin/python"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    manifest = json.loads((V5 / "上传清单.json").read_text(encoding="utf-8"))
    # ① bundle SHA
    for name, info in manifest["tgz"].items():
        p = V5 / name
        actual = sha(p)
        status = "OK" if actual == info["sha256"] else "**MISMATCH**"
        print(f"[bundle] {name} {status} {info['size']}B")
        if actual != info["sha256"]:
            print("!! bundle SHA 不符，停止")
            return 2
    # ② 解包
    orig = V5 / "data" / "_orig"
    orig.mkdir(parents=True, exist_ok=True)
    with tarfile.open(V5 / "v5_bundle_data.tgz") as tf:
        tf.extractall(orig)
    with tarfile.open(V5 / "v5_bundle_code.tgz") as tf:
        tf.extractall(V5 / "code")
    for tgz, dest in (("img448.tgz", V5 / "data" / "img448"), ("img896.tgz", V5 / "data" / "img896")):
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(V5 / tgz) as tf:
            tf.extractall(dest)
    # ②.5 Windows 打包的 arcname 反斜杠归一化（tar 在 Linux 会解成文件名含 '\' 的平铺文件）
    moved = 0
    for p in list((V5 / "code").iterdir()):
        if "\\" in p.name:
            dest = V5 / "code" / p.name.replace("\\", "/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
            moved += 1
    if moved:
        print(f"[fix] 反斜杠文件名归一化 {moved} 个")

    # ③ 全量校验
    bad = 0
    for name, info in manifest["data_files"].items():
        p = orig / name
        if not p.exists() or sha(p) != info["sha256"]:
            print(f"!! 数据文件不符: {name}")
            bad += 1
    for name, info in manifest["code_files"].items():
        p = V5 / "code" / name.replace("\\", "/")   # manifest 键在 Windows 下带反斜杠
        if not p.exists() or sha(p) != info["sha256"]:
            print(f"!! 代码文件不符: {name}")
            bad += 1
    n448 = len(list((V5 / "data" / "img448").glob("*.png")))
    n896 = len(list((V5 / "data" / "img896").glob("*.png")))
    print(f"[extract] 数据 {len(manifest['data_files'])} / 代码 {len(manifest['code_files'])}；"
          f"img448={n448} img896={n896}；不符 {bad}")
    if bad:
        return 2
    # ④ 路径改写
    data_files = [str(orig / n) for n in manifest["data_files"]]
    r = subprocess.run([PY, str(V5 / "code" / "rewrite_paths.py"), "--files", *data_files,
                        "--img448", str(V5 / "data" / "img448"),
                        "--img896", str(V5 / "data" / "img896"),
                        "--root-from", "/WS/v5", "--root-to", str(V5),
                        "--out-dir", str(V5 / "data"), "--log", str(V5 / "data" / "rewrite_log.json")],
                       capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-800:])
        return 3
    cfg_orig = V5 / "code" / "_orig_configs"
    cfg_orig.mkdir(exist_ok=True)
    cfg_files = []
    for p in (V5 / "code" / "configs").glob("*.json"):
        shutil.copyfile(p, cfg_orig / p.name)
        cfg_files.append(str(cfg_orig / p.name))
    r = subprocess.run([PY, str(V5 / "code" / "rewrite_paths.py"), "--files", *cfg_files,
                        "--img448", str(V5 / "data" / "img448"),
                        "--img896", str(V5 / "data" / "img896"),
                        "--root-from", "/WS/v5", "--root-to", str(V5),
                        "--out-dir", str(V5 / "code" / "configs"),
                        "--log", str(V5 / "code" / "configs" / "rewrite_log.json")],
                       capture_output=True, text=True)
    print(r.stdout[-800:])
    if r.returncode != 0:
        print(r.stderr[-800:])
        return 3
    # ⑤ 抽样检查
    import itertools
    for name in ("test_main.jsonl", "sft_S1.jsonl", "rl_candidates800.jsonl"):
        line = (V5 / "data" / name).read_text(encoding="utf-8").splitlines()[0]
        img = json.loads(line)["images"][0]
        ok = os.path.exists(img)
        print(f"[sample] {name}: {img} 存在={ok}")
        if not ok:
            return 4
    print("setup_server 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
