# -*- coding: utf-8 -*-
"""分片114 步骤1：哈希自校验。清单文件由脚本自行发现，内容为唯一权威源。"""
import glob
import hashlib
import json
import os
import sys

ROOT = "D:/pycode/晶圆图研究"
PATTERN = ROOT + "/协作/**/WB_gap1200_分片114_共7图.jsonl"


def main():
    found = sorted(glob.glob(PATTERN, recursive=True))
    print("candidates:", len(found))
    for p in found:
        print("  found:", p)
    if len(found) == 1:
        manifest_path = found[0]
    else:
        sys.exit("manifest not unique or missing: %d" % len(found))

    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print("rows:", len(items))
    for it in items:
        item_id = it["item_id"]
        image_path = it["image_path"]
        expect = it["image_sha256"]
        actual = hashlib.sha256(open(image_path, "rb").read()).hexdigest()
        ok = "MATCH" if actual == expect else "MISMATCH"
        print(item_id, ok)
        if ok == "MISMATCH":
            print("   expect:", expect)
            print("   actual:", actual)


if __name__ == "__main__":
    main()
