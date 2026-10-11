# -*- coding: utf-8 -*-
"""分片114 事实核查：磁盘唯一事实源。结果同时写盘防输出损坏。"""
import glob
import hashlib
import json
import os
import sys

BASE = "D:/pycode/晶圆图研究/协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/workers"

MANIFEST = "D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/客户盲包/WorkBuddy_训练缺口1200/WB_gap1200_分片114_共7图.jsonl"
LOG = os.path.join(BASE, "_w114_truth_out.json")


def main():
    hits = glob.glob("D:/pycode/晶圆图研究/协作/**/WB_gap1200_*114*.jsonl", recursive=True)
    print("manifest hits:", len(hits))
    if len(hits) != 1:
        sys.exit("manifest not unique")
    rows = [json.loads(l) for l in open(hits[0], "r", encoding="utf-8") if l.strip()]
    results = []
    for r in rows:
        iid = r["item_id"]
        path = r["image_path"]
        exp = r["image_sha256"]
        with open(path, "rb") as f:
            act = hashlib.sha256(f.read()).hexdigest()
        status = "MATCH" if act == exp else "MISMATCH"
        results.append({"item_id": iid, "image_path": path,
                        "expected": exp, "actual": act, "status": status})
        print(iid, status)
    with open(LOG, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("LOG_WROTE:", LOG)


if __name__ == "__main__":
    main()
