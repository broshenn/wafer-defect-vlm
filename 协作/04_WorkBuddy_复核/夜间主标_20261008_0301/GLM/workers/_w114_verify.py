# -*- coding: utf-8 -*-
"""分片114 终校验：磁盘为唯一事实源。
1) 权威清单: WB_gap1200_分片114_共7图.jsonl
2) 临时件: _tmp_114.jsonl
3) 输出: 分片114_原答.jsonl (8字段)
"""
import glob
import hashlib
import json
import os
import sys

BASE = "D:/pycode/晶圆图研究/协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/workers"
GLOB = "D:/pycode/晶圆图研究/协作/**/WB_gap1200_分片114_共7图.jsonl"
FIELDS = ("defect_class", "morphology", "radial_zone", "clock_direction",
          "extent_r", "caption_zh", "uncertainty")
FINAL_FIELDS = ("item_id", "image_sha256", "prompt_sha256", "raw_answer",
                "answer_sha256", "model", "answer_time", "check")


def main():
    # 1. 发现权威清单
    hits = glob.glob(GLOB, recursive=True)
    if len(hits) != 1:
        sys.exit("manifest hits != 1: %d" % len(hits))
    with open(hits[0], "r", encoding="utf-8") as f:
        manifest = [json.loads(l) for l in f if l.strip()]
    man = {it["item_id"]: it for it in manifest}
    print("manifest rows:", len(manifest))

    # 2. 读取临时件
    tmp_path = os.path.join(BASE, "_tmp_114.jsonl")
    tmp = []
    with open(tmp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tmp.append(json.loads(line))
    print("tmp rows:", len(tmp))

    # 3. 校验
    problems = []
    seen_ids = set()
    for rec in tmp:
        iid = rec.get("item_id")
        sha = rec.get("image_sha256", "")
        ra = rec.get("raw_answer")
        if iid not in man:
            problems.append((iid, "item_id 不在清单中"))
            continue
        if seen_ids & {iid}:
            problems.append((iid, "item_id 重复"))
        seen_ids.add(iid)
        if len(sha) != 64:
            problems.append((iid, "image_sha256 长度 != 64"))
        elif sha != man[iid]["image_sha256"]:
            problems.append((iid, "image_sha256 与清单不一致"))
        try:
            ans = json.loads(ra)
        except Exception:
            problems.append((iid, "raw_answer 无法解析"))
            continue
        missing = [k for k in FIELDS if k not in ans]
        if missing:
            problems.append((iid, "缺字段: " + ",".join(missing)))
        if ans.get("extent_r") is not None:
            problems.append((iid, "extent_r 应为 null"))
    if len(tmp) != 7:
        problems.append(("*", "行数 != 7"))
    if seen_ids != set(man):
        problems.append(("*", "item_id 集合与清单不一致"))

    # 4. 生成最终件
    if problems:
        print("PROBLEMS:")
        for iid, msg in problems:
            print("  ", iid, msg)
        sys.exit(1)
    final_path = os.path.join(BASE, "分片114_原答.jsonl")
    with open(final_path, "w", encoding="utf-8") as out:
        for rec in tmp:
            iid = rec["item_id"]
            ra = rec["raw_answer"]
            rec_out = {
                "item_id": iid,
                "image_sha256": man[iid]["image_sha256"],
                "prompt_sha256": man[iid].get("prompt_sha256", ""),
                "raw_answer": ra,
                "answer_sha256": hashlib.sha256(ra.encode("utf-8")).hexdigest(),
                "model": "GLM-5.3-Flash",
                "answer_time": "",
                "check": "self_ok",
            }
            out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
    print("WROTE:", final_path)
    print("final rows: 7")


if __name__ == "__main__":
    main()
