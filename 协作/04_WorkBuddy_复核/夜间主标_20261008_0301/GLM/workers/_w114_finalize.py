# -*- coding: utf-8 -*-
"""分片114 终校验并生成最终件：分片114_原答.jsonl"""
import glob
import hashlib
import json
import os
from datetime import datetime, timezone, timedelta

BASE = "D:/pycode/晶圆图研究/协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/workers"
TMP = os.path.join(BASE, "_tmp_114.jsonl")
FINAL = os.path.join(BASE, "分片114_原答.jsonl")
PROMPT_SHA = "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda"
MODEL = "GLM-5.3-Flash"
FIELDS = ("defect_class", "morphology", "radial_zone", "clock_direction",
          "extent_r", "caption_zh", "uncertainty")
TZ = timezone(timedelta(hours=8))


def main():
    hits = glob.glob("D:/pycode/晶圆图研究/协作/**/WB_gap1200_*114*.jsonl", recursive=True)
    assert len(hits) == 1, "manifest hits=%d" % len(hits)
    manifest = {r["item_id"]: r for r in
                (json.loads(l) for l in open(hits[0], encoding="utf-8") if l.strip())}
    assert len(manifest) == 7

    recs = [json.loads(l) for l in open(TMP, encoding="utf-8") if l.strip()]
    problems = []
    seen = set()
    for rec in recs:
        iid = rec.get("item_id")
        sha = rec.get("image_sha256", "")
        if iid not in manifest:
            problems.append((iid, "不在清单"))
            continue
        if iid in seen:
            problems.append((iid, "重复"))
        seen.add(iid)
        if len(sha) != 64:
            problems.append((iid, "sha_len=%d" % len(sha)))
        elif sha != manifest[iid]["image_sha256"]:
            problems.append((iid, "sha与清单不一致"))
        try:
            ans = json.loads(rec.get("raw_answer", ""))
        except Exception:
            problems.append((iid, "raw_answer不可解析"))
            continue
        missing = [k for k in FIELDS if k not in ans]
        if missing:
            problems.append((iid, "缺字段:" + ",".join(missing)))

    if problems:
        for p in problems:
            print("PROBLEM:", p)
        raise SystemExit(1)

    answer_time = datetime.now(TZ).replace(microsecond=0).isoformat()
    with open(FINAL, "w", encoding="utf-8") as f:
        for rec in recs:
            iid = rec["item_id"]
            ra = rec["raw_answer"]
            row = {
                "item_id": iid,
                "image_sha256": manifest[iid]["image_sha256"],
                "prompt_sha256": PROMPT_SHA,
                "raw_answer": ra,
                "answer_sha256": hashlib.sha256(ra.encode("utf-8")).hexdigest(),
                "model": MODEL,
                "answer_time": answer_time,
                "check": "self_ok",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = sum(1 for l in open(FINAL, encoding="utf-8") if l.strip())
    print("FINAL_OK rows=%d" % n)
    print("FINAL_PATH:", FINAL)


if __name__ == "__main__":
    main()
