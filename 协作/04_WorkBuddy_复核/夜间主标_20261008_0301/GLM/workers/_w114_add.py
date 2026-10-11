# -*- coding: utf-8 -*-
"""分片114 追加落盘：每图独立作答后调用一次。"""
import json
import sys

OUT = "D:/pycode/晶圆图研究/协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/workers/_tmp_114.jsonl"

REQUIRED = ("defect_class", "morphology", "radial_zone", "clock_direction",
            "extent_r", "caption_zh", "uncertainty")


def main(argv):
    item_id, image_sha256, raw_answer = argv[1], argv[2], argv[3]
    assert len(image_sha256) == 64
    ans = json.loads(raw_answer)
    missing = [k for k in REQUIRED if k not in ans]
    assert not missing, missing
    record = {"item_id": item_id, "image_sha256": image_sha256,
              "raw_answer": raw_answer}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("APPENDED", item_id, "sha_len", len(image_sha256))


if __name__ == "__main__":
    main(sys.argv)
