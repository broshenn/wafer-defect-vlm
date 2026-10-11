# -*- coding: utf-8 -*-
"""分片114：逐图哈希校验 + 追加落盘。用法：
   python _w114_tool.py hash <image_path> <expected_sha256>
   python _w114_tool.py add  <item_id> <image_sha256> <raw_answer_json>
路径与哈希一律从清单整串复制，脚本不做任何改写。
"""
import hashlib
import json
import sys

TMP = ("D:/pycode/晶圆图研究/协作/04_WorkBuddy_复核/"
       "夜间主标_20261008_0301/GLM/workers/_tmp_114.json")

FIELDS = ("defect_class", "morphology", "radial_zone", "clock_direction",
          "extent_r", "caption_zh", "uncertainty")


def do_hash(image_path, expected):
    actual = hashlib.sha256(open(image_path, "rb").read()).hexdigest()
    print("actual  =", actual)
    print("expected=", expected)
    print("RESULT  =", "MATCH" if actual == expected else "MISMATCH")


def do_add(item_id, sha, raw_answer):
    assert len(sha) == 64, "image_sha256 length != 64"
    ans = json.loads(raw_answer)
    missing = [k for k in FIELDS if k not in ans]
    assert not missing, "missing fields: %s" % missing
    rec = {"item_id": item_id, "image_sha256": sha, "raw_answer": raw_answer}
    line = json.dumps(rec, ensure_ascii=False)
    with open(TMP, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print("APPENDED", item_id, "sha_len=", len(sha))


def main():
    mode = sys.argv[1]
    if mode == "hash":
        do_hash(sys.argv[2], sys.argv[3])
    elif mode == "add":
        do_add(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit("unknown mode: " + mode)


if __name__ == "__main__":
    main()
