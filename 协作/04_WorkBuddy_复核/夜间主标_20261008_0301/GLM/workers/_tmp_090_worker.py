# -*- coding: utf-8 -*-
"""分片090 盲标 worker 辅助脚本（attempt2）。仅服务本分片 7 张图。"""
import json, io, os, sys, hashlib
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"D:/pycode/晶圆图研究"
TMP_JSONL = os.path.join(BASE, "_tmp_090_attempt2.jsonl")
FINAL = os.path.join(BASE, "分片090_原答.jsonl")
PROMPT_SHA = "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda"
MODEL = "GLM-5.3-Flash"
MANIFEST = [
    ("wbg0624", "data/images/wafer_00036754_019.png", "cd9779bfe803ec10f180c971484e20449f8d0ec4cedb9894b2f9c8be4f674f7d"),
    ("wbg0625", "data/images/wafer_00036781_023.png", "2b1acd921b93bb26bddade5b011439c8c97a508d21b776f1bcc5a258c8fd97b3"),
    ("wbg0626", "data/images/wafer_00036812_022.png", "49a2bbdc0940b7d6ad794da0b3640fb1f7448017638c90a72fd923e9d0179e08"),
    ("wbg0627", "data/images/wafer_00036814_005.png", "fb56aa37b9500ed5670e21159a2a0ba2f52bb802d65bba1d2d4acb1e06207edb"),
    ("wbg0628", "data/images/wafer_00036814_011.png", "a14bf7452adcda558b4afe398204e84fe0d9742ef3bced452970a0940db42591"),
    ("wbg0629", "data/images/wafer_00036829_004.png", "ac98d6e2d39f79ac4e770429813c991ae08c5c6984857cb4ed156ef80a90d48b"),
    ("wbg0630", "data/images/wafer_00036847_016.png", "e1fbb86e413126caf1c8e2d4b348d81e6565827c8982013ae29ca6f5e876fb6d"),
]
REQUIRED = ["defect_class", "morphology", "radial_zone", "clock_direction", "extent_r", "caption_zh", "uncertainty"]
CLASSES = {"Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_full", "Random", "Scratch", "none", "unknown"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_hash():
    if os.path.exists(TMP_JSONL):
        os.remove(TMP_JSONL)
        print("RESET_TMP_JSONL")
    for iid, rel, exp in MANIFEST:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print(iid, "MISSING_FILE")
            continue
        got = sha256_file(p)
        print(iid, "HASH_OK" if got == exp else "HASH_MISMATCH got=" + got)


def cmd_append(item_id):
    ans_path = os.path.join(BASE, "_tmp_090_ans_%s.json" % item_id)
    with io.open(ans_path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    if rec.get("item_id") != item_id:
        print("ERR_ITEM_ID_MISMATCH")
        sys.exit(2)
    ans = rec["answer"]
    raw = ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False)
    line = {"item_id": rec["item_id"], "image_sha256": rec["image_sha256"], "raw_answer": raw}
    with io.open(TMP_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    os.remove(ans_path)
    print("APPENDED", item_id)


def check_raw(raw):
    if raw == "ERROR:hash_mismatch":
        return True, "ERROR:hash_mismatch"
    try:
        obj = json.loads(raw)
    except Exception:
        return False, "missing:raw_answer_unparseable"
    if not isinstance(obj, dict):
        return False, "missing:raw_answer_not_object"
    miss = [k for k in REQUIRED if k not in obj]
    if miss:
        return False, "missing:" + ",".join(miss)
    if obj.get("defect_class") not in CLASSES:
        return False, "missing:defect_class_invalid"
    if obj.get("extent_r") is not None:
        return False, "missing:extent_r_not_null"
    return True, "self_ok"


def load_tmp():
    recs = []
    with io.open(TMP_JSONL, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                recs.append(json.loads(ln))
    return recs


def cmd_validate():
    ok = True
    recs = load_tmp()
    print("count", len(recs))
    if len(recs) != 7:
        ok = False
    ids = [r.get("item_id") for r in recs]
    if len(set(ids)) != len(ids):
        ok = False
        print("DUP_ITEM_ID")
    if ids != [m[0] for m in MANIFEST]:
        ok = False
        print("UNEXPECTED_ITEM_IDS")
    exp_sha = dict((m[0], m[2]) for m in MANIFEST)
    for r in recs:
        iid = r.get("item_id")
        if not r.get("image_sha256"):
            ok = False
            print(iid, "EMPTY_SHA")
        elif r["image_sha256"] != exp_sha.get(iid):
            ok = False
            print(iid, "SHA_NOT_COPY_OF_MANIFEST")
        good, check = check_raw(r.get("raw_answer", ""))
        print(iid, check)
        if not good:
            ok = False
    print("VALIDATE_OK" if ok else "VALIDATE_FAIL")
    return ok


def cmd_final():
    recs = load_tmp()
    if len(recs) != 7:
        print("ABORT_TMP_COUNT", len(recs))
        sys.exit(2)
    out_lines = []
    for r in recs:
        raw = r["raw_answer"]
        good, check = check_raw(raw)
        if not good:
            print("ABORT_BAD_RAW", r["item_id"], check)
            sys.exit(2)
        rec = {
            "item_id": r["item_id"],
            "image_sha256": r["image_sha256"],
            "prompt_sha256": PROMPT_SHA,
            "raw_answer": raw,
            "answer_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "model": MODEL,
            "answer_time": datetime.now().astimezone().replace(microsecond=0).isoformat(),
            "check": check,
        }
        out_lines.append(json.dumps(rec, ensure_ascii=False))
    with io.open(FINAL, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print("FINAL_WRITTEN", len(out_lines))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "hash":
        cmd_hash()
    elif c == "append":
        cmd_append(sys.argv[2])
    elif c == "validate":
        sys.exit(0 if cmd_validate() else 3)
    elif c == "final":
        cmd_final()
    else:
        print("UNKNOWN_CMD")
        sys.exit(2)
