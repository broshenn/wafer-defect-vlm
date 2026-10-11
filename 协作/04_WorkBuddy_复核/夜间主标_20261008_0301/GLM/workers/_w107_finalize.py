# -*- coding: utf-8 -*-
import json, hashlib, datetime

PROMPT_SHA = "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda"
EXPECTED_SHA = {
    "wbg0743": "503845111f6896e816d9981661d5114813753203959784254217cca5012a6c93",
    "wbg0744": "aa2d7cd054512c8e2de4d0780d3139eba76a24a544b2ddec07e1cacf334d7ffc",
    "wbg0745": "701ce3466bd4482a9c2ca1b27cfd0c87501591874ff7b3726bfe185a3b60e549",
    "wbg0746": "0c7a19f2347c34a76d9ff8318212daf03b14980b863d8e16c3da45ea88420bae",
    "wbg0747": "42cd7f904d4e6ce08ce76ce74c3f2e270b90ad33f3fa789b2c5b91d628a3430a",
    "wbg0748": "0138f1dbbf3d79fee5fa2f948409313f20078aed2de588edabeef45ee2fc663a",
    "wbg0749": "993ef0b070eaa09d103c892633a9cac7fec03af77aab4b86ce1f532921e93c95",
}
FIELDS = ["defect_class", "morphology", "radial_zone", "clock_direction",
          "extent_r", "caption_zh", "uncertainty"]

tmp_path = "协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/workers/_tmp_107.json"
final_path = "协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/workers/分片107_原答.jsonl"

lines = [json.loads(l) for l in open(tmp_path, encoding="utf-8") if l.strip()]

errors = []
if len(lines) != 7:
    errors.append(f"条数={len(lines)} 不为 7")

seen = set()
raw_answers = set()
records = []
for i, rec in enumerate(lines):
    iid = rec.get("item_id")
    if iid in seen:
        errors.append(f"第{i+1}条 item_id 重复: {iid}")
    seen.add(iid)
    sha = rec.get("image_sha256", "")
    if not sha or len(sha) != 64:
        errors.append(f"{iid} image_sha256 非空或长度!=64")
    if iid in EXPECTED_SHA and sha != EXPECTED_SHA[iid]:
        errors.append(f"{iid} image_sha256 与清单不一致")
    try:
        ans = json.loads(rec.get("raw_answer", ""))
    except Exception as e:
        errors.append(f"{iid} raw_answer 不可解析: {e}")
        continue
    missing = [f for f in FIELDS if f not in ans]
    if missing:
        errors.append(f"{iid} 缺字段: {missing}")
    raw_answers.add(rec.get("raw_answer"))
    check = "self_ok"
    if missing:
        check = "missing:" + ",".join(missing)
    elif sha != EXPECTED_SHA.get(iid):
        check = "ERROR:hash_mismatch"
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    records.append({
        "item_id": iid,
        "image_sha256": sha,
        "prompt_sha256": PROMPT_SHA,
        "raw_answer": rec.get("raw_answer"),
        "answer_sha256": hashlib.sha256(rec.get("raw_answer").encode("utf-8")).hexdigest(),
        "model": "GLM-5.3-Flash",
        "answer_time": now.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "check": check,
    })

if len(raw_answers) != 7:
    errors.append("存在逐字节相同的 raw_answer")

if errors:
    print("校验失败:")
    for e in errors:
        print(" -", e)
    raise SystemExit(1)

with open(final_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("校验通过，已写入:", final_path)
for r in records:
    ans = json.loads(r["raw_answer"])
    print(r["item_id"], ans["defect_class"], r["check"])
