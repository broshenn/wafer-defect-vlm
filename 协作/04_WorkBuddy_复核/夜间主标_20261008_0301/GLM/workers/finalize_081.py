# -*- coding: utf-8 -*-
"""分片081 收尾脚本：
① 若 _tmp_081.json 缺 wbg0567 则补写
② 全量校验（行数/哈希/字段/唯一性）
③ 生成最终 分片081_原答.jsonl
"""
import json, hashlib, os, io
from datetime import datetime

BASE = r"D:\pycode\晶圆图研究"
IMGDIR = os.path.join(BASE, "data", "images")
WORKDIR = os.path.join(BASE, "协作", "04_WorkBuddy_复核", "夜间主标_20261008_0301", "GLM", "workers")
TMP = os.path.join(WORKDIR, "_tmp_081.json")
OUT = os.path.join(WORKDIR, "分片081_原答.jsonl")

PROMPT_SHA = "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda"
MODEL = "GLM-5.3-Flash"
REQ_FIELDS = ["defect_class", "morphology", "radial_zone", "clock_direction", "extent_r", "caption_zh", "uncertainty"]

# 清单（item_id, 文件名, 清单sha256）
MANIFEST = [
    ("wbg0561", "wafer_00026324_022.png", "e9915e0e2e356883bc1118975ec92cd6335f32eeff57c0db9671fcc60ba788eb"),
    ("wbg0562", "wafer_00026624_009.png", "85161b190de0edaa00832116d0daa9addbbfcb31322ff7df55177a90a31ed93d"),
    ("wbg0563", "wafer_00026624_016.png", "4e6d1af0dc32f2580efa988d5e8042f44673a20f41758a45d75890b74c9d4205"),
    ("wbg0564", "wafer_00026627_015.png", "797116f99108453c1959a33f3fe2249ba258299b7ab2f9327f36f5609d294250"),
    ("wbg0565", "wafer_00027252_012.png", "36c5490c587abfb3eb3a8b7d92d335d3ab83a2df3a93b9d4af226130f2857d80"),
    ("wbg0566", "wafer_00028113_023.png", "31ea842cd739e0a319c1f57592521eb8046325d147fbb66e7d82836cb2404245"),
    ("wbg0567", "wafer_00029699_021.png", "82aa871ba91f07fbbfed2a8415c19bddd08e217c9cf202d0e08147062826e3a3"),
]
MAN = {m[0]: m for m in MANIFEST}

# ---- wbg0567 的盲标作答（图7：红色失效约四成多散布全圆，中上偏左中央略集中）----
ANS7 = {
    "defect_class": "Random",
    "morphology": "红色失效die散布全晶圆，约占四成多，多为孤立点与2-3连小簇，中上部偏左的中央区域红色相对集中连片，其余随机散布",
    "radial_zone": "global",
    "clock_direction": None,
    "extent_r": None,
    "caption_zh": "失效die以小簇和孤立点随机散布全晶圆，中上偏左中央区域略集中",
    "uncertainty": "中上偏左聚集边界模糊，整体以Random判类为主",
}

# ---- 第①步：读取临时文件，必要时补写 wbg0567 ----
lines = []
if os.path.exists(TMP):
    with io.open(TMP, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

rows = {}
bad_lines = []
for i, l in enumerate(lines, 1):
    try:
        obj = json.loads(l)
        rows[obj.get("item_id")] = obj
    except Exception as e:
        bad_lines.append((i, str(e)))

have = set(rows)
print("STEP1 现有行数:", len(lines), "| 已有 item_id:", sorted(have))
if bad_lines:
    print("STEP1 坏行:", bad_lines)

if "wbg0567" not in have:
    rec = {"item_id": "wbg0567",
           "image_sha256": MAN["wbg0567"][2],
           "raw_answer": json.dumps(ANS7, ensure_ascii=False)}
    with io.open(TMP, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    rows["wbg0567"] = rec
    print("STEP1 已补写 wbg0567")
else:
    print("STEP1 wbg0567 已存在，跳过")

# ---- 第②步：全量校验 ----
ok = True
final_rows = []
seen_ids = set()

for iid in [m[0] for m in MANIFEST]:
    if iid not in rows:
        print(f"CHECK 缺失 item: {iid}"); ok = False; continue
    if iid in seen_ids:
        print(f"CHECK item_id 重复: {iid}"); ok = False; continue
    seen_ids.add(iid)
    r = rows[iid]
    line_sha = r.get("image_sha256", "")
    raw = r.get("raw_answer", "")

    # 重算图片哈希
    path = os.path.join(IMGDIR, MAN[iid][1])
    with open(path, "rb") as fh:
        re_sha = hashlib.sha256(fh.read()).hexdigest()
    man_sha = MAN[iid][2]
    sha_ok = (line_sha == re_sha == man_sha)
    if not sha_ok:
        print(f"CHECK {iid} 哈希不一致: line={line_sha[:12]}.. file={re_sha[:12]}.. manifest={man_sha[:12]}..")
        ok = False
    else:
        print(f"CHECK {iid} 哈希一致 {re_sha[:12]}..")

    # raw_answer 可解析且七字段齐全
    try:
        inner = json.loads(raw)
    except Exception as e:
        print(f"CHECK {iid} raw_answer 解析失败: {e}"); ok = False; continue
    missing = [k for k in REQ_FIELDS if k not in inner]
    if missing:
        print(f"CHECK {iid} 缺字段: {missing}"); ok = False; continue

    final_rows.append((iid, re_sha, raw, inner))

# raw_answer 两两不同
raws = [x[2] for x in final_rows]
if len(raws) == 7 and len(set(raws)) != 7:
    print("CHECK raw_answer 存在重复"); ok = False

print("STEP2 校验结果:", "PASS" if ok and len(final_rows) == 7 else "FAIL", "| 有效行:", len(final_rows))
if not ok or len(final_rows) != 7:
    raise SystemExit(2)

# ---- 第③步：生成最终文件 ----
answer_time = datetime.now().astimezone().replace(microsecond=0).isoformat()
with io.open(OUT, "w", encoding="utf-8") as f:
    for iid, sha, raw, inner in final_rows:
        missing = [k for k in REQ_FIELDS if k not in inner]
        rec = {
            "item_id": iid,
            "image_sha256": sha,
            "prompt_sha256": PROMPT_SHA,
            "raw_answer": raw,
            "answer_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "model": MODEL,
            "answer_time": answer_time,
            "check": "self_ok" if not missing else "missing:" + ",".join(missing),
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

n = sum(1 for _ in io.open(OUT, "r", encoding="utf-8"))
print("STEP3 最终文件:", OUT)
print("STEP3 行数:", n)
for iid, sha, raw, inner in final_rows:
    print(" ", iid, inner.get("defect_class"))
