# -*- coding: utf-8 -*-
import hashlib, json
M = ("D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/"
     "夜间扩容_20261008_0244/客户盲包/WorkBuddy_训练缺口1200/"
     "WB_gap1200_分片114_共7图.jsonl")
items = []
with open(M, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            items.append(json.loads(line))
print("rows:", len(items))
for it in items:
    iid = it["item_id"]
    p = it["image_path"]
    exp = it["image_sha256"]
    act = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(iid, "MATCH" if act == exp else "MISMATCH")
    if act != exp:
        print("  expect:", exp)
        print("  actual:", act)
