# -*- coding: utf-8 -*-
"""WorkBuddy 夜间主标（1200 缺口）只读汇总审计。
输出：统计.json、合并_去重后原答.jsonl（均在脚本同目录）。
只读其它目录；不修改、不移动任何已有文件。
运行：PYTHONIOENCODING=utf-8 python _审计_WorkBuddy合并.py
"""
import json, glob, os, datetime, hashlib
from collections import Counter, defaultdict

ROOT = r"D:/pycode/晶圆图研究"
WB   = ROOT + r"/协作/04_WorkBuddy_复核/夜间主标_20261008_0301"
GAP  = ROOT + r"/协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/客户端盲包/WorkBuddy_训练缺口1200"
E102 = ROOT + r"/协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/客户端盲包/WorkBuddy_外部评测102"
OLD90= ROOT + r"/协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/客户端盲包/_旧版不派（被更新版任务书取代）/WorkBuddy_90"
POOL = ROOT + r"/协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/数据"
OUT  = os.path.dirname(os.path.abspath(__file__))

ENUM9 = ["Center","Donut","Edge_Loc","Edge_Ring","Loc","Near_full","Random","Scratch","none"]
FIELDS7 = ["defect_class","morphology","radial_zone","clock_direction","extent_r","caption_zh","uncertainty"]

def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")

def read_jsonl(p):
    rows=[]
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line=line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def rel(p):
    return os.path.relpath(p, WB).replace("\\", "/")

snap_start = now()
S = {"读取时刻": {"开始": snap_start, "时区": "本机 +08:00"}}

# ---------------------------------------------------------------- 1. 盲包索引
blind = read_jsonl(GAP + "/WB_gap1200_索引.jsonl")
blind_ids = [r["item_id"] for r in blind]
blind_map = {r["item_id"]: r for r in blind}
S["盲包1200"] = {
    "行数": len(blind),
    "唯一item_id": len(set(blind_ids)),
    "唯一sample_id": len(set(r["sample_id"] for r in blind)),
    "唯一image_sha256": len(set(r["image_sha256"] for r in blind)),
}
# 分片文件核对
shard_files = sorted(glob.glob(GAP + "/WB_gap1200_分片*.jsonl"))
shard_rows = []
for p in shard_files:
    shard_rows += read_jsonl(p)
S["盲包1200"]["分片文件数"] = len(shard_files)
S["盲包1200"]["分片行数合计"] = len(shard_rows)
S["盲包1200"]["分片item_id与索引一致"] = (set(r["item_id"] for r in shard_rows) == set(blind_ids)) and len(shard_rows) == len(blind)
# 盲包内同图多任务（image_sha256 重复）
sha_groups = defaultdict(list)
for r in blind: sha_groups[r["image_sha256"]].append(r["item_id"])
blind_sha_dups = {s: ids for s, ids in sha_groups.items() if len(ids) > 1}
S["盲包1200"]["imgsha重复组数"] = len(blind_sha_dups)
S["盲包1200"]["imgsha重复明细"] = blind_sha_dups
# sample_id 重复
sid_dup = {k: v for k, v in Counter(r["sample_id"] for r in blind).items() if v > 1}
S["盲包1200"]["sample_id重复数"] = len(sid_dup)

# ---------------------------------------------------------------- 2. 产出盘点
inventory = {"最终原答": [], "临时原答jsonl": [], "临时payloadjson": [], "脚本py": [], "其它json": [], "状态与说明": []}
def inv_add(cat, p):
    st = os.stat(p)
    inventory[cat].append({
        "路径": rel(p), "字节": st.st_size,
        "mtime": datetime.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds"),
        "行数": None, "唯一item_id": None,
    })

for p in [WB+"/GLM/原答.jsonl", WB+"/GLM/分片080_原答.jsonl"]:
    rows = read_jsonl(p)
    inventory["最终原答"].append({"路径": rel(p), "字节": os.path.getsize(p),
        "mtime": datetime.datetime.fromtimestamp(os.stat(p).st_mtime).astimezone().isoformat(timespec="seconds"),
        "行数": len(rows), "唯一item_id": len(set(r["item_id"] for r in rows))})
for p in sorted(glob.glob(WB+"/GLM/workers/分片*_原答.jsonl")):
    rows = read_jsonl(p)
    inventory["最终原答"].append({"路径": rel(p), "字节": os.path.getsize(p),
        "mtime": datetime.datetime.fromtimestamp(os.stat(p).st_mtime).astimezone().isoformat(timespec="seconds"),
        "行数": len(rows), "唯一item_id": len(set(r["item_id"] for r in rows))})
for p in sorted(glob.glob(WB+"/GLM/workers/_tmp_*.jsonl")):
    rows = read_jsonl(p)
    inv_add("临时原答jsonl", p)
    inventory["临时原答jsonl"][-1]["行数"]=len(rows)
    inventory["临时原答jsonl"][-1]["唯一item_id"]=len(set(r.get("item_id") for r in rows))
for p in sorted(glob.glob(WB+"/GLM/workers/_tmp_*.json")):
    inv_add("临时payloadjson", p)
for p in sorted(glob.glob(WB+"/GLM/*.py"))+sorted(glob.glob(WB+"/GLM/workers/*.py")):
    inv_add("脚本py", p)
for p in sorted(glob.glob(WB+"/GLM/workers/*.json")):
    if "_tmp_" in os.path.basename(p): continue
    inv_add("其它json", p)
for p in sorted(glob.glob(WB+"/GLM/*")) + sorted(glob.glob(WB+"/*")):
    b=os.path.basename(p)
    if os.path.isfile(p) and (b.endswith(".jsonl") and b not in ("原答.jsonl","分片状态.jsonl","进度检查点.jsonl","隔离清单.jsonl")): continue
    if b in ("分片状态.jsonl","进度检查点.jsonl","隔离清单.jsonl","会话账目.md","锁定题面_结构化_含不确定.txt"):
        inv_add("状态与说明", p)

kimi_files = glob.glob(WB+"/Kimi/**/*", recursive=True)
S["产出盘点"] = {
    "GLM根目录文件": sorted(os.path.basename(p) for p in glob.glob(WB+"/GLM/*") if os.path.isfile(p)),
    "workers目录文件数": len(glob.glob(WB+"/GLM/workers/*")),
    "最终原答文件": inventory["最终原答"],
    "临时原答jsonl数": len(inventory["临时原答jsonl"]),
    "临时payloadjson数": len(inventory["临时payloadjson"]),
    "Kimi目录文件": [rel(p) for p in kimi_files],
    "Kimi结论": "目录为空——复核未执行（not run），非「已完成0图」",
}

# ---------------------------------------------------------------- 3. 去重
# 收集最终文件中同一 item_id 的全部副本
occ = defaultdict(list)   # item_id -> [(source_rel, record)]
for it in inventory["最终原答"]:
    p = WB + "/" + it["路径"]
    for r in read_jsonl(p):
        occ[r["item_id"]].append((it["路径"], r))

def priority(src):
    if "/workers/分片" in src: return 0          # 规范分片文件
    if src.endswith("GLM/原答.jsonl"): return 1
    if src.endswith("GLM/分片080_原答.jsonl"): return 2   # 恢复副本
    return 3

dup_item = {k: v for k, v in occ.items() if len(v) > 1}
kept = {}
for k, v in occ.items():
    v2 = sorted(v, key=lambda t: priority(t[0]))
    kept[k] = (v2[0][0], v2[0][1])

# 080 两份对比
dup_detail = []
for k, v in dup_item.items():
    srcs = [t[0] for t in v]
    ras = [t[1].get("raw_answer") for t in v]
    checks = [t[1].get("check") for t in v]
    dup_detail.append({"item_id": k, "来源": srcs, "raw_answer全同": len(set(ras)) == 1,
                       "check字段": checks, "保留": sorted(v, key=lambda t: priority(t[0]))[0][0]})
S["去重"] = {
    "最终文件item_id总数(含副本)": sum(len(v) for v in occ.values()),
    "唯一item_id数": len(occ),
    "同item_id多副本数": len(dup_item),
    "多副本明细": dup_detail,
    "保留规则": "同一 item_id 取 workers/分片NNN_原答.jsonl（规范分片件）优先；其次 GLM/原答.jsonl；最后 GLM/分片080_原答.jsonl（恢复副本）",
}

# 旧包对照：wb_006/wb_007 与新包 wbg0093/wbg0123 同图？
old90_idx = {r["item_id"]: r for r in read_jsonl(OLD90 + "/WorkBuddy90_索引.jsonl")}
new_by_sha = {r["image_sha256"]: r["item_id"] for r in blind}
old_new_alias = []
for k in ["wb_001","wb_002","wb_003","wb_004","wb_005","wb_006","wb_007"]:
    if k in kept and k in old90_idx:
        sha = old90_idx[k]["image_sha256"]
        alias = new_by_sha.get(sha)
        if alias:
            same = kept[k][1].get("raw_answer") == kept[alias][1].get("raw_answer")
            old_new_alias.append({"旧item": k, "新item": alias, "同sample_id": old90_idx[k]["sample_id"] == blind_map[alias]["sample_id"],
                                  "raw_answer逐字节相同": same})
S["去重"]["旧90包与新1200同图对照"] = old_new_alias

# ---------------------------------------------------------------- 4. 机械检查（只证格式）
mech_fail_json = []; mech_empty = []; mech_notdict = []
field_missing = []; cls_bad = []; cls_unknown = []
raw_not_str = []   # raw_answer 本身已是对象/数组（非字符串）
rz_counter = Counter(); cd_null = 0; ext_nonnull = []
unc_empty = 0
for k, (src, r) in kept.items():
    raw = r.get("raw_answer")
    if isinstance(raw, (dict, list)):
        raw_not_str.append({"item_id": k, "来源文件": src, "类型": type(raw).__name__})
        obj = raw if isinstance(raw, dict) else None
        if obj is None:
            mech_notdict.append(k); continue
    else:
        if raw is None: raw = ""
        if raw.strip() == "":
            mech_empty.append(k); continue
        try:
            obj = json.loads(raw)
        except Exception:
            mech_fail_json.append(k); continue
        if not isinstance(obj, dict):
            mech_notdict.append(k); continue
    miss = [f for f in FIELDS7 if f not in obj]
    if miss: field_missing.append({"item_id": k, "缺": miss})
    c = obj.get("defect_class")
    if c not in ENUM9:
        if c == "unknown": cls_unknown.append(k)
        else: cls_bad.append({"item_id": k, "defect_class": c})
    rz_counter[str(obj.get("radial_zone"))] += 1
    if obj.get("clock_direction") is None: cd_null += 1
    if obj.get("extent_r") is not None: ext_nonnull.append({"item_id": k, "extent_r": obj.get("extent_r")})
    if obj.get("uncertainty") == "": unc_empty += 1
kept_ids = set(kept)
S["机械检查"] = {
    "口径": "仅核格式（可解析/字段齐全/defect_class 枚举），不核内容事实；无法证明形态、位置、尺寸描述正确",
    "检查条数(唯一item_id)": len(kept),
    "json不可解析数": len(mech_fail_json), "json不可解析样例": mech_fail_json[:10],
    "空答数": len(mech_empty), "空答样例": mech_empty[:10],
    "可解析但非对象数": len(mech_notdict),
    "七字段缺字段条数": len(field_missing), "缺字段明细(前10)": field_missing[:10],
    "defect_class非九类且非unknown数": len(cls_bad), "样例": cls_bad[:10],
    "defect_class=unknown数": len(cls_unknown),
    "raw_answer本身为对象而非字符串的条数": len(raw_not_str), "样例": raw_not_str[:10],
    "radial_zone填写值分布(原样,非金标)": dict(rz_counter),
    "clock_direction为null数": cd_null,
    "extent_r非null数": len(ext_nonnull), "extent_r非null样例": ext_nonnull[:10],
    "uncertainty为空串数": unc_empty,
}
# defect_class 原样分布（单独算，简单可靠）
dc_counter = Counter()
for k, (src, r) in kept.items():
    raw = r.get("raw_answer")
    try:
        obj = raw if isinstance(raw, dict) else json.loads(raw or "")
        dc_counter[str(obj.get("defect_class"))] += 1
    except Exception:
        dc_counter["<不可解析>"] += 1
S["机械检查"]["defect_class值分布(模型填写,非金标)"] = dict(dc_counter)
# check 字段分布（文件自带标记）
S["机械检查"]["check字段分布"] = dict(Counter(r.get("check") for _, r in kept.values()))
# 非JSON文本样例（前80字符）
def _first80(k):
    raw = kept[k][1].get("raw_answer")
    return {"item_id": k, "来源文件": kept[k][0], "开头": (raw or "")[:80]}
S["机械检查"]["非JSON文本样例"] = [_first80(k) for k in mech_fail_json[:6]]
# 逐分片问题分布
json_fail_by_shard = Counter(kept[k][0].split("分片")[-1][:3] for k in mech_fail_json)
dict_by_shard = Counter(d["来源文件"].split("分片")[-1][:3] for d in raw_not_str)
S["机械检查"]["非JSON按分片"] = dict(sorted(json_fail_by_shard.items()))
S["机械检查"]["raw对象按分片"] = dict(sorted(dict_by_shard.items()))
S["机械检查"]["非JSON涉及分片数"] = len(json_fail_by_shard)
S["机械检查"]["raw对象涉及分片数"] = len(dict_by_shard)
# radial_zone 字面枚举统计（题面要求 center/middle/edge/global/none/unknown）
RZ_ENUM = {"center","middle","edge","global","none","unknown"}
rz_in = sum(v for k, v in rz_counter.items() if k in RZ_ENUM)
rz_out = sum(v for k, v in rz_counter.items() if k not in RZ_ENUM)
S["机械检查"]["radial_zone字面枚举内数"] = rz_in
S["机械检查"]["radial_zone非字面枚举数"] = rz_out

# ---------------------------------------------------------------- 5. 覆盖与交集
wbg_answered = [k for k in kept if k.startswith("wbg")]
wbg_answered_conv = set(wbg_answered) & set(blind_ids)
missing = [k for k in blind_ids if k not in kept]
extra = [k for k in kept if not k.startswith("wbg")]
S["覆盖"] = {
    "1200中已有原答的唯一item_id数": len(wbg_answered_conv),
    "1200中无任何原答数": len(missing),
    "缺失item_id前30": [k for k in blind_ids if k in set(missing)][:30],
    "缺失item_id全表": [k for k in blind_ids if k in set(missing)],
    "缺失等于wbg1149起连续块": (lambda ml: (ml == ["wbg%04d" % i for i in range(int(ml[0][3:]), int(ml[-1][3:]) + 1)]) if ml else None)([k for k in blind_ids if k in set(missing)]),
    "非1200包的其他原答(旧90包)": sorted(extra),
}
# 原答 image_sha256 与索引一致性
sha_mismatch = []
for k in wbg_answered_conv:
    if kept[k][1].get("image_sha256") != blind_map[k]["image_sha256"]:
        sha_mismatch.append(k)
S["覆盖"]["原答image_sha256与索引不一致数"] = len(sha_mismatch)
S["覆盖"]["原答image_sha256不一致样例"] = sha_mismatch[:10]

# 训练池
poolD = read_jsonl(POOL + "/D_N3072.jsonl")
poolL = read_jsonl(POOL + "/L_N3072.jsonl")
poolD_sids = set(r["sample_id"] for r in poolD)
poolL_sids = set(r["sample_id"] for r in poolL)
e102 = read_jsonl(E102 + "/WB_eval102_索引.jsonl")
e102_sids = set(r["sample_id"] for r in e102)
gap_sids = set(r["sample_id"] for r in blind)
answered_sids = set(blind_map[k]["sample_id"] for k in wbg_answered_conv)
old_sids = set(kept[k][1].get("sample_id") for k in extra if kept[k][1].get("sample_id"))
S["交集"] = {
    "训练池D_N3072": {"行数": len(poolD), "唯一sample_id": len(poolD_sids)},
    "训练池L_N3072": {"行数": len(poolL), "唯一sample_id": len(poolL_sids)},
    "D与L同集": poolD_sids == poolL_sids,
    "gap1200_sid∩pool3072": len(gap_sids & poolD_sids),
    "gap1200_sid∩eval102": len(gap_sids & e102_sids),
    "pool3072_sid∩eval102": len(poolD_sids & e102_sids),
    "已答区间_sid∩pool3072": len(answered_sids & poolD_sids),
    "已答区间_sid∩eval102": len(answered_sids & e102_sids),
    "旧90包7图_sid∩pool3072": len(old_sids & poolD_sids),
    "旧90包7图_sid∩eval102": len(old_sids & e102_sids),
    "eval102_sid∩gap1200明细样例": sorted(gap_sids & e102_sids)[:5],
    "gap1200∩pool3072明细样例": sorted(gap_sids & poolD_sids)[:5],
}

# 已答集合内 image_sha256 重复（同图多任务/同图多item）
ans_sha = defaultdict(list)
for k in kept: ans_sha[kept[k][1].get("image_sha256")].append(k)
ans_sha_dups = {s: ids for s, ids in ans_sha.items() if len(ids) > 1}
S["去重"]["已答集合内image_sha256重复组"] = ans_sha_dups
S["去重"]["已答唯一item_id对应唯一image_sha256数"] = len(ans_sha)
S["去重"]["说明_sha重复"] = "同一 image_sha256 出现在多个 item_id 下 = 同一张图的多条任务记录，晶圆数（唯一图片数）应按 sha 去重后计"

# ---------------------------------------------------------------- 6. 临时文件覆盖交叉检查
tmp_ids = set()
for p in glob.glob(WB+"/GLM/workers/_tmp_*.jsonl"):
    if "作废" in p: continue
    for r in read_jsonl(p): tmp_ids.add(r.get("item_id"))
tmp_json_ids = set(); tmp_json_bad = []
for p in glob.glob(WB+"/GLM/workers/_tmp_*.json"):
    txt = open(p, encoding="utf-8").read()
    ids = set()
    try:
        d = json.loads(txt)
        els = d if isinstance(d, list) else [d]
        for e in els:
            if isinstance(e, dict) and e.get("item_id"): ids.add(e["item_id"])
    except Exception:
        try:
            for l in txt.splitlines():
                if l.strip():
                    e = json.loads(l)
                    if isinstance(e, dict) and e.get("item_id"): ids.add(e["item_id"])
        except Exception:
            tmp_json_bad.append(rel(p)); continue
    tmp_json_ids |= ids
attempt1 = read_jsonl(WB+"/GLM/workers/_tmp_090_attempt1_worker自停作废.jsonl")
S["临时文件"] = {
    "tmp_jsonl唯一item_id": len(tmp_ids),
    "tmp_jsonl中不在最终文件的item_id": sorted(tmp_ids - kept_ids),
    "tmp_payloadjson唯一item_id": len(tmp_json_ids),
    "tmp_payloadjson中不在最终文件的item_id": sorted(tmp_json_ids - kept_ids),
    "tmp_payloadjson解析失败文件数": len(tmp_json_bad), "样例": tmp_json_bad[:5],
    "attempt1作废条数": len(attempt1),
    "attempt1作废item_id": [r.get("item_id") for r in attempt1],
    "attempt1是否被最终文件取代(同id已在最终)": all(r.get("item_id") in kept_ids for r in attempt1),
    "说明": "_tmp_*.jsonl=worker落盘前的临时原答(结构与最终同)；_tmp_*.json=worker暂存payload；attempt1_worker自停作废=按会话账目作废不复用；均不单独计入唯一原答",
}

# ---------------------------------------------------------------- 7. 分片状态
status = read_jsonl(WB+"/GLM/分片状态.jsonl")
by_shard = defaultdict(list)
for r in status: by_shard[r.get("shard")].append(r)
last_status = {s: v[-1].get("status") for s, v in by_shard.items()}
S["分片状态"] = {
    "记录数": len(status),
    "状态分布(按分片末条)": dict(Counter(last_status.values())),
    "末条为done的分片数": sum(1 for v in last_status.values() if v == "done"),
    "workers中最终原答分片数": len(glob.glob(WB+"/GLM/workers/分片*_原答.jsonl")),
    "末条为dispatched的分片": sorted([s for s, v in last_status.items() if v == "dispatched"]),
}

# ---------------------------------------------------------------- 写入输出
merged_path = os.path.join(OUT, "合并_去重后原答.jsonl")
with open(merged_path, "w", encoding="utf-8") as f:
    for k in sorted(kept, key=lambda x: (0, int(x[3:])) if x.startswith("wbg") else (1, int(x[3:])) if x.startswith("wb_") else (2, x)):
        src, r = kept[k]
        rec = {"item_id": k, "sample_id": r.get("sample_id"), "image_sha256": r.get("image_sha256"),
               "raw_answer": r.get("raw_answer"), "来源文件": src}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

snap_end = now()
# 审计期间是否仍在写入：重扫 workers 最新 mtime
newest = None
for p in glob.glob(WB+"/GLM/workers/*") + glob.glob(WB+"/GLM/*"):
    m = os.stat(p).st_mtime
    if newest is None or m > newest[0]:
        newest = (m, rel(p))
S["读取时刻"]["结束"] = snap_end
S["读取时刻"]["最新写入文件"] = {"路径": newest[1], "mtime": datetime.datetime.fromtimestamp(newest[0]).astimezone().isoformat(timespec="seconds")}
S["读取时刻"]["说明"] = "WorkBuddy 侧当时仍在运行（最新原答文件 mtime 距今约在分钟级）；本报告只是读取时刻快照"

with open(os.path.join(OUT, "统计.json"), "w", encoding="utf-8") as f:
    json.dump(S, f, ensure_ascii=False, indent=2)

# 控制台摘要
print("=== 快照", snap_start, "→", snap_end)
print("最终原答文件数:", len(inventory["最终原答"]), " 唯一item_id:", len(kept))
print("wbg已答:", len(wbg_answered_conv), "/1200  缺失:", len(missing), " 缺失前10:", [k for k in blind_ids if k in set(missing)][:10])
print("旧包extra:", sorted(extra))
print("多副本item:", list(dup_item.keys()), {k: [t[0] for t in v] for k, v in dup_item.items()})
print("gap∩pool3072:", S["交集"]["gap1200_sid∩pool3072"], " gap∩eval102:", S["交集"]["gap1200_sid∩eval102"])
print("机械: json失败", len(mech_fail_json), "空答", len(mech_empty), "缺字段", len(field_missing), "非九类", len(cls_bad), "unknown", len(cls_unknown))
print("dc分布:", dict(dc_counter))
print("radial_zone值:", dict(rz_counter))
print("sha重复组:", ans_sha_dups)
print("tmp中不在最终:", sorted(tmp_ids - kept_ids))
print("分片状态:", S["分片状态"]["状态分布(按分片末条)"], "dispatched:", S["分片状态"]["末条为dispatched的分片"])
print("Kimi文件:", kimi_files)
print("080副本明细:", json.dumps(dup_detail, ensure_ascii=False))
print("check分布:", S["机械检查"]["check字段分布"])
print("非JSON按分片:", S["机械检查"]["非JSON按分片"])
print("raw对象按分片:", S["机械检查"]["raw对象按分片"])
print("非JSON样例:", json.dumps(S["机械检查"]["非JSON文本样例"], ensure_ascii=False)[:600])
print("缺失连续块:", S["覆盖"]["缺失等于wbg1149起连续块"])
print("旧90∩pool3072明细:", sorted(old_sids & poolD_sids))
# 分片089/099/100 状态记录
for s in ("089", "099", "100", "090"):
    print("状态-分片%s:" % s, json.dumps(by_shard.get(s), ensure_ascii=False)[:400])
