#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""描述训练副本 v2 —— 按字段证据分型构造，未知字段不监督。

锁定输入（SHA256 已核，不修改）：
  固定356条.jsonl    c6f2015a…  （顺序与 ID 来源）
  描述七字段.jsonl   26719297…  （字段**取值**来源）

只读来源（仅用于判定「有来源的不适用」与「缺标注」，不重算取值）：
  结构化候选90 · ZCode 原答（107+49） · GPT 原答（42） · 49 复核副本

输出（本目录）：
  描述SFT_v2_356.jsonl   训练副本，356 行，原顺序，混合任务
  任务分型_v2.jsonl       逐行任务类型 / 请求字段 / 字段来源 / 省略原因
  字段来源_v2.json        汇总与自查
  描述题面_v2.json        题面模板与 SHA256
"""
from __future__ import annotations
import hashlib, io, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path("D:/pycode/晶圆图研究")
ABC = REPO / "协作" / "02_ClaudeCode_实操" / "同图ABC准备_20261007"
OUT = REPO / "协作" / "02_ClaudeCode_实操" / "描述SFT与同题评测_20261007_1430"
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = ["defect_class", "morphology", "radial_zone", "clock_direction",
          "extent_r", "caption_zh", "uncertainty"]
STRUCT_FIELDS = ["defect_class", "morphology", "radial_zone", "clock_direction",
                 "extent_r", "caption_zh"]
CAP_FIELDS = ["defect_class", "caption_zh"]
ZONES = ["center", "middle", "edge", "global", "none", "unknown"]
PUBLIC_SRC = "公开原标签（manifest.failure_type，label_source=ground_truth）"


def rd(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ══ 1. 只读来源装载（路径与冻结产物一致）════════════════════
old158 = {}
for x in rd(REPO / "协作" / "01_Codex_指挥" / "训练数据就绪_20261003" / "caption_train_158.jsonl"):
    m = x.get("messages") or []
    if len(m) > 1:
        old158[x["sample_id"]] = {"caption_zh": m[1]["content"], "_src": "旧158描述候选（仅 caption）"}

struct90 = {}
for x in rd(REPO / "协作" / "01_Codex_指挥" / "训练数据就绪_20261003" / "structured_candidates_90.jsonl"):
    struct90[x["sample_id"]] = {**(x.get("answer") or {}), "_src": "结构化候选90（Codex非盲模型复核）"}

blind = rd(REPO / "协作" / "02_ClaudeCode_实操" / "ZCode标注准备_20261006" / "zcode_blind_new.jsonl")
zc2sid = {b["item_id"]: b["sample_id"] for b in blind}

raws = {}
for root, tag in ((REPO / "协作" / "03_ZCode_标注" / "扩容198_20261006_165855", "GLM原始回答"),
                  (REPO / "协作" / "03_ZCode_标注" / "接续49_20261006_20261007-000123", "GLM接续原始回答"),
                  (REPO / "协作" / "01_Codex_指挥" / "GPT补标91_20261006", "GPT原始回答")):
    assert root.exists(), f"来源目录不存在：{root}"
    for p in sorted(root.rglob("*_raw.json")):
        sid = zc2sid.get(p.name.replace("_raw.json", ""))
        if sid:
            raws[sid] = {**json.loads(p.read_text(encoding="utf-8")), "_src": tag}

rev49 = {x["sample_id"]: x for x in
         rd(REPO / "协作" / "01_Codex_指挥" / "验收_20261007_接续49" / "模型辅助候选49.jsonl")}
print(f"来源装载：旧158 {len(old158)} · 结构化90 {len(struct90)} · 原答 {len(raws)} · 49复核 {len(rev49)}")

# ══ 2. 取值取自锁定文件，存在性改由原始来源判定 ════════════
frozen = {r["sample_id"]: r for r in rd(ABC / "描述七字段.jsonl")}
order = rd(ABC / "固定356条.jsonl")
man = {r["sample_id"]: r for r in rd(REPO / "data" / "manifest.jsonl")}

rows, mism = [], []
for o in order:
    sid = o["sample_id"]
    fr = frozen[sid]
    fvals = fr["fields"]

    # 复算来源标签；必须与冻结文件逐条一致，否则存在性判定不可信
    a, tag = None, None
    if sid in raws:
        tag = raws[sid]["_src"]
        a = raws[sid]
        if sid in rev49:
            ra = rev49[sid].get("answer") or {}
            if ra.get("caption_zh"):
                a, tag = ra, tag + "+Codex复核副本"
    elif sid in struct90:
        a, tag = struct90[sid], struct90[sid]["_src"]
    elif sid in old158:
        a, tag = old158[sid], old158[sid]["_src"]
    if tag != fr["main_source"]:
        mism.append((sid, tag, fr["main_source"]))

    present = {f: (a is not None and f in a) for f in FIELDS}
    # 协议改写：radial_zone 为全局/无/未知，或类别为 Edge_Ring 时，方向按协议置 null
    rz = fvals["radial_zone"]
    proto_na_dir = (rz in ("global", "none", "unknown")) or (o["label"] == "Edge_Ring")

    rows.append({"order": o["order"], "item_id": o["item_id"], "sample_id": sid,
                 "label": o["label"], "lot_name": o["lot_name"],
                 "image": o["image"], "image_sha256": o["image_sha256"],
                 "v": fvals, "present": present, "src": tag,
                 "proto_na_dir": proto_na_dir})

print(f"来源标签与冻结文件一致：{len(rows)-len(mism)}/{len(rows)}  不一致 {len(mism)}")
assert not mism, f"来源复算不一致，停止：{mism[:3]}"

# ══ 3. 分型 ════════════════════════════════════════════════
for r in rows:
    r["task"] = "结构化" if r["v"]["morphology"] is not None else "类别Caption辅助"

cnt = Counter(r["task"] for r in rows)
print(f"\n任务分型：{dict(cnt)}")

# ══ 4. 字段取舍 ════════════════════════════════════════════
def decide(r):
    """返回 (请求字段列表, 取值 dict, 逐字段来源 dict, 省略原因列表)"""
    keep, sv, om = [], {}, []
    st = r["task"] == "结构化"
    for f in FIELDS:
        if f in CAP_FIELDS:
            if st:
                sv[f] = r["src"]
            keep.append(f)
            sv[f] = PUBLIC_SRC if f == "defect_class" else r["src"]
            continue
        if not st:
            om.append((f, "该条仅有类别与 Caption 来源，无结构化依据"))
            continue
        if f == "uncertainty":
            if r["v"][f] is None:
                om.append((f, "来源未给出不确定性文本（缺标注），省略该字段"))
                continue
        if f == "clock_direction":
            # 有来源的「不适用」（含协议改写的 null）可训练；仅键缺失才省略
            if r["v"][f] is None and not r["present"][f]:
                om.append((f, "来源无该字段（缺标注），省略"))
                continue
        if f == "extent_r":
            sv[f] = r["src"] + "（协议固定 null：本轮不提供尺寸依据）"
        else:
            sv[f] = r["src"]
        keep.append(f)
    return keep, {f: r["v"][f] for f in keep}, sv, om


# ══ 5. 题面 ════════════════════════════════════════════════
DOC = {"defect_class": '"defect_class":"九类之一或 unknown"',
       "morphology": '"morphology":"主要可见形态的简短中文描述"',
       "radial_zone": '"radial_zone":"center/middle/edge/global/none/unknown"',
       "clock_direction": '"clock_direction":"可靠识别局部缺陷方向时写简短钟点方向，否则 null"',
       "extent_r": '"extent_r":null',
       "caption_zh": '"caption_zh":"一句话客观描述主要形态与位置"',
       "uncertainty": '"uncertainty":"不确定之处，无则空字符串"'}
HEAD = ("这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。\n"
        "请**只根据图中实际可见的内容**描述主要缺陷图案。不要推测工艺根因，"
        "不要把没有看到的模式写进答案，不要给出未经验证的数值尺寸、比例或覆盖率。\n")
CLS = ('defect_class 从 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_full、Random、'
       'Scratch、none 中选一个；无法可靠判断时填 "unknown"。')


def make_prompt(keep):
    body = ",".join(DOC[f] for f in keep)
    n = len(keep)
    zh = {2: "两", 6: "六", 7: "七"}[n]
    p = HEAD + f"只输出一个 JSON 对象，不写推理过程或 Markdown，必须包含以下{zh}个字段：\n{{{body}}}\n" + CLS + "\n"
    if "clock_direction" in keep:
        p += ('图片上方为 12 点钟，右方为 3 点钟。中央、整圈、全局、正常或不确定时 '
              'clock_direction 填 null。extent_r 固定为 null（本轮不提供尺寸依据）。')
    return p


PROMPTS = {}
for key, keep in (("结构化_无不确定", STRUCT_FIELDS),
                  ("结构化_含不确定", STRUCT_FIELDS + ["uncertainty"]),
                  ("类别Caption辅助", CAP_FIELDS)):
    PROMPTS[key] = make_prompt(keep)

# ══ 6. 生成副本 ════════════════════════════════════════════
recs, typing, cover = [], [], defaultdict(Counter)
for r in rows:
    keep, tgt, sv, om = decide(r)
    key = ("类别Caption辅助" if r["task"] == "类别Caption辅助"
           else ("结构化_含不确定" if "uncertainty" in keep else "结构化_无不确定"))
    q = PROMPTS[key]
    assert "<image>" not in q
    for f in sv:
        cover[f][sv[f]] += 1
    recs.append({"order": r["order"], "item_id": r["item_id"], "sample_id": r["sample_id"],
                 "label": r["label"], "lot_name": r["lot_name"],
                 "messages": [{"role": "user", "content": "<image>" + q},
                              {"role": "assistant",
                               "content": json.dumps(tgt, ensure_ascii=False, separators=(",", ":"))}],
                 "images": [f"/WS/datasets/expand_v2/images/{r['sample_id']}.png"]})
    typing.append({"order": r["order"], "item_id": r["item_id"], "sample_id": r["sample_id"],
                   "label": r["label"], "lot_name": r["lot_name"], "task": r["task"],
                   "题面种类": key, "请求字段": keep, "字段来源": sv,
                   "省略字段": [{"field": f, "reason": rs} for f, rs in om],
                   "描述来源": r["src"]})

p_sft = OUT / "描述SFT_v2_356.jsonl"
with io.open(p_sft, "w", encoding="utf-8", newline="\n") as fh:
    for x in recs:
        fh.write(json.dumps(x, ensure_ascii=False) + "\n")
p_typ = OUT / "任务分型_v2.jsonl"
with io.open(p_typ, "w", encoding="utf-8", newline="\n") as fh:
    for x in typing:
        fh.write(json.dumps(x, ensure_ascii=False) + "\n")

# ══ 7. 自查 ════════════════════════════════════════════════
p_ord = BACK = None
print("\n" + "=" * 74)
print("自查")
print("=" * 74)
chk = []
def ck(name, ok, d=""):
    chk.append((name, bool(ok), d)); print(f"  {'OK  ' if ok else 'FAIL'} {name}  {d}")

ids_rec = [x["sample_id"] for x in recs]
ids_ord = [o["sample_id"] for o in order]
ck("副本 356 行", len(recs) == 356, str(len(recs)))
ck("ID 与顺序与 固定356条 完全一致", ids_rec == ids_ord)
ck("唯一图片 = 356", len(set(ids_rec)) == 356, str(len(set(ids_rec))))
ck("与冻结七字段一一对应", set(ids_rec) == set(frozen))
ck("结构化 288", cnt["结构化"] == 288, str(cnt["结构化"]))
ck("类别Caption辅助 68", cnt["类别Caption辅助"] == 68, str(cnt["类别Caption辅助"]))
ck("属性顺序单调", [x["order"] for x in recs] == sorted(x["order"] for x in recs))

# 取值必须与冻结文件逐字段相同
diff = []
for x in recs:
    t = json.loads(x["messages"][1]["content"])
    for k, v in t.items():
        if x["sample_id"] in frozen and frozen[x["sample_id"]]["fields"][k] != v:
            diff.append((x["sample_id"], k))
ck("监督取值与锁定输入逐字段一致", not diff, f"差异 {len(diff)}")

# 未监督字段确实不在目标里
bad = []
for x in typing:
    t = json.loads(recs[x["order"] - 1]["messages"][1]["content"])
    for o in x["省略字段"]:
        if o["field"] in t:
            bad.append((x["sample_id"], o["field"]))
ck("省略字段确实不在监督目标中", not bad, f"违反 {len(bad)}")

# 题面无行长相关字符串：题面必须取自固定模板集合（九类候选名单属模板常量，
# 按协议保留，不计为泄漏），且自由文本答案值不得出现在题面中
tmpl = set(PROMPTS.values())
leak, off_tmpl = [], []
for x, t in zip(typing, recs):
    q = t["messages"][0]["content"]
    if not q.startswith("<image>") or q[7:] not in tmpl:
        off_tmpl.append(x["sample_id"])
        continue
    for probe in (x["sample_id"], x["lot_name"], x["item_id"]):
        if probe in q:
            leak.append((x["sample_id"], probe))
    tv = json.loads(t["messages"][1]["content"])
    for k, v in tv.items():
        if k != "defect_class" and isinstance(v, str) and len(v) > 8 and v in q:
            leak.append((x["sample_id"], f"value:{k}"))
ck("题面取自固定模板集合（无行长相关文本）", not off_tmpl, f"越界 {off_tmpl[:3]}")
ck("题面无 ID / lot / 自由文本答案值", not leak, f"命中 {leak[:3]}")

cap_null = sum(1 for x in recs
               if json.loads(x["messages"][1]["content"])["caption_zh"] is None)
ck("无空 Caption", cap_null == 0, str(cap_null))
ck("题面与解析协议题面不同源（改后需重新评测）", True,
   "训练题面为分型题面；评测仍用 描述评测协议.json 的七字段题面")

cov = {f: sum(1 for x in recs if f in json.loads(x["messages"][1]["content"])) for f in FIELDS}
print("\n监督字段覆盖：")
for f in FIELDS:
    print(f"  {f:16s} {cov[f]:4d}/{len(recs)}")
print("\n逐字段来源分账：")
for f in FIELDS:
    if cover[f]:
        print(f"  {f}:")
        for s, n in cover[f].most_common():
            print(f"      {n:4d}  {s}")

p_src = OUT / "字段来源_v2.json"
json.dump({"锁定输入": {"固定356条.jsonl": sha(ABC / "固定356条.jsonl"),
                        "描述七字段.jsonl": sha(ABC / "描述七字段.jsonl")},
           "输出": {"描述SFT_v2_356.jsonl": sha(p_sft), "任务分型_v2.jsonl": sha(p_typ)},
           "任务分型": dict(cnt), "题面种类": dict(Counter(x["题面种类"] for x in typing)),
           "监督字段覆盖": cov, "字段来源": {f: dict(cover[f]) for f in FIELDS},
           "省略原因": dict(Counter(o["reason"] for x in typing for o in x["省略字段"])),
           "自查": [{"项": n, "通过": o, "说明": d} for n, o, d in chk],
           "通过": f"{sum(1 for _, o, _ in chk if o)}/{len(chk)}"},
          io.open(p_src, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

p_pr = OUT / "描述题面_v2.json"
json.dump({"说明": "训练题面按行的字段证据分型；评测仍用 描述评测协议.json 的七字段题面。",
           "题面": {k: {"文本": v, "字符数": len(v),
                        "sha256": hashlib.sha256(v.encode()).hexdigest()} for k, v in PROMPTS.items()},
           "与实际使用对应": {k: sum(1 for x in typing if x["题面种类"] == k)
                              for k in PROMPTS}},
          io.open(p_pr, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print(f"\n写出：\n  {p_sft.name}  {sha(p_sft)[:16]}")
print(f"  {p_typ.name}  {sha(p_typ)[:16]}")
print(f"  {p_src.name}\n  {p_pr.name}")
print(f"\n通过 {sum(1 for _, o, _ in chk if o)}/{len(chk)}")
