#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补交：生成 描述七字段.jsonl / 描述SFT_356.jsonl / 描述评测协议.json 的**转换脚本**。

原脚本未随交付入库，本文件是按原逻辑重建的版本，以**哈希复现**作为证据：
输出写到临时目录，逐文件比对冻结产物的 SHA256，三者全中才通过。
（重建而非原件；比对通过说明逻辑等价，不等于保留了原文件字节。）

原先的加载路径把 GPT 来源写成 `REPO/01_Codex_指挥/...`，漏了 `协作/`，
rglob 在不存在目录上静默返回空，会丢掉 42 份 GPT 原答且不报错。此处已修正。
"""
from __future__ import annotations
import hashlib, io, json, statistics, sys, tempfile
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path("D:/pycode/晶圆图研究")
ABC = Path("D:/pycode/.ssh-tmp/abc")
FROZEN = REPO / "协作" / "02_ClaudeCode_实操" / "同图ABC准备_20261007"
TMP = Path(tempfile.mkdtemp(prefix="repro_"))

FIELDS = ["defect_class", "morphology", "radial_zone", "clock_direction",
          "extent_r", "caption_zh", "uncertainty"]
ZONES = ["center", "middle", "edge", "global", "none", "unknown"]
EXPECT = {
    "描述七字段.jsonl": "26719297d246ef86127ff3b6cb0a8f498d9f2dbf877278bd10b0c802bb46085a",
    "描述SFT_356.jsonl": "1d26dbca6dbf6b7df77e96b4ebf8b3c74d6500c6040cd1b70ace4c151cfbc217",
    "描述评测协议.json": "82cb43cd675fc4a3dc227d8ce89aefa92fa1ea1635d1fd3bc728f645b5d6bcd6",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rd(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


man = {r["sample_id"]: r for r in rd(REPO / "data" / "manifest.jsonl")}
rows = rd(ABC / "items356.jsonl")

# ── 描述来源：三处 ──────────────────────────────────────
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
            try:
                raws[sid] = {**json.loads(p.read_text(encoding="utf-8")), "_src": tag}
            except Exception:
                pass
rev49 = {x["sample_id"]: x for x in rd(
    REPO / "协作" / "01_Codex_指挥" / "验收_20261007_接续49" / "模型辅助候选49.jsonl")}
print(f"描述来源：旧158 {len(old158)} · 结构化90 {len(struct90)} · 新原答 {len(raws)} · 49复核 {len(rev49)}")

# ── 统一描述题面 ────────────────────────────────────────
PROMPT = (
    "这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。\n"
    "请**只根据图中实际可见的内容**描述主要缺陷图案。不要推测工艺根因，"
    "不要把没有看到的模式写进答案，不要给出未经验证的数值尺寸、比例或覆盖率。\n"
    "只输出一个 JSON 对象，不写推理过程或 Markdown，必须包含以下七个字段：\n"
    '{"defect_class":"九类之一或 unknown",'
    '"morphology":"主要可见形态的简短中文描述",'
    '"radial_zone":"center/middle/edge/global/none/unknown",'
    '"clock_direction":"可靠识别局部缺陷方向时写简短钟点方向，否则 null",'
    '"extent_r":null,'
    '"caption_zh":"一句话客观描述主要形态与位置",'
    '"uncertainty":"不确定之处，无则空字符串"}\n'
    "defect_class 从 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_full、Random、"
    "Scratch、none 中选一个；无法可靠判断时填 \"unknown\"。\n"
    "图片上方为 12 点钟，右方为 3 点钟。中央、整圈、全局、正常或不确定时 "
    "clock_direction 填 null。extent_r 固定为 null（本轮不提供尺寸依据）。"
)
print(f"统一描述题面 {len(PROMPT)} 字符")


# ── 构造七字段监督目标 ──────────────────────────────────
def build(sid):
    f = {k: None for k in FIELDS}
    srcs = {}
    a = None
    if sid in raws:
        a = raws[sid]
        if sid in rev49:
            ra = (rev49[sid].get("answer") or {})
            if ra.get("caption_zh"):
                a = {**ra, "_src": raws[sid]["_src"] + "+Codex复核副本"}
    if a is None and sid in struct90:
        a = struct90[sid]
    if a is None and sid in old158:
        a = old158[sid]
    if a is None:
        return f, srcs, None
    for k in FIELDS:
        v = a.get(k)
        if k == "extent_r":
            v = None
        if k == "defect_class":
            v = man[sid]["failure_type"]
        if isinstance(v, str):
            v = v.strip() or None
        f[k] = v
        srcs[k] = a["_src"]
    return f, srcs, a["_src"]


items, cov = [], Counter()
for r in rows:
    sid = r["sample_id"]
    f, srcs, main_src = build(sid)
    rz = f.get("radial_zone")
    if rz in ("global", "none", "unknown") or man[sid]["failure_type"] in ("Edge_Ring",):
        f["clock_direction"] = None
    if rz not in ZONES:
        f["radial_zone"] = None
    for k in FIELDS:
        if f.get(k) is not None:
            cov[k] += 1
    items.append({"item_id": r["item_id"], "sample_id": sid, "label": man[sid]["failure_type"],
                  "lot_name": man[sid]["lot_name"], "fields": f, "field_sources": srcs,
                  "main_source": main_src})

with io.open(TMP / "描述七字段.jsonl", "w", encoding="utf-8", newline="\n") as fh:
    for it in items:
        fh.write(json.dumps(it, ensure_ascii=False) + "\n")

# ── 描述 SFT 数据 ───────────────────────────────────────
recs = []
for it in items:
    tgt = {k: it["fields"][k] for k in FIELDS}
    recs.append({"item_id": it["item_id"], "sample_id": it["sample_id"],
                 "label": it["label"], "lot_name": it["lot_name"],
                 "messages": [{"role": "user", "content": "<image>" + PROMPT},
                              {"role": "assistant",
                               "content": json.dumps(tgt, ensure_ascii=False,
                                                     separators=(",", ":"))}],
                 "images": [f"/WS/datasets/expand_v2/images/{it['sample_id']}.png"]})
with io.open(TMP / "描述SFT_356.jsonl", "w", encoding="utf-8", newline="\n") as fh:
    for r in recs:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

# ── Base 评测协议 ───────────────────────────────────────
proto = {
    "题面": PROMPT, "题面_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
    "主验收": ["形态/位置/方向的事实一致性", "无依据断言", "空答率", "截断率"],
    "分类": "辅助指标（九类 Macro-F1）",
    "解析": {
        "字段": FIELDS,
        "defect_class": "九类之一或 unknown；从 JSON 对象取",
        "未知位置不硬填": "radial_zone 不在枚举内 → 记 null 并计为缺失",
        "extent_r": "必须为 null；若模型给出数值 → 计为**无依据数值断言**",
        "拒绝": ["重复键", "截断", "多对象", "尾随文本"],
    },
    "生成上限": {
        "max_tokens": 256,
        "理由": "七字段目标中位约 %d 字符，最长约 %d；**不能沿分类的 64**，"
                "否则必然截断。640 步实测分类目标只需 8-9 token，描述目标要两个数量级。" % (
                    int(statistics.median([len(json.dumps({k: it['fields'][k] for k in FIELDS},
                                                           ensure_ascii=False)) for it in items])),
                    max(len(json.dumps({k: it["fields"][k] for k in FIELDS}, ensure_ascii=False))
                        for it in items)),
    },
    "评分口径": {
        "有独立 gold 的字段": "才参与自动评分",
        "无独立 gold 的字段": "**只展示输出与模型复核草稿**，不把训练教师文本当唯一评测真值",
        "自由文本事实性": "未获独立依据 → **列未测**",
        "训练规则合规分": "**不能**用来证明总体描述正确",
    },
    "尺寸": "无可靠 gold，extent_r 固定 null，**不改写为 0**",
}
(TMP / "描述评测协议.json").write_text(json.dumps(proto, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8", newline="\n")

# ── 哈希复现比对 ────────────────────────────────────────
print("\n" + "=" * 74)
print("哈希复现比对")
print("=" * 74)
ok = True
for name, exp in EXPECT.items():
    got = sha(TMP / name)
    same = got == exp
    ok &= same
    print(f"  {'一致' if same else '**不一致**'}  {name}")
    print(f"          重建 {got}")
    if not same:
        print(f"          冻结 {exp}")
print(f"\n临时目录 {TMP}")
print(f"结论：{'三份冻结产物均可由本脚本逐字节复现' if ok else '**复现失败**'}")
sys.exit(0 if ok else 1)
