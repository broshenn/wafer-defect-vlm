#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WorkBuddy 1200 最终冻结 · 字段分账 · 下一版候选（纯 CPU，不改任何原件）。

依据 Codex《验收与裁决》「WorkBuddy1200已完整，格式候选不能等同事实gold」与
任务书第三节。硬约束：

  · **只读** 172 份 canonical 分片（`分片N_原答.jsonl`），不改、不删、不移动任何文件
  · 临时件 / 备份 / 中止件**另列**，不吞历史尝试
  · 旧 7 件与新 1200 **分账**
  · item_id / sample_id / 图片 SHA **分别**报告
  · **相同图片 SHA ≠ 相同晶圆 ID** —— 不静默合并，重复组单列
  · **保存记录数 ≠ 合格数 ≠ 事实正确数 ≠ 实际后端调用次数**
  · 不做批量重标、不硬补七字段、不自动入训或回填 3072
  · 纯文本降级为 Caption 候选时：**逐句**判断，删整句必须留
    原文 hash + 原句 + 理由 + 副本 diff；**不许简单删掉数字留下伪精确残句**

跑法：python 冻结与分账.py
"""
from __future__ import annotations
import hashlib, io, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent                      # …/描述事实验收_20261008_1911/收口_v2
ROOT = HERE.parent                                          # …/描述事实验收_20261008_1911
REPO = ROOT.parent.parent.parent                            # 仓库根（02_ClaudeCode_实操 → 协作 → 仓库）
GLM = REPO / "协作" / "04_WorkBuddy_复核" / "夜间主标_20261008_0301" / "GLM"
WORKERS = GLM / "workers"
PKG = (REPO / "协作" / "02_ClaudeCode_实操" / "夜间扩容_20261008_0244"
       / "客户端盲包" / "WorkBuddy_训练缺口1200")
OUT = HERE / "WorkBuddy冻结"
SHARD_RE = re.compile(r"^分片(\d+)_原答\.jsonl$")

CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
ZONES = ["center", "middle", "edge", "global", "none", "unknown"]
FIELDS = ["defect_class", "morphology", "radial_zone", "clock_direction",
          "extent_r", "caption_zh", "uncertainty"]

# 纯文本降级用：先隔离的句子类型（**整个句**走，不做「删数字留残句」）
PAT_NUM = re.compile(r"\d+(\.\d+)?\s*(%|％|个|颗|处|倍|分之|die|片|颗)|[0-9]\.[0-9]")
PAT_ROOT = re.compile(r"污染|光刻|蚀刻|刻蚀|设备故障|工艺(原因|问题|参数)|根因|"
                      r"contamination|lithography|etch|因为.*导致|由于.*导致")
PAT_RADIUS = re.compile(r"半径|直径|R\s*=|覆盖率|占比|面积比")


class DupKey(ValueError):
    pass


def load_jsonl_no_dupkey(path: Path):
    """逐行读 JSONL，**拒绝重复键**。返回 (行列表, 坏行列表)。"""
    rows, bad = [], []
    for ln, line in enumerate(io.open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line, object_pairs_hook=_nodup)
            rows.append((ln, obj))
        except DupKey as e:
            bad.append({"行": ln, "原因": "重复键", "详情": str(e)})
        except Exception as e:
            bad.append({"行": ln, "原因": type(e).__name__, "详情": str(e)[:120]})
    return rows, bad


def _nodup(pairs):
    keys = [p[0] for p in pairs]
    if len(keys) != len(set(keys)):
        dup = [k for k, c in Counter(keys).items() if c > 1]
        raise DupKey(f"重复键 {dup}")
    return dict(pairs)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with io.open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def schema7(obj):
    """与冻结协议 eval_desc_v2.py:schema7 **逐条一致**。"""
    if not isinstance(obj, dict):
        return False, ["未解析"]
    p = []
    miss = [f for f in FIELDS if f not in obj]
    if miss:
        p.append("缺字段:" + ",".join(miss))
    extra = [k for k in obj if k not in FIELDS]
    if extra:
        p.append("多字段:" + ",".join(sorted(extra)))
    if "defect_class" in obj and obj["defect_class"] not in CLASSES + ["unknown"]:
        p.append(f"defect_class越界:{str(obj['defect_class'])[:20]}")
    rz = obj.get("radial_zone")
    if "radial_zone" in obj:
        if not isinstance(rz, str):
            p.append("radial_zone非字符串")
        elif rz not in ZONES:
            p.append(f"radial_zone越界:{rz[:20]}")
    cd = obj.get("clock_direction")
    if "clock_direction" in obj and not (cd is None or isinstance(cd, str)):
        p.append("clock_direction类型错")
    if "extent_r" in obj and obj["extent_r"] is not None:
        p.append(f"extent_r非null:{str(obj['extent_r'])[:12]}")
    for f in ("morphology", "caption_zh", "uncertainty"):
        if f in obj and not isinstance(obj[f], str):
            p.append(f"{f}非字符串")
    return (not p), p


# ---------------- 纯文本 → Caption 候选：逐句隔离 ----------------
SENT = re.compile(r"[^。；;!！?？\n]+[。；;!！?？]?")


def degrade_caption(text: str):
    """把纯文本降级成 Caption 候选。**整句**取舍，返回 (保留文本, 变更记录)。"""
    changes = []
    kept = []
    for s in SENT.findall(text):
        s2 = s.strip()
        if not s2:
            continue
        reasons = []
        if PAT_ROOT.search(s2):
            reasons.append("含未验证的工艺根因")
        if PAT_RADIUS.search(s2):
            reasons.append("含未验证的覆盖率/半径比类表述")
        if PAT_NUM.search(s2):
            reasons.append("含未验证的数量/百分比")
        if reasons:
            changes.append({"动作": "隔离整句", "原句": s2,
                            "原句sha256": sha256_text(s2),
                            "理由": "；".join(reasons)})
        else:
            kept.append(s2)
    new = "".join(kept).strip()
    return new, changes


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict = {"冻结时刻": None, "口径": {
        "只读来源": "GLM/workers/分片NNN_原答.jsonl（172 份 canonical）",
        "拒重复键": "object_pairs_hook 检查，命中即整行计入坏行",
        "不做": ["批量重标", "硬补七字段", "自动入训", "回填 3072", "删改原件"],
        "区分": "保存记录数 ≠ 合格数 ≠ 事实正确数 ≠ 实际后端调用次数",
    }}

    import datetime
    rep["冻结时刻"] = datetime.datetime.now().isoformat(timespec="seconds")

    # ---------- 1. 冻结：172 份 canonical 分片的指纹 ----------
    shards = sorted([p for p in WORKERS.iterdir() if SHARD_RE.match(p.name)],
                    key=lambda p: int(SHARD_RE.match(p.name).group(1)))
    print(f"canonical 分片：{len(shards)} 份")
    fps, all_rows, all_bad = [], [], []
    for p in shards:
        rows, bad = load_jsonl_no_dupkey(p)
        fps.append({"文件": p.name, "字节": p.stat().st_size,
                    "行数": len(rows), "坏行": len(bad),
                    "sha256": sha256_file(p),
                    "mtime": datetime.datetime.fromtimestamp(
                        p.stat().st_mtime).isoformat(timespec="seconds")})
        all_rows += [(p.name, ln, o) for ln, o in rows]
        all_bad += [{"文件": p.name, **b} for b in bad]
    rep["规范分片"] = {"份数": len(shards), "记录": len(all_rows),
                       "拒重复键坏行": len(all_bad), "坏行明细": all_bad[:50],
                       "逐文件指纹": fps}
    print(f"  记录 {len(all_rows)} 行；拒重复键坏行 {len(all_bad)}")

    # ---------- 2. 非规范件另列（不动它们） ----------
    others = [p for p in WORKERS.iterdir() if not SHARD_RE.match(p.name)]
    by_kind = Counter()
    for p in others:
        n = p.name
        if n.startswith("_tmp_"):
            by_kind["临时件 _tmp_*"] += 1
        elif n.endswith(".py"):
            by_kind["脚本 .py"] += 1
        elif "_finalize" in n or "_rebuild" in n or "_build_" in n:
            by_kind["重建/定稿脚本"] += 1
        else:
            by_kind["其他"] += 1
    rep["非规范件"] = {"总数": len(others), "分类": dict(by_kind),
                      "处置": "**只登记，不删除、不移动**（保留历史尝试）"}
    print(f"  非规范件 {len(others)} 个（只登记，不动）")

    # ---------- 3. 索引映射 + 分账 ----------
    idx = [json.loads(l) for l in
           io.open(PKG / "WB_gap1200_索引.jsonl", encoding="utf-8") if l.strip()]
    i2s = {r["item_id"]: r["sample_id"] for r in idx}
    i2h = {r["item_id"]: r["image_sha256"] for r in idx}

    seen, dup_rows = {}, []
    for shard, ln, o in all_rows:
        iid = o.get("item_id")
        if iid in seen:
            dup_rows.append({"item_id": iid, "文件": shard, "行": ln})
            continue
        seen[iid] = o
    rep["去重"] = {"唯一 item_id": len(seen), "重复额外行": len(dup_rows),
                   "重复明细": dup_rows[:20]}

    # ---------- 4. 三档机械分账 ----------
    tiers = {"A_严格七字段机械合格": [], "B_可解析但不合schema": [],
             "C_纯文本或非唯一对象": []}
    schema_problems = Counter()
    raw_type = Counter()
    for iid, o in seen.items():
        a = o.get("raw_answer")
        raw_type[type(a).__name__] += 1
        obj = None
        if isinstance(a, dict):
            obj = a
        elif isinstance(a, str):
            try:
                cand = json.loads(a.strip())
                obj = cand if isinstance(cand, dict) else None
            except Exception:
                obj = None
        if obj is None:
            tiers["C_纯文本或非唯一对象"].append(iid)
            continue
        ok, probs = schema7(obj)
        for pr in probs:
            schema_problems[pr.split(":")[0]] += 1
        (tiers["A_严格七字段机械合格"] if ok else tiers["B_可解析但不合schema"]).append(iid)

    rep["分账"] = {k: len(v) for k, v in tiers.items()}
    rep["分账"]["合计"] = sum(len(v) for v in tiers.values())
    rep["原答类型分布"] = dict(raw_type)
    rep["schema问题分布"] = dict(schema_problems)

    # ---------- 5. 图片 SHA 与晶圆 ID：**不静默合并** ----------
    sha2items = defaultdict(list)
    for iid in seen:
        h = i2h.get(iid) or seen[iid].get("image_sha256")
        if h:
            sha2items[h].append(iid)
    dup_sha = {h: v for h, v in sha2items.items() if len(v) > 1}
    dup_sha_samples = {h: sorted({i2s.get(i, "?") for i in v}) for h, v in dup_sha.items()}
    rep["图片SHA"] = {
        "唯一 item": len(seen),
        "唯一图片SHA": len(sha2items),
        "重复内容组数": len(dup_sha),
        "重复组": {h: {"item_id": v, "sample_id": dup_sha_samples[h]}
                   for h, v in dup_sha.items()},
        "处置": ("**不合并**：相同 SHA 说明像素内容相同，但 item_id / sample_id 不同，"
                 "在弄清映射来由前**两个晶圆 ID 都保留**；本表只登记。"),
    }
    print(f"  唯一 item {len(seen)}；唯一图片SHA {len(sha2items)}；重复组 {len(dup_sha)}")

    # ---------- 6. 下一版候选 + 变更表 ----------
    cand_dir = OUT / "下一版候选"
    cand_dir.mkdir(exist_ok=True)
    A = cand_dir / "A_结构候选.jsonl"
    B = cand_dir / "B_合法字段候选.jsonl"
    C = cand_dir / "C_Caption候选.jsonl"
    changes = []
    with io.open(A, "w", encoding="utf-8", newline="\n") as fa, \
         io.open(B, "w", encoding="utf-8", newline="\n") as fb, \
         io.open(C, "w", encoding="utf-8", newline="\n") as fc:
        for iid in tiers["A_严格七字段机械合格"]:
            obj = seen[iid]["raw_answer"]
            fa.write(json.dumps({"item_id": iid, "sample_id": i2s.get(iid),
                                 "image_sha256": i2h.get(iid),
                                 "obj": obj}, ensure_ascii=False) + "\n")
        for iid in tiers["B_可解析但不合schema"]:
            a = seen[iid]["raw_answer"]
            obj = a if isinstance(a, dict) else json.loads(a)
            keep, drop = {}, []
            for k, v in obj.items():
                if k not in FIELDS:
                    drop.append(k); continue
                if k == "defect_class" and v not in CLASSES + ["unknown"]:
                    drop.append(k); continue
                if k == "radial_zone" and v not in ZONES:
                    drop.append(k); continue
                if k == "extent_r" and v is not None:
                    drop.append(k); continue
                keep[k] = v
            fb.write(json.dumps({"item_id": iid, "sample_id": i2s.get(iid),
                                 "image_sha256": i2h.get(iid),
                                 "obj_合法字段": keep,
                                 "省略字段": sorted(set(drop)),
                                 "原件sha256": sha256_text(
                                     json.dumps(obj, ensure_ascii=False, sort_keys=True))},
                                ensure_ascii=False) + "\n")
            changes.append({"item_id": iid, "档": "B", "省略字段": sorted(set(drop)),
                            "注": "只省略越界/非法字段；**不由类别推断位置**"})
        for iid in tiers["C_纯文本或非唯一对象"]:
            a = seen[iid]["raw_answer"]
            text = a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)
            new, ch = degrade_caption(text)
            fc.write(json.dumps({"item_id": iid, "sample_id": i2s.get(iid),
                                 "image_sha256": i2h.get(iid),
                                 "caption_候选": new,
                                 "原文sha256": sha256_text(text),
                                 "变更数": len(ch)},
                                ensure_ascii=False) + "\n")
            for c in ch:
                changes.append({"item_id": iid, "档": "C", **c})

    rep["下一版候选"] = {
        "A_结构候选": {"条数": len(tiers["A_严格七字段机械合格"]),
                    "性质": "结构候选；**内容仍待核；缺 gold 不放行**"},
        "B_合法字段候选": {"条数": len(tiers["B_可解析但不合schema"]),
                       "规则": "保留合法独立字段，越界字段省略；只允许无歧义、"
                               "提前列明的一对一别名；保原件与 diff"},
        "C_Caption候选": {"条数": len(tiers["C_纯文本或非唯一对象"]),
                       "规则": "逐**整句**隔离含未验证数量/百分比/半径比/工艺根因的句子；"
                               "**不做「删掉数字留残句」**；每条仍须事实复核"},
        "变更表": f"共 {len(changes)} 条变更，明细见 变更表.jsonl",
    }
    with io.open(OUT / "变更表.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for c in changes:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ---------- 7. 撤回漂移解释 ----------
    rep["撤回的解释"] = {
        "原话": "「题面没变，所以是模型自己漂移」",
        "状态": "**撤回为未证解释**",
        "理由": ("`prompt_sha256` 是保存脚本**固定写入**的、`self_ok` 只查缺键 —— "
                 "两者都**不能证明真实题面未变**，也不能证明每条的实际视觉输入。"),
        "可核的": ("磁盘上 172 份分片、1200 条记录、纯文本连续出现在 117–166 片 —— "
                   "这是**记录层面**的事实。"),
        "不可核的": ["实际下发给客户端的题面文本", "每次调用的真实视觉输入链路",
                     "实际后端调用次数", "两个模型/会话是否混用"],
    }

    io.open(OUT / "冻结与分账.json", "w", encoding="utf-8", newline="\n").write(
        json.dumps(rep, ensure_ascii=False, indent=2) + "\n")
    print(f"\n写出 {OUT/'冻结与分账.json'}")
    print(f"  分账：{rep['分账']}")
    print(f"  下一版候选：A={rep['下一版候选']['A_结构候选']['条数']} "
          f"B={rep['下一版候选']['B_合法字段候选']['条数']} "
          f"C={rep['下一版候选']['C_Caption候选']['条数']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
