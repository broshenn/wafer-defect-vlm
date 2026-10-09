#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 交付协作者：对主会话百炼对照产物做**独立机械复核**。

原则：
  · 只读主会话执行目录，不写、不改、不重跑任何请求；
  · **存在 != 成功**：逐条核记录数、唯一 sample_id、图 sha256 是否命中冻结盲包；
  · 解析/内容/连接三层分开计数，不把解析失败算成内容错误，也不把解析成功算成答对；
  · 指标用公开 manifest 的类别独立复算，**不读主会话的类目结果.json 结论**；
  · 输出 JSON + MD，供交付报告引用。

输出：机械核对.json / 机械核对.md
"""
from __future__ import annotations
import io, json, re, hashlib, sys
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent
REPO = D.parent.parent.parent
RUN = REPO / "协作/02_ClaudeCode_实操/百炼多模型业务对照_20261009_1203"
DEP = REPO / "协作/02_ClaudeCode_实操/部署原型_20261009_0120"
BLIND = REPO / "协作/01_Codex_指挥/百炼多模型对照_20261009/blind36.jsonl"
GLM = REPO / "协作/04_WorkBuddy_复核/外部评测102_20261008_201229"

SCHEMA_FIELDS = ["defect_class", "morphology", "radial_zone",
                 "clock_direction", "extent_r", "caption_zh", "uncertainty"]

# 计价（元/百万 token，输入/输出）：**来源** 官方模型页 2026-10-09 读取，
# 华北2北京、低输入区间原价，登记于 `协作/01_Codex_指挥/百炼多模型对照_20261009/模型与接入核验.md`
# 与 `百炼对照.py` 的 MODELS 表。此处用同一价目**独立复算**脚本的费用算术。
# 边界：这是公开价目推算，**不等于平台实际扣款**；地域/分段/优惠/免费额度未核。
PRICE = {
    "qwen3.5-397b-a17b": (1.2, 7.2),
    "qwen3-vl-plus": (1.0, 10.0),
    "qwen3.8-max-0902": (12.0, 36.0),
    "kimi-k3": (20.0, 100.0),
}


def load_jsonl(p):
    out = []
    for l in io.open(p, encoding="utf-8"):
        if l.strip():
            out.append(json.loads(l))
    return out


THINK_RE = re.compile(r"^\s*<think\b[^>]*>.*?</think\s*>\s*", re.S)
FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?\s*```$", re.S)


def strip_think(s):
    """标准口径：只剥**空 think 头**。冻结输出契约要求纯 JSON；空 think = 未思考。"""
    return THINK_RE.sub("", s, count=1)


def strip_fence(s):
    """诊断口径：再剥**一个完整外层** ``` 围栏（长度必须恰好包住整体）。"""
    m = FENCE_RE.match(s)
    return m.group(1).strip() if m else s


def parse_strict(s):
    """严格口径：剥空 think 后整体必须是合法 JSON。返回 (对象, 层级)。"""
    if not isinstance(s, str) or not s.strip():
        return None, "empty"
    try:
        return json.loads(strip_think(s).strip()), "raw"
    except Exception:
        return None, "not_json"


def parse_diag(s):
    """围栏诊断口径：严格口径失败后，再试剥一个完整围栏。"""
    o, k = parse_strict(s)
    if o is not None or k == "empty":
        return o, k
    try:
        return json.loads(strip_fence(strip_think(s).strip())), "fenced"
    except Exception:
        return None, "not_json"


def main():
    blind = load_jsonl(BLIND)
    frozen = {b["sample_id"]: b["image_sha256"] for b in blind}
    frozen_set = set(frozen)
    man = json.loads((DEP / "样本清单.json").read_text(encoding="utf-8"))
    gold = {s["sample_id"]: s["公开类别"] for s in man["样本"]}

    report = {"冻结盲包": {"图数": len(frozen), "唯一sample_id": len(frozen_set),
                          "清单条数": len(man["样本"])}, "模型": {}}

    def add(tag, source, recs, get_raw, get_sha, usage_getter=None):
        """recs: list[dict]"""
        n = len(recs)
        sids = [r.get("sample_id") for r in recs]
        cnt = Counter(sids)
        dup = {k: v for k, v in cnt.items() if v > 1}
        sid_set = set(sids)
        # 图 sha 只对**冻结 36**内的记录比对。
        # 不在本 36 内的记录（如 GLM 的全 102 份）本来就没有可比的冻结值，
        # 把「取不到可比字段」或「本就不该参与比对」记成「sha 不符」是口径错误，
        # 不是指纹断链。故单列 `图sha256未参与比对(非本36)`，不计入 bad。
        sha_ok, sha_bad, sha_missing, sha_outside = 0, [], 0, 0
        for r in recs:
            sid = r.get("sample_id")
            if sid not in frozen_set:
                sha_outside += 1
                continue
            sh = get_sha(r)
            if sh is None:
                sha_missing += 1
            elif frozen.get(sid) == sh:
                sha_ok += 1
            else:
                sha_bad.append(sid)
        lvl = Counter()
        parsed = {}
        for r in recs:
            o, k = parse_strict(get_raw(r))
            lvl[k] += 1
            parsed[r.get("sample_id")] = o
        strict_ok = lvl["raw"]
        # schema（只在**严格口径**解析出的对象上判）
        schema_ok, schema_bad = 0, []
        for sid, o in parsed.items():
            if isinstance(o, dict) and all(f in o for f in SCHEMA_FIELDS):
                schema_ok += 1
            elif isinstance(o, dict):
                schema_bad.append(sid)
        # 类别准确率（严格口径：解析不出或字段缺失即算错；分母=冻结36）
        correct, errs = 0, []
        for r in recs:
            sid = r.get("sample_id")
            o = parsed.get(sid)
            g = gold.get(sid)
            if o is None or "defect_class" not in o:
                errs.append({"sample_id": sid, "gold": g, "pred": None,
                             "why": "未解析/缺字段"})
                continue
            p = o["defect_class"]
            if p == g:
                correct += 1
            else:
                errs.append({"sample_id": sid, "gold": g, "pred": p, "why": "错类"})
        acc = correct / len(frozen_set) if frozen_set else None
        # 围栏诊断口径：统一对全部候选实施，**不因失败而只挑好的剥**
        dlvl = Counter()
        dcorrect = 0
        for r in recs:
            sid = r.get("sample_id")
            o, k = parse_diag(get_raw(r))
            dlvl[k] += 1
            if isinstance(o, dict) and o.get("defect_class") == gold.get(sid):
                dcorrect += 1
        dacc = dcorrect / len(frozen_set) if frozen_set else None
        entry = {
            "来源": source, "记录数": n, "唯一sample_id": len(sid_set),
            "冻结36缺图": sorted(frozen_set - sid_set),
            "不在冻结36的记录": sorted(sid_set - frozen_set),
            "与冻结36交集": len(sid_set & frozen_set),
            "重复": dup,
            "图sha256命中": sha_ok, "图sha256不符": sha_bad, "图sha256缺失": sha_missing,
            "图sha256未参与比对(非本36)": sha_outside,
            "解析层级": dict(lvl), "严格JSON可解析": strict_ok,
            "诊断层级": dict(dlvl), "围栏诊断可解析": sum(
                v for k, v in dlvl.items() if k in ("raw", "fenced")),
            "完整schema": schema_ok, "schema不合": schema_bad,
            "类别_严格Acc": None if acc is None else round(acc, 4),
            "类别_严格_正确数": correct,
            "类别_围栏诊断Acc": None if dacc is None else round(dacc, 4),
            "类别_错误明细": errs,
        }
        if usage_getter:
            pin = pout = 0
            nok = 0
            think_tok = 0
            for r in recs:
                u = usage_getter(r)
                if not u:
                    continue
                nok += 1
                pin += u.get("prompt_tokens", 0) or 0
                pout += u.get("completion_tokens", 0) or 0
                think_tok += ((u.get("completion_tokens_details") or {})
                              .get("reasoning_tokens", 0) or 0)
            pr = PRICE.get(tag)
            cost = None
            if pr:
                cost = round(pin / 1e6 * pr[0] + pout / 1e6 * pr[1], 4)
            entry["usage"] = {"有条数": nok, "输入token": pin, "输出token": pout,
                              "思考token": think_tok,
                              "复算费用_元": cost,
                              "计价(元/百万token)": pr}
        report["模型"][tag] = entry

    # ---- 百炼四个：原答_API/*.jsonl ----
    for fn in sorted((RUN / "原答_API").glob("*_raw.jsonl")):
        tag = fn.stem.replace("_raw", "")
        add(tag, f"百炼API · {fn.name}", load_jsonl(fn),
            lambda r: r.get("content"), lambda r: r.get("image_sha256"),
            lambda r: (r.get("usage") or None) if r.get("status") == "ok" else None)

    # ---- GPU：Base / L ----
    for tag, fn in (("Base", "Base_raw.jsonl"), ("L-N3072-3407", "L-N3072-3407_raw.jsonl")):
        add(tag, f"租机GPU · {fn}", load_jsonl(RUN / "原答_GPU" / fn),
            lambda r: r.get("raw"), lambda r: r.get("image_sha256"))

    # ---- D：部署服务 HTTP 回放 ----
    # 图 sha 是**嵌套**字段：response.image.image_sha256。
    # 首版只取顶层 image_sha256，取到 36 个 None → 误报「D 图 hash 缺 36」；
    # 这是取字段口径问题，不是哈希链断裂（修正后 36/36 命中）。
    add("D-N3072-3407", "部署服务HTTP · 性能与自测/http_replay.jsonl",
        load_jsonl(DEP / "性能与自测/http_replay.jsonl"),
        lambda r: (r.get("response") or {}).get("raw_answer"),
        lambda r: ((r.get("response") or {}).get("image") or {}).get("image_sha256"))

    # ---- GLM：外部 102 的子集 ----
    glm_recs = []
    for d in load_jsonl(GLM / "原答清单.jsonl"):
        f = GLM / "raw" / (d["item_id"] + ".txt")
        d["_raw"] = f.read_text(encoding="utf-8") if f.exists() else None
        glm_recs.append(d)
    add("GLM-WorkBuddy", "WorkBuddy客户端 · 外部评测102（全102，非36子集）",
        glm_recs, lambda r: r.get("_raw"), lambda r: r.get("image_sha256"))

    # ---- 请求日志费用总账（独立复算）----
    log = load_jsonl(RUN / "请求日志.jsonl")
    lvl = Counter()
    cost_sum = {}
    toks = Counter()
    for r in log:
        st = r.get("status")
        lvl[st] += 1
        if st == "ok":
            u = r.get("usage") or {}
            m = r.get("model") or r.get("model_id")
            pin = u.get("prompt_tokens", 0) or 0
            pout = u.get("completion_tokens", 0) or 0
            toks["输入"] += pin
            toks["输出"] += pout
            toks["思考"] += ((u.get("completion_tokens_details") or {})
                            .get("reasoning_tokens", 0) or 0)
            if m in PRICE:
                c = pin / 1e6 * PRICE[m][0] + pout / 1e6 * PRICE[m][1]
                cost_sum[m] = round(cost_sum.get(m, 0) + c, 4)
    report["请求日志"] = {
        "文件": "请求日志.jsonl", "条数": len(log),
        "状态分布": dict(lvl),
        "token合计": dict(toks),
        "按模型复算费用_元": cost_sum,
        "复算总费用_元": round(sum(cost_sum.values()), 4),
    }

    (D / "机械核对.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- 生成 MD ----
    L = []
    L.append("# 机械核对（CPU 交付协作者独立复算）\n")
    L.append(f"- 冻结盲包：{report['冻结盲包']['图数']} 图 / "
             f"{report['冻结盲包']['唯一sample_id']} 唯一 sample_id\n")
    L.append("\n## 逐模型\n")
    L.append("| 模型 | 记录数 | 唯一id | 图中sha命中 | 严格JSON | 完整schema | "
             "类别严格Acc | 围栏诊断Acc | 复算费用(元) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for tag, e in report["模型"].items():
        u = e.get("usage") or {}
        L.append(f"| `{tag}` | {e['记录数']} | {e['唯一sample_id']} | "
                 f"{e['图sha256命中']} | {e['严格JSON可解析']} | {e['完整schema']} | "
                 f"{e['类别_严格Acc']} | {e['类别_围栏诊断Acc']} | "
                 f"{u.get('复算费用_元')} |\n")
    L.append("\n> 「图中sha命中」的分母是**本 36 图内参与比对的记录**；"
             "非本轮 36 图的记录（GLM 有 66 条）不参与 sha 比对，见下表，**不是 sha 不符**。\n")
    L.append("\n## 异常项（逐模型机械检查）\n")
    for tag, e in report["模型"].items():
        flags = []
        if e["冻结36缺图"]:
            flags.append(f"**冻结36缺图 {len(e['冻结36缺图'])} 条**")
        if e["不在冻结36的记录"]:
            flags.append(f"额外记录 {len(e['不在冻结36的记录'])} 条（非本轮36图）")
        if e["重复"]:
            flags.append(f"重复 sample_id {e['重复']}")
        if e["图sha256不符"]:
            flags.append(f"**图sha256不符 {e['图sha256不符']}**")
        if e["图sha256缺失"]:
            flags.append(f"图sha256缺失 {e['图sha256缺失']} 条（限本 36 图内）")
        if e.get("图sha256未参与比对(非本36)"):
            flags.append(f"非本轮 36 图、不参与 sha 比对 {e['图sha256未参与比对(非本36)']} 条"
                         f"（**不是 sha 不符**）")
        if e["schema不合"]:
            flags.append(f"schema不合 {len(e['schema不合'])} 条")
        L.append(f"- `{tag}`：{'；'.join(flags) if flags else '无'}\n")
    L.append("\n## 请求日志总账\n")
    rl = report["请求日志"]
    L.append(f"- 条数 {rl['条数']}，状态分布 {rl['状态分布']}\n")
    L.append(f"- token 合计 {rl['token合计']}（思考占输出 "
             f"{(rl['token合计'].get('思考',0)/max(1,rl['token合计'].get('输出',1))):.1%}）\n")
    L.append(f"- **复算总费用 ≈ {rl['复算总费用_元']} 元**（分模型 {rl['按模型复算费用_元']}）\n")
    (D / "机械核对.md").write_text("".join(L), encoding="utf-8")
    print("写出 机械核对.json / 机械核对.md")
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk in ("记录数", "唯一sample_id", "图sha256命中",
                                    "严格JSON可解析", "完整schema", "类别_严格Acc",
                                    "类别_围栏诊断Acc")}
                      for k, v in report["模型"].items()},
                     ensure_ascii=False, indent=1))
    print("请求日志:", json.dumps(report["请求日志"], ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
