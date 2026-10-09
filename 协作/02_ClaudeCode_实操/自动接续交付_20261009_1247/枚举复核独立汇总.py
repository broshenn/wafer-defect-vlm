#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 交付协作者：**只读**独立复算 8 模型 × 36 图的冻结枚举描述复核。

为什么另写一份：
  主会话 `汇总枚举复核.py` 的 `extract()` 只读 `d["条目"]` 并只从每条的 `答案集` 取答案
  —— 这对**分组格式**分片有效，对**扁平格式**分片会读出 0 行。
  实测 6 个分片里 5 个是扁平格式、1 个（分片2）是分组格式并另存 `扁平条目`。
  因此本脚本**同时吃两种形态**，并**分别报告**两种读法各自的覆盖量，把差异摊开。

本脚本**不写主会话目录**，只在自己目录输出；不解析自由文本，只统计枚举值。
输出：枚举复核独立汇总.json / .md
"""
from __future__ import annotations
import io, json
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
RUN = D.parent / "百炼多模型业务对照_20261009_1203"
RES = RUN / "复核结果"
KEY = RUN / "盲号对照_key.json"

DIMS = [("形态", "主要形态是否有图像支持"), ("位置", "位置是否吻合"),
        ("遗漏", "主要结构是否有遗漏"), ("断言", "是否无依据断言"),
        ("格式", "格式约定问题")]
VERDICTS = ["SUPPORTED", "PARTIAL", "CONTRADICTED", "UNCERTAIN", "MISSING",
            "NEEDS_REVIEW"]
DVERDICTS = VERDICTS + ["NO_ANSWER"]
EXPECT_FIGS, EXPECT_MODELS = 36, 8


def cell(v, allowed):
    if isinstance(v, dict):
        v = v.get("verdict")
    if v is None:
        return "MISSING"
    s = str(v).strip().upper()
    return s if s in allowed else "NEEDS_REVIEW"


def norm_answers(shard):
    """把一片统一成 [(图号, 盲号, 答案dict), ...]，两种形态都吃。"""
    out = []
    for it in shard.get("条目", []):
        sid = it.get("sample_id") or it.get("图号")
        if isinstance(it.get("答案集"), list):          # 分组格式
            for a in it["答案集"]:
                out.append((sid, a.get("盲号"), a))
        elif "盲号" in it:                              # 扁平格式
            out.append((sid, it.get("盲号"), it))
    for it in shard.get("扁平条目", []):                 # 分组格式的扁平副本
        out.append((it.get("sample_id") or it.get("图号"), it.get("盲号"), it))
    return out


def main():
    key = {k["sample_id"]: k["盲号对照"]
           for k in json.loads(KEY.read_text(encoding="utf-8"))}
    rows, shard_shape, seen = [], {}, {}
    for i in range(1, 7):
        p = RES / f"分片{i}_复核.json"
        if not p.exists():
            shard_shape[f"分片{i}"] = "缺文件"
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        grouped = any(isinstance(x.get("答案集"), list) for x in d.get("条目", []))
        shard_shape[f"分片{i}"] = {
            "条目数": len(d.get("条目", [])),
            "形态": "分组(条目含答案集)" if grouped else "扁平(条目即逐答案)",
            "有扁平条目标签": "扁平条目" in d,
        }
        for sid, lab, a in norm_answers(d):
            # 同一 (图, 盲号) 只收一次：分片2 的分组条目与其 `扁平条目` 是同一批答案的两份副本
            if (sid, lab) in seen:
                continue
            seen[(sid, lab)] = True
            rows.append((sid, lab, a))
    uniq = set(seen)

    # ---- 主会话脚本的读法（只读模拟，不执行其写文件）----
    main_read_n = 0
    for i in range(1, 7):
        p = RES / f"分片{i}_复核.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for it in d.get("条目", []):
            main_read_n += len(it.get("答案集", []) or [])

    per = defaultdict(lambda: defaultdict(Counter))
    dir_stats = defaultdict(Counter)
    unmapped = []
    for sid, lab, a in rows:
        model = (key.get(sid) or {}).get(lab)
        if model is None:
            unmapped.append((sid, lab))
            continue
        for short, full in DIMS:
            v = cell(a.get(short) if short in a else a.get(full), VERDICTS)
            per[model][short][v] += 1
        dv = a.get("方向") or {}
        app = dv.get("applicable") if isinstance(dv, dict) else None
        app = ("true" if app is True else "false" if app is False
               else "unknown" if app == "unknown" else "MISSING")
        vv = cell(dv.get("verdict") if isinstance(dv, dict) else None, DVERDICTS)
        dir_stats[model][f"applicable={app}"] += 1
        dir_stats[model][f"verdict={vv}"] += 1

    # 覆盖度：每模型每维应得 36
    coverage = {}
    for m in sorted(per):
        coverage[m] = {d: sum(c.values()) for d, c in per[m].items()}

    rep = {
        "分片结构": shard_shape,
        "覆盖": {
            "本脚本(兼容两种形态)读到答案单元": len(rows),
            "去重后(图,盲号)": len(uniq),
            "期望": EXPECT_FIGS * EXPECT_MODELS,
            "主会话脚本原样读法会读到": main_read_n,
            "未映射盲号": unmapped,
        },
        "逐模型逐维度": {m: {d: dict(c) for d, c in v.items()}
                    for m, v in per.items()},
        "方向分列": {m: dict(c) for m, c in dir_stats.items()},
        "每模型每维覆盖数": coverage,
    }
    (D / "枚举复核独立汇总.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L = ["# 冻结枚举描述复核：独立复算（CPU 交付协作者，只读）\n",
         "\n> **性质：模型复核草稿，不是人工 gold。真人两列永久留空。**\n",
         "\n## 一、分片结构（先说数据形状）\n\n",
         "| 分片 | 条目数 | 形态 | 有扁平条目标签 |\n|---|---:|---|---|\n"]
    for k, v in shard_shape.items():
        if isinstance(v, dict):
            L.append(f"| {k} | {v['条目数']} | {v['形态']} | {v['有扁平条目标签']} |\n")
        else:
            L.append(f"| {k} | — | {v} | — |\n")
    c = rep["覆盖"]
    L.append(f"\n## 二、覆盖度\n\n- 期望答案单元：{c['期望']}（36 图 × 8 模型）\n")
    L.append(f"- 本脚本兼容两种形态读到：**{c['本脚本(兼容两种形态)读到答案单元']}**，"
             f"去重后 **{c['去重后(图,盲号)']}**\n")
    L.append(f"- **主会话脚本原样读法（只读模拟）会读到：{c['主会话脚本原样读法会读到']}**\n")
    if c["未映射盲号"]:
        L.append(f"- 未映射盲号：{c['未映射盲号']}\n")
    L.append("\n## 三、逐模型逐维度（枚举计数）\n\n")
    for m in sorted(per):
        L.append(f"\n### `{m}`（每维覆盖 {coverage[m]}）\n\n")
        L.append("| 维度 | " + " | ".join(VERDICTS) + " |\n|---|" +
                 "---:|" * len(VERDICTS) + "\n")
        for short, _ in DIMS:
            cc = per[m][short]
            L.append(f"| {short} | " + " | ".join(str(cc.get(v, 0))
                                                  for v in VERDICTS) + " |\n")
    L.append("\n## 四、方向三分列\n\n")
    for m in sorted(dir_stats):
        L.append(f"- `{m}`：{dict(dir_stats[m])}\n")
    (D / "枚举复核独立汇总.md").write_text("".join(L), encoding="utf-8")

    print(json.dumps(rep["分片结构"], ensure_ascii=False, indent=1))
    print(json.dumps(rep["覆盖"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
