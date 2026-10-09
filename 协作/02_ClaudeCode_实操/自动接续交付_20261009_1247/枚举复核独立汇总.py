#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 交付协作者：**只读**独立复算 8 模型 × 36 图的冻结枚举描述复核。

为什么另写一份：
  核对当时（12:47–12:49 之间）主会话 `汇总枚举复核.py` 的 `extract()` 只从每条的 `答案集`
  取答案 —— 这对**嵌套格式**分片有效，对**扁平格式**分片会读出 0 行。
  当时 6 个分片里多数是扁平格式。因此本脚本**同时吃两种形态**，
  并**分别报告**两种读法各自的覆盖量，把差异摊开。

⚠️ **时点说明（重要，避免留下过时指控）**：
  主会话已在 **12:49** 修复该读法（`汇总枚举复核.py` 现同时处理扁平与嵌套，
  mtime 12:49、体积 8237→9350），并重新产出 `描述对照_枚举.json`。
  本脚本里的「主会话原样读法」**是修复前读法的静态只读模拟，属历史记录**，
  **不代表主会话当前脚本**。当前有效性以本脚本第四节的**逐格直接对比**为准。

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

    # ---- 主会话**修复前**读法的静态只读模拟（历史记录，不是当前脚本）----
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

    # ---- 第四节：与主会话**12:49 修复后**的成品逐格直接对比 ----
    MAIN_OUT = RUN / "描述对照_枚举.json"
    diff, cmp_note = [], "主会话成品缺失，无法逐格对比"
    if MAIN_OUT.exists():
        mo = json.loads(MAIN_OUT.read_text(encoding="utf-8"))
        m_dims = mo.get("逐模型逐维度") or {}
        cmp_note = f"对比对象：{MAIN_OUT.name}（mtime 见文件）"
        for m in sorted(set(per) | set(m_dims)):
            if m not in per:
                diff.append(f"仅主会话有模型 {m}")
                continue
            if m not in m_dims:
                diff.append(f"仅本脚本有模型 {m}")
                continue
            for short, _full in DIMS:
                mine = dict(per[m].get(short) or {})
                theirs = {k: v for k, v in (m_dims[m].get(short) or {}).items()}
                # 主会话可能把 0 计数省略，补齐再比
                keys = set(mine) | set(theirs)
                for k in sorted(keys):
                    a, b = mine.get(k, 0), theirs.get(k, 0)
                    if a != b:
                        diff.append(f"{m}/{short}/{k}: 本脚本 {a} vs 主会话 {b}")
    rep = {
        "分片结构": shard_shape,
        "覆盖": {
            "本脚本(兼容两种形态)读到答案单元": len(rows),
            "去重后(图,盲号)": len(uniq),
            "期望": EXPECT_FIGS * EXPECT_MODELS,
            "主会话修复前读法(静态只读模拟，历史记录)会读到": main_read_n,
            "未映射盲号": unmapped,
        },
        "与主会话成品的逐格对比": {
            "说明": cmp_note,
            "差异条数": len(diff),
            "差异明细": diff[:50],
            "结论": "一致" if not diff else "存在差异，需人工定性",
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
    L.append(f"- **主会话「修复前」读法（静态只读模拟，历史记录）会读到："
             f"{c['主会话修复前读法(静态只读模拟，历史记录)会读到']}**"
             f" —— 主会话已于 12:49 修复该读法，**此数不代表当前脚本**\n")
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
    d4 = rep["与主会话成品的逐格对比"]
    L.append("\n## 四、与主会话**成品**的逐格对比（当前有效性以此节为准）\n\n")
    L.append(f"- {d4['说明']}\n")
    L.append(f"- 差异条数：**{d4['差异条数']}** → 结论：**{d4['结论']}**\n")
    for x in d4["差异明细"]:
        L.append(f"  - {x}\n")
    L.append("\n## 五、方向三分列\n\n")
    for m in sorted(dir_stats):
        L.append(f"- `{m}`：{dict(dir_stats[m])}\n")
    (D / "枚举复核独立汇总.md").write_text("".join(L), encoding="utf-8")

    print(json.dumps(rep["分片结构"], ensure_ascii=False, indent=1))
    print(json.dumps(rep["覆盖"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
